# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT alternating-attention Aggregator with the Plann3r goal token added.

Upstream VGGT code. Plann3r adds enable_nav_tokens: when anchor_patch_idx marks
the goal patch on a frame, a learned anchor embedding is added to that patch
token and a learned goal embedding, weighted by a Gaussian around the goal
patch, is added to every patch token of that frame. With nav tokens off or no
anchor it behaves as upstream. VGGTNav in vggt/models/vggt_nav.py turns it on.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_nav_tokens=False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.enable_nav_tokens = enable_nav_tokens
        if self.enable_nav_tokens:
            self.anchor_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.goal_gaussian_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.goal_gaussian_log_sigma = nn.Parameter(torch.log(torch.tensor(2.0)))
        else:
            self.register_parameter("anchor_embed", None)
            self.register_parameter("goal_gaussian_embed", None)
            self.register_parameter("goal_gaussian_log_sigma", None)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        if self.enable_nav_tokens:
            nn.init.normal_(self.anchor_embed, std=1e-6)
            nn.init.normal_(self.goal_gaussian_embed, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        anchor_patch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        tokens = self._apply_nav_token_conditioning(
            tokens=tokens,
            B=B,
            S=S,
            anchor_patch_idx=anchor_patch_idx,
            patch_h=patch_h,
            patch_w=patch_w,
        )

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _apply_nav_token_conditioning(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        anchor_patch_idx: Optional[torch.Tensor],
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """
        Applies goal-conditioned embeddings.

                This method is opt-in and fully backward-compatible:
                - If nav tokens are disabled, tokens are returned unchanged.
                - If anchor_patch_idx is None, tokens are returned unchanged.
                - Anchor placement is controlled solely by anchor_patch_idx:
                  any frame with a non-negative anchor index receives goal conditioning.
        """
        if not self.enable_nav_tokens:
            return tokens

        device = tokens.device
        token_dtype = tokens.dtype
        patch_count = tokens.shape[1] - self.patch_start_idx
        if patch_h * patch_w != patch_count:
            raise ValueError(
                f"Patch grid mismatch: patch_h*patch_w={patch_h * patch_w}, patch_count={patch_count}"
            )
        if anchor_patch_idx is None:
            return tokens

        anchor_patch_idx = anchor_patch_idx.to(device=device)
        if anchor_patch_idx.ndim == 1:
            if anchor_patch_idx.shape[0] == B:
                anchor_map = torch.full((B, S), -1, dtype=torch.long, device=device)
                anchor_map[:, 0] = anchor_patch_idx.long()
            elif anchor_patch_idx.shape[0] == B * S:
                anchor_map = anchor_patch_idx.long().view(B, S)
            else:
                raise ValueError(
                    f"Expected anchor_patch_idx length {B} or {B*S}, got {anchor_patch_idx.shape[0]}"
                )
        elif anchor_patch_idx.ndim == 2:
            if anchor_patch_idx.shape != (B, S):
                raise ValueError(
                    f"Expected anchor_patch_idx shape {(B, S)}, got {tuple(anchor_patch_idx.shape)}"
                )
            anchor_map = anchor_patch_idx.long()
        else:
            raise ValueError(
                f"Expected anchor_patch_idx ndim in [1,2], got {anchor_patch_idx.ndim}"
            )

        anchor_flat = anchor_map.reshape(B * S)
        valid_anchor = anchor_flat >= 0
        if not valid_anchor.any():
            return tokens

        if (anchor_flat[valid_anchor] >= patch_count).any():
            max_idx = int(anchor_flat[valid_anchor].max().item())
            raise ValueError(
                f"anchor_patch_idx out of range: max={max_idx}, patch_count={patch_count}"
            )

        rows = torch.nonzero(valid_anchor, as_tuple=False).squeeze(1)
        anchor_indices = anchor_flat[valid_anchor]
        cols = self.patch_start_idx + anchor_indices
        anchor = self.anchor_embed.to(dtype=token_dtype).view(1, -1).expand(rows.shape[0], -1)
        tokens[rows, cols, :] = tokens[rows, cols, :] + anchor

        # Dense goal conditioning: inject a Gaussian map centered at goal anchor.
        gaussian_map = self._build_goal_gaussian_map(
            anchor_indices=anchor_indices,
            patch_h=patch_h,
            patch_w=patch_w,
            dtype=token_dtype,
            device=device,
        )
        gaussian_embed = self.goal_gaussian_embed.to(dtype=token_dtype)
        tokens[rows, self.patch_start_idx:, :] = (
            tokens[rows, self.patch_start_idx:, :]
            + gaussian_map.unsqueeze(-1) * gaussian_embed
        )

        return tokens

    def _build_goal_gaussian_map(
        self,
        anchor_indices: torch.Tensor,
        patch_h: int,
        patch_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        patch_count = patch_h * patch_w
        if anchor_indices.numel() == 0:
            return torch.zeros((0, patch_count), dtype=dtype, device=device)

        ys = torch.arange(patch_h, dtype=dtype, device=device)
        xs = torch.arange(patch_w, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = grid_y.reshape(1, patch_count)
        grid_x = grid_x.reshape(1, patch_count)

        anchor_x = (anchor_indices % patch_w).to(dtype=dtype).unsqueeze(1)
        anchor_y = (anchor_indices // patch_w).to(dtype=dtype).unsqueeze(1)
        dist_sq = (grid_x - anchor_x) ** 2 + (grid_y - anchor_y) ** 2

        sigma = torch.exp(self.goal_gaussian_log_sigma).to(dtype=dtype, device=device)
        sigma = torch.clamp(sigma, min=0.5, max=float(max(patch_h, patch_w)))
        gaussian = torch.exp(-0.5 * dist_sq / (sigma * sigma + 1e-6))
        return gaussian

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
