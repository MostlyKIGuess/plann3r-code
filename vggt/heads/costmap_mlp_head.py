"""Plann3r costmap head that maps each patch token to one cost value.

A LayerNorm, an MLP and a linear layer run on the patch tokens of one
Aggregator layer, giving one value per 14x14 patch (16x16 for a 224 input)
after the output activation, gelu in the released checkpoint. VGGTNav in
vggt/models/vggt_nav.py builds it as costmap_head.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.heads.head_act import inverse_log_transform
from vggt.layers import Mlp


class CostmapMLPHead(nn.Module):
    """
    MLP head for dense costmap prediction from patch tokens.

    The output contract is a single regression map with shape [B, S, H, W, C].
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "gelu",
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        layer_idx: int = -1,
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1")

        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.layer_idx = layer_idx
        self.down_ratio = down_ratio

        self.norm = nn.LayerNorm(dim_in)
        hidden_dim = max(1, int(dim_in * mlp_ratio))
        self.mlp = Mlp(in_features=dim_in, hidden_features=hidden_dim, out_features=dim_in, drop=drop)
        self.proj = nn.Linear(dim_in, output_dim)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        B, S, _, _, _ = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        assert frames_chunk_size > 0

        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_preds = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
            )
            all_preds.append(chunk_preds)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        tokens = aggregated_tokens_list[self.layer_idx]
        if frames_start_idx is not None and frames_end_idx is not None:
            tokens = tokens[:, frames_start_idx:frames_end_idx]

        patch_tokens = tokens[:, :, patch_start_idx:]
        patch_tokens = self.norm(patch_tokens)
        patch_tokens = self.mlp(patch_tokens)
        patch_tokens = self.proj(patch_tokens)

        out = patch_tokens.view(B * S, patch_h, patch_w, self.output_dim).permute(0, 3, 1, 2)

        if self.down_ratio != 1:
            target_h = max(1, int(round(patch_h / self.down_ratio)))
            target_w = max(1, int(round(patch_w / self.down_ratio)))
            out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=True)

        preds = self._activate(out)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(B, S, *preds.shape[1:])
        return preds

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return x
        if self.activation == "exp":
            return torch.exp(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "inv_log":
            return inverse_log_transform(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported activation: {self.activation}")
