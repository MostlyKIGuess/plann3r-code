# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGTNav, the Plann3r model that predicts a goal costmap for the query frame.

It is VGGT with the goal token turned on in the Aggregator, so the goal pixel
is marked on its frame by anchor_patch_idx, and a CostmapMLPHead that decodes
the query frame's tokens into a costmap. An optional frozen DPT point head gives
the auxiliary pointmap used in training. Built by training/config/nav_costmap.yaml
for training and by libs/experiments/vggtnav_inference.py for navigation, the
propagation maps and MARD.
"""

import logging
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.heads.camera_head import CameraHead
from vggt.heads.costmap_mlp_head import CostmapMLPHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.aggregator import Aggregator


logger = logging.getLogger(__name__)


class VGGTNav(nn.Module, PyTorchModelHubMixin):
    """
    VGGT variant for query-conditioned navigation costmap prediction.

    Key differences from the base VGGT architecture:
    - Aggregator supports optional query/submap type embeddings and goal-anchor embedding.
    - A dedicated MLP head predicts query-frame costmap only.
    - Optional query-frame point head prediction for auxiliary pointmap supervision.
    - Point head is frozen by default; only costmap head remains trainable.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        init_costmap_from_depth: bool = False,
        costmap_activation: str = "gelu",
        costmap_head_type: str = "mlp",
        costmap_mlp_ratio: float = 4.0,
        costmap_mlp_drop: float = 0.0,
        costmap_mlp_layer_idx: int = -1,
        enable_point_aux: bool = False,
        train_point_head: bool = False,
        enable_camera: bool = False,
        enable_track: bool = False,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_nav_tokens=True,
        )

        # costmap_head_type stays a parameter because checkpoints and configs
        # record it, but the MLP head is the only released costmap head.
        self.costmap_activation = costmap_activation
        self.costmap_head_type = str(costmap_head_type).lower()
        if self.costmap_head_type != "mlp":
            raise ValueError(
                f"Unsupported costmap_head_type: {costmap_head_type} (only 'mlp' is supported)"
            )
        self.costmap_head = CostmapMLPHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            output_dim=1,
            activation=self.costmap_activation,
            mlp_ratio=costmap_mlp_ratio,
            drop=costmap_mlp_drop,
            layer_idx=costmap_mlp_layer_idx,
        )

        # Depth-head initialization copies DPT weights, so it cannot apply to
        # the MLP head. It stays a parameter for older configs that set it.
        if init_costmap_from_depth:
            logger.warning(
                "init_costmap_from_depth ignored for costmap_head_type=%s",
                self.costmap_head_type,
            )

        self.train_point_head = bool(train_point_head)
        self.point_head = None
        if enable_point_aux:
            self.point_head = DPTHead( 
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            self._freeze_point_head()

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def _freeze_point_head(self) -> None:
        if self.point_head is None or self.train_point_head:
            return
        self.point_head.eval()
        for param in self.point_head.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        out = super().train(mode)
        # Keep point head frozen unless explicitly enabled for training.
        self._freeze_point_head()
        return out

    @staticmethod
    def _checkpoint_has_point_head_weights(state_dict) -> bool:
        if not isinstance(state_dict, dict):
            return False

        for key in state_dict.keys():
            if not isinstance(key, str):
                continue
            if key.startswith("point_head.") or key.startswith("module.point_head."):
                return True

        return False

    @staticmethod
    def _checkpoint_has_camera_head_weights(state_dict) -> bool:
        if not isinstance(state_dict, dict):
            return False

        for key in state_dict.keys():
            if not isinstance(key, str):
                continue
            if key.startswith("camera_head.") or key.startswith("module.camera_head."):
                return True

        return False

    @staticmethod
    def _checkpoint_has_track_head_weights(state_dict) -> bool:
        if not isinstance(state_dict, dict):
            return False

        for key in state_dict.keys():
            if not isinstance(key, str):
                continue
            if key.startswith("track_head.") or key.startswith("module.track_head."):
                return True

        return False

    def load_state_dict(self, state_dict, strict: bool = True):
        checkpoint_has_point_head = self._checkpoint_has_point_head_weights(state_dict)
        checkpoint_has_camera_head = self._checkpoint_has_camera_head_weights(state_dict)
        checkpoint_has_track_head = self._checkpoint_has_track_head_weights(state_dict)
        load_result = super().load_state_dict(state_dict, strict=strict)

        if self.point_head is not None:
            if checkpoint_has_point_head:
                point_head_missing = [
                    key for key in getattr(load_result, "missing_keys", []) if key.startswith("point_head.")
                ]
                if len(point_head_missing) == 0:
                    logger.info("Loaded point_head weights from checkpoint")
                else:
                    logger.warning(
                        "Checkpoint had point_head weights but some point_head keys are missing after load: %s",
                        point_head_missing,
                    )
            else:
                logger.warning("Checkpoint does not contain point_head weights")

            self._freeze_point_head()

        if self.camera_head is not None:
            if checkpoint_has_camera_head:
                camera_head_missing = [
                    key for key in getattr(load_result, "missing_keys", []) if key.startswith("camera_head.")
                ]
                if len(camera_head_missing) == 0:
                    logger.info("Loaded camera_head weights from checkpoint")
                else:
                    logger.warning(
                        "Checkpoint had camera_head weights but some camera_head keys are missing after load: %s",
                        camera_head_missing,
                    )
            else:
                logger.warning("Checkpoint does not contain camera_head weights")

        if self.track_head is not None:
            if checkpoint_has_track_head:
                track_head_missing = [
                    key for key in getattr(load_result, "missing_keys", []) if key.startswith("track_head.")
                ]
                if len(track_head_missing) == 0:
                    logger.info("Loaded track_head weights from checkpoint")
                else:
                    logger.warning(
                        "Checkpoint had track_head weights but some track_head keys are missing after load: %s",
                        track_head_missing,
                    )
            else:
                logger.warning("Checkpoint does not contain track_head weights")

        return load_result

    def _resolve_query_frames(
        self,
        query_frame_idx: Optional[Union[int, str, Sequence[int], torch.Tensor]],
        num_frames: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Normalize `query_frame_idx` to a 1-D LongTensor of frame slots.

        Returns None when the selection covers every frame in slot order, which
        lets the caller skip indexing altogether.
        """
        if query_frame_idx is None:
            return torch.zeros(1, dtype=torch.long, device=device)

        if isinstance(query_frame_idx, str):
            if query_frame_idx != "all":
                raise ValueError(f"query_frame_idx string must be 'all', got {query_frame_idx!r}")
            return None

        if isinstance(query_frame_idx, torch.Tensor):
            sel = query_frame_idx.to(device=device, dtype=torch.long).reshape(-1)
        else:
            # Accept any iterable of indices, or any integer scalar (including numpy
            # ints, which are not instances of `int`).
            try:
                values = [int(v) for v in query_frame_idx]
            except TypeError:
                values = [int(query_frame_idx)]
            sel = torch.tensor(values, dtype=torch.long, device=device)

        if sel.numel() == 0:
            raise ValueError("query_frame_idx selected no frames")
        if int(sel.min()) < 0 or int(sel.max()) >= num_frames:
            raise ValueError(
                f"query_frame_idx out of range: got [{int(sel.min())}, {int(sel.max())}] "
                f"for a sequence of {num_frames} frames"
            )

        if sel.numel() == num_frames and bool(
            torch.equal(sel, torch.arange(num_frames, dtype=torch.long, device=device))
        ):
            return None

        return sel

    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        anchor_patch_idx: Optional[torch.Tensor] = None,
        query_frame_idx: Optional[Union[int, str, Sequence[int], torch.Tensor]] = None,
    ) -> dict:
        """
        Args:
            images: [S, 3, H, W] or [B, S, 3, H, W] in [0, 1].
            query_points: Kept for API compatibility (unused in this class).
            anchor_patch_idx:
                Optional goal-anchor patch index metadata (can be on query or submap frame).
                Supported shapes: [B], [B, S], or flattened [B*S].
                Prefer the [B, S] form: the [B] form hard-binds the anchor to slot 0.
            query_frame_idx:
                Which frames the costmap (and point aux) head runs on. The aggregator
                always processes the whole sequence; this only selects which frames'
                tokens are decoded.
                    None (default) -> slot 0 only, identical to the historical behaviour
                    int            -> that one slot
                    list / 1-D tensor -> those slots, **in the order given**
                    "all"          -> every slot, in slot order
                **The output K-order follows the selection order, not sorted order.**
                Callers building a dict must zip against their own selection list.

                Note that decoding a non-zero slot is an approximation: slot 0 receives
                a distinct camera/register token (see `slice_expand_and_flatten`) and the
                costmap head was only ever trained on slot-0 tokens.

        Returns:
            dict with:
                - costmap: [B, K, H, W, 1]  (K = number of selected frames; 1 by default)
                - world_points: [B, K, H, W, 3] (if point aux is enabled)
                - pose_enc: [B, S, 9] (if camera head is enabled)
                - track: [B, S, N, 2] (if track head is enabled and query_points given)
                - images: [B, S, 3, H, W] (in eval mode only)
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            anchor_patch_idx=anchor_patch_idx,
        )

        sel = self._resolve_query_frames(query_frame_idx, images.shape[1], images.device)
        if sel is None:
            # Whole sequence: pass through rather than copying all 24 token tensors.
            query_tokens_list: List[torch.Tensor] = aggregated_tokens_list
            query_images = images
        else:
            # `images` and the token tensors must be sliced with the same index, since
            # the heads derive their frame count from `images.shape[1]`.
            query_tokens_list = [tokens.index_select(1, sel) for tokens in aggregated_tokens_list]
            query_images = images.index_select(1, sel)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            # The MLP head is cheap enough that chunking is pure overhead, so
            # decode all K frames in one call.
            costmap = self.costmap_head(
                query_tokens_list,
                images=query_images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=query_images.shape[1],
            )
            predictions["costmap"] = costmap

            if self.point_head is not None:
                world_points, _ = self.point_head(
                    query_tokens_list,
                    images=query_images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = world_points

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions
