# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training dataset for the Plann3r planner, read from subtrajectory folders.

Each sample is a query frame followed by submap frames from one subtrajectory
(VGGTNAV_DIR/<scene>/subtraj_<start>_<end>[_goal_<k>]/), with the goal frame
kept in the submap when possible. It returns the images, the ground-truth query
costmap, the goal anchor patch index on the goal frame (-1 on other frames), and
optionally the query pointmap for the auxiliary point loss. Scenes are split
into train and val. training/config/nav_costmap.yaml builds it as
data.datasets.vggtnav.VGGTNavDataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SubtrajRecord:
    scene_name: str
    subtraj_name: str
    subtraj_dir: Path
    frame_ids: Tuple[int, ...]
    query_idx: int
    goal_frame_idx: int
    goal_pixel_uv: Tuple[int, int]
    image_paths: Dict[int, Path]
    costmap_paths: Dict[int, Path]
    pointmap_paths: Dict[int, Path]
    base_subtraj_key: str
    costmap_mode: str

    @property
    def sequence_name(self) -> str:
        return f"vggtnav_{self.scene_name}_{self.subtraj_name}"


class VGGTNavDataset(BaseDataset):
    """Dataset loader for generated VGGTNav subtrajectory folders.

        Expected structure for each sample:
            VGGTNAV_DIR/<scene>/subtraj_<start>_<end>[_goal_<k>]/
                images_fov90/
                gt_costmaps/
                pointmaps_fov90/
                subtraj_info.json
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        VGGTNAV_DIR: Optional[str] = None,
        load_query_pointmap: bool = True,
        ensure_goal_frame_in_context: bool = True,
        len_train: int = 100000,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        val_scenes: Optional[Sequence[str]] = None,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = bool(common_conf.debug)
        self.training = bool(common_conf.training)
        self.inside_random = bool(common_conf.inside_random)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.get_nearby = bool(common_conf.get_nearby)

        if VGGTNAV_DIR is None:
            raise ValueError("VGGTNAV_DIR must be provided")

        self.root_dir = Path(VGGTNAV_DIR)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"VGGTNAV_DIR does not exist: {self.root_dir}")

        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self.load_query_pointmap = bool(load_query_pointmap)
        self.ensure_goal_frame_in_context = bool(ensure_goal_frame_in_context)
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.val_scenes = set(val_scenes) if val_scenes else None

        all_records = self._discover_subtrajectories(self.root_dir)
        if len(all_records) == 0:
            raise RuntimeError(f"No subtrajectories found under: {self.root_dir}")

        self.sequence_list = self._select_split(all_records)
        if len(self.sequence_list) == 0:
            raise RuntimeError(
                f"Split '{self.split}' has no samples under {self.root_dir}. "
                "Adjust val_ratio or verify data layout."
            )

        self.sequence_map = {record.sequence_name: record for record in self.sequence_list}
        self.sequence_list_len = len(self.sequence_list)

        if self.split == "train":
            if self.inside_random:
                self.len_train = int(len_train)
            else:
                self.len_train = self.sequence_list_len
        else:
            # For val/test, iterate each available sequence once per epoch.
            self.len_train = self.sequence_list_len

        status = "Training" if self.training else "Testing"
        logger.info("%s: VGGTNav root: %s", status, self.root_dir)
        if hasattr(self, "scene_order"):
            logger.info(
                "%s: VGGTNav scene split (total=%d, train=%d, val=%d)",
                status,
                len(self.scene_order),
                len(self.train_scenes),
                len(self.val_scenes),
            )
            logger.info("%s: Train scenes: %s", status, self.train_scenes)
            logger.info("%s: Val scenes: %s", status, self.val_scenes)
        logger.info("%s: VGGTNav split '%s' size: %d", status, self.split, self.sequence_list_len)
        logger.info("%s: VGGTNav dataset length: %d", status, len(self))

    @staticmethod
    def _collect_frame_file_map(folder: Path, exts: Sequence[str]) -> Dict[int, Path]:
        frame_map: Dict[int, Path] = {}
        if not folder.exists():
            return frame_map

        for ext in exts:
            for path in folder.glob(f"*{ext}"):
                if not path.stem.isdigit():
                    continue
                frame_map[int(path.stem)] = path
        return frame_map

    def _discover_subtrajectories(self, root_dir: Path) -> List[_SubtrajRecord]:
        records: List[_SubtrajRecord] = []

        for scene_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
            for subtraj_dir in sorted(scene_dir.glob("subtraj_*")):
                info_path = subtraj_dir / "subtraj_info.json"
                image_dir = subtraj_dir / "images_fov90"
                costmap_dir = subtraj_dir / "gt_costmaps"
                pointmap_dir = subtraj_dir / "pointmaps_fov90"

                if not info_path.exists() or not image_dir.exists() or not costmap_dir.exists():
                    continue

                with info_path.open("r", encoding="utf-8") as f:
                    info = json.load(f)

                frame_ids = tuple(int(fid) for fid in info.get("frame_ids", []))
                query_idx = int(info.get("query_idx"))
                goal_frame_idx = int(info.get("goal_frame_idx", query_idx))
                goal_uv = info.get("goal_pixel_uv", [0, 0])
                goal_u, goal_v = int(goal_uv[0]), int(goal_uv[1])

                costmap_mode = str(info.get("costmap_mode", "window")).lower().strip()
                if costmap_mode not in {"window", "query"}:
                    logger.warning(
                        "Unknown costmap_mode '%s' in %s; defaulting to 'window'",
                        costmap_mode,
                        subtraj_dir,
                    )
                    costmap_mode = "window"

                base_subtraj_key = None
                try:
                    start_idx = info.get("start_idx", None)
                    end_idx = info.get("end_idx", None)
                    if start_idx is not None and end_idx is not None:
                        base_subtraj_key = f"subtraj_{int(start_idx)}_{int(end_idx)}"
                except (TypeError, ValueError):
                    base_subtraj_key = None

                if base_subtraj_key is None:
                    match = re.match(r"^subtraj_(\d+)_(\d+)(?:_goal_(\d+))?$", subtraj_dir.name)
                    if match:
                        base_subtraj_key = f"subtraj_{match.group(1)}_{match.group(2)}"
                    else:
                        base_subtraj_key = subtraj_dir.name

                image_paths = self._collect_frame_file_map(image_dir, (".jpg", ".png", ".jpeg"))
                costmap_paths = self._collect_frame_file_map(costmap_dir, (".npy",))
                pointmap_paths = self._collect_frame_file_map(pointmap_dir, (".npy",))

                if query_idx not in image_paths or query_idx not in costmap_paths:
                    continue

                if len(frame_ids) == 0:
                    # Fallback if metadata is missing frame_ids.
                    if costmap_mode == "query":
                        frame_ids = tuple(sorted(image_paths.keys()))
                    else:
                        frame_ids = tuple(sorted(set(image_paths.keys()) & set(costmap_paths.keys())))

                if costmap_mode == "query":
                    frame_ids = tuple(fid for fid in frame_ids if fid in image_paths)
                else:
                    frame_ids = tuple(fid for fid in frame_ids if fid in image_paths and fid in costmap_paths)
                if len(frame_ids) == 0:
                    continue

                if goal_frame_idx not in frame_ids:
                    logger.warning(
                        "goal_frame_idx=%d missing in %s; falling back to query_idx=%d",
                        goal_frame_idx,
                        subtraj_dir,
                        query_idx,
                    )
                    goal_frame_idx = query_idx

                records.append(
                    _SubtrajRecord(
                        scene_name=scene_dir.name,
                        subtraj_name=subtraj_dir.name,
                        subtraj_dir=subtraj_dir,
                        frame_ids=frame_ids,
                        query_idx=query_idx,
                        goal_frame_idx=goal_frame_idx,
                        goal_pixel_uv=(goal_u, goal_v),
                        image_paths=image_paths,
                        costmap_paths=costmap_paths,
                        pointmap_paths=pointmap_paths,
                        base_subtraj_key=base_subtraj_key,
                        costmap_mode=costmap_mode,
                    )
                )

        return records

    def _split_bucket(self, key: str) -> float:
        payload = f"{self.split_seed}:{key}".encode("utf-8")
        digest = hashlib.md5(payload).hexdigest()
        # Map hash to [0, 1).
        return int(digest[:8], 16) / float(0x100000000)

    def _select_split(self, records: List[_SubtrajRecord]) -> List[_SubtrajRecord]:
        scene_names = sorted({record.scene_name for record in records})

        # Explicit, deterministic scene split takes precedence over the hash-based ratio split.
        if self.val_scenes is not None:
            val_scenes = {s for s in scene_names if s in self.val_scenes}
            if len(val_scenes) == 0:
                logger.warning(
                    "None of val_scenes=%s found among discovered scenes=%s",
                    sorted(self.val_scenes),
                    scene_names,
                )
            train_scenes = set(scene_names) - val_scenes
            scene_order = scene_names
        elif self.val_ratio <= 0:
            if self.split == "train":
                return records
            return []
        else:
            scene_scores = {scene: self._split_bucket(scene) for scene in scene_names}
            scene_order = sorted(scene_names, key=lambda name: scene_scores[name])

            if self.val_ratio >= 1:
                val_scenes = set(scene_order)
            else:
                num_val = max(1, int(math.ceil(len(scene_order) * self.val_ratio)))
                val_scenes = set(scene_order[:num_val])

            train_scenes = set(scene_order) - val_scenes
        self.scene_order = scene_order
        self.val_scenes = sorted(val_scenes)
        self.train_scenes = sorted(train_scenes)

        val_selected: List[_SubtrajRecord] = []
        train_selected: List[_SubtrajRecord] = []

        for record in records:
            if record.scene_name in val_scenes:
                val_selected.append(record)
            else:
                train_selected.append(record)

        if self.split in {"val", "test"}:
            return val_selected if len(val_selected) > 0 else records

        return train_selected if len(train_selected) > 0 else records

    def _sample_frame_ids(self, record: _SubtrajRecord, img_per_seq: int) -> List[int]:
        img_per_seq = max(1, int(img_per_seq))

        # Query frame must be first for the VGGTNav forward path.
        query_id = record.query_idx
        others = [fid for fid in record.frame_ids if fid != query_id]
        num_submap = max(0, img_per_seq - 1)

        if num_submap == 0:
            return [query_id]

        if len(others) == 0:
            return [query_id] + [query_id] * num_submap

        if self.get_nearby:
            # Frame ids in subtrajectory are already local to the query window.
            candidates = others
        else:
            candidates = others

        if num_submap <= len(candidates) and not self.allow_duplicate_img:
            selected = random.sample(candidates, k=num_submap)
        else:
            selected = [random.choice(candidates) for _ in range(num_submap)]

        if self.ensure_goal_frame_in_context:
            goal_id = record.goal_frame_idx
            if goal_id != query_id and goal_id in record.frame_ids and goal_id not in selected and len(selected) > 0:
                # Keep sample length fixed: replace one submap frame with the goal frame.
                selected[random.randrange(len(selected))] = goal_id

        # Keep temporal ordering for submap frames before passing to the model.
        selected = sorted(int(fid) for fid in selected)
        return [query_id] + selected

    @staticmethod
    def _resize_image(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if image.shape[0] == out_h and image.shape[1] == out_w:
            return image
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _resize_map(map_hw: np.ndarray, out_h: int, out_w: int, interpolation: int) -> np.ndarray:
        if map_hw.shape[0] == out_h and map_hw.shape[1] == out_w:
            return map_hw
        return cv2.resize(map_hw, (out_w, out_h), interpolation=interpolation)

    def _compute_anchor_patch_idx(
        self,
        goal_u: int,
        goal_v: int,
        src_w: int,
        src_h: int,
        out_w: int,
        out_h: int,
    ) -> int:
        scale_x = out_w / max(src_w, 1)
        scale_y = out_h / max(src_h, 1)

        u = int(np.clip(np.floor(goal_u * scale_x), 0, out_w - 1))
        v = int(np.clip(np.floor(goal_v * scale_y), 0, out_h - 1))

        patch_w = max(1, out_w // self.patch_size)
        patch_x = min(u // self.patch_size, patch_w - 1)
        patch_y = min(v // self.patch_size, max(1, out_h // self.patch_size) - 1)

        return int(patch_y * patch_w + patch_x)

    def get_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[Sequence[int]] = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training and seq_index is None and seq_name is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is not None:
            if seq_name not in self.sequence_map:
                raise KeyError(f"Unknown sequence name: {seq_name}")
            record = self.sequence_map[seq_name]
        else:
            if seq_index is None:
                seq_index = random.randint(0, self.sequence_list_len - 1)
            seq_index = int(seq_index) % self.sequence_list_len
            record = self.sequence_list[seq_index]

        if img_per_seq is None:
            img_per_seq = len(record.frame_ids)

        if ids is not None and len(ids) > 0:
            selected_ids = [int(fid) for fid in ids]
            if record.query_idx not in selected_ids:
                selected_ids = [record.query_idx] + selected_ids
            else:
                # Keep query first for query-conditioned heads.
                selected_ids = [record.query_idx] + [fid for fid in selected_ids if fid != record.query_idx]

            if self.ensure_goal_frame_in_context:
                goal_id = record.goal_frame_idx
                if goal_id != record.query_idx and goal_id in record.frame_ids and goal_id not in selected_ids:
                    if len(selected_ids) > 1:
                        selected_ids[-1] = goal_id
                    else:
                        selected_ids.append(goal_id)

            if len(selected_ids) > 1:
                # Keep temporal ordering for submap frames while preserving query first.
                selected_ids = [record.query_idx] + sorted(int(fid) for fid in selected_ids[1:])
        else:
            selected_ids = self._sample_frame_ids(record, int(img_per_seq))

        target_shape = self.get_target_shape(aspect_ratio)
        out_h, out_w = int(target_shape[0]), int(target_shape[1])

        images: List[np.ndarray] = []
        original_hw: List[Tuple[int, int]] = []

        for fid in selected_ids:
            image_path = record.image_paths.get(fid)
            if image_path is None:
                raise FileNotFoundError(f"Missing RGB for frame {fid} in {record.subtraj_dir}")

            image = read_image_cv2(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            src_h, src_w = image.shape[:2]
            original_hw.append((int(src_h), int(src_w)))

            image = self._resize_image(image, out_h=out_h, out_w=out_w)
            images.append(image)

        if len(images) == 0:
            raise RuntimeError(f"No frames loaded for sequence: {record.sequence_name}")

        query_costmap_path = record.costmap_paths.get(record.query_idx)
        if query_costmap_path is None:
            raise FileNotFoundError(f"Missing query costmap for frame {record.query_idx} in {record.subtraj_dir}")

        gt_costmap = np.load(query_costmap_path).astype(np.float32)
        gt_costmap = self._resize_map(gt_costmap, out_h=out_h, out_w=out_w, interpolation=cv2.INTER_NEAREST)
        gt_costmap_mask = np.isfinite(gt_costmap)
        gt_costmap = np.nan_to_num(gt_costmap, nan=0.0, posinf=0.0, neginf=0.0)

        if record.goal_frame_idx == record.query_idx:
            goal_frame_pos = 0
        elif record.goal_frame_idx in selected_ids:
            goal_frame_pos = int(selected_ids.index(record.goal_frame_idx))
        else:
            logger.warning(
                "Goal frame %d not selected in %s; falling back to query frame anchor",
                record.goal_frame_idx,
                record.sequence_name,
            )
            goal_frame_pos = 0

        goal_src_h, goal_src_w = original_hw[goal_frame_pos]
        anchor_patch_idx_goal = self._compute_anchor_patch_idx(
            goal_u=record.goal_pixel_uv[0],
            goal_v=record.goal_pixel_uv[1],
            src_w=goal_src_w,
            src_h=goal_src_h,
            out_w=out_w,
            out_h=out_h,
        )

        is_query_frame = np.zeros((len(selected_ids),), dtype=np.bool_)
        is_query_frame[0] = True

        anchor_patch_idx = np.full((len(selected_ids),), -1, dtype=np.int64)
        anchor_patch_idx[goal_frame_pos] = anchor_patch_idx_goal

        batch = {
            "seq_name": record.sequence_name,
            "ids": np.asarray(selected_ids, dtype=np.int64),
            "frame_num": len(selected_ids),
            "images": images,
            "gt_costmap": gt_costmap,
            "gt_costmap_mask": gt_costmap_mask,
            "is_query_frame": is_query_frame,
            "anchor_patch_idx": anchor_patch_idx,
            "goal_frame_idx": np.int64(record.goal_frame_idx),
            "goal_frame_pos": np.int64(goal_frame_pos),
        }

        if self.load_query_pointmap:
            query_pointmap_path = record.pointmap_paths.get(record.query_idx)
            if query_pointmap_path is not None and os.path.exists(query_pointmap_path):
                query_pointmap = np.load(query_pointmap_path).astype(np.float32)
                query_pointmap = self._resize_map(
                    query_pointmap,
                    out_h=out_h,
                    out_w=out_w,
                    interpolation=cv2.INTER_LINEAR,
                )
                query_point_mask = np.all(np.isfinite(query_pointmap), axis=-1)
                query_pointmap = np.nan_to_num(query_pointmap, nan=0.0, posinf=0.0, neginf=0.0)

                # Keep query-only point supervision to match the query-only prediction head.
                batch["world_points"] = [query_pointmap]
                batch["point_masks"] = [query_point_mask]

        return batch
