#!/usr/bin/env python3
"""HTTP server for real-world Plann3r + learned controller inference.

Runs on the GPU machine. It loads a recorded traversal map (images/, frames.jsonl,
vggt_propagation_costs.npy), the Plann3r planner and the GNM controller, then
answers POST /predict with a {v, w} command for each RGB frame the robot sends.
GET /health and POST /reset are also served. See docs/real-world.md.

Run from the repository root inside the Pixi environment:

    PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
    PLANN3R_REAL_CONTROLLER_RUN=/path/to/controller_run_dir \
    pixi run python real_world/plann3r_realworld_server.py \
        --device cuda --map-dir /path/to/map --port 8088 --retrieval odom
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def yaw_from_xyzw(q: List[float]) -> float:
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_from_odom(odom: Optional[dict]) -> Optional[np.ndarray]:
    if not odom:
        return None
    position = odom.get("position")
    quat = odom.get("orientation_xyzw")
    if position is None or quat is None:
        return None
    return np.array([float(position[0]), float(position[1]), yaw_from_xyzw(quat)], dtype=np.float32)


def relative_pose(pose: np.ndarray, origin: np.ndarray) -> np.ndarray:
    dx = float(pose[0] - origin[0])
    dy = float(pose[1] - origin[1])
    c = math.cos(-float(origin[2]))
    s = math.sin(-float(origin[2]))
    return np.array(
        [
            c * dx - s * dy,
            s * dx + c * dy,
            wrap_angle(float(pose[2] - origin[2])),
        ],
        dtype=np.float32,
    )


def add_repo_to_path(repo_root: Path) -> None:
    repo = str(repo_root.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if width > 0 and height > 0:
        img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def decode_rgb_jpeg(payload: str, width: int, height: int) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    import io

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if width > 0 and height > 0:
        img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    min_val = float(np.min(arr[finite]))
    max_val = float(np.max(arr[finite]))
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - min_val) / (max_val - min_val), 0.0, 1.0)


def turbo_colormap(norm: np.ndarray) -> np.ndarray:
    norm = np.nan_to_num(np.asarray(norm, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    try:
        import cv2

        values = np.round(norm * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:  # pylint: disable=broad-except
        x = norm
        v4 = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=-1)
        v2 = np.stack([x**4, x**5], axis=-1)
        red = v4 @ np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234]) + v2 @ np.array(
            [-152.94239396, 59.28637943]
        )
        green = v4 @ np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333]) + v2 @ np.array(
            [4.27729857, 2.82956604]
        )
        blue = v4 @ np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771]) + v2 @ np.array(
            [-89.90310912, 27.34824973]
        )
        return (np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize_costmap(
    costmap: np.ndarray,
    size: Tuple[int, int],
    resample=Image.BILINEAR,
    draw_grid: bool = False,
) -> Image.Image:
    valid = np.isfinite(costmap)
    norm = normalize_array(costmap)
    rgb = turbo_colormap(norm)
    rgb[~valid] = 255
    image = Image.fromarray(rgb, "RGB").resize(size, resample)
    if draw_grid and costmap.ndim == 2:
        draw = ImageDraw.Draw(image)
        h, w = costmap.shape
        if h > 0 and w > 0:
            for x in range(1, w):
                px = int(round(x * size[0] / w))
                draw.line((px, 0, px, size[1]), fill=(20, 20, 20), width=1)
            for y in range(1, h):
                py = int(round(y * size[1] / h))
                draw.line((0, py, size[0], py), fill=(20, 20, 20), width=1)
    return image


def costmap_extrema(costmap: np.ndarray) -> dict:
    arr = np.asarray(costmap, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return {}
    min_flat = int(np.nanargmin(arr))
    max_flat = int(np.nanargmax(arr))
    min_y, min_x = np.unravel_index(min_flat, arr.shape)
    max_y, max_x = np.unravel_index(max_flat, arr.shape)
    return {
        "min_xy": (int(min_x), int(min_y)),
        "max_xy": (int(max_x), int(max_y)),
        "min_value": float(arr[min_y, min_x]),
        "max_value": float(arr[max_y, max_x]),
    }


def draw_costmap_extrema(image: Image.Image, costmap: np.ndarray, extrema: dict) -> Image.Image:
    if not extrema or costmap.ndim != 2:
        return image
    out = image.copy()
    draw = ImageDraw.Draw(out)
    h, w = costmap.shape

    def patch_center(x: int, y: int) -> Tuple[int, int]:
        return (
            int(round((x + 0.5) * out.width / max(1, w))),
            int(round((y + 0.5) * out.height / max(1, h))),
        )

    for label, key, color in [("min", "min_xy", (0, 255, 255)), ("max", "max_xy", (255, 255, 255))]:
        x, y = extrema[key]
        px, py = patch_center(x, y)
        r = 8
        draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=3)
        draw.text((px + r + 2, py - r), label, fill=color)
    return out


def waypoint_summary(action_pred: Optional[np.ndarray]) -> dict:
    if action_pred is None:
        return {}
    arr = np.asarray(action_pred, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return {}
    first = arr[0, :2]
    last = arr[-1, :2]
    return {
        "first_forward_lateral": (float(first[0]), float(first[1])),
        "last_forward_lateral": (float(last[0]), float(last[1])),
    }


def draw_waypoints(action_pred: Optional[np.ndarray], size: Tuple[int, int] = (320, 240)) -> Image.Image:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    w, h = size
    origin = (w // 2, h - 30)
    draw.line((origin[0], origin[1], origin[0], 20), fill=(180, 180, 180), width=1)
    draw.line((20, origin[1], w - 20, origin[1]), fill=(180, 180, 180), width=1)
    draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(20, 20, 20))

    if action_pred is not None and len(action_pred) > 0:
        pts = []
        scale = 16.0
        for waypoint in np.asarray(action_pred):
            if len(waypoint) < 2:
                continue
            forward = float(waypoint[0])
            lateral = float(waypoint[1])
            x = int(origin[0] - lateral * scale)
            y = int(origin[1] - forward * scale)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=(0, 130, 255), width=3)
        for idx, pt in enumerate(pts):
            r = 4
            draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=(255, 80, 0))
            draw.text((pt[0] + 5, pt[1] - 5), str(idx), fill=(0, 0, 0))
    draw.text((8, 8), "waypoints topdown: right=-lateral", fill=(0, 0, 0))
    return image


def label_panel(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return out


class RealWorldNavigator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).expanduser().resolve()
        add_repo_to_path(self.repo_root)

        from libs.control.learnt_controller import ObjRelLearntController
        from libs.experiments.vggtnav_inference import load_vggtnav_model, predict_vggtnav_costmap

        self.predict_vggtnav_costmap = predict_vggtnav_costmap

        self.map_dir = Path(args.map_dir).expanduser().resolve()
        self.image_paths = sorted((self.map_dir / "images").glob("*"))
        self.image_paths = [p for p in self.image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not self.image_paths:
            raise FileNotFoundError(f"No map images found under {self.map_dir / 'images'}")

        self.map_images = [load_rgb(p, args.width, args.height) for p in self.image_paths]
        self.thumbnails = self._build_thumbnails(self.map_images)
        self.cursor = int(args.start_frame)
        self.map_odom = self._load_map_odom()
        self.map_odom_origin_frame = self._select_map_odom_origin_frame()
        self.live_odom_origin = None
        self.map_odom_rel = self._compute_map_odom_relative()

        prop_path = self.map_dir / args.prop_costs_name
        self.prop_costs = np.load(prop_path).astype(np.float32) if prop_path.exists() else None
        if self.prop_costs is not None and self.prop_costs.shape[0] != len(self.map_images):
            raise ValueError(
                f"Propagation cost count mismatch: {self.prop_costs.shape[0]} vs {len(self.map_images)} images"
            )
        self.goal_frame = self._load_goal_frame()

        controller_config = Path(args.controller_config).expanduser()
        if not controller_config.is_absolute():
            controller_config = self.repo_root / controller_config

        plann3r_ckpt = Path(args.plann3r_ckpt).expanduser()
        if not plann3r_ckpt.is_absolute():
            plann3r_ckpt = self.repo_root / plann3r_ckpt

        vggtnav_cfg = {
            "checkpoint_path": str(plann3r_ckpt),
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "init_costmap_from_depth": False,
            "costmap_activation": args.costmap_activation,
            "costmap_head_type": "mlp",
            "costmap_mlp_ratio": 4.0,
            "costmap_mlp_drop": 0.0,
            "costmap_mlp_layer_idx": -1,
            "enable_point_aux": False,
            "train_point_head": False,
        }
        self.model = load_vggtnav_model(vggtnav_cfg, args.device)
        self.controller = ObjRelLearntController(
            config=str(controller_config.resolve()),
        )
        self.controller.reset_params()
        self.vis_dir = Path(args.vis_dir).expanduser().resolve() if args.vis_dir else None
        self.vis_counter = 0
        if self.vis_dir is not None:
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _load_map_odom(self) -> np.ndarray:
        frames_path = self.map_dir / "frames.jsonl"
        odom = np.full((len(self.image_paths), 3), np.nan, dtype=np.float32)
        if not frames_path.exists():
            return odom

        latest_by_idx = {}
        with frames_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame_idx = int(record.get("frame_idx", -1))
                if 0 <= frame_idx < len(self.image_paths):
                    latest_by_idx[frame_idx] = record

        for frame_idx, record in latest_by_idx.items():
            pose = pose_from_odom(record.get("odom"))
            if pose is not None:
                odom[frame_idx] = pose
        return odom

    def _valid_map_odom_mask(self) -> np.ndarray:
        return np.isfinite(self.map_odom[:, 0]) & np.isfinite(self.map_odom[:, 1]) & np.isfinite(self.map_odom[:, 2])

    def _select_map_odom_origin_frame(self) -> Optional[int]:
        valid = np.where(self._valid_map_odom_mask())[0]
        if len(valid) == 0:
            return None
        requested = int(self.args.map_odom_origin_frame)
        if requested >= 0 and requested in set(int(x) for x in valid):
            return requested
        start_frame = int(np.clip(self.args.start_frame, 0, len(self.image_paths) - 1))
        if start_frame in set(int(x) for x in valid):
            return start_frame
        return int(valid[0])

    def _compute_map_odom_relative(self) -> np.ndarray:
        rel = np.full_like(self.map_odom, np.nan, dtype=np.float32)
        if self.map_odom_origin_frame is None:
            return rel
        origin = self.map_odom[self.map_odom_origin_frame]
        for idx, pose in enumerate(self.map_odom):
            if np.isfinite(pose).all():
                rel[idx] = relative_pose(pose, origin)
        return rel

    def _load_goal_frame(self) -> int:
        if int(self.args.goal_frame) >= 0:
            return int(np.clip(int(self.args.goal_frame), 0, len(self.map_images) - 1))

        meta_path = self.map_dir / f"{Path(self.args.prop_costs_name).stem}_meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                return int(np.clip(int(meta.get("goal_img_idx", len(self.map_images) - 1)), 0, len(self.map_images) - 1))
            except Exception:  # pylint: disable=broad-except
                pass

        goal_path = self.map_dir / "goal.json"
        if goal_path.exists():
            try:
                with goal_path.open("r", encoding="utf-8") as f:
                    goal = json.load(f)
                return int(np.clip(int(goal.get("image_idx", len(self.map_images) - 1)), 0, len(self.map_images) - 1))
            except Exception:  # pylint: disable=broad-except
                pass

        return len(self.map_images) - 1

    @staticmethod
    def _build_thumbnails(images: List[np.ndarray]) -> np.ndarray:
        thumbs = []
        for img in images:
            pil = Image.fromarray(img).resize((64, 48), Image.BILINEAR).convert("L")
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            thumbs.append(arr.reshape(-1))
        return np.stack(thumbs, axis=0)

    def reset(self, odom: Optional[dict] = None, map_frame: Optional[int] = None) -> None:
        self.controller.reset_params()
        self.cursor = int(self.args.start_frame)
        live_pose = pose_from_odom(odom)
        if live_pose is not None:
            self.live_odom_origin = live_pose
        else:
            self.live_odom_origin = None

        if map_frame is not None:
            map_frame = int(np.clip(map_frame, 0, len(self.image_paths) - 1))
            if np.isfinite(self.map_odom[map_frame]).all():
                self.map_odom_origin_frame = map_frame
                self.map_odom_rel = self._compute_map_odom_relative()

    def retrieve_frame_by_image(self, query_rgb: np.ndarray) -> Tuple[int, dict]:
        query_thumb = np.asarray(Image.fromarray(query_rgb).resize((64, 48), Image.BILINEAR).convert("L"), dtype=np.float32)
        query_thumb = (query_thumb.reshape(-1) / 255.0)[None]
        dists = np.mean((self.thumbnails - query_thumb) ** 2, axis=1)
        idx = int(np.argmin(dists))
        return idx, {"image_distance": float(dists[idx])}

    def retrieve_frame_by_odom(self, odom: Optional[dict]) -> Tuple[Optional[int], dict]:
        live_pose = pose_from_odom(odom)
        if live_pose is None or self.map_odom_origin_frame is None:
            return None, {"odom_available": False}
        if self.live_odom_origin is None:
            self.live_odom_origin = live_pose

        live_rel = relative_pose(live_pose, self.live_odom_origin)
        valid = self._valid_map_odom_mask() & np.isfinite(self.map_odom_rel[:, 0])
        if not valid.any():
            return None, {"odom_available": False}

        delta_xy = self.map_odom_rel[valid, :2] - live_rel[:2]
        d_xy = np.linalg.norm(delta_xy, axis=1)
        d_yaw = np.abs([wrap_angle(float(yaw - live_rel[2])) for yaw in self.map_odom_rel[valid, 2]])
        scores = d_xy + float(self.args.odom_yaw_weight) * d_yaw
        valid_indices = np.where(valid)[0]
        best_pos = int(np.argmin(scores))
        idx = int(valid_indices[best_pos])
        return idx, {
            "odom_available": True,
            "odom_distance_m": float(d_xy[best_pos]),
            "odom_yaw_error_rad": float(d_yaw[best_pos]),
            "odom_score": float(scores[best_pos]),
            "live_odom_rel": [float(x) for x in live_rel.tolist()],
            "map_odom_origin_frame": int(self.map_odom_origin_frame),
        }

    def goal_distance_by_odom(self, odom: Optional[dict]) -> dict:
        live_pose = pose_from_odom(odom)
        if live_pose is None or self.map_odom_origin_frame is None:
            return {"goal_odom_available": False, "goal_frame": int(self.goal_frame)}
        if self.live_odom_origin is None:
            self.live_odom_origin = live_pose
        if not np.isfinite(self.map_odom_rel[self.goal_frame]).all():
            return {"goal_odom_available": False, "goal_frame": int(self.goal_frame)}

        live_rel = relative_pose(live_pose, self.live_odom_origin)
        goal_rel = self.map_odom_rel[self.goal_frame]
        goal_distance_m = float(np.linalg.norm(goal_rel[:2] - live_rel[:2]))
        return {
            "goal_odom_available": True,
            "goal_frame": int(self.goal_frame),
            "goal_distance_m": goal_distance_m,
            "goal_rel_xy": [float(goal_rel[0]), float(goal_rel[1])],
            "live_rel_xy": [float(live_rel[0]), float(live_rel[1])],
        }

    def retrieve_frame(self, query_rgb: np.ndarray, odom: Optional[dict] = None) -> Tuple[int, dict]:
        mode = self.args.retrieval
        if mode == "cursor":
            idx = int(np.clip(self.cursor, 0, len(self.map_images) - 1))
            self.cursor = min(len(self.map_images) - 1, self.cursor + int(self.args.cursor_step))
            return idx, {"retrieval": "cursor"}
        if mode == "fixed":
            return int(np.clip(self.args.start_frame, 0, len(self.map_images) - 1)), {"retrieval": "fixed"}
        if mode == "odom":
            idx, meta = self.retrieve_frame_by_odom(odom)
            if idx is not None:
                meta["retrieval"] = "odom"
                return idx, meta
            fallback_idx, fallback_meta = self.retrieve_frame_by_image(query_rgb)
            fallback_meta.update(meta)
            fallback_meta["retrieval"] = "nearest_fallback"
            return fallback_idx, fallback_meta

        idx, meta = self.retrieve_frame_by_image(query_rgb)
        meta["retrieval"] = "nearest"
        return idx, meta

    def build_submap_indices(self, center_idx: int) -> List[int]:
        k = max(1, int(self.args.submap_size))
        half = k // 2
        start = max(0, int(center_idx) - half)
        end = min(len(self.map_images), start + k)
        start = max(0, end - k)
        return list(range(start, end))

    def choose_anchor(self, submap_indices: List[int]) -> Tuple[int, Tuple[int, int], int]:
        if self.prop_costs is not None:
            local_costs = self.prop_costs[submap_indices]
            if not np.isfinite(local_costs).any():
                local_frame_pos = len(submap_indices) - 1
                return int(local_frame_pos), (int(self.args.goal_pixel_x), int(self.args.goal_pixel_y)), int(submap_indices[local_frame_pos])
            flat_idx = int(np.nanargmin(local_costs))
            local_frame_pos, patch_y, patch_x = np.unravel_index(flat_idx, local_costs.shape)
            grid_h, grid_w = local_costs.shape[1], local_costs.shape[2]
            px = int(round((patch_x + 0.5) * self.args.width / grid_w))
            py = int(round((patch_y + 0.5) * self.args.height / grid_h))
            px = int(np.clip(px, 0, self.args.width - 1))
            py = int(np.clip(py, 0, self.args.height - 1))
            return int(local_frame_pos), (px, py), int(submap_indices[local_frame_pos])

        goal_frame = self.args.goal_frame
        if goal_frame < 0:
            goal_frame = len(self.map_images) - 1
        goal_frame = int(np.clip(goal_frame, 0, len(self.map_images) - 1))
        if goal_frame not in submap_indices:
            submap_indices[-1] = goal_frame
        local_frame_pos = submap_indices.index(goal_frame)
        return int(local_frame_pos), (int(self.args.goal_pixel_x), int(self.args.goal_pixel_y)), goal_frame

    def maybe_save_visualization(
        self,
        seq,
        query_rgb: np.ndarray,
        submap_images: List[np.ndarray],
        submap_indices: List[int],
        anchor_global_idx: int,
        anchor_pixel: Tuple[int, int],
        raw_costmap: np.ndarray,
        v: float,
        w: float,
        retrieval_meta: dict,
        query_stamp=None,
    ) -> Optional[str]:
        if self.vis_dir is None or int(self.args.vis_every) <= 0:
            return None

        self.vis_counter += 1
        if (self.vis_counter - 1) % int(self.args.vis_every) != 0:
            return None

        panel_size = (320, 240)
        query_panel = label_panel(Image.fromarray(query_rgb).resize(panel_size), "query")
        extrema = costmap_extrema(raw_costmap)
        native_cost_panel = label_panel(
            draw_costmap_extrema(
                colorize_costmap(raw_costmap, panel_size, resample=Image.NEAREST, draw_grid=True),
                raw_costmap,
                extrema,
            ),
            f"native {raw_costmap.shape[0]}x{raw_costmap.shape[1]} min=cyan max=white",
        )
        approx_ros_w = float(w) * float(self.args.debug_angular_scale) * float(self.args.debug_angular_sign)
        cost_panel = label_panel(
            draw_costmap_extrema(
                colorize_costmap(raw_costmap, panel_size, resample=Image.BILINEAR),
                raw_costmap,
                extrema,
            ),
            f"upscaled response_w={w:.2f} ros_w~{approx_ros_w:.2f}",
        )
        action_pred = getattr(self.controller, "action_pred", None)
        wp_summary = waypoint_summary(action_pred)
        waypoint_panel = label_panel(draw_waypoints(action_pred, panel_size), "controller waypoints")

        submap_thumb_w = panel_size[0] // max(1, min(4, len(submap_images)))
        submap_thumb_h = panel_size[1] // 2
        submap_panel = Image.new("RGB", panel_size, (25, 25, 25))
        draw = ImageDraw.Draw(submap_panel)
        for pos, (idx, img) in enumerate(zip(submap_indices[:8], submap_images[:8])):
            col = pos % 4
            row = pos // 4
            thumb = Image.fromarray(img).resize((submap_thumb_w, submap_thumb_h), Image.BILINEAR)
            x0 = col * submap_thumb_w
            y0 = row * submap_thumb_h
            submap_panel.paste(thumb, (x0, y0))
            label = f"{idx}"
            if idx == anchor_global_idx:
                label += f" anchor {anchor_pixel[0]},{anchor_pixel[1]}"
                draw.rectangle((x0, y0, x0 + submap_thumb_w - 1, y0 + submap_thumb_h - 1), outline=(255, 80, 0), width=3)
            draw.rectangle((x0, y0, x0 + submap_thumb_w, y0 + 17), fill=(0, 0, 0))
            draw.text((x0 + 3, y0 + 3), label, fill=(255, 255, 255))
        submap_panel = label_panel(submap_panel, f"submap {submap_indices}")

        debug_text = Image.new("RGB", panel_size, (250, 250, 250))
        draw = ImageDraw.Draw(debug_text)
        lines = [
            f"seq: {seq}",
            f"query_stamp: {query_stamp}",
            f"retrieval: {retrieval_meta.get('retrieval')}",
            f"nearest: {retrieval_meta}",
            f"anchor_frame: {anchor_global_idx}",
            f"anchor_pixel: {anchor_pixel}",
            f"goal_dist_m: {retrieval_meta.get('goal_distance_m', 'na')}",
            f"http_response_v_w: {v:.3f}, {w:.3f}",
            f"client_ros_w_est: {approx_ros_w:.3f}",
            f"client_est_sign_scale: {self.args.debug_angular_sign:.1f}, {self.args.debug_angular_scale:.1f}",
            f"cost_min_xy_val: {extrema.get('min_xy', 'na')}, {extrema.get('min_value', 'na')}",
            f"cost_max_xy_val: {extrema.get('max_xy', 'na')}, {extrema.get('max_value', 'na')}",
            f"wp_first_fwd_lat: {wp_summary.get('first_forward_lateral', 'na')}",
            f"wp_last_fwd_lat: {wp_summary.get('last_forward_lateral', 'na')}",
        ]
        for i, line in enumerate(lines):
            draw.text((8, 8 + i * 18), str(line)[:80], fill=(0, 0, 0))
        debug_text = label_panel(debug_text, "debug")

        canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1] * 2), (255, 255, 255))
        for i, panel in enumerate([query_panel, native_cost_panel, cost_panel, waypoint_panel, submap_panel, debug_text]):
            x = (i % 3) * panel_size[0]
            y = (i // 3) * panel_size[1]
            canvas.paste(panel, (x, y))

        safe_seq = self.vis_counter if seq is None else seq
        out_path = self.vis_dir / f"rrc_vis_{int(safe_seq):06d}.jpg"
        canvas.save(out_path, quality=90)
        latest_path = self.vis_dir / "latest.jpg"
        canvas.save(latest_path, quality=90)
        return str(out_path)

    def predict(self, query_rgb: np.ndarray, odom: Optional[dict] = None, seq=None, query_stamp=None) -> dict:
        start = time.time()
        nearest_idx, retrieval_meta = self.retrieve_frame(query_rgb, odom)
        retrieval_meta.update(self.goal_distance_by_odom(odom))
        submap_indices = self.build_submap_indices(nearest_idx)
        anchor_local_pos, anchor_pixel, anchor_global_idx = self.choose_anchor(submap_indices)
        submap_images = [self.map_images[idx] for idx in submap_indices]

        costmap, raw_costmap = self.predict_vggtnav_costmap(
            self.model,
            query_rgb,
            submap_images,
            anchor_frame_index=1 + anchor_local_pos,
            anchor_pixel=anchor_pixel,
            img_size=int(self.args.img_size),
            patch_size=int(self.args.patch_size),
            normalize=False,
            upsample_size=60,
            device=self.args.device,
        )
        v, w = self.controller.predict(query_rgb, raw_costmap)
        v = float(np.clip(v, -self.args.max_v, self.args.max_v))
        w = float(np.clip(w, -self.args.max_w, self.args.max_w))
        vis_path = self.maybe_save_visualization(
            seq,
            query_rgb,
            submap_images,
            submap_indices,
            anchor_global_idx,
            anchor_pixel,
            raw_costmap,
            v,
            w,
            retrieval_meta,
            query_stamp=query_stamp,
        )

        return {
            "v": v,
            "w": w,
            "debug": {
                "nearest_frame": int(nearest_idx),
                "query_stamp": query_stamp,
                "retrieval": retrieval_meta,
                "submap_indices": [int(x) for x in submap_indices],
                "anchor_frame": int(anchor_global_idx),
                "anchor_pixel": [int(anchor_pixel[0]), int(anchor_pixel[1])],
                "has_propagation_costs": self.prop_costs is not None,
                "latency_s": time.time() - start,
                "costmap_min": float(np.nanmin(raw_costmap)),
                "costmap_max": float(np.nanmax(raw_costmap)),
                "server_response_v": float(v),
                "server_response_w": float(w),
                "client_ros_w_est": float(w) * float(self.args.debug_angular_scale) * float(self.args.debug_angular_sign),
                "vis_path": vis_path,
            },
        }


def make_handler(navigator: RealWorldNavigator, width: int, height: int):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "map_frames": len(navigator.map_images),
                        "map_odom_frames": int(navigator._valid_map_odom_mask().sum()),
                        "map_odom_origin_frame": navigator.map_odom_origin_frame,
                    },
                )
            else:
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})

        def do_POST(self):  # noqa: N802
            if self.path == "/reset":
                payload = {}
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                navigator.reset(payload.get("odom"), payload.get("map_frame"))
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "map_odom_origin_frame": navigator.map_odom_origin_frame,
                        "has_live_odom_origin": navigator.live_odom_origin is not None,
                    },
                )
                return
            if self.path != "/predict":
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                query_rgb = decode_rgb_jpeg(payload["rgb_jpeg_b64"], width, height)
                result = navigator.predict(query_rgb, payload.get("odom"), payload.get("seq"), payload.get("stamp"))
                result["ok"] = True
                result["seq"] = payload.get("seq")
                self._send_json(200, result)
            except Exception as exc:  # pylint: disable=broad-except
                self._send_json(500, {"ok": False, "error": str(exc), "v": 0.0, "w": 0.0})

        def log_message(self, fmt, *args):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=os.environ.get("PLANN3R_REPO", str(Path.cwd())))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--prop-costs-name", default="vggt_propagation_costs.npy")
    parser.add_argument("--controller-config", default="real_world/configs/gnm_gt_navmesh_costmap_history5.yaml")
    parser.add_argument(
        "--plann3r-ckpt",
        default=os.environ.get("PLANN3R_CKPT", ""),
        help="Plann3r planner checkpoint. Defaults to $PLANN3R_CKPT.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--costmap-activation", default="gelu")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--submap-size", type=int, default=8)
    parser.add_argument("--retrieval", choices=["nearest", "odom", "cursor", "fixed"], default="odom")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--cursor-step", type=int, default=1)
    parser.add_argument("--map-odom-origin-frame", type=int, default=-1)
    parser.add_argument("--odom-yaw-weight", type=float, default=0.25)
    parser.add_argument("--goal-frame", type=int, default=-1)
    parser.add_argument("--goal-pixel-x", type=int, default=160)
    parser.add_argument("--goal-pixel-y", type=int, default=120)
    parser.add_argument("--max-v", type=float, default=0.20)
    parser.add_argument("--max-w", type=float, default=0.60)
    parser.add_argument("--vis-dir", default="", help="Optional directory for query/submap/costmap/waypoint debug panels.")
    parser.add_argument("--vis-every", type=int, default=0, help="Save one visualization every N predictions. 0 disables.")
    parser.add_argument("--debug-angular-scale", type=float, default=3.0, help="Visualization-only angular scale used to display approximate ROS angular.z.")
    parser.add_argument("--debug-angular-sign", type=float, default=-1.0, help="Visualization-only angular sign used to display approximate ROS angular.z.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plann3r_ckpt:
        raise SystemExit("Set --plann3r-ckpt or the PLANN3R_CKPT environment variable.")
    navigator = RealWorldNavigator(args)
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(navigator, args.width, args.height))
    print(f"Plann3r real-world server listening on http://{args.host}:{args.port}")
    print(f"Map: {Path(args.map_dir).resolve()} ({len(navigator.map_images)} frames)")
    server.serve_forever()


if __name__ == "__main__":
    main()
