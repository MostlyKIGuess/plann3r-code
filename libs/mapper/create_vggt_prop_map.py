"""Build the Plann3r propagation map costmaps for every frame of a mapped episode.

Navigation needs a goal cost on every map frame, but VGGTNav sees only a short
window. So the frames are split into windows of 9 with stride 8, run backward
and forward from the goal frame. In each window VGGTNav predicts costmaps from
the goal pixel, and the lowest-cost pixel of the shared frame becomes the goal
for the next window, with its cost added as an offset. Each episode gets an
(N, 16, 16) .npy file and a JSON sidecar. task_setup.py reads them at run time.

Usage:
    pixi run --frozen python -m libs.mapper.create_vggt_prop_map vggtnav.checkpoint_path=... prop_map.multi_query=true ...
    (baseline/build_prop_maps.sh wraps this per task)
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from libs.common.geometry_utils import get_goal_info, get_mask_centroid
from libs.experiments.vggtnav_inference import (
	load_vggtnav_model,
	preprocess_rgb_images,
	predict_vggtnav_costmap,
	predict_vggtnav_costmaps_batched_exact,
	predict_vggtnav_costmaps_multi,
)


def _load_rgb_image(img_path: Path, width: int, height: int) -> np.ndarray:
	img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError(f"Failed to read image: {img_path}")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
	return img


def _patch_to_pixel(
	patch_y: int,
	patch_x: int,
	meta: dict,
	patch_size: int,
	width: int,
	height: int,
) -> Tuple[int, int]:
	u = (patch_x + 0.5) * patch_size
	v = (patch_y + 0.5) * patch_size
	px = int(round((u - meta["pad_left"]) / meta["scale"]))
	py = int(round((v - meta["pad_top"]) / meta["scale"]))
	px = int(np.clip(px, 0, width - 1))
	py = int(np.clip(py, 0, height - 1))
	return px, py


def _select_min_patch(costmap: np.ndarray) -> Tuple[int, int, float]:
	idx = int(np.nanargmin(costmap))
	patch_y, patch_x = np.unravel_index(idx, costmap.shape)
	return int(patch_y), int(patch_x), float(costmap[patch_y, patch_x])


def _build_windows(num_images: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
	if num_images <= 0:
		return []

	last_end = num_images - 1
	start = last_end - window_size + 1
	if start <= 0:
		return [(0, min(window_size - 1, last_end))]

	starts = []
	current = start
	while current >= 0:
		starts.append(current)
		current -= stride

	if starts[-1] != 0:
		starts.append(0)

	return [(s, min(s + window_size - 1, last_end)) for s in starts]


def _build_forward_windows(
	num_images: int,
	start_idx: int,
	window_size: int,
	stride: int,
) -> List[Tuple[int, int]]:
	if num_images <= 0 or start_idx >= num_images:
		return []

	windows = []
	start = start_idx
	end = min(start + window_size - 1, num_images - 1)
	windows.append((start, end))

	while end < num_images - 1:
		next_start = start + stride
		next_end = next_start + window_size - 1
		if next_end >= num_images:
			next_end = num_images - 1
			next_start = max(start_idx, next_end - window_size + 1)

		if next_start <= start and next_end <= end:
			break

		windows.append((next_start, next_end))
		start, end = next_start, next_end

	return windows


def _get_scene_list(cfg: DictConfig) -> List[Path]:
	base_dir = Path(cfg.scenes.base_dir)

	if not cfg.scenes.multi_scene:
		scene_path = base_dir / cfg.scenes.scene_name
		print(f"Single scene mode: {scene_path.name}")
		return [scene_path]

	list_file = cfg.scenes.get("scene_list_file")
	if list_file and Path(list_file).exists():
		with open(list_file, "r") as f:
			names = [line.strip() for line in f if line.strip()]
		scenes = [base_dir / name for name in names if (base_dir / name).exists()]
		print(f"Multi-scene mode (from file): {len(scenes)} scenes")
		return scenes

	all_scenes = natsorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
	start = cfg.scenes.get("start_idx", 0)
	end = cfg.scenes.get("end_idx", -1)
	step = cfg.scenes.get("step", 1)
	if end == -1:
		end = len(all_scenes)

	scenes = all_scenes[start:end:step]
	print(f"Multi-scene mode: {len(scenes)} scenes (indices {start}:{end}:{step})")
	return scenes


def _get_goal_from_episode(scene_dir: Path, cfg: DictConfig, width: int, height: int) -> Tuple[int, int, int]:
	task_type = cfg.goal.get("task_type", "original")
	goal_img_idx, goal_mask, _ = get_goal_info(str(scene_dir), task_type)

	if goal_mask.shape != (height, width):
		goal_mask = cv2.resize(goal_mask, (width, height), interpolation=cv2.INTER_NEAREST)

	centroid = get_mask_centroid(goal_mask)
	if centroid is None:
		raise ValueError(f"Goal mask is empty in episode {scene_dir}")

	goal_px, goal_py = centroid
	return int(goal_img_idx), int(goal_px), int(goal_py)


def _compute_image_metadata(
	get_image: Callable[[int], np.ndarray],
	num_images: int,
	img_size: int,
	patch_size: int,
) -> List[dict]:
	metadata = []
	for idx in range(num_images):
		img = get_image(idx)
		_, meta = preprocess_rgb_images([img], target_size=img_size, patch_size=patch_size)
		metadata.append(meta[0])
	return metadata


def _resolve_device(device: str) -> str:
	if device.startswith("cuda") and not torch.cuda.is_available():
		return "cpu"
	return device


def _window_frame_order(window_indices: List[int], slot0_idx: Optional[int]) -> List[int]:
	"""Put `slot0_idx` first, keeping the remaining frames in temporal order."""
	if slot0_idx is None or slot0_idx == window_indices[0]:
		return list(window_indices)
	return [slot0_idx] + [i for i in window_indices if i != slot0_idx]


def _compute_window_costmaps(
	model: torch.nn.Module,
	window_indices: List[int],
	goal_idx: int,
	goal_pixel: Tuple[int, int],
	get_image: Callable[[int], np.ndarray],
	img_size: int,
	patch_size: int,
	upsample_size: int,
	device: str,
	normalize: bool,
	multi_query: bool = False,
	slot0_idx: Optional[int] = None,
	batched_exact: bool = False,
	batch_chunk: int = 3,
) -> Dict[int, np.ndarray]:
	"""Costmaps for every frame of one window, keyed by global image index.

	Three strategies, all producing the same dict:

	`multi_query=False, batched_exact=False` (default)
		One model call per frame, each with that frame as the query at slot 0. This is
		the historical path and the exact reference.

	`batched_exact=True`
		Same computation, but rows are batched along B and preprocessing happens once.
		Mathematically identical to the default -- every row still puts its query at
		slot 0 -- just faster.

	`multi_query=True`
		A single aggregator pass for the whole window, decoding every frame's tokens
		through the costmap head. ~window_size times less compute, but only the slot-0
		frame is in-distribution; see `predict_vggtnav_costmaps_multi`.

	Note both anchor branches of the per-query path mean the same thing -- "anchor on the
	goal frame" -- so the single per-slot anchor vector used by the other two paths
	reproduces their conditioning exactly.
	"""
	if multi_query or batched_exact:
		frame_order = _window_frame_order(window_indices, slot0_idx)
		if goal_idx not in frame_order:
			raise ValueError(
				f"goal_idx={goal_idx} is not in window {window_indices[0]}:{window_indices[-1]}; "
				f"check that prop_map.stride < prop_map.window_size"
			)
		anchor_slot = frame_order.index(goal_idx)
		frame_images = [get_image(i) for i in frame_order]

		predict = (
			predict_vggtnav_costmaps_multi
			if multi_query
			else predict_vggtnav_costmaps_batched_exact
		)
		kwargs = {} if multi_query else {"batch_chunk": batch_chunk}
		raw_costmaps = predict(
			model,
			frame_images,
			anchor_slot,
			goal_pixel,
			img_size=img_size,
			patch_size=patch_size,
			device=device,
			**kwargs,
		)
		# Zip against frame_order -- the returned order is the selection order.
		return {idx: raw_costmaps[slot].astype(np.float32) for slot, idx in enumerate(frame_order)}

	costmaps: Dict[int, np.ndarray] = {}
	for idx in window_indices:
		query_image = get_image(idx)
		submap_indices = [i for i in window_indices if i != idx]
		submap_images = [get_image(i) for i in submap_indices]
		if idx == goal_idx:
			anchor_frame_index = 0
		else:
			if goal_idx not in submap_indices:
				raise ValueError(
					f"goal_idx={goal_idx} is not in window {window_indices[0]}:{window_indices[-1]}; "
					f"check that prop_map.stride < prop_map.window_size"
				)
			anchor_frame_index = 1 + submap_indices.index(goal_idx)

		_, raw_costmap = predict_vggtnav_costmap(
			model,
			query_image,
			submap_images,
			anchor_frame_index,
			goal_pixel,
			img_size=img_size,
			patch_size=patch_size,
			normalize=normalize,
			upsample_size=upsample_size,
			device=device,
		)
		costmaps[idx] = raw_costmap.astype(np.float32)

	return costmaps


def _process_scene(
	scene_dir: Path,
	cfg: DictConfig,
	model: torch.nn.Module,
	device: str,
) -> bool:
	img_dir = scene_dir / "images"
	if not img_dir.exists():
		print(f"⚠ Skipping {scene_dir.name}: no images/ folder")
		return False

	base_out_dir = cfg.scenes.get("base_out_dir", None)
	if base_out_dir is not None:
		out_dir = Path(base_out_dir) / scene_dir.name
		out_dir.mkdir(parents=True, exist_ok=True)
	else:
		out_dir = scene_dir

	output_path = out_dir / cfg.prop_map.get("save_filename", "vggt_propagation_costs.npy")
	meta_path = out_dir / cfg.prop_map.get("metadata_filename", "vggt_propagation_costs_meta.json")
	if output_path.exists() and not cfg.prop_map.get("overwrite", False):
		print(f"Skipping {scene_dir.name}: output exists")
		return True

	image_paths = natsorted([p for p in img_dir.iterdir() if p.is_file()])
	if len(image_paths) == 0:
		print(f"⚠ Skipping {scene_dir.name}: no images found")
		return False

	width = int(cfg.image.width)
	height = int(cfg.image.height)
	cache_images = bool(cfg.prop_map.get("cache_images", True))
	images: List[np.ndarray] = []

	def get_image(idx: int) -> np.ndarray:
		if cache_images:
			return images[idx]
		return _load_rgb_image(image_paths[idx], width, height)

	if cache_images:
		images = [_load_rgb_image(p, width, height) for p in image_paths]

	goal_mode = cfg.goal.get("mode", "episode")
	if goal_mode == "episode":
		goal_img_idx, goal_px, goal_py = _get_goal_from_episode(scene_dir, cfg, width, height)
	elif goal_mode == "config":
		goal_img_idx = int(cfg.goal.image_idx)
		goal_px = int(cfg.goal.pixel_x)
		goal_py = int(cfg.goal.pixel_y)
	else:
		raise ValueError(
			f"Unsupported goal.mode={goal_mode!r}; expected 'episode' or 'config'"
		)

	if goal_img_idx >= len(image_paths):
		raise ValueError(f"Goal image index {goal_img_idx} out of range for {scene_dir.name}")

	truncate_to_goal = bool(cfg.prop_map.get("truncate_to_goal", True))
	if truncate_to_goal and goal_img_idx < len(image_paths) - 1:
		print(
			"truncate_to_goal is enabled, but bidirectional propagation needs full sequence; "
			"disabling truncation."
		)
		truncate_to_goal = False
	if truncate_to_goal and goal_img_idx < len(image_paths) - 1:
		image_paths = image_paths[: goal_img_idx + 1]
		if cache_images:
			images = images[: goal_img_idx + 1]
		goal_img_idx = len(image_paths) - 1

	num_images = len(image_paths)
	img_size = int(cfg.vggtnav.get("img_size", 224))
	patch_size = int(cfg.vggtnav.get("patch_size", 14))
	upsample_size = int(cfg.vggtnav.get("upsample_size", 60))
	grid = max(1, img_size // patch_size)

	image_meta = _compute_image_metadata(get_image, num_images, img_size, patch_size)

	window_size = int(cfg.prop_map.get("window_size", 9))
	stride = int(cfg.prop_map.get("stride", 8))
	windows_backward = _build_windows(goal_img_idx + 1, window_size, stride)
	windows_forward = _build_forward_windows(num_images, goal_img_idx, window_size, stride)
	windows = sorted(
		{tuple(win) for win in (windows_backward + windows_forward)},
		key=lambda x: (x[0], x[1]),
	)

	global_costmaps = np.full((num_images, grid, grid), np.nan, dtype=np.float32)
	global_prop = np.full((num_images,), np.nan, dtype=np.float32)
	normalize = bool(cfg.prop_map.get("normalize_costmap", False))
	# multi_query built every released propagation map (mapper_config.yaml).
	multi_query = bool(cfg.prop_map.get("multi_query", True))
	batched_exact = bool(cfg.prop_map.get("batched_exact", False))
	batch_chunk = int(cfg.prop_map.get("batch_chunk", 3))
	if multi_query and batched_exact:
		raise ValueError("prop_map.multi_query and prop_map.batched_exact are mutually exclusive")

	def _overlap_index(windows_seq: List[Tuple[int, int]], window_idx: int, direction: str) -> Optional[int]:
		"""Frame shared with the next window, or None for the last window of a sequence."""
		if window_idx + 1 >= len(windows_seq):
			return None
		start, end = windows_seq[window_idx]
		if direction == "backward":
			return start
		if direction == "forward":
			return end
		raise ValueError(f"Unknown direction: {direction}")

	def _run_window_sequence(
		windows_seq: List[Tuple[int, int]],
		direction: str,
		init_goal_idx: int,
		init_goal_pixel: Tuple[int, int],
	) -> None:
		if not windows_seq:
			return

		goal_idx = init_goal_idx
		goal_pixel = init_goal_pixel
		window_offset = 0.0

		for window_idx, (start, end) in enumerate(windows_seq):
			window_indices = list(range(start, end + 1))
			if goal_idx < start or goal_idx > end:
				print(
					f"⚠ Goal index {goal_idx} not in window {start}:{end} for {scene_dir.name}"
				)

			overlap_idx = _overlap_index(windows_seq, window_idx, direction)

			# Slot 0 is the only in-distribution costmap in multi-query mode, so spend it
			# on the overlap frame: its argmin sets the next window's goal pixel and its
			# min value becomes the offset added to every downstream window. Errors on any
			# other frame stay local to one row of global_costmaps.
			window_costmaps = _compute_window_costmaps(
				model,
				window_indices,
				goal_idx,
				goal_pixel,
				get_image,
				img_size,
				patch_size,
				upsample_size,
				device,
				normalize,
				multi_query=multi_query,
				slot0_idx=overlap_idx,
				batched_exact=batched_exact,
				batch_chunk=batch_chunk,
			)

			for idx in window_indices:
				adjusted = window_costmaps[idx] + window_offset
				if np.isnan(global_costmaps[idx, 0, 0]):
					global_costmaps[idx] = adjusted.astype(np.float32)
					global_prop[idx] = float(window_offset)

			if overlap_idx is not None:
				overlap_costmap = window_costmaps[overlap_idx] + window_offset
				patch_y, patch_x, min_val = _select_min_patch(overlap_costmap)
				goal_pixel = _patch_to_pixel(
					patch_y,
					patch_x,
					image_meta[overlap_idx],
					patch_size,
					width,
					height,
				)
				goal_idx = overlap_idx
				window_offset = float(min_val)

	_run_window_sequence(windows_backward, "backward", goal_img_idx, (goal_px, goal_py))
	_run_window_sequence(windows_forward, "forward", goal_img_idx, (goal_px, goal_py))

	missing = np.isnan(global_costmaps[:, 0, 0])
	if np.any(missing):
		missing_count = int(np.sum(missing))
		print(f"⚠ {scene_dir.name}: {missing_count} frames missing costmaps, filling zeros")
		global_costmaps[missing] = 0.0
		global_prop[missing] = 0.0

	np.save(output_path, global_costmaps.astype(np.float32))
	if cfg.prop_map.get("save_metadata", True):
		windows_backward_serialized = [list(win) for win in windows_backward]
		windows_forward_serialized = [list(win) for win in windows_forward]
		windows_serialized = [list(win) for win in windows]
		meta = {
			"scene": scene_dir.name,
			"goal_img_idx": int(goal_img_idx),
			"goal_pixel": [int(goal_px), int(goal_py)],
			"image_paths": [str(p) for p in image_paths],
			"windows": windows_serialized,
			"windows_backward": windows_backward_serialized,
			"windows_forward": windows_forward_serialized,
			"propagation_costs": global_prop.tolist(),
			"shape": list(global_costmaps.shape),
			"cfg": OmegaConf.to_container(cfg, resolve=True),
		}
		with open(meta_path, "w") as f:
			json.dump(meta, f, indent=2)

	print(f"Saved propagation costmaps to {output_path}")
	return True


@hydra.main(version_base=None, config_path="../../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
	print("\n" + "=" * 80)
	print("VGGT PROPAGATION COSTMAP CREATION")
	print("=" * 80)
	print(f"\nUsing configuration from: {cfg}")
	print("=" * 80 + "\n")

	scenes = _get_scene_list(cfg)
	if len(scenes) == 0:
		raise ValueError("No scenes found to process")

	vggtnav_cfg = OmegaConf.to_container(cfg.vggtnav, resolve=True)
	device = _resolve_device(str(vggtnav_cfg.get("device", "cuda")))
	model = load_vggtnav_model(vggtnav_cfg, device)

	results = {}
	for scene_dir in tqdm(scenes, desc="Processing scenes", unit="scene"):
		print(f"\nScene: {scene_dir.name}")
		start_time = time.time()
		ok = _process_scene(scene_dir, cfg, model, device)
		elapsed = time.time() - start_time
		print(f"Scene {scene_dir.name} complete in {elapsed:.2f}s")
		results[scene_dir.name] = ok

	successful = sum(results.values())
	print(f"\n{'=' * 80}")
	print(f"✓ Completed {successful}/{len(results)} scenes")
	print(f"{'=' * 80}\n")


if __name__ == "__main__":
	main()
