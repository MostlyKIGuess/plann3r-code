"""Compute the Euclidean baseline costmaps for MARD from base VGGT pointmaps.

This is the second MARD stage. For each 9-frame subtrajectory, base VGGT
predicts pointmaps, and the cost of each query pixel is its straight-line 3D
distance to the anchor point. The anchor is the lowest-cost pixel of the
navmesh_extractor.py costmaps, so that stage must run first. The local VGGT
checkpoint at $PLANN3R_ROOT/models/vggt/model.pt is used unless
--checkpoint-path is given. PYTHONNOUSERSITE=1 keeps a torch in ~/.local from
shadowing the pixi one.

Usage:
    cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
    PYTHONNOUSERSITE=1 pixi run python euclidean_extractor.py --scene-list "$PLANN3R_ROOT/plann3r-code/episodes_removing_blacklist.txt" \
      --navmesh-root "$PLANN3R_ROOT/evaluation/mard" --output_dir "$PLANN3R_ROOT/evaluation/mard"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _plann3r_root() -> Path:
	root = os.environ.get("PLANN3R_ROOT")
	if not root:
		raise RuntimeError(
			"PLANN3R_ROOT is not set; export it to the Plann3r release bundle root."
		)
	return Path(root)


# The goal is not read from a graph file: the anchor pixel comes from the
# navmesh costmaps produced by navmesh_extractor.py, which now infers the goal
# from the episode folder (see gt_mesh_generator.load_goal_pixel_from_episode).
SCENE_LIST_DEFAULT = (
	Path(__file__).resolve().parents[1] / "episodes_removing_blacklist.txt"
)
DEFAULT_PARENT_DIR = (
	_plann3r_root() / "evaluation/datasets/hm3d_navigation/hm3d_iin_val_320x240"
)
NAVMESH_ROOT_DEFAULT = _plann3r_root() / "evaluation/mard"
DEFAULT_OUTPUT_DIR = _plann3r_root() / "evaluation/mard"
DEFAULT_CHECKPOINT_PATH = _plann3r_root() / "models/vggt/model.pt"


def _ensure_vggt_on_path() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.append(str(repo_root))


_ensure_vggt_on_path()
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute Euclidean costmaps from goal pixel for query frames."
	)
	parser.add_argument(
		"parent_dir",
		type=Path,
		nargs="?",
		default=DEFAULT_PARENT_DIR,
		help="Parent directory containing scene subdirectories.",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Output root directory where costmaps will be written.",
	)
	parser.add_argument(
		"--navmesh-root",
		type=Path,
		default=NAVMESH_ROOT_DEFAULT,
		help="Root directory containing navmesh costmaps per scene.",
	)
	parser.add_argument(
		"--scene-list",
		type=Path,
		default=SCENE_LIST_DEFAULT,
		help="Path to scene list file; one scene path per line.",
	)
	parser.add_argument(
		"--checkpoint-path",
		type=Path,
		default=DEFAULT_CHECKPOINT_PATH,
		help="Local VGGT checkpoint path. Must exist; nothing is downloaded.",
	)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--img-size", type=int, default=518)
	parser.add_argument("--patch-size", type=int, default=14)
	return parser.parse_args()


def load_scene_names(scene_list_path: Path) -> set:
	if not scene_list_path.exists():
		raise ValueError(f"Scene list file not found: {scene_list_path}")

	lines = scene_list_path.read_text().splitlines()
	return {
		Path(line.strip()).name
		for line in lines
		if line.strip() and not line.strip().startswith("#")
	}


def resolve_navmesh_arrays_dir(navmesh_root: Path, scene_name: str) -> Path:
	arrays_dir = navmesh_root / scene_name / "navmesh_costmaps" / "arrays"
	if not arrays_dir.is_dir():
		raise FileNotFoundError(f"Navmesh arrays not found: {arrays_dir}")
	return arrays_dir


def load_num_frames(session_folder: Path) -> int:
	meta_path = session_folder / "metadata.json"
	if meta_path.exists():
		with open(meta_path, "r") as f:
			meta = json.load(f)
		if "num_frames" in meta:
			return int(meta["num_frames"])
	return -1


def resolve_images_dir(session_folder: Path) -> Path:
	for name in (
		"images_fov90",
		"images",
		"images_downsampled_fov120",
		"images_downsampled",
		"rgb",
		"color",
	):
		candidate = session_folder / name
		if candidate.is_dir():
			return candidate
	raise FileNotFoundError(f"No images directory found in {session_folder}")


def resolve_image_path(images_dir: Path, frame_idx: int) -> Path | None:
	for ext in ("jpg", "png", "jpeg"):
		path = images_dir / f"{frame_idx:05d}.{ext}"
		if path.exists():
			return path
	return None


def find_anchor_pixel(navmesh_arrays_dir: Path, submap_indices: list[int]) -> tuple[int, tuple[int, int]]:
	best_cost = np.inf
	best_pixel = None
	best_frame = None

	for frame_idx in submap_indices:
		cm_path = navmesh_arrays_dir / f"{frame_idx:05d}.npy"
		if not cm_path.exists():
			continue
		costmap = np.load(cm_path)
		finite = np.isfinite(costmap)
		if not np.any(finite):
			continue
		local_min = float(np.min(costmap[finite]))
		if local_min < best_cost:
			flat_idx = int(np.argmin(np.where(finite, costmap, np.inf)))
			y, x = np.unravel_index(flat_idx, costmap.shape)
			best_cost = local_min
			best_pixel = (int(x), int(y))
			best_frame = frame_idx

	if best_pixel is None or best_frame is None:
		raise ValueError("No valid anchor pixel found in submap costmaps")

	return best_frame, best_pixel


def compute_query_costmap(
	anchor_pmap: np.ndarray,
	query_pmap: np.ndarray,
	anchor_pixel: tuple[int, int],
) -> np.ndarray:
	anchor_u, anchor_v = anchor_pixel
	if (
		anchor_v < 0
		or anchor_v >= anchor_pmap.shape[0]
		or anchor_u < 0
		or anchor_u >= anchor_pmap.shape[1]
	):
		raise ValueError(f"Anchor pixel out of bounds: {anchor_pixel}")

	anchor_point = anchor_pmap[anchor_v, anchor_u]
	if not np.all(np.isfinite(anchor_point)):
		raise ValueError(f"Anchor point invalid at pixel {anchor_pixel}")

	valid = np.all(np.isfinite(query_pmap), axis=2)
	costmap = np.full(query_pmap.shape[:2], np.inf, dtype=np.float32)
	if np.any(valid):
		diffs = query_pmap[valid] - anchor_point
		costmap[valid] = np.linalg.norm(diffs, axis=1).astype(np.float32)
	return costmap


def map_pixel_to_model(
	pixel: tuple[int, int],
	coords: np.ndarray,
	target_size: int,
) -> tuple[int, int]:
	x1, y1, x2, y2, width, height = [float(v) for v in coords.tolist()]
	width = max(width, 1.0)
	height = max(height, 1.0)
	scale_x = (x2 - x1) / width
	scale_y = (y2 - y1) / height

	u = float(pixel[0])
	v = float(pixel[1])
	u_resized = x1 + u * scale_x
	v_resized = y1 + v * scale_y
	u_resized = int(np.clip(round(u_resized), 0, target_size - 1))
	v_resized = int(np.clip(round(v_resized), 0, target_size - 1))
	return u_resized, v_resized


def load_vggt_model(
	checkpoint_path: Path,
	device: str,
	img_size: int,
	patch_size: int,
) -> VGGT:
	"""Load VGGT from a local checkpoint. Never downloads from HuggingFace."""
	model = VGGT(
		img_size=img_size,
		patch_size=patch_size,
		enable_camera=False,
		enable_depth=False,
		enable_track=False,
		enable_point=True,
	)

	if checkpoint_path is None:
		raise ValueError(
			"A local VGGT checkpoint is required; remote downloads are disabled. "
			f"Expected: {DEFAULT_CHECKPOINT_PATH}"
		)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint_path}")

	state_dict = torch.load(str(checkpoint_path), map_location="cpu")
	if isinstance(state_dict, dict) and "model" in state_dict:
		state_dict = state_dict["model"]

	print(f"Loaded VGGT checkpoint: {checkpoint_path}")
	model.load_state_dict(state_dict, strict=False)
	model = model.to(device)
	model.eval()
	return model


def predict_vggt_pointmaps(
	model: VGGT,
	image_paths: list[Path],
	img_size: int,
	device: str,
) -> tuple[np.ndarray, np.ndarray]:
	images, coords = load_and_preprocess_images_square(
		[str(p) for p in image_paths],
		target_size=img_size,
	)
	images = images.to(device)

	use_amp = device.startswith("cuda") and torch.cuda.is_available()
	if use_amp:
		cap = torch.cuda.get_device_capability()
		dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
	else:
		dtype = torch.float32

	with torch.no_grad():
		if use_amp:
			with torch.cuda.amp.autocast(dtype=dtype):
				outputs = model(images)
		else:
			outputs = model(images)

	if "world_points" not in outputs:
		raise ValueError("VGGT output does not contain world_points")

	world_points = outputs["world_points"]
	if world_points.ndim == 5:
		world_points = world_points[0]
	world_points = world_points.detach().cpu().float().numpy()

	coords = coords.detach().cpu().numpy()
	return world_points, coords


def main() -> None:
	args = parse_args()
	parent_dir = args.parent_dir
	output_dir = args.output_dir
	scene_list_path = args.scene_list
	navmesh_root = args.navmesh_root
	checkpoint_path = args.checkpoint_path
	img_size = args.img_size
	patch_size = args.patch_size

	if not parent_dir.exists() or not parent_dir.is_dir():
		raise ValueError(f"Parent directory not found: {parent_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	allowed_scene_names = load_scene_names(scene_list_path)
	if not allowed_scene_names:
		raise ValueError(f"Scene list is empty: {scene_list_path}")

	scene_dirs = sorted([p for p in parent_dir.iterdir() if p.is_dir()])
	if not scene_dirs:
		raise ValueError(f"No scene directories found under: {parent_dir}")

	scene_dirs = [p for p in scene_dirs if p.name in allowed_scene_names]
	if not scene_dirs:
		raise ValueError(
			"No matching scene directories found under: "
			f"{parent_dir} (scene list: {scene_list_path})"
		)

	device = args.device
	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model = load_vggt_model(checkpoint_path, device, img_size, patch_size)

	processed = 0
	for scene_dir in scene_dirs:
		try:
			images_dir = resolve_images_dir(scene_dir)
			navmesh_arrays_dir = resolve_navmesh_arrays_dir(navmesh_root, scene_dir.name)
			num_frames = load_num_frames(scene_dir)
			if num_frames <= 0:
				frame_files = sorted(images_dir.glob("*.jpg"))
				if not frame_files:
					frame_files = sorted(images_dir.glob("*.png"))
				num_frames = len(frame_files)
			if num_frames <= 0:
				raise ValueError("num_frames is not available")

			num_subtrajs = num_frames // 9
			if num_subtrajs == 0:
				raise ValueError("Not enough frames for a full subtrajectory")

			out_dir = output_dir / scene_dir.name / "euclidean_costmap"
			out_dir.mkdir(parents=True, exist_ok=True)

			for sub_idx in range(num_subtrajs):
				start_idx = sub_idx * 9
				end_idx = start_idx + 8
				query_frame_idx = start_idx + 4
				submap_indices = [
					i for i in range(start_idx, start_idx + 9) if i != query_frame_idx
				]
				frame_indices = list(range(start_idx, start_idx + 9))

				image_paths = []
				for frame_idx in frame_indices:
					path = resolve_image_path(images_dir, frame_idx)
					if path is None:
						raise FileNotFoundError(f"Missing image {frame_idx}")
					image_paths.append(path)

				anchor_frame_idx, anchor_pixel = find_anchor_pixel(
					navmesh_arrays_dir, submap_indices
				)

				pointmaps, coords = predict_vggt_pointmaps(
					model,
					image_paths,
					img_size=img_size,
					device=device,
				)

				frame_pos = {idx: pos for pos, idx in enumerate(frame_indices)}
				anchor_pos = frame_pos[anchor_frame_idx]
				query_pos = frame_pos[query_frame_idx]

				anchor_pixel_model = map_pixel_to_model(
					anchor_pixel,
					coords[anchor_pos],
					img_size,
				)

				costmap = compute_query_costmap(
					pointmaps[anchor_pos],
					pointmaps[query_pos],
					anchor_pixel_model,
				)

				out_path = out_dir / (
					f"subtraj_{sub_idx}_st_{start_idx}_end_{end_idx}.npy"
				)
				np.save(out_path, costmap)
				processed += 1
		except Exception as exc:
			print(f"[WARN] Failed {scene_dir}: {exc}")

	if processed == 0:
		raise ValueError("No scenes processed successfully.")

	print(f"Processed {processed} scene(s).")


if __name__ == "__main__":
	main()
