"""Compute Plann3r costmaps for MARD with the VGGTNav planner checkpoint.

This is the third MARD stage. For each 9-frame subtrajectory, VGGTNav predicts
the query costmap from the submap with the goal given as the anchor pixel. The
anchor is the lowest-cost pixel of the navmesh_extractor.py costmaps, so that
stage must run first. The checkpoint is
$PLANN3R_ROOT/models/planner/checkpoint_best.pt unless --checkpoint-path is
given. PYTHONNOUSERSITE=1 keeps a torch in ~/.local from shadowing the pixi one.

Usage:
    cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
    PYTHONNOUSERSITE=1 pixi run python vggtnav_extractor.py --scene-list "$PLANN3R_ROOT/plann3r-code/episodes_removing_blacklist.txt" \
      --navmesh-root "$PLANN3R_ROOT/evaluation/mard" --output_dir "$PLANN3R_ROOT/evaluation/mard"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
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
CHECKPOINT_DEFAULT = _plann3r_root() / "models/planner/checkpoint_best.pt"
OUT_SUBDIR_NAME_DEFAULT = "vggtnav_costmaps"

OUT_DIR_DEFAULT = _plann3r_root() / "evaluation/mard"

def _ensure_repo_on_path() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()
from libs.experiments.vggtnav_inference import load_vggtnav_model, predict_vggtnav_costmap


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run VGGTNav inference over subtrajectories."
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
		default=OUT_DIR_DEFAULT,
		help="Output root directory where costmaps will be written.",
	)
	parser.add_argument(
		"--scene-list",
		type=Path,
		default=SCENE_LIST_DEFAULT,
		help="Path to scene list file; one scene path per line.",
	)
	parser.add_argument(
		"--navmesh-root",
		type=Path,
		default=NAVMESH_ROOT_DEFAULT,
		help="Root directory containing navmesh costmaps per scene.",
	)
	parser.add_argument(
		"--checkpoint-path",
		type=Path,
		default=CHECKPOINT_DEFAULT,
		help="Local VGGTNav checkpoint path. Must exist; nothing is downloaded.",
	)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--img-size", type=int, default=224)
	parser.add_argument("--patch-size", type=int, default=14)
	parser.add_argument("--upsample-size", type=int, default=60)
	parser.add_argument("--normalize", action="store_true", default=False)
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


def load_num_frames(session_folder: Path) -> int:
	meta_path = session_folder / "metadata.json"
	if meta_path.exists():
		with open(meta_path, "r") as f:
			meta = json.load(f)
		if "num_frames" in meta:
			return int(meta["num_frames"])
	return -1


def load_rgb_image(images_dir: Path, frame_idx: int) -> np.ndarray | None:
	for ext in ("jpg", "png", "jpeg"):
		path = images_dir / f"{frame_idx:05d}.{ext}"
		if not path.exists():
			continue
		img = cv2.imread(str(path))
		if img is None:
			continue
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def main() -> None:
	args = parse_args()
	parent_dir = args.parent_dir
	output_dir = args.output_dir
	scene_list_path = args.scene_list
	navmesh_root = args.navmesh_root

	if not parent_dir.exists() or not parent_dir.is_dir():
		raise ValueError(f"Parent directory not found: {parent_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	allowed_scene_names = load_scene_names(scene_list_path)
	if not allowed_scene_names:
		raise ValueError(f"Scene list is empty: {scene_list_path}")

	device = args.device
	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"

	if not args.checkpoint_path.exists():
		raise FileNotFoundError(
			f"VGGTNav checkpoint not found: {args.checkpoint_path}. "
			"A local checkpoint is required; nothing is downloaded."
		)
	print(f"Loading VGGTNav checkpoint: {args.checkpoint_path}")

	model_cfg = {
		"checkpoint_path": str(args.checkpoint_path),
		"img_size": args.img_size,
		"patch_size": args.patch_size,
	}
	model = load_vggtnav_model(model_cfg, device=device)

	scene_dirs = sorted([p for p in parent_dir.iterdir() if p.is_dir()])
	if not scene_dirs:
		raise ValueError(f"No scene directories found under: {parent_dir}")

	scene_dirs = [p for p in scene_dirs if p.name in allowed_scene_names]
	if not scene_dirs:
		raise ValueError(
			"No matching scene directories found under: "
			f"{parent_dir} (scene list: {scene_list_path})"
		)

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

			out_dir = output_dir / scene_dir.name / OUT_SUBDIR_NAME_DEFAULT
			out_dir.mkdir(parents=True, exist_ok=True)

			for sub_idx in range(num_subtrajs):
				start_idx = sub_idx * 9
				end_idx = start_idx + 8
				query_frame_idx = start_idx + 4
				submap_indices = [
					i for i in range(start_idx, start_idx + 9) if i != query_frame_idx
				]

				anchor_frame_idx, anchor_pixel = find_anchor_pixel(
					navmesh_arrays_dir, submap_indices
				)

				query_img = load_rgb_image(images_dir, query_frame_idx)
				if query_img is None:
					raise FileNotFoundError(f"Missing query image {query_frame_idx}")

				submap_images = []
				anchor_submap_pos = None
				for pos, frame_idx in enumerate(submap_indices):
					img = load_rgb_image(images_dir, frame_idx)
					if img is None:
						raise FileNotFoundError(f"Missing submap image {frame_idx}")
					submap_images.append(img)
					if frame_idx == anchor_frame_idx:
						anchor_submap_pos = pos

				if anchor_submap_pos is None:
					raise ValueError("Anchor frame not found in submap indices")

				anchor_frame_index = 1 + anchor_submap_pos
				_, raw_costmap = predict_vggtnav_costmap(
					model,
					query_img,
					submap_images,
					anchor_frame_index,
					anchor_pixel,
					img_size=args.img_size,
					patch_size=args.patch_size,
					normalize=args.normalize,
					upsample_size=args.upsample_size,
					device=device,
				)

				out_path = out_dir / (
					f"subtraj_{sub_idx}_st_{start_idx}_end_{end_idx}.npy"
				)
				np.save(out_path, raw_costmap)
				processed += 1
		except Exception as exc:
			print(f"[WARN] Failed {scene_dir}: {exc}")

	if processed == 0:
		raise ValueError("No scenes processed successfully.")

	print(f"Processed {processed} subtrajectories.")


if __name__ == "__main__":
	main()
