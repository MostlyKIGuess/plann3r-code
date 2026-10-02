"""Build the topological graph of a mapped episode from base VGGT 3D points.

Image pixels become graph nodes. Frames are joined by VGGT point tracks, and
pixels in one image are joined by 3D distance. With a goal, Dijkstra gives a
graph distance costmap for every map image. The graph costmaps are an older
alternative to the Plann3r propagation maps and are not used for the reported
results. Nothing in the released pipeline imports this file, see
history/README.md.

Usage (historical, the mapper config no longer carries the graph keys):
    pixi run python -m history.create_topomap model.path=... scenes.base_dir=... goal.mode=episode ...
"""

import os
import time
import json
import sys
import shutil
import pickle
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import networkx as nx
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import blosc2

from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Add third-party libraries to path
BASE_DIR = Path(__file__).parent.parent.parent

# Import geometry utilities
from libs.common.geometry_utils import get_mask_centroid
from history.graph_utils import load_compressed_graph_chunked


class NodeCullingMode(Enum):
	NONE = "NONE"
	FPS = "FPS"


class EdgeCullingMode(Enum):
	NONE = "NONE"
	EMST_SINGLE = "EMST_SINGLE"
	DELAUNAY_3D = "DELAUNAY_3D"


def _ensure_vggt_on_path() -> None:
	"""Try to locate the VGGTNav repo and add it to sys.path for import."""
	candidates = []

	env_root = (
		os.environ.get("VGGTNAV_ROOT")
		or os.environ.get("VGGT_ROOT")
		or os.environ.get("VGGT_PATH")
	)
	if env_root:
		candidates.append(Path(env_root).expanduser().resolve())

	repo_root = Path(__file__).resolve().parents[2]
	candidates.append(repo_root)
	candidates.append(repo_root.parent / "VGGTNav")
	candidates.append(repo_root.parent / "vggt")
	candidates.append(repo_root.parent.parent / "VGGTNav")
	candidates.append(repo_root.parent.parent / "vggt")

	for base in candidates:
		if not base.exists():
			continue

		if (base / "vggt").is_dir():
			path_to_add = base
		elif base.name == "vggt" and (base / "__init__.py").exists():
			path_to_add = base.parent
		else:
			continue

		if str(path_to_add) not in sys.path:
			sys.path.append(str(path_to_add))
		return


try:
	from vggt.models.vggt import VGGT
	from vggt.utils.pose_enc import pose_encoding_to_extri_intri
	from vggt.utils.geometry import depth_to_world_coords_points
except ModuleNotFoundError as exc:
	if exc.name != "vggt":
		raise
	_ensure_vggt_on_path()
	from vggt.models.vggt import VGGT
	from vggt.utils.pose_enc import pose_encoding_to_extri_intri
	from vggt.utils.geometry import depth_to_world_coords_points


def _strip_module_prefix(state_dict: dict) -> dict:
	if all(key.startswith("module.") for key in state_dict.keys() if isinstance(key, str)):
		return {key[len("module."):]: value for key, value in state_dict.items()}
	return state_dict


def _compute_resize_params(
	image: Image.Image,
	target_size: int,
	patch_size: int,
) -> Tuple[int, int, float, int, int]:
	width, height = image.size
	scale = float(target_size) / float(max(width, height))
	resized_w = int(round(width * scale))
	resized_h = int(round(height * scale))
	resized_w = min(resized_w, target_size)
	resized_h = min(resized_h, target_size)
	resized_w = max(patch_size, int(round(resized_w / patch_size)) * patch_size)
	resized_h = max(patch_size, int(round(resized_h / patch_size)) * patch_size)
	resized_w = min(resized_w, target_size)
	resized_h = min(resized_h, target_size)
	pad_left = (target_size - resized_w) // 2
	pad_top = (target_size - resized_h) // 2
	return resized_w, resized_h, scale, pad_left, pad_top


def _preprocess_vggt_images(
	images: List[np.ndarray],
	target_size: int,
	patch_size: int,
) -> Tuple[torch.Tensor, List[dict]]:
	if len(images) == 0:
		raise ValueError("At least one image is required for VGGT inference.")

	processed = []
	metadata = []

	for img in images:
		if img.dtype != np.uint8:
			img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
		pil_img = Image.fromarray(img).convert("RGB")

		resized_w, resized_h, scale, pad_left, pad_top = _compute_resize_params(
			pil_img, target_size, patch_size
		)

		resized_img = pil_img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
		square_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
		square_img.paste(resized_img, (pad_left, pad_top))

		tensor_img = torch.from_numpy(np.array(square_img)).float() / 255.0
		tensor_img = tensor_img.permute(2, 0, 1).contiguous()

		processed.append(tensor_img)
		metadata.append({
			"scale": scale,
			"pad_left": pad_left,
			"pad_top": pad_top,
			"resized_w": resized_w,
			"resized_h": resized_h,
			"orig_w": pil_img.width,
			"orig_h": pil_img.height,
		})

	images_tensor = torch.stack(processed)
	return images_tensor, metadata


def _transform_points_to_vggt(points_xy: np.ndarray, meta: dict) -> np.ndarray:
	if points_xy.size == 0:
		return points_xy.astype(np.float32)

	points = points_xy.astype(np.float32)
	points[:, 0] = points[:, 0] * meta["scale"] + meta["pad_left"]
	points[:, 1] = points[:, 1] * meta["scale"] + meta["pad_top"]
	return points


def _transform_points_from_vggt(points_xy: np.ndarray, meta: dict) -> np.ndarray:
	if points_xy.size == 0:
		return points_xy.astype(np.float32)

	points = points_xy.astype(np.float32)
	points[:, 0] = (points[:, 0] - meta["pad_left"]) / meta["scale"]
	points[:, 1] = (points[:, 1] - meta["pad_top"]) / meta["scale"]
	return points


def _remap_world_points_to_original(
	world_points: np.ndarray,
	meta: dict,
	out_h: int,
	out_w: int,
) -> np.ndarray:
	if world_points is None:
		return None

	grid_x, grid_y = np.meshgrid(np.arange(out_w), np.arange(out_h))
	map_x = grid_x.astype(np.float32) * meta["scale"] + meta["pad_left"]
	map_y = grid_y.astype(np.float32) * meta["scale"] + meta["pad_top"]

	remapped = []
	for c in range(3):
		remapped_c = cv2.remap(
			world_points[..., c].astype(np.float32),
			map_x,
			map_y,
			interpolation=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=np.nan,
		)
		remapped.append(remapped_c)

	return np.stack(remapped, axis=-1)


class MapTopological3DPoints:
	def __init__(self, img_dir: str, out_dir: str, cfg: DictConfig):
		self.cfg = cfg

		print("\n" + "=" * 80)
		print("INITIALIZING TOPOLOGICAL MAPPER (VGGT)")
		print("=" * 80)
		print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

		self.img_dir = Path(img_dir)
		self.scene_dir = self.img_dir.parent
		self.out_dir = Path(out_dir)
		self.out_dir.mkdir(parents=True, exist_ok=True)

		self.W = cfg.image.width
		self.H = cfg.image.height
		self.device = cfg.model.device
		self.img_match_window_size = cfg.graph.inter_image_match_window_size

		self.vggt_img_size = int(cfg.model.get("img_size", 518))
		self.vggt_patch_size = int(cfg.model.get("patch_size", 14))
		self.track_conf_threshold = float(cfg.model.get("track_conf_threshold", 0.0))
		self.track_vis_threshold = 0.0
		self.query_stride = int(cfg.model.get("query_stride", 8))
		self.max_query_points = int(cfg.model.get("max_query_points", 4000))
		self.infer_batch_size = int(cfg.model.get("batch_size", 1))

		self.img_names = natsorted(os.listdir(self.img_dir))
		self.img_paths = [self.img_dir / img_name for img_name in self.img_names]
		print(f"Found {len(self.img_paths)} images in {self.img_dir}")

		self.img_paths = self._subsample_images()
		print(f"After subsampling, {len(self.img_paths)} images will be used.")

		if cfg.processing.copy_images:
			self._copy_images_to_output_dir()

		self.G = None
		self.nodeID_to_imgRegionIdx = None
		self.inter_image_edges = {}
		self.intra_image_edges = {}
		self.pixel_to_node_id = {}
		self.force_recompute_graph = cfg.processing.force_recompute_graph
		self.query_points_cache = {}

		self.pc_npz_path = self.out_dir / "nodes_vggt_points.npz"
		self.graph_intra_path = self.out_dir / "graph_intra_edges.pickle"
		self.graph_inter_path = self.out_dir / "graph_just_inter_edges.pickle"
		self.graph_path = self.out_dir / "graph_vggt_intra_edges_with_weights.pickle"

		self.model_path = cfg.model.path
		self.vggt = self._load_vggt_model(self.model_path)

	def _load_vggt_model(self, model_path: str):
		model = VGGT(
			img_size=self.vggt_img_size,
			patch_size=self.vggt_patch_size,
			enable_camera=True,
			enable_point=False,
			enable_depth=True,
			enable_track=True,
		)

		checkpoint_path = Path(model_path)
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint_path}")

		checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
		if isinstance(checkpoint, dict) and "model" in checkpoint:
			state_dict = checkpoint["model"]
		else:
			state_dict = checkpoint

		state_dict = _strip_module_prefix(state_dict)
		model.load_state_dict(state_dict, strict=False)
		model = model.to(self.device)
		model.eval()
		return model

	def _get_base_graph_filename(self):
		ec_mode = self.cfg.graph.edge_culling_mode
		nc_mode = self.cfg.graph.node_culling_mode
		nc_factor = self.cfg.graph.node_culling_factor
		w, h = self.cfg.image.width, self.cfg.image.height

		return (
			f"graph_base_"
			f"{w}x{h}_"
			f"EC_{ec_mode}_"
			f"NC_{nc_mode}_"
			f"NCF_{nc_factor}.pkl"
		)

	def _get_goal_graph_filename(self, goal_img_idx: int = None):
		ec_mode = self.cfg.graph.edge_culling_mode
		nc_mode = self.cfg.graph.node_culling_mode
		nc_factor = self.cfg.graph.node_culling_factor
		w, h = self.cfg.image.width, self.cfg.image.height

		return (
			f"graph_with_distances_to_goal_"
			f"{w}x{h}_"
			f"EC_{ec_mode}_"
			f"NC_{nc_mode}_"
			f"NCF_{nc_factor}.pkl"
		)

	def _get_costmap_filename(self):
		ec_mode = self.cfg.graph.edge_culling_mode
		nc_mode = self.cfg.graph.node_culling_mode
		nc_factor = self.cfg.graph.node_culling_factor
		w, h = self.cfg.image.width, self.cfg.image.height

		return (
			f"costmaps_"
			f"{w}x{h}_"
			f"EC_{ec_mode}_"
			f"NC_{nc_mode}_"
			f"NCF_{nc_factor}"
		)

	def _subsample_images(self) -> list:
		start_idx = self.cfg.processing.subsample_start_idx
		end_idx = self.cfg.processing.subsample_end_idx
		step = self.cfg.processing.subsample_step

		return self.img_paths[start_idx:end_idx:step]

	def _copy_images_to_output_dir(self):
		out_img_dir = self.out_dir / "images"
		out_img_dir.mkdir(parents=True, exist_ok=True)

		extension = Path(self.img_paths[0]).suffix

		print(f"Copying {len(self.img_paths)} images to {out_img_dir}")
		for i, img_path in enumerate(self.img_paths):
			output_path = out_img_dir / f"{i:04d}{extension}"
			shutil.copy2(img_path, output_path)

	def _load_image_for_mapper(self, img_path: Path) -> np.ndarray:
		img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
		if img is None:
			raise ValueError(f"Failed to read image: {img_path}")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
		return img

	def _get_query_points_for_image(self, img_idx: int) -> np.ndarray:
		if img_idx in self.query_points_cache:
			return self.query_points_cache[img_idx]

		xs = np.arange(0, self.W, self.query_stride, dtype=np.int32)
		ys = np.arange(0, self.H, self.query_stride, dtype=np.int32)
		grid_x, grid_y = np.meshgrid(xs, ys)
		points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)

		if self.max_query_points > 0 and len(points) > self.max_query_points:
			indices = np.linspace(0, len(points) - 1, self.max_query_points).astype(np.int64)
			points = points[indices]

		self.query_points_cache[img_idx] = points
		return points

	def _get_track_matches_for_window(
		self,
		anchor_img_idx: int,
		target_img_indices: List[int],
		query_points_xy: np.ndarray,
	) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
		if query_points_xy.size == 0 or len(target_img_indices) == 0:
			return []

		window_img_indices = [anchor_img_idx] + target_img_indices
		window_images = [
			self._load_image_for_mapper(self.img_paths[img_idx])
			for img_idx in window_img_indices
		]

		images_tensor, metadata = _preprocess_vggt_images(
			window_images,
			target_size=self.vggt_img_size,
			patch_size=self.vggt_patch_size,
		)
		images_tensor = images_tensor.to(self.device)

		query_points_vggt = _transform_points_to_vggt(query_points_xy, metadata[0])
		query_points_tensor = torch.from_numpy(query_points_vggt).float().to(self.device)

		with torch.no_grad():
			outputs = self.vggt(images_tensor, query_points=query_points_tensor)

		tracks = outputs["track"][0].detach().cpu().numpy()
		vis = outputs["vis"][0].detach().cpu().numpy()
		conf = outputs["conf"][0].detach().cpu().numpy()

		matches = []
		max_matches = 100
		for window_offset, target_img_idx in enumerate(target_img_indices, start=1):
			tracks_j = tracks[window_offset]
			vis_j = vis[window_offset]
			conf_j = conf[window_offset]

			valid = (vis_j >= self.track_vis_threshold) & (conf_j >= self.track_conf_threshold)
			if not np.any(valid):
				continue

			query_points_valid = query_points_xy[valid]
			tracks_j_valid = tracks_j[valid]
			conf_valid = conf_j[valid]

			if len(conf_valid) > max_matches:
				top_indices = np.argsort(-conf_valid)[:max_matches]
				query_points_valid = query_points_valid[top_indices]
				tracks_j_valid = tracks_j_valid[top_indices]

			tracks_orig = _transform_points_from_vggt(
				tracks_j_valid,
				metadata[window_offset],
			)

			for src, dst in zip(query_points_valid, tracks_orig):
				x_src, y_src = int(src[0]), int(src[1])
				x_dst, y_dst = int(round(dst[0])), int(round(dst[1]))

				if x_dst < 0 or x_dst >= self.W or y_dst < 0 or y_dst >= self.H:
					continue

				matches.append((
					(anchor_img_idx, x_src, y_src),
					(target_img_idx, x_dst, y_dst),
				))

		return matches

	def create_map_topo(self):
		if (not self.pc_npz_path.exists() and not self.graph_path.exists()) or self.force_recompute_graph:
			pc_dict = self.compute_and_save_point_clouds(save_as_npz=True)
		else:
			print(f"Using Precomputed point clouds from {self.pc_npz_path}")
			pc_dict = np.load(self.pc_npz_path)

		self.G = self.create_base_graph_with_nodes(pc_dict)
		print(f"\nGraph just after creation: {self.G}")

		if not self.graph_intra_path.exists() or self.force_recompute_graph:
			self.G = self.add_inter_image_edges_to_graph()

		if not self.graph_inter_path.exists() or self.force_recompute_graph:
			self.G = self.add_intra_image_edges_to_graph()

	def compute_and_save_point_clouds(self, save_as_npz: bool = True) -> dict:
		pc_dict = {}
		total_images = len(self.img_paths)

		for start in tqdm(range(0, total_images, self.infer_batch_size), desc="VGGT depth batches"):
			end = min(start + self.infer_batch_size, total_images)
			batch_paths = self.img_paths[start:end]
			batch_images = [self._load_image_for_mapper(path) for path in batch_paths]

			images_tensor, metadata = _preprocess_vggt_images(
				batch_images,
				target_size=self.vggt_img_size,
				patch_size=self.vggt_patch_size,
			)
			images_tensor = images_tensor.to(self.device)

			with torch.no_grad():
				outputs = self.vggt(images_tensor)

			depth = outputs["depth"].detach().float()
			pose_enc = outputs["pose_enc"].detach().float()

			extrinsics, intrinsics = pose_encoding_to_extri_intri(
				pose_enc,
				image_size_hw=(self.vggt_img_size, self.vggt_img_size),
			)

			if depth.ndim == 5:
				depth_seq = depth[0]
			elif depth.ndim == 4:
				depth_seq = depth
			else:
				raise ValueError(f"Unexpected VGGT depth shape: {depth.shape}")

			if extrinsics.ndim == 4:
				extrinsics_seq = extrinsics[0]
				intrinsics_seq = intrinsics[0]
			elif extrinsics.ndim == 3:
				extrinsics_seq = extrinsics
				intrinsics_seq = intrinsics
			else:
				raise ValueError(f"Unexpected VGGT extrinsics shape: {extrinsics.shape}")

			seq_len = depth_seq.shape[0]
			if seq_len != len(batch_paths):
				print(
					f"Warning: VGGT depth seq_len={seq_len} != batch size={len(batch_paths)}"
				)

			for idx, img_path in enumerate(batch_paths):
				depth_map = depth_seq[idx]
				if depth_map.ndim == 3:
					depth_map = depth_map.squeeze(-1)
				depth_map = depth_map.detach().cpu().numpy()

				extr = extrinsics_seq[idx]
				intr = intrinsics_seq[idx]
				if isinstance(extr, torch.Tensor):
					extr = extr.detach().cpu().numpy()
				if isinstance(intr, torch.Tensor):
					intr = intr.detach().cpu().numpy()

				world_points, _, point_mask = depth_to_world_coords_points(
					depth_map,
					extr,
					intr,
				)
				if point_mask is not None:
					world_points = world_points.astype(np.float32)
					world_points[~point_mask] = np.nan

				wp_remap = _remap_world_points_to_original(world_points, metadata[idx], self.H, self.W)
				pc_dict[str(img_path)] = wp_remap

		if save_as_npz:
			np.savez_compressed(self.pc_npz_path, **pc_dict)

		return pc_dict

	def pixel_coord_to_global_node_id(self, img_idx, px, py):
		H, W = self.H, self.W
		img_st_node_id = img_idx * (H * W)
		node_id = img_st_node_id + (py * W + px)
		return node_id

	def create_base_graph_with_nodes(self, pc_dict):
		G = nx.Graph()
		G.graph["cfg"] = OmegaConf.to_container(self.cfg, resolve=True)

		all_match_pairs = []
		num_matches_per_pair = []
		nc_factor = self.cfg.graph.node_culling_factor

		print("First pass: collecting all track match pairs...")
		img_st_idx = 0
		img_end_idx = len(self.img_paths)
		match_window_size = self.img_match_window_size

		for i in tqdm(range(img_st_idx, img_end_idx), desc="Collecting track pairs"):
			query_points = self._get_query_points_for_image(i)
			target_img_indices = list(range(i + 1, min(i + 1 + match_window_size, img_end_idx)))
			if len(target_img_indices) == 0:
				continue

			try:
				window_match_pairs = self._get_track_matches_for_window(
					i,
					target_img_indices,
					query_points,
				)

				matches_by_target = {}
				for pair in window_match_pairs:
					target_img_idx = pair[1][0]
					matches_by_target.setdefault(target_img_idx, []).append(pair)

				for j in target_img_indices:
					matches_per_pair = matches_by_target.get(j, [])
					num_matches_per_pair.append(len(matches_per_pair))
					all_match_pairs.extend(matches_per_pair[::nc_factor])

			except Exception as e:
				print(f"Error getting matches from {i} to {target_img_indices}: {e}")
				continue

		print(f"Found {len(all_match_pairs)} match pairs across all images")
		if len(num_matches_per_pair) > 0:
			avg_matches = np.mean(num_matches_per_pair)
			min_matches = np.min(num_matches_per_pair)
			max_matches = np.max(num_matches_per_pair)
			print(f"Matches per image pair - Avg: {avg_matches:.1f}, Min: {min_matches}, Max: {max_matches}")

		sampled_match_pairs = all_match_pairs
		print(f"Sampled {len(sampled_match_pairs)} match pairs (every {nc_factor}th pair)")

		unique_pixels = set()
		for pair in sampled_match_pairs:
			unique_pixels.add(pair[0])
			unique_pixels.add(pair[1])

		print(f"Extracted {len(unique_pixels)} unique pixels from sampled pairs")

		pixel_to_node_id = {}
		for img_idx, px, py in tqdm(unique_pixels, desc="Creating DA nodes"):
			node_id = self.pixel_coord_to_global_node_id(img_idx, px, py)

			key = str(self.img_paths[img_idx])
			pcd = pc_dict[key]
			coord_3d = pcd[py, px]
			if not np.all(np.isfinite(coord_3d)):
				continue
			if coord_3d[2] <= 1e-8:
				continue

			node_attrs = {
				"map": [img_idx, py * self.W + px],
				"coord_mast3r": coord_3d,
				"pixel": [px, py],
				"type": "da",
			}

			G.add_node(node_id, **node_attrs)
			pixel_to_node_id[(img_idx, px, py)] = node_id

		self.pixel_to_node_id = pixel_to_node_id
		self.sampled_match_pairs = sampled_match_pairs

		print(f"Created sparse graph with {G.number_of_nodes()} DA nodes using old node IDs")
		print(f"Stored {len(self.sampled_match_pairs)} match pairs for edge creation")

		nodes_per_image = {}
		for img_idx, px, py in unique_pixels:
			nodes_per_image[img_idx] = nodes_per_image.get(img_idx, 0) + 1

		if len(nodes_per_image) > 0:
			counts = list(nodes_per_image.values())
			avg_nodes = np.mean(counts)
			min_nodes = np.min(counts)
			max_nodes = np.max(counts)
			print(f"DA nodes per image - Avg: {avg_nodes:.1f}, Min: {min_nodes}, Max: {max_nodes}")

		return G

	def add_inter_image_edges_to_graph(self):
		da_edges = []

		print(f"Adding Inter-Image edges from {len(self.sampled_match_pairs)} stored match pairs...")

		for pair in tqdm(self.sampled_match_pairs, desc="Creating DA edges from match pairs"):
			pixel_i, pixel_j = pair

			node_i = self.pixel_coord_to_global_node_id(pixel_i[0], pixel_i[1], pixel_i[2])
			node_j = self.pixel_coord_to_global_node_id(pixel_j[0], pixel_j[1], pixel_j[2])

			if node_i in self.G.nodes and node_j in self.G.nodes:
				da_edges.append((int(node_i), int(node_j), {"edge_type": "da", "weight": 0}))
			else:
				print(f"Warning: Missing nodes {node_i} or {node_j} for pair {pair}")

		print(f"Created {len(da_edges)} DA edges from stored pairs")
		print(f"Expected {len(self.sampled_match_pairs)} edges, got {len(da_edges)} edges")

		self.G.add_edges_from(da_edges)
		print(f"\n\nNumber of nodes and edges: {len(self.G.nodes())}, {self.G.number_of_edges()}")

		return self.G

	def add_intra_image_edges_to_graph(self):
		da_nodes_per_img = {}
		for node_id in self.G.nodes():
			node = self.G.nodes[node_id]
			img_id = node["map"][0]

			if img_id not in da_nodes_per_img:
				da_nodes_per_img[img_id] = []
			da_nodes_per_img[img_id].append(node_id)
		print(f"Adding intra-image edges for {len(da_nodes_per_img)} images")

		for img_id in tqdm(
			da_nodes_per_img.keys(), desc="connecting da nodes intra edges"
		):
			da_nodes = da_nodes_per_img[img_id]
			edge_culling_mode = EdgeCullingMode(self.cfg.graph.edge_culling_mode)
			if edge_culling_mode == EdgeCullingMode.EMST_SINGLE:
				edges = self._create_emst_edges(da_nodes, img_id)
			elif edge_culling_mode == EdgeCullingMode.DELAUNAY_3D:
				edges = self._create_delaunay_3d_edges(da_nodes, img_id)
			else:
				edges = self._create_complete_graph_edges(da_nodes, img_id)

			self.G.add_edges_from(edges)

		print(
			f"Final graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges"
		)

		return self.G

	def _create_emst_edges(self, da_nodes: list, img_id: int) -> list:
		if len(da_nodes) <= 1:
			return []

		coords = np.array([
			self.G.nodes[nid]["coord_mast3r"]
			for nid in da_nodes
		])

		dist_matrix = np.linalg.norm(
			coords[:, None, :] - coords[None, :, :],
			axis=2,
		)

		mst = minimum_spanning_tree(dist_matrix).toarray()

		edges = []
		for i in range(len(da_nodes)):
			for j in range(len(da_nodes)):
				if i != j and mst[i, j] > 0:
					edge_weight = dist_matrix[i, j]
					edges.append((
						da_nodes[i],
						da_nodes[j],
						{
							"edge_type": "da_intra",
							"weight": edge_weight,
						},
					))

		return edges

	def _create_delaunay_3d_edges(self, da_nodes: list, img_id: int) -> list:
		if len(da_nodes) <= 3:
			print(f"Not enough DA nodes for 3D Delaunay in image {img_id}")
			return []

		coords = np.array([
			self.G.nodes[nid]["coord_mast3r"]
			for nid in da_nodes
		])

		dist_matrix = np.linalg.norm(
			coords[:, None, :] - coords[None, :, :],
			axis=2,
		)

		try:
			tri = Delaunay(coords)

			edges_set = set()
			for simplex in tri.simplices:
				for i in range(4):
					for j in range(i + 1, 4):
						a, b = simplex[i], simplex[j]
						node_a = da_nodes[a]
						node_b = da_nodes[b]
						edge = tuple(sorted((node_a, node_b)))
						edge_with_attrs = edge + ({"edge_type": "da_intra", "weight": dist_matrix[a, b]},)
						edges_set.add(edge_with_attrs)

			edges = list(edges_set)
			return edges

		except Exception as e:
			print(f"3D Delaunay failed for image {img_id}: {e}")
			return []

	def _create_complete_graph_edges(self, da_nodes: list, img_id: int) -> list:
		if len(da_nodes) <= 1:
			print(f"Only one DA node in image {img_id}, no intra-image edges needed")
			return []

		coords = np.array([
			self.G.nodes[nid]["coord_mast3r"]
			for nid in da_nodes
		])

		dist_matrix = np.linalg.norm(
			coords[:, None, :] - coords[None, :, :],
			axis=2,
		)

		edges = [
			(da_nodes[i], da_nodes[j], {"edge_type": "da_intra", "weight": dist_matrix[i, j]})
			for i, j in combinations(range(len(da_nodes)), 2)
		]

		return edges

	def get_goal_from_episode(self):
		from libs.common.geometry_utils import get_goal_info

		task_type = self.cfg.goal.get("task_type", "original")
		goal_img_idx, goal_mask, goal_instance_id = get_goal_info(
			str(self.scene_dir),
			task_type,
		)

		if goal_mask.shape != (self.H, self.W):
			print(f"Resizing goal mask from {goal_mask.shape} to ({self.H}, {self.W})")
			goal_mask = cv2.resize(goal_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

		centroid = get_mask_centroid(goal_mask)
		if centroid is None:
			raise ValueError(f"Goal mask is empty in episode {self.scene_dir}")

		goal_px, goal_py = centroid

		print(f"Inferred goal from episode: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")

		self.inferred_goal_img_idx = goal_img_idx
		self.inferred_goal_px = goal_px
		self.inferred_goal_py = goal_py
		self.inferred_goal_mask = goal_mask

		return goal_img_idx, goal_px, goal_py

	def compute_distances_to_goal_node(
		self,
		goal_img_idx: int = None,
		goal_px: int = None,
		goal_py: int = None,
	):
		if goal_img_idx is None:
			if hasattr(self, "inferred_goal_img_idx"):
				goal_img_idx = self.inferred_goal_img_idx
				goal_px = self.inferred_goal_px
				goal_py = self.inferred_goal_py
			else:
				goal_img_idx = self.cfg.goal.image_idx
				goal_px = self.cfg.goal.pixel_x
				goal_py = self.cfg.goal.pixel_y

		print(f"Goal: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")

		if os.path.exists(self.pc_npz_path):
			pc_dict = np.load(self.pc_npz_path)
		else:
			pc_dict = self.compute_and_save_point_clouds(save_as_npz=False)

		expected_goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)

		if expected_goal_node_id in self.G.nodes:
			print(f"Goal node {expected_goal_node_id} already exists")
			goal_node_id = expected_goal_node_id
		else:
			goal_node_id = self.add_goal_node(goal_img_idx, goal_px, goal_py, pc_dict)
			print(f"Added goal node {goal_node_id} at pixel ({goal_px}, {goal_py}) in image {goal_img_idx}")

		self.G.graph["goal_img_idx"] = goal_img_idx
		self.G.graph["goal_node_coords"] = (goal_px, goal_py)
		self.G.graph["goal_node_id"] = goal_node_id

		self.connect_da_to_goal_node(goal_img_idx, goal_node_id)

		path_lengths = self.get_single_source_paths(self.G, source_node=goal_node_id, weight="weight")
		self.all_path_lengths = path_lengths
		self.G.graph["all_path_lengths"] = {"weight": path_lengths}

		if self.cfg.goal.get("compute_costmaps", True):
			print(f"\nComputing distance-to-goal costmaps for all images...")
			img_costmaps = self.compute_all_image_costmaps(pc_dict)
			metadata = {
				"goal_img_idx": goal_img_idx,
				"goal_pixel": (goal_px, goal_py),
				"goal_node_id": goal_node_id,
				"cfg": OmegaConf.to_container(self.cfg, resolve=True),
				"goal_coord_3d": self.G.nodes[goal_node_id]["coord_mast3r"].tolist(),
				"image_paths": [str(path) for path in self.img_paths],
				"shape": list(img_costmaps.shape),
			}
			self.save_costmaps(img_costmaps, metadata)
			return img_costmaps

		return None

	def add_goal_node(self, goal_img_idx, goal_px, goal_py, pc_dict):
		key = str(self.img_paths[goal_img_idx])
		pcd = pc_dict[key]
		coord_3d = pcd[goal_py, goal_px]

		goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)

		goal_attrs = {
			"map": [goal_img_idx, goal_py * self.W + goal_px],
			"coord_mast3r": coord_3d,
			"pixel": [goal_px, goal_py],
			"type": "goal",
		}

		self.G.add_node(goal_node_id, **goal_attrs)

		self.pixel_to_node_id[(goal_img_idx, goal_px, goal_py)] = goal_node_id

		self.G.graph["goal_img_idx"] = goal_img_idx
		self.G.graph["goal_node_coords"] = (goal_px, goal_py)
		self.G.graph["goal_node_id"] = goal_node_id

		return goal_node_id

	def connect_da_to_goal_node(self, img_idx, goal_node_id):
		goal_node = self.G.nodes[goal_node_id]

		img_da_node_ids = []
		for node_id, node_data in self.G.nodes(data=True):
			if node_data["map"][0] == img_idx and node_data.get("type") == "da":
				img_da_node_ids.append(node_id)

		print(f"Found {len(img_da_node_ids)} DA nodes in image {img_idx}")

		if goal_node["type"] == "da":
			print(f"goal_node_id={goal_node_id} is a DA node")
			self.G.nodes[goal_node_id]["type"] = "goal"

			for da_node_id in img_da_node_ids:
				if self.G.has_edge(goal_node_id, da_node_id):
					self.G.edges[goal_node_id, da_node_id]["edge_type"] = "goal_da_intra"
			return

		self.G.nodes[goal_node_id]["type"] = "goal"
		pts3d_goal_node = goal_node["coord_mast3r"]

		edge_weights = {}

		for da_node_id in img_da_node_ids:
			da_node = self.G.nodes[da_node_id]
			pts3d_da_node = da_node["coord_mast3r"]
			edge_weights[da_node_id] = np.linalg.norm(pts3d_goal_node - pts3d_da_node)

		self.G.add_edges_from(
			[
				(
					goal_node_id,
					da_node_id,
					{
						"edge_type": "goal_da_intra",
						"weight": edge_weights[da_node_id],
					},
				)
				for da_node_id in img_da_node_ids
			]
		)

		print(f"Connected goal node {goal_node_id} to {len(img_da_node_ids)} DA nodes")
		return self.G

	def get_single_source_paths(self, G, source_node, weight=None, maxVal=1e6):
		path_lengths = nx.single_source_dijkstra_path_length(
			G, source_node, weight=weight
		)

		for node in G.nodes():
			if node not in path_lengths:
				path_lengths[node] = maxVal

		return path_lengths

	def compute_all_image_costmaps(self, pc_dict):
		img_costmaps = []
		num_images = len(self.img_paths)
		for i in tqdm(range(0, num_images), desc="computing non-da to goal distances"):
			pts3d = pc_dict[str(self.img_paths[i])]
			costmap = self.compute_single_image_costmap(i, pts3d)
			img_costmaps.append(costmap)

		img_costmaps = np.stack(img_costmaps, axis=0)
		return img_costmaps

	def compute_single_image_costmap(self, img_idx, pts3d, max_dist=1e6):
		H, W = self.H, self.W

		costmap = np.full((H, W), max_dist, dtype=np.float32)

		pts3d_flat = pts3d.reshape(H * W, 3)

		da_pixel_indices = []
		da_distances = []
		nonda_pixel_indices = []

		for y in range(H):
			for x in range(W):
				node_id = self.pixel_coord_to_global_node_id(img_idx, x, y)
				linear_idx = y * W + x

				if node_id in self.G.nodes:
					node_type = self.G.nodes[node_id].get("type", "unknown")

					if node_type in ["da", "goal"]:
						da_pixel_indices.append(linear_idx)
						da_distances.append(self.all_path_lengths[node_id])
						costmap[y, x] = self.all_path_lengths[node_id]
					else:
						nonda_pixel_indices.append(linear_idx)
				else:
					nonda_pixel_indices.append(linear_idx)

		da_pixel_indices = np.array(da_pixel_indices, dtype=np.int32)
		da_distances = np.array(da_distances, dtype=np.float32)
		nonda_pixel_indices = np.array(nonda_pixel_indices, dtype=np.int32)

		if len(da_pixel_indices) == 0:
			print(f"  Warning: No DA nodes in image {img_idx}, returning max distances")
			return costmap

		if len(nonda_pixel_indices) == 0:
			return costmap

		nonda_distances = self.compute_nonda_distances(
			pts3d_flat,
			nonda_pixel_indices,
			da_pixel_indices,
			da_distances,
			max_dist,
		)

		nonda_y = nonda_pixel_indices // W
		nonda_x = nonda_pixel_indices % W
		costmap[nonda_y, nonda_x] = nonda_distances

		return costmap

	def compute_nonda_distances(
		self,
		pts3d_flat: np.ndarray,
		nonda_indices: np.ndarray,
		da_indices: np.ndarray,
		da_distances: np.ndarray,
		max_dist: float = 1e6,
	) -> np.ndarray:
		device = torch.device(self.device)

		nonda_pts3d = pts3d_flat[nonda_indices]
		da_pts3d = pts3d_flat[da_indices]

		nonda_pts3d = torch.from_numpy(nonda_pts3d).float().to(device)
		da_pts3d = torch.from_numpy(da_pts3d).float().to(device)
		da_distances = torch.from_numpy(da_distances).float().to(device)

		nonda_valid = nonda_pts3d[:, 2] >= 0
		da_valid = da_pts3d[:, 2] >= 0

		diff = nonda_pts3d.unsqueeze(1) - da_pts3d.unsqueeze(0)
		euclidean_dists = torch.norm(diff, dim=2)

		total_dists = euclidean_dists + da_distances.unsqueeze(0)

		total_dists[:, ~da_valid] = max_dist

		min_dists, _ = torch.min(total_dists, dim=1)

		min_dists[~nonda_valid] = max_dist

		nonda_distances = min_dists.cpu().numpy()

		return nonda_distances

	def save_costmaps(self, costmaps: np.ndarray, metadata: dict, filename=None):
		if filename is None:
			filename = self._get_costmap_filename()

		save_path = self.out_dir / filename

		np.savez_compressed(
			save_path,
			costmaps=costmaps,
			metadata=json.dumps(metadata),
		)

		print(f"✓ Saved costmaps to {save_path}")

	@staticmethod
	def load_costmap_file(filepath):
		data = np.load(filepath, allow_pickle=True)
		costmap = data["costmaps"]
		metadata = json.loads(data["metadata"].item())
		return costmap, metadata

	def save_compressed_graph_chunked(self, path, graph=None):
		if graph is None:
			graph = self.G

		t_start = time.time()
		graph_data = self.decompose_graph_data(graph)

		serialized = pickle.dumps(graph_data, protocol=pickle.HIGHEST_PROTOCOL)
		original_size = len(serialized)

		compressed_path = f"{path}.b2s"

		BLOSC2_MAX_SIZE = 2000000000

		if original_size <= BLOSC2_MAX_SIZE:
			compressed = blosc2.compress(serialized, codec=blosc2.Codec.ZSTD, clevel=9)

			with open(compressed_path, "wb") as f:
				f.write(b"SINGLE")
				f.write(len(compressed).to_bytes(8, "little"))
				f.write(compressed)

			compressed_size = len(compressed)
			compression_method = "single_blosc2"

		else:
			chunk_size = 1024 * 1024 * 1024
			num_chunks = (len(serialized) + chunk_size - 1) // chunk_size
			compressed_chunks = []

			print(f"Data too large, using {num_chunks} chunks...")
			for i in range(num_chunks):
				start_idx = i * chunk_size
				end_idx = min((i + 1) * chunk_size, len(serialized))
				chunk = serialized[start_idx:end_idx]

				compressed_chunk = blosc2.compress(
					chunk, codec=blosc2.Codec.ZSTD, clevel=9
				)
				compressed_chunks.append(compressed_chunk)

			with open(compressed_path, "wb") as f:
				f.write(b"CHUNKS")
				f.write(num_chunks.to_bytes(4, "little"))
				f.write(original_size.to_bytes(8, "little"))

				for chunk in compressed_chunks:
					f.write(len(chunk).to_bytes(4, "little"))

				for chunk in compressed_chunks:
					f.write(chunk)

			compressed_size = sum(len(chunk) for chunk in compressed_chunks)
			compression_method = "chunked_blosc2"

		compression_ratio = (1 - compressed_size / original_size) * 100

		print(f"Saved to: {compressed_path} | Total Time: {time.time() - t_start:.3f}s")

		return compressed_path

	def decompose_graph_data(self, graph=None):
		t_start = time.time()

		if graph is None:
			graph = self.G

		graph_data = {"directed": graph.is_directed(), "graph_attrs": dict(graph.graph)}

		nodes_list = list(graph.nodes(data=True))
		node_ids = []
		node_maps = []
		node_coords = []
		node_pixels = []
		node_types = []

		print(f"Processing {len(nodes_list)} nodes...")
		for node_id, attrs in nodes_list:
			node_ids.append(node_id)
			node_maps.append(attrs.get("map", [0, 0]))
			node_coords.append(
				attrs.get("coord_mast3r", np.array([0, 0, 0], dtype=np.float64))
			)
			node_pixels.append(attrs.get("pixel", [0, 0]))
			node_types.append(attrs.get("type", "unknown"))

		graph_data["node_ids"] = np.array(node_ids)
		graph_data["node_maps"] = np.array(node_maps, dtype=np.int32)
		graph_data["node_coords"] = np.array(node_coords, dtype=np.float64)
		graph_data["node_pixels"] = np.array(node_pixels, dtype=np.int32)
		graph_data["node_types"] = np.array(node_types, dtype="U20")

		edges_list = list(graph.edges(data=True))
		edge_sources = []
		edge_targets = []
		edge_types = []
		edge_weights = []

		print(f"Processing {len(edges_list)} edges...")
		for source, target, attrs in edges_list:
			edge_sources.append(source)
			edge_targets.append(target)
			edge_types.append(attrs.get("edge_type", "unknown"))
			edge_weights.append(attrs.get("weight", 0.0))

		graph_data["edge_sources"] = np.array(edge_sources, dtype=np.int32)
		graph_data["edge_targets"] = np.array(edge_targets, dtype=np.int32)
		graph_data["edge_types"] = np.array(edge_types, dtype="U10")
		graph_data["edge_weights"] = np.array(edge_weights, dtype=np.float64)

		if "all_path_lengths" in graph_data["graph_attrs"]:
			path_lengths = graph_data["graph_attrs"]["all_path_lengths"]["weight"]
			path_nodes = np.array(list(path_lengths.keys()))
			path_distances = np.array(list(path_lengths.values()), dtype=np.float64)

			graph_data["path_nodes"] = path_nodes
			graph_data["path_distances"] = path_distances
			del graph_data["graph_attrs"]["all_path_lengths"]
			print(f"Converted {len(path_lengths)} path lengths to arrays")

		print(f"Data preparation took: {time.time() - t_start:.3f}s")
		return graph_data

	def load_base_graph_and_add_goal(self):
		base_graph_path = self.scene_dir / self.cfg.goal.base_graph_path
		if not base_graph_path.exists():
			raise FileNotFoundError(f"Base graph not found: {base_graph_path}")

		print(f"Loading base graph from: {base_graph_path}")
		self.G = load_compressed_graph_chunked(str(base_graph_path))

		goal_img_idx = self.cfg.goal.image_idx
		goal_px = self.cfg.goal.pixel_x
		goal_py = self.cfg.goal.pixel_y

		self.compute_distances_to_goal_node(goal_img_idx, goal_px, goal_py)

		graph_filename = self._get_goal_graph_filename(goal_img_idx)
		self.save_compressed_graph_chunked(str(self.out_dir / graph_filename))

		print("✓ Updated graph saved with goal node")
		return self.G


def make_topo_map(scene_dir: Path, img_dir: Path, out_dir: Path, cfg: DictConfig) -> nx.Graph:
	print(f"\n{'=' * 80}")
	print(f"PROCESSING SCENE: {scene_dir.name}")
	print(f"{'=' * 80}\n")

	goal_mode = cfg.goal.get("mode", "config")
	print(f"Goal mode: {goal_mode}")

	start_time = time.time()
	topo_map = MapTopological3DPoints(str(img_dir), str(out_dir), cfg)

	if goal_mode == "update_graph":
		print("\nMode: UPDATE_GRAPH - Loading base graph and adding goal")
		topo_map.load_base_graph_and_add_goal()
		total_time = time.time() - start_time
		print(f"\n{'=' * 80}")
		print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
		print(f"{'=' * 80}\n")
		return topo_map.G

	print(f"\nMode: {goal_mode.upper()} - Creating topological map...")
	t1 = time.time()
	topo_map.create_map_topo()
	print(f"✓ Created topological map in {time.time() - t1:.2f}s")
	print(f"  Graph: {topo_map.G.number_of_nodes()} nodes, {topo_map.G.number_of_edges()} edges")

	print("\nSaving base graph...")
	base_filename = topo_map._get_base_graph_filename()
	if cfg.compression.enabled:
		topo_map.save_compressed_graph_chunked(str(out_dir / base_filename))
	else:
		with open(out_dir / base_filename, "wb") as f:
			pickle.dump(topo_map.G, f)

	if goal_mode == "episode":
		print("\nInferring goal from episode folder...")
		topo_map.get_goal_from_episode()

		print("Computing distances to goal node...")
		t2 = time.time()
		topo_map.compute_distances_to_goal_node()
		print(f"✓ Computed distances in {time.time() - t2:.2f}s")

		print("\nSaving goal graph...")
		goal_filename = topo_map._get_goal_graph_filename()
		if cfg.compression.enabled:
			topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
		else:
			with open(out_dir / goal_filename, "wb") as f:
				pickle.dump(topo_map.G, f)

	elif goal_mode == "config":
		print("\nComputing distances to goal node...")
		t2 = time.time()
		topo_map.compute_distances_to_goal_node()
		print(f"✓ Computed distances in {time.time() - t2:.2f}s")

		print("\nSaving goal graph...")
		goal_filename = topo_map._get_goal_graph_filename()
		if cfg.compression.enabled:
			topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
		else:
			with open(out_dir / goal_filename, "wb") as f:
				pickle.dump(topo_map.G, f)

	elif goal_mode == "none":
		pass

	else:
		raise ValueError(f"Unknown goal mode: {goal_mode}. Expected: none, config, episode, update_graph")

	total_time = time.time() - start_time
	print(f"\n{'=' * 80}")
	print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
	print(f"{'=' * 80}\n")

	return topo_map.G


def get_scene_list(cfg: DictConfig) -> list:
	base_dir = Path(cfg.scenes.base_dir)

	if not cfg.scenes.multi_scene:
		scene_path = base_dir / cfg.scenes.scene_name
		print(f"Single scene mode: {scene_path.name}")
		return [scene_path]

	list_file = cfg.scenes.get("scene_list_file")
	if list_file and Path(list_file).exists():
		with open(list_file, "r") as f:
			names = [line.strip() for line in f if line.strip()]
		all_scenes = [base_dir / name for name in names if (base_dir / name).exists()]
		source = f"from file {list_file}"
	else:
		all_scenes = natsorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
		source = "from directory"

	start = cfg.scenes.get("start_idx", 0)
	end = cfg.scenes.get("end_idx", -1)
	step = cfg.scenes.get("step", 1)

	if end == -1:
		end = len(all_scenes)

	scenes = all_scenes[start:end:step]
	print(f"Multi-scene mode ({source}): {len(scenes)} scenes (indices {start}:{end}:{step})")
	return scenes


@hydra.main(version_base=None, config_path="../../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
	print("\n" + "=" * 80)
	print("TOPOLOGICAL MAP CREATION (VGGT)")
	print("=" * 80)
	print(f"\nUsing configuration from: {cfg}")
	print("=" * 80 + "\n")

	os.environ["BASE_DIR"] = str(BASE_DIR)

	scenes = get_scene_list(cfg)

	if len(scenes) == 0:
		raise ValueError("No scenes found to process")

	if cfg.scenes.multi_scene:
		print("Multi-scene mode: disabling goal computation (base graph only)")

	results = {}
	base_out_dir = cfg.scenes.get("base_out_dir", None)

	for scene_num, scene_dir in enumerate(tqdm(scenes, desc="Processing scenes", unit="scene")):
		img_dir = scene_dir / "images"
		if scene_dir.name == "q5QZSEeHe5g_0000000_plant_32_":
			continue

		task_type = cfg.goal.get("task_type", "original")
		goal_mode = cfg.goal.get("mode", "config")
		if (
			goal_mode == "episode"
			and task_type in {"original_reverse", "reverse"}
			and not (scene_dir / "reverse_goal.npy").exists()
		):
			print(f"Skipping {scene_dir.name}: no reverse_goal.npy")
			results[scene_dir.name] = False
			continue

		if base_out_dir is not None:
			out_dir = Path(base_out_dir) / scene_dir.name
			out_dir.mkdir(parents=True, exist_ok=True)
		else:
			out_dir = scene_dir

		if not img_dir.exists():
			print(f"⚠ Skipping {scene_dir.name}: no images/ folder")
			results[scene_dir.name] = False
			continue

		print(f"\nScene: {scene_dir.name}")
		print(f"Images: {img_dir}")
		print(f"Output: {out_dir}\n")

		try:
			graph = make_topo_map(scene_dir, img_dir, out_dir, cfg)
			results[scene_dir.name] = graph is not None
		except Exception as exc:
			print(f"Skipping {scene_dir.name}: {type(exc).__name__}: {exc}")
			results[scene_dir.name] = False
			continue

	successful = sum(results.values())
	print(f"\n{'=' * 80}")
	print(f"✓ COMPLETE: {successful}/{len(results)} scenes processed successfully")
	print(f"{'=' * 80}")


if __name__ == "__main__":
	main()
