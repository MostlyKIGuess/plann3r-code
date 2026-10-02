"""Set up and step one navigation episode for run_nav.py.

The Episode class loads the Habitat scene, places the start and goal for the
task (original, reverse, alt_goal_v2 or via_alt_goal), and runs the per-step
loop. At each step it localizes the query against the map frames, by ground
truth pose or MegaLoc retrieval, picks the lowest-cost map pixel from the
Plann3r propagation map costmaps as the goal anchor, predicts the query costmap
with VGGTNav, and passes it to the GNM costmap controller. Imitate and Shortcut
take the goal frame from the propagation costmap metadata. run_nav.py also
calls init_results_dir_and_save_cfg from here.
"""

import os

# IMPORTANT: Set habitat-sim env vars BEFORE importing habitat_sim
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import numpy as np
from pathlib import Path
from natsort import natsorted
# Not used here directly. Kept so torch loads before habitat_sim, the import
# order every reported run used, since the two ship their own CUDA libraries.
import torch  # noqa: F401
import cv2
import time
from datetime import datetime
from typing import Tuple, Optional
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import habitat_sim

from libs.simulation.habitat_utils import get_sim_agent
from libs.experiments.episode_utils import (
    pick_random_start_state,
    select_trajectory_start_state,
    calculate_path_distance,
    find_shortest_path,
    initialize_results,
    write_results,
    write_final_meta_results
)
from libs.mapper.costmap_data import CostmapData, load_costmap_metadata
from libs.common.geometry_utils import get_goal_info
from libs.common.utils_sim import build_intrinsics, apply_velocity
from libs.experiments.vggtnav_inference import (
    load_vggtnav_model,
    predict_vggtnav_costmap,
)
from libs.collision_avoidance.care import care_step

from libs.control.learnt_controller import ObjRelLearntController

import logging
logger = logging.getLogger("[Task Setup]") # logger level is explicitly set below by LOG_LEVEL

class Episode:
    def __init__(self, cfg: DictConfig, episode_path, scene_glb_path, episode_results_path, preload_data={}):

        self.cfg = cfg
        self.device = cfg.device
        self.H = self.cfg.sim.height
        self.W = self.cfg.sim.width

        self.episode_path = Path(episode_path)
        self.episode_results_path = Path(episode_results_path)
        logger.info(f"Running {self.episode_path=}...")

        self.scene_glb_path = scene_glb_path
        image_dirname = getattr(self.cfg, "scenes", {}).get("image_dirname", "images")
        self.episode_img_dir = self.episode_path / image_dirname

        self.start_idx = cfg.start_idx
        self.loc_radius = self.cfg.localizer.loc_radius
        self.subsample_ref = self.cfg.localizer.subsample_ref

        self.vggtnav_enabled = bool(getattr(self.cfg, "vggtnav", {}).get("enabled", False))
        self.vggtnav_cfg = getattr(self.cfg, "vggtnav", None)
        self.gt_navmesh_costmap_cfg = self.cfg.get("gt_navmesh_costmap", {})
        self.gt_navmesh_costmap_enabled = bool(
            self.gt_navmesh_costmap_cfg.get("enabled", False)
        )
        self.gt_navmesh_goal_snapped = None
        if self.gt_navmesh_costmap_enabled:
            self.vggtnav_enabled = False
            logger.info("GT navmesh costmap enabled; bypassing VGGTNav")
        self.vggtnav_model = preload_data.get("vggtnav_model", None)
        self.costmap_data = None
        self.costmap_file_path = self._resolve_costmap_file_path()
        self.vggtnav_anchor_source = "topomap"
        self.navmesh_min_pixels = None
        self.navmesh_min_costs = None
        self.navmesh_costmaps = None
        self.vggtnav_map_dir = None
        self.query_anchor_pointmap_cache = {}
        self.query_anchor_pointmap_archive = None
        self.query_anchor_pointmap_archive_keys = None
        self.collision_avoidance_cfg = self.cfg.get("collision_avoidance", {})
        self.collision_avoidance_enabled = bool(
            self.collision_avoidance_cfg.get("enabled", False)
        )
        self.collision_avoidance_method = str(
            self.collision_avoidance_cfg.get("method", "care")
        )
        if self.collision_avoidance_enabled:
            logger.info(
                "Collision avoidance enabled: "
                f"{self.collision_avoidance_method}"
            )

        self.retriever = None
        if self.gt_navmesh_costmap_enabled:
            self._setup_gt_navmesh_costmap()
        elif not self.vggtnav_enabled:
            raise ValueError(
                "No goal source: set vggtnav.enabled=true (Plann3r costmaps) or "
                "gt_navmesh_costmap.enabled=true (simulator NavMesh costmaps)"
            )
        else:
            self.map_img_paths = self._episode_image_paths()
            self._setup_vggtnav()

            if self.vggtnav_anchor_source == "topomap":
                if not self.costmap_file_path.exists():
                    raise FileNotFoundError(f"Costmap file not found: {self.costmap_file_path}")

                logger.info(f"Loading costmap from: {self.costmap_file_path}")
                self.costmap_data = CostmapData.from_file(
                    self.costmap_file_path,
                    self._resolve_costmap_metadata_path(),
                )

        self.init_controller_params()

        self.setup_sim_agent()
        self.ready_agent()

        # robot intrinsics in simulator
        self.agent_intrinsics = build_intrinsics(
            image_width=self.W,
            image_height=self.H,
            field_of_view_radians_u=self.hfov_radians,
            device=self.device
        )

        # Direct GT navmesh mode builds its goal mask online from simulator
        # depth and pathfinder geodesics, so it needs no submap localization.
        if self.vggtnav_enabled:
            self._check_vggtnav_goal_config()

        # Set the controller
        self.set_controller()

    def init_controller_params(self):
        # The agent intrinsics back-project simulator depth for CARE and the GT
        # navmesh paths, so they use the simulator camera's field of view.
        self.fov_deg = float(self.cfg.sim.hfov)
        self.hfov_radians = np.pi * self.fov_deg / 180

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = np.nan

        # Recorded in metadata.txt. The learnt controller has no PID steering.
        self.pid_steer_values = []
        self.discrete_action = -1
        self.controller_logs = None

    def _setup_vggtnav(self):
        """Load the VGGTNav model and the optional privileged anchor files."""
        if self.vggtnav_cfg is None:
            raise ValueError("Missing vggtnav config for VGGTNav inference")

        if self.vggtnav_model is None:
            self.vggtnav_model = load_vggtnav_model(self.vggtnav_cfg, self.device)

        # The episode's map directory holds the propagation costmaps and, for
        # the privileged anchor options, the navmesh and pointmap files.
        self.vggtnav_map_dir = self.costmap_file_path.parent
        if not self.vggtnav_map_dir.is_dir():
            raise FileNotFoundError(f"Map directory not found: {self.vggtnav_map_dir}")

        self.vggtnav_anchor_source = str(self.vggtnav_cfg.get("anchor_source", "topomap")).lower()
        if self.vggtnav_anchor_source not in {"topomap", "navmesh_min"}:
            raise ValueError(f"Unknown vggtnav.anchor_source: {self.vggtnav_anchor_source}")

        if self.vggtnav_anchor_source == "navmesh_min":
            navmesh_min_path = str(self.vggtnav_cfg.get("navmesh_min_pixels_path", "")).strip()
            if navmesh_min_path == "":
                navmesh_min_path = str(self.vggtnav_map_dir / "navmesh_min_pixels.npy")
            navmesh_costmaps_path = str(self.vggtnav_cfg.get("navmesh_costmaps_path", "")).strip()
            if navmesh_costmaps_path == "":
                navmesh_costmaps_path = str(self.vggtnav_map_dir / "navmesh_costmaps.npy")
            if Path(navmesh_costmaps_path).exists():
                self._load_navmesh_costmaps(navmesh_costmaps_path)
            elif Path(navmesh_min_path).exists():
                self._load_navmesh_min_pixels(navmesh_min_path)
            else:
                raise FileNotFoundError(
                    "Navmesh anchor requires navmesh_costmaps.npy or navmesh_min_pixels.npy. "
                    f"Missing: {navmesh_costmaps_path} and {navmesh_min_path}"
                )

        logger.info(f"Loaded VGGTNav model: {self.vggtnav_model.__class__.__name__}")
        logger.info(f"VGGTNav anchor source: {self.vggtnav_anchor_source}")

        self._setup_submap_retriever()

    def _setup_submap_retriever(self) -> None:
        """Build the MegaLoc submap retriever when configured, replacing the
        oracle (GT pose) submap selection. Left as None for the oracle path so
        get_goal keeps using ground-truth localization.
        """
        self.retriever = None
        retrieval_method = str(self.cfg.localizer.get("retrieval", "oracle")).lower()
        if retrieval_method not in {"oracle", "megaloc"}:
            raise ValueError(
                f"Unknown localizer.retrieval={retrieval_method!r}; expected 'oracle' or 'megaloc'"
            )
        if retrieval_method == "megaloc":
            from libs.localizer.megaloc_retrieval import MegaLocRetriever

            self.retriever = MegaLocRetriever(
                map_image_paths=self.map_img_paths,
                model_source=str(self.cfg.localizer.megaloc_model_source),
                weights_path=str(self.cfg.localizer.megaloc_weights_path),
                cache_root=str(self.cfg.localizer.megaloc_cache_root),
                device=self.device,
                top_k=int(self.cfg.localizer.get("retrieval_top_k", 8)),
                batch_size=int(self.cfg.localizer.get("megaloc_batch_size", 16)),
            )
            logger.info("Submap retrieval: MegaLoc (oracle localization bypassed)")

    def _check_vggtnav_goal_config(self) -> None:
        """Plann3r needs a submap for every query, from GT pose or MegaLoc."""
        if self.cfg.goal_source != "topological_pixelwise":
            raise ValueError(
                f"Unknown goal_source={self.cfg.goal_source!r}; expected 'topological_pixelwise'"
            )
        if self.cfg.localizer.name != "topological":
            raise ValueError(f"Unknown localizer: {self.cfg.localizer.name}")
        if self.retriever is None and not bool(
            self.cfg.localizer.get("use_gt_localization", False)
        ):
            raise ValueError(
                "VGGT-Nav requires pose localization or a configured RGB retriever"
            )

    def _resolve_costmap_file_path(self) -> Path:
        costmap_base_dir = self.cfg.get("costmap_base_dir", None)
        costmap_filename = self.cfg.costmap_filename

        if costmap_base_dir:
            return Path(costmap_base_dir) / self.episode_path.name / costmap_filename
        return self.episode_path / costmap_filename

    def _resolve_costmap_metadata_path(self) -> Optional[Path]:
        """Sidecar JSON path, or None to use the `<stem>_meta.json` default.

        The sidecar names differ across tasks, so evaluate.sh sets
        `costmap_metadata_filename` per task.
        """
        metadata_filename = self.cfg.get("costmap_metadata_filename", None)
        if not metadata_filename:
            return None
        return self.costmap_file_path.parent / metadata_filename

    def _episode_image_paths(self) -> list[str]:
        if not self.episode_img_dir.exists():
            raise FileNotFoundError(f"Episode image directory not found: {self.episode_img_dir}")

        image_paths = [
            p for p in natsorted(self.episode_img_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No map images found in: {self.episode_img_dir}")
        return [str(p) for p in image_paths]

    def _setup_gt_navmesh_costmap(self) -> None:
        """Prepare direct GT navmesh-costmap mode.

        Reverse navigation only needs the recorded trajectory to set the start
        and goal states, so the propagation costmap file is optional. Other
        tasks load it when available, since Imitate and Shortcut take the goal
        frame from its metadata.
        """
        source = str(
            self.gt_navmesh_costmap_cfg.get("source", "online_navmesh")
        ).lower()
        if source != "online_navmesh":
            raise ValueError(
                "Only gt_navmesh_costmap.source=online_navmesh is currently "
                f"wired for direct evaluation, got {source}"
            )

        self.map_img_paths = self._episode_image_paths()
        if self.costmap_file_path.exists():
            logger.info(
                "Loading metadata costmap for GT navmesh setup from: "
                f"{self.costmap_file_path}"
            )
            self.costmap_data = CostmapData.from_file(
                self.costmap_file_path,
                self._resolve_costmap_metadata_path(),
            )
        elif not self.cfg.reverse:
            raise FileNotFoundError(
                "GT navmesh costmap mode needs the costmap metadata "
                f"for non-reverse goals, missing: {self.costmap_file_path}"
            )

    def _fill_costmap_holes_nearest(self, costmap: np.ndarray) -> np.ndarray:
        valid_mask = np.isfinite(costmap)
        if valid_mask.all() or not valid_mask.any():
            return costmap

        y_valid, x_valid = np.where(valid_mask)
        y_missing, x_missing = np.where(~valid_mask)
        tree = cKDTree(np.stack([y_valid, x_valid], axis=1))
        _, indices = tree.query(np.stack([y_missing, x_missing], axis=1), k=1)

        filled = costmap.copy()
        filled[y_missing, x_missing] = costmap[y_valid[indices], x_valid[indices]]
        return filled

    def _sensor_pose(self):
        agent_state = self.agent.get_state()
        sensor_state = agent_state.sensor_states.get("depth_sensor")
        if sensor_state is None:
            sensor_state = agent_state.sensor_states.get("color_sensor")
        if sensor_state is not None:
            return np.asarray(sensor_state.position, dtype=np.float32), sensor_state.rotation

        sensor_offset = np.array(
            [0.0, float(self.cfg.sim.sensor_height), 0.0],
            dtype=np.float32,
        )
        q = agent_state.rotation
        q_scipy = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        rot = R.from_quat(q_scipy).as_matrix()
        position = np.asarray(agent_state.position, dtype=np.float32) + rot @ sensor_offset
        return position.astype(np.float32), q

    def _depth_to_world_points(self, depth: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        intrinsics = self.agent_intrinsics.detach().cpu().numpy()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        z = depth[rows, cols].astype(np.float32)
        x = (cols.astype(np.float32) - cx) * z / fx
        y = (rows.astype(np.float32) - cy) * z / fy

        # x, y, z above are image coordinates (x right, y down, z forward).
        # Habitat camera local coordinates are x right, y up, -z forward.
        local_points = np.stack([x, -y, -z], axis=1)
        cam_pos, cam_rot = self._sensor_pose()
        q_scipy = np.array([cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w], dtype=np.float64)
        rot = R.from_quat(q_scipy).as_matrix()
        return (local_points @ rot.T + cam_pos).astype(np.float32)

    def _snap_depth_point_to_navmesh(self, point_3d: np.ndarray, camera_height: float) -> np.ndarray:
        point_3d = np.asarray(point_3d, dtype=np.float32)
        if not np.all(np.isfinite(point_3d)):
            return np.full(3, np.nan, dtype=np.float32)

        if point_3d[1] > camera_height + 0.5:
            candidate = point_3d.copy()
            candidate[1] = camera_height - 0.5
            snapped = self.sim.pathfinder.snap_point(candidate)
        else:
            snapped = self.sim.pathfinder.snap_point(point_3d)

        snapped = np.asarray(snapped, dtype=np.float32)
        if not np.all(np.isfinite(snapped)) or abs(float(snapped[1]) - camera_height) > 3.0:
            candidate = point_3d.copy()
            candidate[1] = camera_height
            snapped = np.asarray(self.sim.pathfinder.snap_point(candidate), dtype=np.float32)

        return snapped

    def _get_gt_navmesh_goal(self) -> np.ndarray:
        if self.gt_navmesh_goal_snapped is None:
            goal = np.asarray(self.final_goal_position, dtype=np.float32)
            snapped = np.asarray(self.sim.pathfinder.snap_point(goal), dtype=np.float32)
            if np.all(np.isfinite(snapped)):
                self.gt_navmesh_goal_snapped = snapped
            else:
                self.gt_navmesh_goal_snapped = goal
        return self.gt_navmesh_goal_snapped

    def _get_gt_navmesh_costmap(self, depth: np.ndarray) -> Tuple[np.ndarray, dict]:
        if depth is None:
            raise ValueError("GT navmesh costmap mode requires depth observations")

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        height, width = depth.shape
        stride = max(1, int(self.gt_navmesh_costmap_cfg.get("stride", 1)))
        fill_holes = bool(self.gt_navmesh_costmap_cfg.get("fill_holes", True))

        grid_rows, grid_cols = np.mgrid[0:height:stride, 0:width:stride]
        rows = grid_rows.reshape(-1)
        cols = grid_cols.reshape(-1)
        valid_depth = np.isfinite(depth[rows, cols]) & (depth[rows, cols] > 0)
        rows = rows[valid_depth]
        cols = cols[valid_depth]

        costmap = np.full((height, width), np.inf, dtype=np.float32)
        if len(rows) == 0:
            logger.warning("GT navmesh costmap has no valid depth pixels")
            costmap.fill(1e6)
            return costmap, {"valid_pixels": 0, "stride": stride}

        cam_pos, _ = self._sensor_pose()
        camera_height = float(cam_pos[1])
        world_points = self._depth_to_world_points(depth, rows, cols)
        goal = self._get_gt_navmesh_goal()

        path = habitat_sim.ShortestPath()
        path.requested_end = goal
        valid_count = 0
        best_distance = float("inf")
        best_pixel = None
        best_world_point = None
        best_snapped_point = None
        for row, col, point_3d in zip(rows, cols, world_points):
            snapped = self._snap_depth_point_to_navmesh(point_3d, camera_height)
            if not np.all(np.isfinite(snapped)):
                continue

            path.requested_start = snapped
            if self.sim.pathfinder.find_path(path):
                distance = float(path.geodesic_distance)
                costmap[int(row), int(col)] = distance
                valid_count += 1
                if distance < best_distance:
                    best_distance = distance
                    best_pixel = (int(col), int(row))
                    best_world_point = np.asarray(point_3d, dtype=np.float32).copy()
                    best_snapped_point = np.asarray(snapped, dtype=np.float32).copy()

        if fill_holes:
            costmap = self._fill_costmap_holes_nearest(costmap)

        if not np.isfinite(costmap).any():
            logger.warning("GT navmesh costmap pathfinder returned no valid pixels")
            costmap.fill(1e6)

        vis_data = {
            "gt_navmesh_costmap": True,
            "gt_navmesh_source": "online_navmesh",
            "gt_navmesh_stride": stride,
            "gt_navmesh_valid_pixels": valid_count,
            "gt_navmesh_goal_position": goal.tolist(),
            "gt_navmesh_min_cost": best_distance,
            "gt_navmesh_min_pixel": best_pixel,
            "gt_navmesh_min_world_point": (
                best_world_point.tolist() if best_world_point is not None else None
            ),
            "gt_navmesh_min_snapped_point": (
                best_snapped_point.tolist() if best_snapped_point is not None else None
            ),
        }
        return costmap.astype(np.float32), vis_data

    def _load_query_anchor_pointmap(self, image_idx: int):
        image_idx = int(image_idx)
        if image_idx in self.query_anchor_pointmap_cache:
            return self.query_anchor_pointmap_cache[image_idx]

        pointmap = None
        for dirname in ("gt_pointmaps_fov90", "pointmaps"):
            pointmap_path = self.episode_path / dirname / f"{image_idx:05d}.npy"
            if pointmap_path.exists():
                pointmap = np.load(pointmap_path)
                break

        if pointmap is None and self.vggtnav_map_dir is not None:
            archive_path = self.vggtnav_map_dir / "nodes_vggt_points.npz"
            if archive_path.exists():
                if self.query_anchor_pointmap_archive is None:
                    self.query_anchor_pointmap_archive = np.load(archive_path)
                    self.query_anchor_pointmap_archive_keys = {
                        int(Path(key).stem): key
                        for key in self.query_anchor_pointmap_archive.files
                        if Path(key).stem.isdigit()
                    }
                key = self.query_anchor_pointmap_archive_keys.get(image_idx)
                if key is not None:
                    pointmap = self.query_anchor_pointmap_archive[key]

        if pointmap is None:
            return None
        pointmap = np.asarray(pointmap, dtype=np.float32)
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            logger.warning(
                "Ignoring invalid query-anchor pointmap %d with shape %s",
                image_idx,
                pointmap.shape,
            )
            return None

        valid = np.all(np.isfinite(pointmap), axis=2)
        if not np.any(valid):
            return None
        rows, cols = np.where(valid)
        cached = (cKDTree(pointmap[rows, cols]), rows, cols)
        self.query_anchor_pointmap_cache[image_idx] = cached
        return cached

    def _select_query_gt_anchor(self, depth: np.ndarray, candidate_img_indices):
        """Select the live query's best navmesh point, preferring a submap match."""
        _, query_vis = self._get_gt_navmesh_costmap(depth)
        query_pixel = query_vis.get("gt_navmesh_min_pixel")
        query_world = query_vis.get("gt_navmesh_min_world_point")
        if query_pixel is None or query_world is None:
            logger.warning("Query GT anchor has no valid navmesh minimum")
            return None

        query_world = np.asarray(query_world, dtype=np.float32)
        match_threshold = float(
            self.vggtnav_cfg.get("query_gt_anchor", {}).get("match_distance", 0.35)
        )
        best_match = None
        for image_idx in candidate_img_indices:
            cached = self._load_query_anchor_pointmap(int(image_idx))
            if cached is None:
                continue
            tree, rows, cols = cached
            distance, point_idx = tree.query(query_world, k=1)
            distance = float(distance)
            if distance > match_threshold:
                continue
            if best_match is None or distance < best_match[0]:
                best_match = (
                    distance,
                    int(image_idx),
                    (int(cols[int(point_idx)]), int(rows[int(point_idx)])),
                )

        query_cost = float(query_vis.get("gt_navmesh_min_cost", float("inf")))
        if best_match is not None:
            distance, image_idx, pixel = best_match
            return image_idx, pixel, query_cost, query_world, distance
        return None, tuple(query_pixel), query_cost, query_world, None

    def _load_navmesh_min_pixels(self, navmesh_min_path: str) -> None:
        navmesh_min_path = Path(navmesh_min_path)
        if not navmesh_min_path.exists():
            raise FileNotFoundError(f"Navmesh min pixels not found: {navmesh_min_path}")

        data = np.load(navmesh_min_path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            payload = {key: data[key] for key in data.files}
        elif isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            payload = data.item()
        elif isinstance(data, dict):
            payload = data
        else:
            raise ValueError(f"Unsupported navmesh min pixels format: {type(data)}")

        min_pixels = np.array(payload.get("min_pixels"))
        min_costs = np.array(payload.get("min_costs"))

        if min_pixels.ndim != 2 or min_pixels.shape[1] != 2:
            raise ValueError(f"Invalid min_pixels shape: {min_pixels.shape}")

        if min_costs.shape[0] != min_pixels.shape[0]:
            raise ValueError("min_costs length does not match min_pixels")

        if len(self.map_img_paths) != min_pixels.shape[0]:
            logger.warning(
                "Navmesh min pixels count does not match map images: "
                f"{min_pixels.shape[0]} vs {len(self.map_img_paths)}"
            )

        self.navmesh_min_pixels = min_pixels
        self.navmesh_min_costs = min_costs
        logger.warning(
            "GT anchor source: navmesh minima fallback %s (frames=%d)",
            navmesh_min_path,
            len(min_pixels),
        )

    def _load_navmesh_costmaps(self, navmesh_costmaps_path: str) -> None:
        navmesh_costmaps_path = Path(navmesh_costmaps_path)
        if not navmesh_costmaps_path.exists():
            raise FileNotFoundError(f"Navmesh costmaps not found: {navmesh_costmaps_path}")

        costmaps = np.load(navmesh_costmaps_path)
        if costmaps.ndim != 3:
            raise ValueError(f"Invalid navmesh costmaps shape: {costmaps.shape}")

        self.navmesh_costmaps = costmaps
        logger.warning(
            "GT anchor source: full Habitat navmesh costmaps %s (shape=%s)",
            navmesh_costmaps_path,
            costmaps.shape,
        )

    def _map_costmap_pixel_to_rgb(
        self,
        px: int,
        py: int,
        costmap_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Map a costmap-grid pixel to RGB-image coordinates, resolution-aware.

        The precomputed costmaps can be at any resolution (e.g. 16x16 or 224x224). We
        scale the selected cell center by the ratio of image size to costmap size so the anchor
        pixel lands in the (W, H) image frame regardless of the map's resolution. When the costmap
        already matches the image resolution this is a no-op.
        """
        costmap_h, costmap_w = int(costmap_shape[0]), int(costmap_shape[1])
        if (costmap_h, costmap_w) == (self.H, self.W):
            return int(px), int(py)

        scale_x = self.W / float(costmap_w)
        scale_y = self.H / float(costmap_h)
        rgb_x = int(round((px + 0.5) * scale_x - 0.5))
        rgb_y = int(round((py + 0.5) * scale_y - 0.5))
        rgb_x = int(np.clip(rgb_x, 0, self.W - 1))
        rgb_y = int(np.clip(rgb_y, 0, self.H - 1))
        return rgb_x, rgb_y

    def _select_min_cost_pixel_from_costmaps_array(
        self,
        costmaps: np.ndarray,
        candidate_img_indices,
        closest_map_img_idx: Optional[int],
    ) -> Tuple[int, Tuple[int, int], float]:
        if costmaps.ndim != 3:
            raise ValueError(f"Unexpected costmap shape: {costmaps.shape}")

        best_img_idx = None
        best_pixel = None
        best_cost = float("inf")

        for img_idx in candidate_img_indices:
            if img_idx < 0 or img_idx >= costmaps.shape[0]:
                continue

            costmap = costmaps[img_idx]
            if not np.isfinite(costmap).any():
                continue

            flat_idx = np.nanargmin(costmap)
            py, px = np.unravel_index(flat_idx, costmap.shape)
            cost = float(costmap[py, px])
            if not np.isfinite(cost):
                continue

            if best_img_idx is None or cost < best_cost:
                best_cost = cost
                best_img_idx = img_idx
                best_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)
            elif cost == best_cost and best_img_idx is not None and img_idx < best_img_idx:
                best_img_idx = img_idx
                best_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)

        if best_img_idx is not None and best_pixel is not None:
            return best_img_idx, best_pixel, best_cost

        fallback_idx = closest_map_img_idx
        if fallback_idx is None:
            fallback_idx = candidate_img_indices[0] if candidate_img_indices else 0

        fallback_pixel = None
        fallback_cost = float("inf")
        if 0 <= fallback_idx < costmaps.shape[0]:
            costmap = costmaps[fallback_idx]
            if np.isfinite(costmap).any():
                flat_idx = np.nanargmin(costmap)
                py, px = np.unravel_index(flat_idx, costmap.shape)
                fallback_cost = float(costmap[py, px])
                if np.isfinite(fallback_cost):
                    fallback_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)

        if fallback_pixel is None:
            logger.warning("Costmap min pixel missing for fallback image; using center pixel")
            height, width = costmaps.shape[1], costmaps.shape[2]
            fallback_pixel = self._map_costmap_pixel_to_rgb(width // 2, height // 2, (height, width))

        return int(fallback_idx), fallback_pixel, float(fallback_cost)

    def _select_min_median_costmap_pixel_from_costmaps_array(
        self,
        costmaps: np.ndarray,
        candidate_img_indices,
        closest_map_img_idx: Optional[int],
    ) -> Tuple[int, Tuple[int, int], float]:
        if costmaps.ndim != 3:
            raise ValueError(f"Unexpected costmap shape: {costmaps.shape}")

        if candidate_img_indices is None or len(candidate_img_indices) == 0:
            candidate_img_indices = [closest_map_img_idx] if closest_map_img_idx is not None else []

        best_img_idx = None
        best_median = float("inf")

        for img_idx in candidate_img_indices:
            if img_idx < 0 or img_idx >= costmaps.shape[0]:
                continue

            costmap = costmaps[img_idx]
            if not np.isfinite(costmap).any():
                continue

            median_cost = float(np.median(costmap))
            if median_cost < best_median:
                best_median = median_cost
                best_img_idx = img_idx

        if best_img_idx is None:
            logger.warning("No valid median costmaps found; falling back to min pixel")
            return self._select_min_cost_pixel_from_costmaps_array(
                costmaps,
                candidate_img_indices,
                closest_map_img_idx,
            )

        costmap = costmaps[best_img_idx]
        if not np.isfinite(costmap).any():
            logger.warning("Median-selected costmap invalid; falling back to min pixel")
            return self._select_min_cost_pixel_from_costmaps_array(
                costmaps,
                candidate_img_indices,
                closest_map_img_idx,
            )

        flat_idx = np.nanargmin(costmap)
        py, px = np.unravel_index(flat_idx, costmap.shape)
        best_cost = float(costmap[py, px])
        if not np.isfinite(best_cost):
            logger.warning("Median-selected costmap min pixel invalid; falling back to min pixel")
            return self._select_min_cost_pixel_from_costmaps_array(
                costmaps,
                candidate_img_indices,
                closest_map_img_idx,
            )

        return int(best_img_idx), self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape), best_cost

    def _select_anchor_from_navmesh_min(
        self,
        candidate_img_indices,
        closest_map_img_idx: Optional[int],
    ) -> Tuple[int, Tuple[int, int], float]:
        """Select anchor pixel using precomputed per-image navmesh minima."""
        if self.navmesh_min_pixels is None or self.navmesh_min_costs is None:
            raise ValueError("Navmesh min pixels are not loaded")

        tiebreaker = str(self.vggtnav_cfg.get("navmesh_min_tiebreaker", "max_img_idx")).lower()

        best_img_idx = None
        best_pixel = None
        best_cost = float("inf")

        for img_idx in candidate_img_indices:
            if img_idx < 0 or img_idx >= len(self.navmesh_min_costs):
                continue

            cost = float(self.navmesh_min_costs[img_idx])
            if not np.isfinite(cost):
                continue

            pixel = self.navmesh_min_pixels[img_idx]
            px, py = int(pixel[0]), int(pixel[1])
            if px < 0 or py < 0 or px >= self.W or py >= self.H:
                continue

            if best_img_idx is None or cost < best_cost:
                best_cost = cost
                best_img_idx = img_idx
                best_pixel = (px, py)
            elif cost == best_cost:
                if tiebreaker == "min_img_idx" and img_idx < best_img_idx:
                    best_img_idx = img_idx
                    best_pixel = (px, py)
                elif tiebreaker != "min_img_idx" and img_idx > best_img_idx:
                    best_img_idx = img_idx
                    best_pixel = (px, py)

        if best_img_idx is not None and best_pixel is not None:
            return best_img_idx, best_pixel, best_cost

        fallback_idx = closest_map_img_idx
        if fallback_idx is None:
            fallback_idx = candidate_img_indices[0] if candidate_img_indices else 0

        fallback_pixel = None
        fallback_cost = float("inf")
        if 0 <= fallback_idx < len(self.navmesh_min_costs):
            fallback_cost = float(self.navmesh_min_costs[fallback_idx])
            pixel = self.navmesh_min_pixels[fallback_idx]
            px, py = int(pixel[0]), int(pixel[1])
            if px >= 0 and py >= 0 and px < self.W and py < self.H and np.isfinite(fallback_cost):
                fallback_pixel = (px, py)

        if fallback_pixel is None:
            logger.warning("Navmesh min pixels missing for fallback image; using center pixel")
            fallback_pixel = (self.W // 2, self.H // 2)

        return int(fallback_idx), fallback_pixel, float(fallback_cost)

    def _select_min_cost_pixel_from_costmap(
        self,
        candidate_img_indices,
        closest_map_img_idx: Optional[int],
    ) -> Tuple[int, Tuple[int, int], float]:
        """Select the minimum-cost pixel from costmap data among candidate images."""
        if self.costmap_data is None:
            raise ValueError("Costmap data is required for topomap anchor selection")

        costmaps = self.costmap_data.get_costmap()
        if costmaps.ndim != 3:
            raise ValueError(f"Unexpected costmap shape: {costmaps.shape}")

        best_img_idx = None
        best_pixel = None
        best_cost = float("inf")

        for img_idx in candidate_img_indices:
            if img_idx < 0 or img_idx >= costmaps.shape[0]:
                continue

            costmap = costmaps[img_idx]
            if not np.isfinite(costmap).any():
                continue

            flat_idx = np.nanargmin(costmap)
            py, px = np.unravel_index(flat_idx, costmap.shape)
            cost = float(costmap[py, px])
            if not np.isfinite(cost):
                continue

            if best_img_idx is None or cost < best_cost:
                best_cost = cost
                best_img_idx = img_idx
                best_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)
            elif cost == best_cost and best_img_idx is not None and img_idx < best_img_idx:
                best_img_idx = img_idx
                best_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)

        if best_img_idx is not None and best_pixel is not None:
            return best_img_idx, best_pixel, best_cost

        fallback_idx = closest_map_img_idx
        if fallback_idx is None:
            fallback_idx = candidate_img_indices[0] if candidate_img_indices else 0

        fallback_pixel = None
        fallback_cost = float("inf")
        if 0 <= fallback_idx < costmaps.shape[0]:
            costmap = costmaps[fallback_idx]
            if np.isfinite(costmap).any():
                flat_idx = np.nanargmin(costmap)
                py, px = np.unravel_index(flat_idx, costmap.shape)
                fallback_cost = float(costmap[py, px])
                if np.isfinite(fallback_cost):
                    fallback_pixel = self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape)

        if fallback_pixel is None:
            logger.warning("Costmap min pixel missing for fallback image; using center pixel")
            height, width = costmaps.shape[1], costmaps.shape[2]
            fallback_pixel = self._map_costmap_pixel_to_rgb(width // 2, height // 2, (height, width))

        return int(fallback_idx), fallback_pixel, float(fallback_cost)

    def _select_min_median_costmap_pixel(
        self,
        candidate_img_indices,
        closest_map_img_idx: Optional[int],
    ) -> Tuple[int, Tuple[int, int], float]:
        """Select the min-cost pixel from the costmap with the lowest median cost."""
        if self.costmap_data is None:
            raise ValueError("Costmap data is required for topomap anchor selection")

        costmaps = self.costmap_data.get_costmap()
        if costmaps.ndim != 3:
            raise ValueError(f"Unexpected costmap shape: {costmaps.shape}")

        selected_imgs = self._select_min_median_path_length(candidate_img_indices)
        if len(selected_imgs) == 0:
            selected_imgs = [closest_map_img_idx] if closest_map_img_idx is not None else []

        if len(selected_imgs) == 0:
            height, width = costmaps.shape[1], costmaps.shape[2]
            return 0, self._map_costmap_pixel_to_rgb(width // 2, height // 2, (height, width)), float("inf")

        best_img_idx = int(selected_imgs[0])
        costmap = costmaps[best_img_idx]
        if not np.isfinite(costmap).any():
            logger.warning("Median-selected costmap has no finite values; falling back to min pixel")
            return self._select_min_cost_pixel_from_costmap(candidate_img_indices, closest_map_img_idx)

        flat_idx = np.nanargmin(costmap)
        py, px = np.unravel_index(flat_idx, costmap.shape)
        best_cost = float(costmap[py, px])
        if not np.isfinite(best_cost):
            logger.warning("Median-selected costmap min pixel invalid; falling back to min pixel")
            return self._select_min_cost_pixel_from_costmap(candidate_img_indices, closest_map_img_idx)

        return best_img_idx, self._map_costmap_pixel_to_rgb(int(px), int(py), costmap.shape), best_cost

    def _load_map_image(self, image_idx):
        image_path = self.map_img_paths[image_idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load map image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def set_controller(self):
        method_name = self.cfg.controller.name
        self.collided = None
        controller_cfg = self.cfg.controller

        if method_name in {'learnt', 'gnm', 'object_react', 'vggt_nav'}:
            goal_controller = ObjRelLearntController(
                config=controller_cfg.config_file,
            )
            goal_controller.reset_params()
        else:
            raise NotImplementedError("Other controller methods have not been implemented yet")
        
        self.goal_controller = goal_controller

    def setup_sim_agent(self):
        # Note: MAGNUM_LOG and HABITAT_SIM_LOG are set at module level before import
        sim_cfg = self.cfg.sim

        # Initialize Habitat Sim and Agent
        self.sim, self.agent, self.vel_control = get_sim_agent(
            scene_path=self.scene_glb_path,
            update_nav_mesh=sim_cfg.update_nav_mesh,
            width=sim_cfg.width,
            height=sim_cfg.height,
            hfov=sim_cfg.hfov,
            sensor_height=sim_cfg.sensor_height,
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control
    
    def ready_agent(self):

        # 1. Load agent trajectory
        agent_states_path = self.episode_path / 'agent_states.npy'
        if not agent_states_path.exists():
            raise FileNotFoundError(f"Agent states file not found: {agent_states_path}")

        self.agent_states = np.load(str(agent_states_path), allow_pickle=True)
        self.agent_positions_in_map = np.array([state.position for state in self.agent_states])

        # 2. Set goal based on task type
        self._set_goal_state()

        # 3. Select and set start state
        self._set_start_state()

        # 4. calculate distance metric
        self.distance_to_final_goal = calculate_path_distance(
            self.sim,
            self.start_position,
            self.final_goal_position,
        )
        self.goal_distance_threshold = float(self.cfg.goal_distance_threshold)
        if (
            self.cfg.task_type == 'alt_goal_v2'
            and np.isfinite(self.distance_to_final_goal)
            and self.distance_to_final_goal > 5.0
        ):
            # Alt-goal starts are intended to be approximately 5 m from the
            # target. Preserve the semantic goal used by the planner, but do
            # not require navigation through any excess Habitat snap distance.
            self.goal_distance_threshold += self.distance_to_final_goal - 5.0
            logger.info(
                "Alt-goal effective success threshold: %.2fm "
                "(base %.2fm, initial distance %.2fm)",
                self.goal_distance_threshold,
                float(self.cfg.goal_distance_threshold),
                self.distance_to_final_goal,
            )
        self.agent_state_history = []

        logger.info(f"Agent ready: task_type={self.cfg.task_type}, "
            f"reverse={self.cfg.reverse}, "
            f"start_idx={self.start_idx}, "
            f"Start Position={self.start_position}, "
            f"Goal Position={self.final_goal_position}, "
            f"goal_distance={self.distance_to_final_goal:.2f}m")
    
    def _set_goal_state(self):
        """Set goal state based on task type.

        Alt goal always uses the annotated semantic object, so it accepts the
        default "trajectory" as well as "semantic_instance", which evaluate.sh
        records for it. Every other task uses the agent position at the goal
        frame and needs "trajectory".
        """
        goal_method = str(self.cfg.get("goal_position_method", "trajectory"))
        if goal_method not in {"trajectory", "semantic_instance"}:
            raise ValueError(
                f"Unsupported goal_position_method={goal_method!r}. "
                "Expected trajectory, or semantic_instance for alt_goal_v2."
            )
        is_alt_goal = not self.cfg.reverse and self.cfg.task_type == 'alt_goal_v2'
        if not is_alt_goal and goal_method != "trajectory":
            raise ValueError(
                f"goal_position_method={goal_method!r} is only valid for "
                "alt_goal_v2. Imitate, Reverse and Shortcut use trajectory."
            )

        if self.cfg.reverse:
            self._set_reverse_goal()
        elif is_alt_goal:
            self._set_alt_goal()
        elif self.cfg.task_type not in ('original', 'via_alt_goal'):
            raise ValueError(
                f"Unsupported task_type={self.cfg.task_type!r}. "
                "Expected original, alt_goal_v2 or via_alt_goal."
            )
        else:
            self._set_topological_goal()

    def _set_reverse_goal(self):
        """Set goal for reverse navigation task."""
        self.final_goal_position = self.agent_states[0].position
        self.final_goal_image_idx = len(self.agent_states) - 1

        logger.debug(f"Reverse goal set: image_idx={self.final_goal_image_idx}")
    
    def _set_alt_goal(self):
        """Alt-goal protocol: the goal is the annotated semantic object itself.

        The AABB center of the object is moved to the trajectory floor height and
        snapped to the NavMesh. The annotation file holds the object instance and
        the frame it was seen in.
        """
        goal_info_episode_path = Path(self.episode_path)
        annotation_name = "seen_but_unvisited_object_v2.npy"
        if not (goal_info_episode_path / annotation_name).exists():
            # The ObjectReact download keeps the alt-goal annotation and semantic
            # masks in object-rel-nav/hm3d_iin_val instead of the episode folder.
            datasets_root = goal_info_episode_path.parent.parent.parent
            object_rel_episode_path = (
                datasets_root / "object-rel-nav" / "hm3d_iin_val" / goal_info_episode_path.name
            )
            if not (object_rel_episode_path / annotation_name).exists():
                raise FileNotFoundError(
                    f"Alt-goal annotation {annotation_name} missing from both "
                    f"{goal_info_episode_path} and {object_rel_episode_path}"
                )
            goal_info_episode_path = object_rel_episode_path

        self.final_goal_image_idx, goal_mask, goal_instance_id = get_goal_info(
            str(goal_info_episode_path), self.cfg.task_type
        )

        instance_position = None
        for instance in self.sim.semantic_scene.objects:
            if int(instance.semantic_id) == int(goal_instance_id):
                instance_position = instance.aabb.center
                break

        if instance_position is None:
            raise ValueError(f'Goal instance {goal_instance_id} not found in scene')

        avg_floor_height = self.agent_positions_in_map[:, 1].mean()
        instance_position = np.array(instance_position, dtype=np.float32)
        instance_position[1] = avg_floor_height
        self.final_goal_position = self.sim.pathfinder.snap_point(instance_position)
        # Match the original alt-goal evaluation: frames after the annotated
        # goal observation are not valid localization/submap candidates.
        self.agent_positions_in_map = self.agent_positions_in_map[
            : self.final_goal_image_idx + 1
        ]

        logger.info(
            "Alt goal semantic instance: image_idx=%d instance_id=%d center=%s snapped=%s",
            self.final_goal_image_idx,
            goal_instance_id,
            instance_position.tolist(),
            np.asarray(self.final_goal_position).tolist(),
        )

    def _set_topological_goal(self):
        """Set the Imitate or Shortcut goal from the propagation costmap metadata.

        The goal frame stored by create_vggt_prop_map.py is the frame the map
        costmaps were propagated from. The goal position is the recorded agent
        position at that frame.
        """
        if self.costmap_data is not None:
            metadata = self.costmap_data.get_metadata()
        else:
            # anchor_source=navmesh_min does not load the costmap array, but the
            # goal frame still comes from its sidecar.
            metadata = load_costmap_metadata(
                self.costmap_file_path, self._resolve_costmap_metadata_path()
            )
        if "goal_img_idx" not in metadata:
            raise KeyError(
                f"goal_img_idx missing from the costmap metadata of {self.costmap_file_path}"
            )
        self.final_goal_image_idx = int(metadata["goal_img_idx"])
        self.final_goal_position = np.array(
            self.agent_states[self.final_goal_image_idx].position, dtype=np.float32
        )

    def _set_start_state(self):
        """
        Select and set the agent's starting state based on start_state_mode.
        
        Modes:
            - "random": Sample random navigable points with distance constraints
            - "trajectory": Select from recorded trajectory at target distance from goal
            - "fixed_idx": Use specific trajectory index (from start_idx config)
        """
        mode = getattr(self.cfg, 'start_state_mode', 'random')
        
        if mode == "random":
            # Random start state with distance constraints
            start_state = pick_random_start_state(
                sim=self.sim,
                cfg=self.cfg,
                final_goal_position=self.final_goal_position,
                agent_positions_in_map=self.agent_positions_in_map,
                max_tries=100
            )
            logger.debug("Random start state selected")
            
        elif mode == "trajectory":
            # Select from recorded trajectory at target distance from goal
            trajectory_states = self.agent_states
            if self.cfg.task_type == 'alt_goal_v2':
                # An alt-goal is annotated in a frame where the object was seen.
                # Do not choose a start after that observation: doing so can place
                # the robot beyond the retained map and turn the task into a
                # reverse traversal.
                trajectory_states = self.agent_states[
                    : self.final_goal_image_idx + 1
                ]
            start_state = select_trajectory_start_state(
                sim=self.sim,
                cfg=self.cfg,
                agent_states=trajectory_states,
                goal_position=self.final_goal_position
            )
            logger.debug("Trajectory-based start state selected")
            
        elif mode == "fixed_idx":
            # Use specific trajectory index
            if self.start_idx >= len(self.agent_states):
                raise ValueError(f"start_idx {self.start_idx} out of range "
                               f"(trajectory has {len(self.agent_states)} states)")
            start_state = self.agent_states[self.start_idx]
            logger.debug(f"Fixed index start state: idx={self.start_idx}")
            
        else:
            raise ValueError(f"Unknown start_state_mode: {mode}. "
                           f"Expected: random, trajectory, fixed_idx")
        
        if start_state is None:
            raise ValueError(f'Could not find valid start state for {self.episode_path}')
        
        self.agent.set_state(start_state)
        self.start_position = start_state.position
        
        logger.debug(f"Start state set: mode={mode}, position={self.start_position}")
        
        return start_state
    
    def get_goal(self, rgb, depth, pose=None, return_vis_data=False):
        """
        Get goal mask for the current observation.

        Args:
            rgb: RGB observation (H, W, 3)
            depth: Depth map (H, W), only used by the privileged GT navmesh paths
            pose: Agent pose (optional)
            return_vis_data: If True, return (goal_mask, vis_data) tuple

        Returns:
            goal_mask: Distance-to-goal costmap (H, W)
            OR
            (goal_mask, vis_data): If return_vis_data=True, dict contains match data
        """
        if self.gt_navmesh_costmap_enabled:
            self.goal_mask, vis_data = self._get_gt_navmesh_costmap(depth)
            self.control_input_learnt = self.goal_mask
            if return_vis_data:
                return self.goal_mask, vis_data
            return self.goal_mask

        # Getting the closest reference image to the particular query image.
        # MegaLoc, when configured, replaces the oracle (GT pose) localization.
        if self.retriever is not None:
            goal_frame_idx = int(self.final_goal_image_idx)
            # The goal frame index is part of the alt-goal task. Oracle
            # localization never considers frames after it, so MegaLoc gets
            # the same candidates.
            is_alt_goal = self.cfg.task_type == "alt_goal_v2"
            goal_lock = bool(self.cfg.localizer.get("megaloc_goal_lock", False))
            localized_img_idxs, closest_map_img_idx = self.retriever.retrieve(
                rgb,
                last_frame_idx=goal_frame_idx if is_alt_goal else None,
                lock_frame_idx=goal_frame_idx if goal_lock else None,
            )
        elif self.cfg.localizer.use_gt_localization:
            if pose is not None:
                localized_img_idxs, closest_map_img_idx = self.get_closest_map_img_from_odometry(
                    pose, self.episode_path
                )
            else:
                localized_img_idxs, closest_map_img_idx = self.get_gt_closest_map_img()
                closest_map_img_idx = self._select_min_median_path_length(localized_img_idxs)[0]
        else:
            raise ValueError(
                "VGGT-Nav requires pose localization or a configured RGB retriever"
            )

        if self.vggtnav_enabled:
            if len(localized_img_idxs) == 0:
                localized_img_idxs = [closest_map_img_idx]

            anchor_cost = None
            anchor_world_position = None
            anchor_match_distance = None
            query_anchor = None
            if bool(self.vggtnav_cfg.get("query_gt_anchor", {}).get("enabled", False)):
                if depth is None:
                    raise ValueError("vggtnav.query_gt_anchor requires GT depth")
                query_anchor = self._select_query_gt_anchor(depth, localized_img_idxs)

            if query_anchor is not None:
                (
                    anchor_img_idx,
                    anchor_pixel,
                    anchor_cost,
                    anchor_world_position,
                    anchor_match_distance,
                ) = query_anchor
            elif self.vggtnav_anchor_source == "navmesh_min":
                topomap_mode = str(self.vggtnav_cfg.get("topomap_anchor_mode", "min_pixel")).lower()
                if self.navmesh_costmaps is not None and topomap_mode in {"min_pixel", "min_cost_pixel"}:
                    anchor_img_idx, anchor_pixel, anchor_cost = self._select_min_cost_pixel_from_costmaps_array(
                        self.navmesh_costmaps,
                        localized_img_idxs,
                        closest_map_img_idx,
                    )
                elif self.navmesh_costmaps is not None and topomap_mode in {"min_median_costmap", "median_costmap"}:
                    anchor_img_idx, anchor_pixel, anchor_cost = self._select_min_median_costmap_pixel_from_costmaps_array(
                        self.navmesh_costmaps,
                        localized_img_idxs,
                        closest_map_img_idx,
                    )
                else:
                    anchor_img_idx, anchor_pixel, anchor_cost = self._select_anchor_from_navmesh_min(
                        localized_img_idxs,
                        closest_map_img_idx,
                    )
            else:
                topomap_mode = str(self.vggtnav_cfg.get("topomap_anchor_mode", "min_pixel")).lower()
                if topomap_mode in {"min_pixel", "min_cost_pixel"}:
                    anchor_img_idx, anchor_pixel, anchor_cost = self._select_min_cost_pixel_from_costmap(
                        localized_img_idxs,
                        closest_map_img_idx,
                    )
                elif topomap_mode in {"min_median_costmap", "median_costmap"}:
                    anchor_img_idx, anchor_pixel, anchor_cost = self._select_min_median_costmap_pixel(
                        localized_img_idxs,
                        closest_map_img_idx,
                    )
                else:
                    raise ValueError(f"Unknown vggtnav.topomap_anchor_mode: {topomap_mode}")
            if anchor_img_idx is not None and anchor_img_idx not in localized_img_idxs:
                localized_img_idxs = [anchor_img_idx]

            candidate_images = [self._load_map_image(idx) for idx in localized_img_idxs]
            anchor_frame_index = (
                0 if anchor_img_idx is None
                else localized_img_idxs.index(anchor_img_idx) + 1
            )

            if anchor_cost is None:
                logger.debug(
                    f"VGGTNav anchor source={self.vggtnav_anchor_source}, "
                    f"image idx={anchor_img_idx}, anchor_pixel={anchor_pixel}, "
                    f"candidate images={localized_img_idxs}"
                )
            else:
                logger.debug(
                    f"VGGTNav anchor source={self.vggtnav_anchor_source}, "
                    f"image idx={anchor_img_idx}, anchor_pixel={anchor_pixel}, "
                    f"cost={anchor_cost:.3f}, candidate images={localized_img_idxs}"
                )

            costmap_forward_start = time.perf_counter()
            self.goal_mask, self.goal_mask_raw = predict_vggtnav_costmap(
                model=self.vggtnav_model,
                query_image=rgb,
                submap_images=candidate_images,
                anchor_frame_index=anchor_frame_index,
                anchor_pixel=anchor_pixel,
                img_size=int(self.vggtnav_cfg.get("img_size", 224)),
                patch_size=int(self.vggtnav_cfg.get("patch_size", 14)),
                normalize=True,
                upsample_size=int(self.vggtnav_cfg.get("upsample_size", 60)),
                device=self.device
            )
            self.vggtnav_costmap_forward_count = (
                getattr(self, "vggtnav_costmap_forward_count", 0) + 1
            )
            self.vggtnav_costmap_forward_seconds = (
                getattr(self, "vggtnav_costmap_forward_seconds", 0.0)
                + time.perf_counter() - costmap_forward_start
            )

            self.control_input_learnt = self.goal_mask

            if return_vis_data:
                vis_data = {
                    'localized_img_idxs': localized_img_idxs,
                    'closest_map_img_idx': closest_map_img_idx,
                    'vggtnav_anchor_img_idx': -1 if anchor_img_idx is None else anchor_img_idx,
                    'vggtnav_anchor_pixel': tuple(anchor_pixel),
                    'vggtnav_anchor_frame_index': int(anchor_frame_index),
                    'vggtnav_anchor_source': self.vggtnav_anchor_source,
                    'vggtnav_topomap_anchor_mode': str(self.vggtnav_cfg.get("topomap_anchor_mode", "min_pixel")),
                    'vggtnav_anchor_cost': anchor_cost,
                    'vggtnav_anchor_world_position': anchor_world_position,
                    'vggtnav_anchor_match_distance': anchor_match_distance,
                }
                return self.goal_mask, vis_data

            return self.goal_mask

        # __init__ rejects configs with neither goal source, so this is a bug.
        raise RuntimeError("get_goal reached with neither VGGTNav nor GT navmesh costmaps enabled")

    def get_gt_closest_map_img(self):
        dists = np.linalg.norm(
            self.agent_positions_in_map - self.agent.get_state().position, axis=1)
        
        top_k = 2 * self.loc_radius
        closest_idxs = np.argsort(dists)[:top_k]
        closest_idxs = sorted(closest_idxs)[::self.subsample_ref]
        logger.info(f"Top K closest idxs: {closest_idxs = }")
        closest_idx = np.argmin(dists)
        return closest_idxs, closest_idx

    def get_closest_map_img_from_odometry(self, odom_pose, episode_path, position_weight=1.0, rotation_weight=1.0):
        """
        Given a robot odometry pose (x, y, z, qx, qy, qz, qw), find the image index in poses_odom.txt
        that is closest to this pose using both translation and rotation.

        Returns:
            closest_idx: int, index of the closest image
            localized_img_inds: list of indices, sorted by combined distance (topK, subsampled)
        """
        # Mapping sessions store poses as text, while the benchmark episodes
        # store the same Habitat states in agent_states.npy.
        odom_file = Path(episode_path) / 'poses_odom.txt'
        poses = []
        quats = []
        if odom_file.exists():
            with open(odom_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    poses.append([x, y, z])
                    quats.append([qx, qy, qz, qw])
        else:
            states = getattr(self, 'agent_states', None)
            if states is None or len(states) == 0:
                states_path = Path(episode_path) / 'agent_states.npy'
                if not states_path.exists():
                    raise FileNotFoundError(
                        f"Neither {odom_file} nor {states_path} exists."
                    )
                states = np.load(str(states_path), allow_pickle=True)
            for state in states:
                position = state.position
                rotation = state.rotation
                poses.append([float(position[0]), float(position[1]), float(position[2])])
                quats.append([
                    float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
                ])
        poses = np.array(poses)  # shape (N, 3)
        quats = np.array(quats)  # shape (N, 4)

        if (
            self.cfg.task_type == 'alt_goal_v2'
            and self.final_goal_image_idx is not None
        ):
            end_idx = min(len(poses), int(self.final_goal_image_idx) + 1)
            poses = poses[:end_idx]
            quats = quats[:end_idx]

        pos_query = np.array(odom_pose[:3])
        quat_query = np.array(odom_pose[3:7])

        trans_dists = np.linalg.norm(poses - pos_query, axis=1)

        def quat_angle(q1, q2):
            dot = np.abs(np.sum(q1 * q2, axis=-1))
            dot = np.clip(dot, -1.0, 1.0)
            return 2 * np.arccos(dot)  # angle in radians

        rot_dists = quat_angle(quats, quat_query)
        total_dists = position_weight * trans_dists + rotation_weight * rot_dists

        closest_idx = int(np.argmin(total_dists))
        # Sort all indices by distance, take topK and subsample
        topK = 2 * self.loc_radius
        sorted_idxs = np.argsort(total_dists)[:topK]
        localized_img_idxs = sorted(sorted_idxs.tolist())[::self.subsample_ref]
        
        return localized_img_idxs, closest_idx

    def _select_min_median_path_length(self, candidate_img_indices):
        if self.costmap_data is None or len(candidate_img_indices) == 0:
            return [candidate_img_indices[0]] if candidate_img_indices else []

        min_median_path_length = 100 # Max Value

        best_ref_img_idx = None
        img_costmaps = self.costmap_data.get_costmap()
        for ref_img_idx in candidate_img_indices:
            # (H, W)
            img_pls = img_costmaps[ref_img_idx]

            median_path_length = np.median(img_pls)
            logger.info(f"FOUND Image {ref_img_idx} has {median_path_length} median path length")
            if median_path_length < min_median_path_length:
                min_median_path_length = median_path_length
                best_ref_img_idx = ref_img_idx
        
        if best_ref_img_idx is not None:
            logger.info(f"Selected Image {best_ref_img_idx} with {min_median_path_length} median path length")
            return [best_ref_img_idx]
        else:
            logger.warning(f"No good matches found, using the 0th index image: {candidate_img_indices[0] = }")
            return [candidate_img_indices[0]]

    def _care_intrinsics(self):
        intrinsics = self.agent_intrinsics.detach().cpu().numpy()
        return {
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        }

    def _apply_collision_avoidance(self, rgb, depth):
        self.care_debug = None
        if not self.collision_avoidance_enabled:
            return
        if self.collision_avoidance_method != "care":
            raise NotImplementedError(
                f"{self.collision_avoidance_method} collision avoidance is not implemented"
            )
        if depth is None:
            logger.warning("CARE skipped because depth is None")
            return
        if not hasattr(self.goal_controller, "action_pred") or self.goal_controller.action_pred is None:
            logger.warning("CARE skipped because controller waypoints are unavailable")
            return

        params = dict(
            OmegaConf.to_container(
                self.collision_avoidance_cfg.get("care", {}),
                resolve=True,
            )
        )
        original_waypoints = np.asarray(self.goal_controller.action_pred)
        output = care_step(
            rgb=rgb,
            depth=np.asarray(depth),
            waypoints=original_waypoints,
            intrinsics=self._care_intrinsics(),
            params=params,
        )

        self.care_debug = {
            "original_waypoints": original_waypoints,
            "adjusted_waypoints": np.asarray(output["adjusted_waypoints"]),
            "obstacles": np.asarray(output["obstacles"]),
            "theta_rot": float(output["theta_rot"]),
            "k_star": -1 if output["k_star"] is None else int(output["k_star"]),
            "v": float(output["v"]),
            "omega": float(output["omega"]),
        }

        if self.collision_avoidance_cfg.get("override_velocity", True):
            self.velocity_control = float(output["v"])
        # execute_action applies steer=-theta_control, while CARE omega is the
        # desired local heading sign, so store the negated value here.
        self.theta_control = -float(output["omega"])

        if self.controller_logs is not None and len(self.controller_logs) > 0:
            self.controller_logs[-1]["care"] = {
                "v": float(output["v"]),
                "omega": float(output["omega"]),
                "theta_rot": float(output["theta_rot"]),
                "k_star": output["k_star"],
                "num_obstacles": int(output["obstacles"].shape[0]),
            }
        logger.info(
            "CARE adjusted control: "
            f"v={self.velocity_control:.3f}, "
            f"theta_control={self.theta_control:.3f}, "
            f"obstacles={output['obstacles'].shape[0]}"
        )

    def get_control_signal(self, rgb, depth):
        control_method = self.cfg.controller.name

        if control_method in ('learnt', 'gnm', 'object_react', 'vggt_nav'):
            if self.control_input_learnt[0] is None or self.control_input_learnt[1] is None:
                self.velocity_control, self.theta_control = 0, 0
            else:
                self.velocity_control, self.theta_control = self.goal_controller.predict(
                    rgb, self.control_input_learnt)
            
            self.controller_logs = self.goal_controller.controller_logs
            self._apply_collision_avoidance(rgb, depth)
            # NOTE: In simulation, theta is NOT negated here.
            # It's only negated for real robot (env != 'sim').
            # The negation for sim happens in execute_action with steer=-self.theta_control
        else:
            raise NotImplementedError(f"{control_method} is not available...")

    def execute_action(self):
        control_method = self.cfg.controller.name
        if control_method in ('learnt', 'gnm', 'object_react', 'vggt_nav'):
            self.agent, self.sim, self.collided = apply_velocity(
                vel_control=self.vel_control,
                agent=self.agent,
                sim=self.sim,
                velocity=self.velocity_control,
                steer=-self.theta_control,  # opposite y axis
                time_step=self.time_delta
            )  # will add velocity control once steering is working
        else:
            raise NotImplementedError("Other controller methods task not implemented yet.")
        
        self.agent_state_history.append(self.agent.get_state())
    
    def update_distance_to_goal(self):
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position)[0]
        return self.distance_to_goal

    def is_done(self):
        done = False
        self.update_distance_to_goal()
        if self.distance_to_goal <= self.goal_distance_threshold:
            logger.info(
                f'\nWinner! dist to goal: {self.distance_to_goal:.6f}\n')
            self.success_status = 'success'
            done = True
        return done
    
    def setup_logging(self):
        self.episode_metadata_filepath = self.episode_results_path / 'metadata.txt'
        self.episode_results_csv = self.episode_results_path / 'results.csv'

        # Initialize results files
        initialize_results(
            metadata_file=self.episode_metadata_filepath,
            results_csv=self.episode_results_csv,
            method=self.cfg.controller.name,
            goal_source=self.cfg.goal_source,
            max_steps=self.cfg.max_steps,
            goal_distance_threshold=self.goal_distance_threshold,
            pid_steer_values=self.pid_steer_values,
            hfov_degrees=self.fov_deg,
            time_delta=self.time_delta,
            velocity_control=self.velocity_control,
            goal_position=self.final_goal_position,
        )

        # Initialize results dictionary for accumulating per-step data
        results_dict_keys = [
            "step",
            "distance_to_goal",
            "velocity_control",
            "theta_control",
            "collided",
            "discrete_action",
            "agent_states",
            "controller_logs",
        ]
        self.results_dict = {k: [] for k in results_dict_keys}

    def log_results(self, step: int, final: bool = False) -> None:
        """
        Log per-step or final results to files.
        
        Args:
            step: Current step number
            final: If True, write final metadata and save results_dict.npz
        """
        if not final:
            # Write per-step results to CSV
            write_results(
                results_csv=self.episode_results_csv,
                step=step,
                current_robot_state=self.agent.get_state() if self.agent is not None else None,
                distance_to_goal=self.distance_to_goal,
                velocity_control=self.velocity_control,
                theta_control=self.theta_control,
                collided=self.collided,
                discrete_action=self.discrete_action
            )
            
            # Accumulate results in results_dict
            results_dict_curr = {
                "step": step,
                "distance_to_goal": self.distance_to_goal,
                "velocity_control": self.velocity_control,
                "theta_control": self.theta_control,
                "collided": self.collided,
                "discrete_action": self.discrete_action,
                "agent_states": self.agent.get_state() if self.agent is not None else None,
                "controller_logs": self.controller_logs[-1] if self.controller_logs is not None and len(self.controller_logs) > 0 else None,
            }
            self.update_results_dict(results_dict_curr)
        else:
            # Write final metadata
            write_final_meta_results(
                metadata_file=self.episode_metadata_filepath,
                success_status=self.success_status,
                final_distance=self.distance_to_goal,
                step=step,
                distance_to_final_goal=self.distance_to_final_goal
            )
            
            # Save accumulated results dictionary as npz
            np.savez(
                self.episode_results_path / 'results_dict.npz',
                **self.results_dict
            )
    
    def update_results_dict(self, curr_dict: dict) -> None:
        """
        Append current step's data to the results dictionary.
        
        Args:
            curr_dict: Dictionary with current step's data
        """
        for k, v in curr_dict.items():
            self.results_dict[k].append(v)

def init_results_dir_and_save_cfg(cfg: DictConfig, default_logger=None):
    # Build results path from config
    results_path = Path(cfg.results_dirpath) if cfg.results_dirpath.startswith('/') else Path.cwd() / cfg.results_dirpath

    # Create structured folder path
    task_str = cfg.task_type
    if cfg.get('reverse', False):
        task_str += '_reverse'
    
    results_dirpath = (results_path / task_str / cfg.exp_name /
    f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{cfg.controller.name}_{cfg.goal_source}')
    results_dirpath.mkdir(exist_ok=True, parents=True)

    # Update logger file handler
    if default_logger is not None:
        default_logger.update_file_handler_root(results_dirpath / 'output.log')
    
    logger.info(f'Logging to {results_dirpath}')

    # Save config
    OmegaConf.save(cfg, results_dirpath / 'config.yaml')
    return results_dirpath
