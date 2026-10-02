"""Compute ground-truth geodesic distance costmaps on the Habitat NavMesh.

Each frame's pointmap is snapped to the NavMesh floor and the geodesic distance
to the goal is computed per pixel, in parallel worker processes. Pixels on
disconnected NavMesh islands are filled from their nearest valid neighbor.
mard_benchmark/navmesh_extractor.py imports the worker and goal functions from
here. It can also run alone. --parent_dir defaults to the path below and needs
PLANN3R_ROOT, and --scene-list defaults to episodes_removing_blacklist.txt.

Usage:
    cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
    PYTHONNOUSERSITE=1 pixi run python gt_mesh_generator.py \
      --parent_dir "$PLANN3R_ROOT/evaluation/datasets/hm3d_navigation/hm3d_iin_val_320x240" \
      --output_dir "$PLANN3R_ROOT/evaluation/mard"
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, Any

import matplotlib
import numpy as np
import habitat_sim
from scipy.spatial import cKDTree  # Added for fast hole filling
from tqdm import tqdm


# Agg backend for headless plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables for worker processes
_worker_navmesh: Optional[habitat_sim.PathFinder] = None

GOAL_TASK_TYPE = "original"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_LIST_DEFAULT = REPO_ROOT / "episodes_removing_blacklist.txt"


def _ensure_agent_state_unpickle():
    import sys
    import types

    try:
        import habitat_sim  # noqa: F401
    except Exception:
        habitat_sim = types.ModuleType("habitat_sim")
        agent_mod = types.ModuleType("habitat_sim.agent")
        agent_agent_mod = types.ModuleType("habitat_sim.agent.agent")

        class AgentState:
            def __init__(self):
                pass

        agent_agent_mod.AgentState = AgentState
        agent_mod.agent = agent_agent_mod
        habitat_sim.agent = agent_mod

        sys.modules.setdefault("habitat_sim", habitat_sim)
        sys.modules.setdefault("habitat_sim.agent", agent_mod)
        sys.modules.setdefault("habitat_sim.agent.agent", agent_agent_mod)

    try:
        import quaternion  # noqa: F401
    except Exception:
        quat_mod = types.ModuleType("quaternion")
        class quaternion:
            def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
                self.w = w
                self.x = x
                self.y = y
                self.z = z
            def __iter__(self):
                return iter((self.w, self.x, self.y, self.z))
            def __repr__(self):
                return f"quaternion({self.w}, {self.x}, {self.y}, {self.z})"
        quat_mod.quaternion = quaternion
        sys.modules.setdefault("quaternion", quat_mod)


def load_scene_names(scene_list_path: Path) -> set:
    if not scene_list_path.exists():
        raise ValueError(f"Scene list file not found: {scene_list_path}")

    lines = scene_list_path.read_text().splitlines()
    return {
        Path(line.strip()).name
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


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


def count_frames_from_images(images_dir: Path) -> int:
    count = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        count += len(list(images_dir.glob(ext)))
    return count


def ensure_pointmaps(session_folder: Path) -> Optional[Path]:
    pointmap_dir = session_folder / "gt_pointmaps_fov90"
    if pointmap_dir.is_dir() and any(pointmap_dir.glob("*.npy")):
        return pointmap_dir

    fallback_dir = session_folder / "pointmaps"
    if fallback_dir.is_dir() and any(fallback_dir.glob("*.npy")):
        return fallback_dir

    logger.info("Pointmaps missing in %s. Generating...", session_folder)
    try:
        from write_gt_pointmaps import process_session
    except Exception as exc:
        raise RuntimeError(f"Failed to import pointmap generator: {exc}") from exc

    process_session(str(session_folder))

    if pointmap_dir.is_dir() and any(pointmap_dir.glob("*.npy")):
        return pointmap_dir
    if fallback_dir.is_dir() and any(fallback_dir.glob("*.npy")):
        return fallback_dir

    return None


def init_worker(navmesh_path: str):
    """Initialize the worker process with a cached NavMesh."""
    global _worker_navmesh
    
    try:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(config)
        
        if not sim.pathfinder.load_nav_mesh(navmesh_path):
            raise RuntimeError(f"Worker failed to load navmesh: {navmesh_path}")
            
        _worker_navmesh = sim.pathfinder
        
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def snap_point_to_floor(
    point_3d: np.ndarray,
    navmesh: habitat_sim.PathFinder,
    camera_height: float,
    height_tolerance: float = 0.5
) -> np.ndarray:
    """
    Snap a 3D point to the navmesh, preferring the floor at camera level.
    
    For ceiling points (above camera), force snapping downward to avoid
    snapping to upper floors in multi-story buildings.
    
    Args:
        point_3d: 3D point to snap
        navmesh: Habitat PathFinder
        camera_height: Y-coordinate of camera position
        height_tolerance: Vertical tolerance for considering same floor (meters)
        
    Returns:
        Snapped 3D point on the navigable surface
    """
    # Check if point is above camera (likely ceiling)
    is_ceiling = point_3d[1] > camera_height + height_tolerance
    
    if is_ceiling:
        # For ceiling points, create a point below and snap that
        # This forces snapping to the floor below rather than floor above
        point_below = point_3d.copy()
        point_below[1] = camera_height - 0.5  # Force it below camera
        snapped = navmesh.snap_point(point_below)
    else:
        # For floor/wall points, normal snapping is fine
        snapped = navmesh.snap_point(point_3d)
    
    # Verify the snapped point is on the correct floor
    # If it's too far vertically from camera level, try to re-snap
    if abs(snapped[1] - camera_height) > 3.0:  # More than 3m away vertically
        # Try snapping with constrained height
        point_constrained = point_3d.copy()
        point_constrained[1] = camera_height
        snapped = navmesh.snap_point(point_constrained)
    
    return snapped


def fill_holes_nearest(costmap: np.ndarray, invalid_val: float = np.inf) -> np.ndarray:
    """
    Fills 'inf' values (holes) in the costmap using the value of the 
    nearest valid pixel (Euclidean distance in pixel space).
    
    This fixes issues where points snap to disconnected NavMesh islands 
    (like tables) by assigning them the cost of the adjacent floor.
    """
    H, W = costmap.shape
    
    # Identify valid and invalid pixels
    # We consider anything finite as valid.
    valid_mask = np.isfinite(costmap)
    
    # If the whole image is empty or full, return as is
    if valid_mask.all():
        return costmap
    if not valid_mask.any():
        return costmap

    # Get coordinates
    # y_valid, x_valid = (N, ), (N, )
    y_valid, x_valid = np.where(valid_mask)
    y_missing, x_missing = np.where(~valid_mask)
    
    # Stack into (N, 2) arrays for KDTree
    coords_valid = np.stack([y_valid, x_valid], axis=1)
    coords_missing = np.stack([y_missing, x_missing], axis=1)
    
    # Build Tree on valid pixels
    tree = cKDTree(coords_valid)
    
    # Query nearest neighbor for every missing pixel
    # k=1 returns (distances, indices)
    _, indices = tree.query(coords_missing, k=1)
    
    # Fill values
    # The nearest valid value for each missing pixel
    filled_costmap = costmap.copy()
    filled_costmap[y_missing, x_missing] = costmap[y_valid[indices], x_valid[indices]]
    
    return filled_costmap


def compute_frame_costmap(
    frame_idx: int,
    pointmap_path: Path,
    goal_snapped: np.ndarray,
    camera_positions: dict
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Worker function to process a single frame.
    1. Projects 3D points to NavMesh (floor-aware).
    2. Computes Geodesic distance.
    3. Fills holes (disconnected islands) via Nearest Neighbor.
    """
    global _worker_navmesh

    if _worker_navmesh is None:
        return frame_idx, None, None

    try:
        if not pointmap_path.exists():
            return frame_idx, None, None
        
        pointmap = np.load(pointmap_path) # (H, W, 3)
        H, W = pointmap.shape[:2]
        
        costmap = np.full((H, W), np.inf, dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)

        # 1. Identify valid 3D points
        valid_pixels_mask = np.all(np.isfinite(pointmap), axis=2)
        
        if not np.any(valid_pixels_mask):
            return frame_idx, costmap, valid_mask

        rows, cols = np.where(valid_pixels_mask)
        points_3d = pointmap[rows, cols]

        # 2. Compute Geodesic with floor-aware snapping
        goal_pos = goal_snapped
        nav = _worker_navmesh
        distances = np.full(len(rows), np.inf, dtype=np.float32)
        
        # Get camera position for this frame (if available)
        camera_pos = None
        if camera_positions is not None and frame_idx in camera_positions:
            camera_pos = camera_positions[frame_idx]
        
        path = habitat_sim.ShortestPath()
        path.requested_end = goal_pos

        for i in range(len(rows)):
            pt = points_3d[i]
            # Snap to navigable surface (floor-aware if camera position available)
            if camera_pos is not None:
                snapped_pt = snap_point_to_floor(pt, nav, camera_pos[1])
            else:
                snapped_pt = nav.snap_point(pt)
            
            path.requested_start = snapped_pt
            found = nav.find_path(path)
            
            if found:
                distances[i] = path.geodesic_distance
            # If not found, it remains inf (will be filled later)

        costmap[rows, cols] = distances
        
        # 3. Fill Holes (In-painting)
        # We perform this on the 2D image to fix table-tops/islands
        costmap_filled = fill_holes_nearest(costmap)
        
        # Re-compute valid mask based on the filled map
        # Now valid means "reachable or near something reachable"
        final_valid_mask = np.isfinite(costmap_filled) & valid_pixels_mask

        return frame_idx, costmap_filled, final_valid_mask

    except Exception as e:
        return frame_idx, None, None


def save_visualization(
    costmap: np.ndarray, 
    output_path: Path, 
    cmap: str = 'turbo'
) -> None:
    """Render visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    finite_mask = np.isfinite(costmap)
    
    if not np.any(finite_mask):
        plt.close(fig)
        return

    valid_costs = costmap[finite_mask]
    
    vmin = np.percentile(valid_costs, 1)
    vmax = np.percentile(valid_costs, 99)

    costmap_masked = np.ma.masked_where(~finite_mask, costmap)

    im = ax.imshow(costmap_masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Geodesic Distance (m)', rotation=270, labelpad=20)

    ax.set_title(
        f"Geodesic Costmap (Filled)\n"
        f"Range: [{np.min(valid_costs):.2f}, {np.max(valid_costs):.2f}] m"
    )
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def load_camera_poses(session_folder: Path) -> dict:
    """
    Load camera positions from agent_states.npy.

    Returns:
        Dictionary mapping frame_idx -> camera_position (x, y, z)
    """
    _ensure_agent_state_unpickle()
    states_path = session_folder / "agent_states.npy"
    camera_positions = {}

    if not states_path.exists():
        logger.warning(f"agent_states.npy not found at {states_path}")
        logger.warning("Using standard snapping (ceiling may snap to upper floors)")
        return camera_positions

    states = np.load(states_path, allow_pickle=True)
    for idx, state in enumerate(states):
        try:
            camera_positions[idx] = np.array(state.position, dtype=np.float32)
        except Exception:
            continue

    logger.info(f"Loaded {len(camera_positions)} camera positions from agent_states.npy")
    logger.info("Floor-aware snapping enabled (ceiling → ground floor)")
    return camera_positions


def load_goal_pixel_from_episode(
    session_folder: Path,
    task_type: str = GOAL_TASK_TYPE,
) -> Tuple[int, Tuple[int, int]]:
    """
    Infer the goal frame and pixel directly from the episode folder.

    Mirrors goal.mode="episode" in libs/mapper/create_vggt_prop_map.py: read the goal
    object's semantic mask via get_goal_info, then take its centroid as the goal
    pixel. Returns (frame_idx, (u, v)).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from libs.common.geometry_utils import get_goal_info, get_mask_centroid

    goal_frame_idx, goal_mask, _ = get_goal_info(str(session_folder), task_type)

    # The mask must be in the same resolution as the pointmaps we index into.
    pointmap_path = session_folder / "gt_pointmaps_fov90" / f"{goal_frame_idx:05d}.npy"
    if not pointmap_path.exists():
        pointmap_path = session_folder / "pointmaps" / f"{goal_frame_idx:05d}.npy"
    if pointmap_path.exists():
        target_h, target_w = np.load(pointmap_path).shape[:2]
        if goal_mask.shape[:2] != (target_h, target_w):
            import cv2

            logger.info(
                "Resizing goal mask from %s to (%d, %d)",
                goal_mask.shape[:2],
                target_h,
                target_w,
            )
            goal_mask = cv2.resize(
                goal_mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

    centroid = get_mask_centroid(goal_mask)
    if centroid is None:
        raise ValueError(f"Goal mask is empty in episode {session_folder}")

    goal_u, goal_v = int(centroid[0]), int(centroid[1])
    return int(goal_frame_idx), (goal_u, goal_v)


def load_goal_pixel(session_folder: Path) -> Tuple[int, Tuple[int, int]]:
    """Resolve the goal frame/pixel from the episode folder's semantic mask."""
    return load_goal_pixel_from_episode(session_folder)


def create_sim_and_load_navmesh(
    navmesh_path: Path,
) -> Tuple[habitat_sim.Simulator, habitat_sim.PathFinder]:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    if not sim.pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")
    return sim, sim.pathfinder


def run_session_parallel(
    session_folder: Path,
    navmesh_path: Path,
    output_dir: Optional[Path] = None,
    num_workers: int = 8,
    save_arrays: bool = True,
    save_pngs: bool = True,
    executor: Optional[ProcessPoolExecutor] = None,
    main_sim: Optional[habitat_sim.Simulator] = None,
    main_navmesh: Optional[habitat_sim.PathFinder] = None,
) -> None:
    
    # 1. Setup paths
    images_dir = resolve_images_dir(session_folder)
    num_frames = count_frames_from_images(images_dir)
    if num_frames <= 0:
        raise ValueError(f"No images found in {images_dir}")

    pointmap_dir = ensure_pointmaps(session_folder)
    if pointmap_dir is None:
        logger.warning("No pointmaps available for %s, skipping", session_folder)
        return
    
    if output_dir is None:
        output_dir = session_folder / "navmesh_costmaps"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "arrays"
    if save_arrays:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    # 2. Main Thread Setup
    logger.info("Initializing Main Process & Goal Snap...")
    
    owns_sim = False
    if main_navmesh is None:
        sim, navmesh = create_sim_and_load_navmesh(navmesh_path)
        owns_sim = True
    else:
        sim = main_sim
        navmesh = main_navmesh
    
    # Load goal pixel from graph (goal node) and read 3D point from pointmap
    goal_frame_idx, (goal_u, goal_v) = load_goal_pixel(session_folder)
    goal_pmap_path = pointmap_dir / f"{goal_frame_idx:05d}.npy"
    if not goal_pmap_path.exists():
        pointmap_dir = ensure_pointmaps(session_folder)
        if pointmap_dir is None:
            logger.warning("No pointmaps available for %s, skipping", session_folder)
            return
        goal_pmap_path = pointmap_dir / f"{goal_frame_idx:05d}.npy"
        if not goal_pmap_path.exists():
            logger.warning("Goal pointmap missing for %s, skipping", session_folder)
            return
    goal_pmap = np.load(goal_pmap_path)
    goal_raw = goal_pmap[goal_v, goal_u]
    if not np.all(np.isfinite(goal_raw)):
        raise ValueError(f"Goal point {goal_raw} is invalid/infinite.")
    goal_snapped = navmesh.snap_point(goal_raw)
    logger.info(
        f"Goal frame={goal_frame_idx} pixel=({goal_u}, {goal_v}) | "
        f"Goal: {goal_raw} -> Snapped: {goal_snapped}"
    )
    
    # Load camera poses for floor-aware snapping
    camera_positions = load_camera_poses(session_folder)
    
    # 3. Parallel Execution
    if executor is None:
        logger.info(f"Starting pool with {num_workers} workers...")
    else:
        logger.info("Using shared pool for navmesh reuse")

    tasks = []
    owned_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(str(navmesh_path),)
        )
        owned_executor = True

    try:
        for i in range(num_frames):
            p_path = pointmap_dir / f"{i:05d}.npy"
            tasks.append(
                executor.submit(
                    compute_frame_costmap, i, p_path, goal_snapped, camera_positions
                )
            )

        success = 0
        failed = 0

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Computing Costmaps"):
            f_idx, costmap, valid_mask = future.result()

            if costmap is None:
                failed += 1
                continue

            if save_arrays:
                np.save(arrays_dir / f"{f_idx:05d}.npy", costmap)
            if save_pngs:
                save_visualization(costmap, output_dir / f"{f_idx:05d}.png")
            success += 1
    finally:
        if owned_executor:
            executor.shutdown()

    logger.info(f"Done. Success: {success}, Failed/Skipped: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute geodesic costmaps (Parallel + Hole Filling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=None,
        help="Episode root; defaults to $PLANN3R_ROOT/evaluation/datasets/"
        "hm3d_navigation/hm3d_iin_val_320x240.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root directory (per-scene subfolders created).",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=SCENE_LIST_DEFAULT,
        help="Path to scene list file; one scene path per line.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    parent_dir = args.parent_dir
    if parent_dir is None:
        plann3r_root = os.environ.get("PLANN3R_ROOT")
        if not plann3r_root:
            raise RuntimeError(
                "PLANN3R_ROOT is not set; export it to the Plann3r release "
                "bundle root or pass --parent_dir."
            )
        parent_dir = (
            Path(plann3r_root)
            / "evaluation/datasets/hm3d_navigation/hm3d_iin_val_320x240"
        )
    output_root = args.output_dir
    scene_list_path = args.scene_list
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise ValueError(f"parent_dir not found: {parent_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    allowed_scene_names = load_scene_names(scene_list_path)
    if not allowed_scene_names:
        raise ValueError(f"Scene list is empty: {scene_list_path}")

    session_dirs = [p for p in parent_dir.iterdir() if p.is_dir()]
    if not session_dirs:
        raise ValueError(f"No session directories found under: {parent_dir}")

    session_dirs = [p for p in session_dirs if p.name in allowed_scene_names]
    if not session_dirs:
        raise ValueError(
            "No matching scene directories found under: "
            f"{parent_dir} (scene list: {scene_list_path})"
        )

    for session_folder in sorted(session_dirs):
        navmesh_candidates = list(session_folder.glob("*.basis.navmesh"))
        if not navmesh_candidates:
            logger.warning(f"No navmesh in {session_folder}, skipping")
            continue
        navmesh_path = navmesh_candidates[0]

        try:
            run_session_parallel(
                session_folder=session_folder,
                navmesh_path=navmesh_path,
                output_dir=output_root / session_folder.name / "navmesh_costmaps",
                num_workers=args.workers,
            )
        except Exception as exc:
            logger.warning(f"Failed {session_folder}: {exc}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()