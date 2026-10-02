"""Run VGGT-Nav navigation episodes in Habitat and write per-episode metrics.

Each episode loads a mapped trajectory, steps the agent with the Plann3r planner
and the GNM costmap controller, and stops at the goal threshold or max_steps.
The VGGTNav model is loaded once and shared across episodes. Results go to
results_summary.txt per episode and results_summary.csv and metrics_summary.txt
(success rate, SPL, SSPL) per run. baseline/evaluate.sh calls this script for
every reported result.

Usage:
    pixi run python run_nav.py experiment=vggt_nav_baseline task_type=original ...
"""

import os

# Suppress habitat-sim logs - MUST be before importing habitat_sim
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# CRITICAL: Import habitat_sim FIRST before numpy, torch, scipy, etc.
# It is not used in this file, only loaded early for its libraries.
import habitat_sim  # noqa: F401

# Now import everything else
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from libs.experiments import task_setup
from libs.logger import default_logger
from libs.experiments.episode_utils import (
    compute_soft_spl,
    compute_spl,
    compute_travelled_distance,
)
from libs.visualizations import (
    VisualizationDataCollector,
    VisualizationRenderer,
    VggtNavCompactVisualizer,
)
from libs.common.gpu_memory_utils import clear_gpu_cache

# Setup logging
default_logger.setup_logging(level=logging.INFO, console=True)
logger = logging.getLogger("[RunNav]")
logging.getLogger("[Task Setup]").setLevel(logging.WARNING)

# ==============================================================================
# Utility Functions
# ==============================================================================

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_observations(observations: dict) -> tuple:
    """
    Extract RGB, depth, and semantic from simulator observations.

    Returns:
        rgb: (H, W, 3) uint8 array
        depth: (H, W) float32 array
        semantic: (H, W) int32 array or None
    """
    rgb = observations["color_sensor"][:, :, :3]  # Drop alpha channel
    depth = observations["depth_sensor"]
    # semantic = observations.get("semantic_sensor", None)
    return rgb, depth

# ==============================================================================
# Episode Runner
# ==============================================================================

def compact_visualization_enabled(cfg: DictConfig) -> bool:
    """The compact video needs a VGGT-Nav or GT navmesh costmap to draw."""
    compact_cfg = cfg.visualization.get("vggt_nav_compact", {})
    return bool(compact_cfg.get("enabled", False)) and (
        bool(getattr(cfg, "vggtnav", {}).get("enabled", False))
        or bool(getattr(cfg, "gt_navmesh_costmap", {}).get("enabled", False))
    )


def run_episode(
    cfg: DictConfig,
    episode_path: Path,
    episode_results_path: Path,
    vggtnav_model=None,
    stopping_depth_model=None,
) -> dict:
    """
    Run a single navigation episode.

    If the episode raises, its Habitat simulator is closed before the exception
    propagates. Otherwise the traceback keeps it alive until a later garbage
    collection, and closing it then, after a retry has opened a new simulator,
    aborts the process.

    Args:
        cfg: Hydra config
        episode_path: Path to episode data
        episode_results_path: Path to save results
    Returns:
        dict with episode results (success_status, steps, distance_to_goal)
    """
    opened = []
    try:
        return _run_episode(
            cfg,
            episode_path,
            episode_results_path,
            vggtnav_model,
            stopping_depth_model,
            opened,
        )
    except BaseException:
        for episode in opened:
            sim = getattr(episode, "sim", None)
            if sim is not None:
                try:
                    sim.close()
                except Exception as close_error:
                    logger.warning(f"Closing simulator after failure raised: {close_error}")
        raise


def _run_episode(
    cfg: DictConfig,
    episode_path: Path,
    episode_results_path: Path,
    vggtnav_model,
    stopping_depth_model,
    opened: list,
) -> dict:
    """Body of run_episode. Appends the Episode to `opened` once created."""
    # Construct scene path
    hm3d_root_path = Path(cfg.hm3d_root_path)
    episode_name = str(episode_path.parts[-1].split('_')[0])
    hm3d_scene_path = sorted(hm3d_root_path.glob(f"*{episode_name}"))[0]
    scene_glb_path = str(sorted(hm3d_scene_path.glob('*basis.glb'))[0])

    logger.info(f"Scene GLB PATH: {scene_glb_path} | {os.path.exists(scene_glb_path) = }")

    # Create episode runner
    logger.info(f"Initializing episode: {episode_path}")
    episode = task_setup.Episode(
        cfg=cfg,
        episode_path=episode_path,
        scene_glb_path=scene_glb_path,
        episode_results_path=episode_results_path,
        preload_data={"vggtnav_model": vggtnav_model}
    )
    opened.append(episode)

    # Setup logging directories
    episode.setup_logging()

    # Initialize visualization system if enabled
    data_collector = None
    vis_renderer = None
    compact_visualizer = None
    compact_submap_cache = {}
    compact_cfg = cfg.visualization.get("vggt_nav_compact", {})
    if compact_visualization_enabled(cfg):
        start_pos = np.asarray(episode.agent.get_state().position)
        goal_pos = np.asarray(episode.final_goal_position)
        compact_visualizer = VggtNavCompactVisualizer(
            episode_results_path=episode_results_path,
            sim=episode.sim,
            start_position=start_pos,
            goal_position=goal_pos,
            map_positions=episode.agent_positions_in_map,
            goal_frame_index=int(episode.final_goal_image_idx),
            width=int(compact_cfg.get("width", 1280)),
            height=int(compact_cfg.get("height", 720)),
            fps=float(compact_cfg.get("video_fps", 5)),
            topdown_meters_per_pixel=float(
                compact_cfg.get("topdown_meters_per_pixel", 0.05)
            ),
            save_frames=bool(compact_cfg.get("save_frames", True)),
        )
        logger.info("Compact VGGT-Nav visualization initialized: %s", episode_results_path)
    if cfg.visualization.save_raw_data.enabled:
        start_pos = episode.agent.get_state().position
        goal_pos = episode.final_goal_position

        data_collector = VisualizationDataCollector(
            episode_results_path=episode_results_path,
            cfg=cfg,
            start_position=start_pos,
            goal_position=goal_pos
        )

        vis_renderer = VisualizationRenderer(
            vis_cfg=cfg.visualization,
            episode_results_path=episode_results_path
        )

        # Save topdown base map and metadata for offline rendering
        meters_per_pixel = getattr(cfg.visualization.render_visualizations, 'topdown_meters_per_pixel', 0.025)
        data_collector.save_topdown_data(
            sim=episode.sim,
            start_position=np.array(start_pos),
            goal_position=np.array(goal_pos),
            meters_per_pixel=meters_per_pixel
        )
        (episode_results_path / "frame_count.txt").write_text(
            str(len(episode.map_img_paths))
        )
        (episode_results_path / "goal_frame_idx.txt").write_text(
            str(int(episode.final_goal_image_idx))
        )

        logger.info("Raw data collector and visualization renderer initialized")

    # Navigation loop
    step = 0
    max_steps = cfg.max_steps
    pts3d_source = cfg.get("pts3d_source", "none")

    logger.info(f"Starting navigation loop (max_steps={max_steps}, pts3d_source={pts3d_source})")

    online_stopper = None
    stopping_cfg = cfg.get("online_stopping", {})
    if bool(stopping_cfg.get("enabled", False)):
        from stopping_condition.online_stopping import OnlineStoppingCondition

        online_stopper = OnlineStoppingCondition(
            model=stopping_depth_model,
            eps=stopping_cfg.get("eps", 0.2),
            n=stopping_cfg.get("N", 100),
            n_depth=stopping_cfg.get("N_depth", 100),
            depth_m=stopping_cfg.get("depth_m", 1.0),
            map_frame_slack=stopping_cfg.get("map_frame_slack", 1),
        )

    pbar = tqdm(total=max_steps, desc="Navigation", leave=False)

    while step < max_steps:
        step_start = time.time()

        # Check if goal reached
        if online_stopper is None:
            if episode.is_done():
                logger.info(f"Episode succeeded at step {step}!")
                break
        else:
            episode.update_distance_to_goal()

        # 1. Get sensor observations
        observations = episode.sim.get_sensor_observations()
        rgb, depth_gt = split_observations(observations)

        # 2. Get pts3d based on source
        if pts3d_source == "gt_depth":
            # GT depth feeds the privileged navmesh paths (gt_navmesh_costmap and
            # vggtnav.query_gt_anchor), which build costmaps from simulator depth.
            pts3d = None
            depth = depth_gt
        elif pts3d_source == "none":
            # Image-only policies (GNM and VGGT-Nav) do not use query geometry.
            pts3d = None
            depth = None
        else:
            raise ValueError(f"Unknown pts3d_source: {pts3d_source}")

        # 3. Get goal mask (localization + planning)
        goal_pose = None
        oracle_mode = str(cfg.localizer.get("oracle_mode", "legacy"))
        if oracle_mode not in {"legacy", "odometry"}:
            raise ValueError(
                f"Unsupported localizer.oracle_mode={oracle_mode!r}; "
                "expected 'legacy' or 'odometry'"
            )
        # Without a pose, Episode.get_goal uses the legacy position-only
        # ranking that produced the reported results.
        if (
            getattr(cfg.localizer, "use_gt_localization", False)
            and oracle_mode == "odometry"
        ):
            state = episode.agent.get_state()
            quat = state.rotation
            goal_pose = np.array(
                [
                    state.position[0],
                    state.position[1],
                    state.position[2],
                    quat.x,
                    quat.y,
                    quat.z,
                    quat.w,
                ],
                dtype=np.float32,
            )
        capture_vis_data = (
            cfg.visualization.save_raw_data.enabled or compact_visualizer is not None
        )
        if capture_vis_data:
            # The costmap is read back from episode.goal_mask below.
            _, vis_data = episode.get_goal(
                rgb=rgb, depth=depth, pose=goal_pose, return_vis_data=True
            )
        else:
            episode.get_goal(rgb=rgb, depth=depth, pose=goal_pose)
            vis_data = None

        # 4. Get control signal
        # Collision avoidance uses the raw sensor depth.
        episode.get_control_signal(rgb, depth_gt)

        # 5. Execute action in simulator
        episode.execute_action()
        if online_stopper is not None:
            episode.update_distance_to_goal()

        inferred_stop = None
        if online_stopper is not None and vis_data is not None:
            inferred_stop = online_stopper.observe(
                step=step,
                rgb=rgb,
                costmap=getattr(episode, "goal_mask_raw", episode.goal_mask),
                localized_img_idxs=vis_data.get("localized_img_idxs", []),
                goal_frame_idx=episode.final_goal_image_idx,
            )

        if compact_visualizer is not None and vis_data is not None:
            localized_indices = list(vis_data.get("localized_img_idxs", []))
            submap_images = []
            for submap_index in localized_indices:
                submap_index = int(submap_index)
                if submap_index not in compact_submap_cache:
                    submap_image = cv2.imread(str(episode.map_img_paths[submap_index]))
                    if submap_image is None:
                        raise FileNotFoundError(
                            f"Could not read localized map image: {episode.map_img_paths[submap_index]}"
                        )
                    compact_submap_cache[submap_index] = cv2.cvtColor(
                        submap_image, cv2.COLOR_BGR2RGB
                    )
                submap_images.append(compact_submap_cache[submap_index])
            controller_waypoints = getattr(
                getattr(episode, "goal_controller", None), "action_pred", None
            )
            compact_visualizer.render(
                step=step,
                query_rgb=rgb,
                submap_images_rgb=submap_images,
                submap_indices=localized_indices,
                anchor_index=int(vis_data.get("vggtnav_anchor_img_idx", -1)),
                anchor_pixel=tuple(vis_data.get("vggtnav_anchor_pixel", (-1, -1))),
                costmap=episode.goal_mask,
                costmap_title=(
                    "GT NAVMESH QUERY COSTMAP"
                    if bool(getattr(cfg, "gt_navmesh_costmap", {}).get("enabled", False))
                    else "PLANN3R QUERY COSTMAP"
                ),
                anchor_world_position=vis_data.get(
                    "vggtnav_anchor_world_position", None
                ),
                waypoints=controller_waypoints,
                trajectory_history=episode.agent_state_history,
                distance_to_goal=float(episode.distance_to_goal),
                velocity=float(episode.velocity_control),
                angular_velocity=float(episode.theta_control),
                collided=bool(episode.collided),
                closest_index=int(vis_data.get("closest_map_img_idx", -1)),
                inferred_stop=inferred_stop is not None,
            )

        # 6. Save raw data if enabled
        if cfg.visualization.save_raw_data.enabled and data_collector is not None and vis_data is not None:
            # Prepare matches data for saving
            matches_data = {
                'qry_img_idx': step,
                'closest_map_img_idx': vis_data.get('closest_map_img_idx', -1),
                'localized_img_idxs': vis_data.get('localized_img_idxs', np.array([])),
                'qry_mkpts': vis_data.get('qry_mkpts', np.empty((0, 2))),
                'ref_mkpts': vis_data.get('ref_mkpts', np.empty((0, 3))),
                'confidences': vis_data.get('confidences', np.array([])),
                'vggtnav_anchor_img_idx': vis_data.get('vggtnav_anchor_img_idx', -1),
                'vggtnav_anchor_pixel': vis_data.get('vggtnav_anchor_pixel', np.array([-1, -1])),
                'vggtnav_anchor_source': vis_data.get('vggtnav_anchor_source', ''),
                'vggtnav_topomap_anchor_mode': vis_data.get('vggtnav_topomap_anchor_mode', ''),
                'vggtnav_anchor_cost': vis_data.get('vggtnav_anchor_cost', np.nan),
            }

            # The saved waypoints are the GNM controller's predicted waypoints.
            waypoints = None
            if hasattr(episode, 'goal_controller') and hasattr(episode.goal_controller, 'action_pred'):
                if episode.goal_controller.action_pred is not None:
                    waypoints = episode.goal_controller.action_pred  # (N, 2) predicted waypoints

            # Get agent state for saving
            agent_state_obj = episode.agent.get_state()
            agent_state = {
                'position': np.array(agent_state_obj.position),
                'rotation': np.array([agent_state_obj.rotation.w, agent_state_obj.rotation.x,
                                      agent_state_obj.rotation.y, agent_state_obj.rotation.z])
            }

            # Save raw data
            saved_costmap = episode.goal_mask
            if bool(cfg.visualization.save_raw_data.get("unnormalized_costmaps", False)):
                saved_costmap = getattr(episode, "goal_mask_raw", episode.goal_mask)
            data_collector.save_step_data(
                step=step,
                rgb=rgb,
                depth=depth,
                pts3d=pts3d,
                costmap=saved_costmap,
                matches_data=matches_data,
                velocity=episode.velocity_control,
                theta=episode.theta_control,
                waypoints=waypoints,
                care_data=getattr(episode, "care_debug", None),
                agent_state=agent_state,
                collided=episode.collided,
                distance_to_goal=episode.distance_to_goal,
            )

            # Optionally render visualizations online
            if vis_renderer is not None and vis_renderer.online_render:
                localized_idxs = vis_data.get('localized_img_idxs', [])
                submap_imgs = []
                for submap_idx in localized_idxs:
                    submap_img = cv2.imread(str(episode.map_img_paths[submap_idx]))
                    if submap_img is not None:
                        submap_imgs.append(cv2.cvtColor(submap_img, cv2.COLOR_BGR2RGB))
                vis_renderer.render_step_visualizations(
                    step=step,
                    rgb=rgb,
                    costmap=episode.goal_mask,
                    matches_data=matches_data,
                    ref_img_path=episode.map_img_paths[vis_data['closest_map_img_idx']],
                    sim=episode.sim,
                    trajectory_history=episode.agent_state_history,
                    start_position=np.array(episode.start_position),
                    goal_position=np.array(episode.final_goal_position),
                    waypoints=waypoints,
                    agent_position=agent_state['position'],
                    agent_rotation=agent_state['rotation'],
                    distance_to_goal=episode.distance_to_goal,
                    care_data=getattr(episode, "care_debug", None),
                    submap_imgs=submap_imgs,
                    submap_idxs=list(localized_idxs),
                    submap_images=submap_imgs,
                    map_positions=episode.agent_positions_in_map,
                )

        # The result CSV is required even when raw visualization arrays are disabled.
        episode.log_results(step, final=False)

        # Log progress
        step_time = time.time() - step_start
        pbar.set_postfix({
            "dist": f"{episode.distance_to_goal:.2f}m",
            "v": f"{episode.velocity_control:.3f}",
            "w": f"{episode.theta_control:.3f}",
            "t": f"{step_time:.2f}s"
        })
        pbar.update(1)

        step += 1
        clear_gpu_cache()
        if inferred_stop is not None:
            import json

            success_distance = float(stopping_cfg.get("success_distance", 1.0))
            is_true_positive = episode.distance_to_goal < success_distance
            episode.success_status = "success" if is_true_positive else "false_positive_stop"
            (episode_results_path / "online_stopping.json").write_text(
                json.dumps(
                    {
                        **inferred_stop,
                        "distance_to_goal": float(episode.distance_to_goal),
                        "success_distance": success_distance,
                        "true_positive": bool(is_true_positive),
                    },
                    indent=2,
                )
            )
            logger.info(
                "Online inferred stop at step %d (candidate=%d, distance=%.3fm, %s)",
                inferred_stop["decision_step"],
                inferred_stop["candidate_step"],
                episode.distance_to_goal,
                "TP" if is_true_positive else "FP",
            )
            break

    pbar.close()

    # Finalize
    if step >= max_steps:
        episode.success_status = "exceeded_steps"
        logger.warning(f"Episode exceeded max steps ({max_steps})")

    # Log final results to metadata file
    episode.log_results(step, final=True)

    # Save episode visualization metadata if enabled
    if cfg.visualization.save_raw_data.enabled and data_collector is not None:
        data_collector.save_episode_metadata(
            success_status=episode.success_status,
            total_distance=episode.distance_to_final_goal,
            final_distance_to_goal=episode.distance_to_goal
        )

    # Save results
    # `agent_state_history` only records post-action states, so the start pose
    # has to be prepended for the trajectory length to cover the first step.
    trajectory = [np.asarray(episode.start_position)] + [
        np.asarray(state.position) for state in episode.agent_state_history
    ]
    travelled_distance = compute_travelled_distance(
        trajectory, episode.distance_to_goal
    )
    success = episode.success_status == "success"
    results = {
        "success_status": episode.success_status,
        "steps": step,
        "distance_to_goal": episode.distance_to_goal,
        "distance_to_final_goal": episode.distance_to_final_goal,
        "travelled_distance": travelled_distance,
        "spl": compute_spl(
            success, episode.distance_to_final_goal, travelled_distance
        ),
        "sspl": compute_soft_spl(
            success,
            episode.distance_to_final_goal,
            episode.distance_to_goal,
            travelled_distance,
        ),
        "vggtnav_costmap_forward_count": int(
            getattr(episode, "vggtnav_costmap_forward_count", 0)
        ),
        "vggtnav_costmap_forward_seconds": float(
            getattr(episode, "vggtnav_costmap_forward_seconds", 0.0)
        ),
    }

    if compact_visualizer is not None:
        compact_video_path = compact_visualizer.close()
        logger.info("Saved VGGT-Nav compact video: %s", compact_video_path)

    # Close simulator
    episode.sim.close()

    return results


def _free_cuda():
    """Release cached CUDA memory between episodes to limit fragmentation when
    sharing the GPU with other jobs."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Episode Discovery
# ==============================================================================

def get_episode_list(cfg: DictConfig) -> list:
    """
    Get list of episode paths based on config.

    Priority order:
    1. If episode_list is non-empty: use only those specific episodes
    2. If multi_episode is true: get episodes from episodes_dir with start/end filtering
    3. Otherwise: use single episode_path

    Args:
        cfg: Hydra config with episode_path, episodes_dir, multi_episode, etc.

    Returns:
        List of Path objects for episodes to process
    """
    from natsort import natsorted

    # Single episode mode - just use episode_path
    if not cfg.multi_episode:
        episode_path = Path(cfg.episode_path)
        if not episode_path.exists():
            raise ValueError(f"Episode path does not exist: {episode_path}")
        return [episode_path]

    # Multi-episode mode
    episodes_dir = Path(cfg.episodes_dir) if cfg.get("episodes_dir") else None
    if not episodes_dir or not episodes_dir.exists():
        raise ValueError(f"Episodes directory does not exist: {episodes_dir}")

    # Priority 1: episode_list_file (txt file with episode names)
    episode_list_file = cfg.get("episode_list_file", None)
    if episode_list_file:
        file_path = Path(episode_list_file)
        if not file_path.exists():
            raise ValueError(f"Episode list file not found: {file_path}")

        with open(file_path, 'r') as f:
            episode_names = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(episode_names)} episodes from {file_path}")

        episodes = []
        for ep_name in episode_names:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")

        return natsorted(episodes, key=lambda x: x.name)

    # Priority 2: episode_list array (if non-empty)
    episode_list = cfg.get("episode_list", [])
    if episode_list and len(episode_list) > 0:
        logger.info(f"Using episode_list with {len(episode_list)} episodes")

        episodes = []
        for ep_name in episode_list:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")

        return natsorted(episodes, key=lambda x: x.name)

    # Priority 3: All episodes from directory with start/end filtering
    all_episodes = [p for p in episodes_dir.iterdir() if p.is_dir()]
    all_episodes = natsorted(all_episodes, key=lambda x: x.name)

    logger.info(f"Found {len(all_episodes)} episodes in {episodes_dir}")

    # Apply start/end index filtering
    start_idx = cfg.get("episode_start_idx", 0)
    end_idx = cfg.get("episode_end_idx", -1)

    if start_idx > 0:
        all_episodes = all_episodes[start_idx:]
    if end_idx > 0:
        all_episodes = all_episodes[:end_idx - start_idx]

    logger.info(f"After filtering (start={start_idx}, end={end_idx}): {len(all_episodes)} episodes")

    # Apply blacklist filtering
    blacklist = cfg.get("episode_blacklist", [])
    if blacklist:
        episodes = [ep for ep in all_episodes
                   if not any(bl in ep.name for bl in blacklist)]
        logger.info(f"After blacklist filtering: {len(episodes)} episodes")
    else:
        episodes = all_episodes

    return episodes


# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""

    # Print config
    logger.info("=" * 60)
    logger.info("VGGT-Nav navigation evaluation")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    seed_everything(int(cfg.seed))

    # The online stopper reads the localized map frames from the per-step
    # vis_data, which is only collected for raw-data saving or the compact
    # video. Without it the stopper would never fire.
    if bool(cfg.get("online_stopping", {}).get("enabled", False)) and not (
        cfg.visualization.save_raw_data.enabled or compact_visualization_enabled(cfg)
    ):
        raise ValueError(
            "online_stopping.enabled=true needs visualization.save_raw_data.enabled=true "
            "(or visualization.vggt_nav_compact.enabled=true), which collects the "
            "per-step localization data the stopper reads"
        )

    # Load VGGTNav once and reuse it across episodes.
    vggtnav_model = None
    if bool(getattr(cfg, "vggtnav", {}).get("enabled", False)):
        vggtnav_model = task_setup.load_vggtnav_model(cfg.vggtnav, cfg.device)

    stopping_depth_model = None
    if bool(cfg.get("online_stopping", {}).get("enabled", False)):
        from stopping_condition.model_loading import VGGTMetricModel

        stopping_depth_model = VGGTMetricModel(
            pretrained_model=str(
                cfg.online_stopping.get("vggt_checkpoint", "facebook/VGGT-1B")
            )
        )

    # Initialize results directory
    results_path = task_setup.init_results_dir_and_save_cfg(cfg, default_logger)
    logger.info(f"Results will be saved to: {results_path}")

    # A missing episode source raises here, so the run exits nonzero instead
    # of reporting an empty evaluation as a success.
    episodes = get_episode_list(cfg)

    logger.info(f"Processing {len(episodes)} episode(s)")

    # Results tracking
    results_summary = {
        'total_episodes': len(episodes),
        'successful_episodes': 0,
        'failed_episodes': 0,
        'exceeded_steps': 0,
        'stuck': 0,
        'success_rate': 0.0,
        'spl': 0.0,
        'sspl': 0.0,
        'episode_results': []
    }

    # Process each episode
    for ei, episode_path in enumerate(tqdm(episodes, desc="Processing Episodes")):
        episode_name = episode_path.parts[-1]

        logger.info("=" * 60)
        logger.info(f"Episode {ei+1}/{len(episodes)}: {episode_name}")
        logger.info("=" * 60)

        # Create episode results directory
        episode_results_path = results_path / f"{episode_name}_{cfg.controller.name}_{cfg.goal_source}"
        episode_results_path.mkdir(exist_ok=True, parents=True)

        try:
            # Run episode, retrying on transient CUDA OOM (the GPU may be shared
            # with other jobs whose memory spikes can briefly starve us).
            oom_retries = int(cfg.get("oom_retries", 3))
            attempt = 0
            while True:
                try:
                    results = run_episode(
                        cfg,
                        episode_path,
                        episode_results_path,
                        vggtnav_model,
                        stopping_depth_model,
                    )
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    attempt += 1
                    _free_cuda()
                    if attempt > oom_retries:
                        raise
                    wait_s = 15 * attempt
                    logger.warning(
                        f"CUDA OOM on {episode_name} (attempt {attempt}/{oom_retries}); "
                        f"freed cache, waiting {wait_s}s then retrying. ({oom})"
                    )
                    time.sleep(wait_s)

            # Track results
            results['episode_name'] = episode_name
            results_summary['episode_results'].append(results)

            if results['success_status'] == 'success':
                results_summary['successful_episodes'] += 1
            elif results['success_status'] == 'exceeded_steps':
                results_summary['exceeded_steps'] += 1
                results_summary['failed_episodes'] += 1
            elif 'stuck' in results['success_status']:
                results_summary['stuck'] += 1
                results_summary['failed_episodes'] += 1
            else:
                results_summary['failed_episodes'] += 1

            # Save individual episode results
            results_file = episode_results_path / "results_summary.txt"
            with open(results_file, "w") as f:
                f.write(f"episode_name: {episode_name}\n")
                f.write(f"success_status: {results['success_status']}\n")
                f.write(f"steps: {results['steps']}\n")
                f.write(f"distance_to_goal: {results['distance_to_goal']:.4f}\n")
                f.write(f"distance_to_final_goal: {results['distance_to_final_goal']:.4f}\n")
                f.write(f"travelled_distance: {results['travelled_distance']:.4f}\n")
                f.write(f"spl: {results['spl']:.4f}\n")
                f.write(f"sspl: {results['sspl']:.4f}\n")

            logger.info(f"Episode {episode_name}: {results['success_status']} "
                       f"(steps={results['steps']}, dist={results['distance_to_goal']:.2f}m, "
                       f"spl={results['spl']:.3f}, sspl={results['sspl']:.3f})")
            costmap_forwards = int(results.get("vggtnav_costmap_forward_count", 0))
            if costmap_forwards:
                costmap_seconds = float(results.get("vggtnav_costmap_forward_seconds", 0.0))
                logger.info(
                    "VGGTNav costmaps: %d forwards, %.3fs average, %.2fs total",
                    costmap_forwards,
                    costmap_seconds / costmap_forwards,
                    costmap_seconds,
                )

        except Exception as e:
            logger.error(f"Error processing episode {episode_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary['failed_episodes'] += 1
            results_summary['episode_results'].append({
                'episode_name': episode_name,
                'success_status': f'error: {str(e)}',
                'steps': 0,
                'distance_to_goal': float('nan'),
                'distance_to_final_goal': float('nan'),
                'travelled_distance': float('nan'),
                'spl': 0.0,
                'sspl': 0.0
            })
        finally:
            _free_cuda()

    # Calculate success rate and path-weighted metrics. Episodes that errored
    # out contribute 0 to both averages rather than being dropped.
    total_episodes = results_summary['total_episodes']
    results_summary['success_rate'] = (
        results_summary['successful_episodes'] / total_episodes * 100
        if total_episodes > 0 else 0
    )
    results_summary['spl'] = (
        sum(ep['spl'] for ep in results_summary['episode_results']) / total_episodes * 100
        if total_episodes > 0 else 0
    )
    results_summary['sspl'] = (
        sum(ep['sspl'] for ep in results_summary['episode_results']) / total_episodes * 100
        if total_episodes > 0 else 0
    )

    # Print final summary
    logger.info("=" * 60)
    logger.info("Final Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Episodes: {results_summary['total_episodes']}")
    logger.info(f"Successful: {results_summary['successful_episodes']}")
    logger.info(f"Failed: {results_summary['failed_episodes']}")
    logger.info(f"  - Exceeded Steps: {results_summary['exceeded_steps']}")
    logger.info(f"  - Stuck: {results_summary['stuck']}")
    logger.info(f"Success Rate: {results_summary['success_rate']:.2f}%")
    logger.info(f"SPL: {results_summary['spl']:.2f}%")
    logger.info(f"SSPL: {results_summary['sspl']:.2f}%")
    logger.info("=" * 60)

    # Save overall results summary
    summary_file = results_path / "results_summary.csv"
    import csv
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_name", "success_status", "steps", "distance_to_goal",
            "distance_to_final_goal", "travelled_distance", "spl", "sspl",
        ])
        for ep_result in results_summary['episode_results']:
            writer.writerow([
                ep_result['episode_name'],
                ep_result['success_status'],
                ep_result['steps'],
                f"{ep_result['distance_to_goal']:.4f}",
                f"{ep_result['distance_to_final_goal']:.4f}",
                f"{ep_result['travelled_distance']:.4f}",
                f"{ep_result['spl']:.4f}",
                f"{ep_result['sspl']:.4f}"
            ])

    logger.info(f"Results summary saved to: {summary_file}")

    # Save aggregated metrics
    metrics_file = results_path / "metrics_summary.txt"
    with open(metrics_file, "w") as f:
        f.write(f"total_episodes: {results_summary['total_episodes']}\n")
        f.write(f"successful_episodes: {results_summary['successful_episodes']}\n")
        f.write(f"failed_episodes: {results_summary['failed_episodes']}\n")
        f.write(f"exceeded_steps: {results_summary['exceeded_steps']}\n")
        f.write(f"stuck: {results_summary['stuck']}\n")
        f.write(f"success_rate: {results_summary['success_rate']:.2f}\n")
        f.write(f"spl: {results_summary['spl']:.2f}\n")
        f.write(f"sspl: {results_summary['sspl']:.2f}\n")

    logger.info(f"Metrics summary saved to: {metrics_file}")


if __name__ == "__main__":
    main()
