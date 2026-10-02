"""Goal lookup and small geometry helpers for episode folders.

get_goal_info returns the goal frame, goal mask and object instance id for a
task type (original, via_alt_goal, alt_goal_v2 or original_reverse), and
get_mask_centroid turns the mask into a goal pixel. Used by task_setup.py,
create_vggt_prop_map.py and mard_benchmark/gt_mesh_generator.py.
"""

import numpy as np
import cv2
import os
from pathlib import Path


def get_mask_centroid(object_mask):
    """
    Get centroid of binary mask using OpenCV moments.

    Args:
        object_mask: Binary mask (0s and 1s) of shape (H, W)

    Returns:
        (cx, cy): Centroid coordinates as (x, y) pixel coordinates
        Returns None if mask is empty
    """
    # Ensure mask is uint8
    if object_mask.dtype != np.uint8:
        object_mask = object_mask.astype(np.uint8)

    # Calculate moments
    moments = cv2.moments(object_mask)

    # Check if mask is not empty
    if moments["m00"] == 0:
        return None

    # Calculate centroid
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy)


def _parse_goal_instance_id_from_path(episode_path: str) -> int:
    return int(episode_path.rstrip("/").split("_")[-2])


def _load_goal_instance_id(episode_path: str) -> int:
    episode_data_path = os.path.join(episode_path, "episode.npy")
    if os.path.exists(episode_data_path):
        try:
            episode = np.load(episode_data_path, allow_pickle=True)[()]
        except ModuleNotFoundError:
            return _parse_goal_instance_id_from_path(episode_path)
        if isinstance(episode, dict):
            return int(episode["goal_object_id"])

        goal_instance_id = getattr(episode, "goal_object_id", None)
        if goal_instance_id is None:
            goal_instance_id = getattr(episode, "object_id", None)
        if goal_instance_id is None:
            raise ValueError(f"Could not read goal object id from {episode_data_path}")
        return int(goal_instance_id)

    return _parse_goal_instance_id_from_path(episode_path)


def _as_binary_goal_mask(instance_mask: np.ndarray, goal_instance_id: int) -> np.ndarray:
    unique_vals = np.unique(instance_mask)
    if instance_mask.dtype == bool or set(unique_vals).issubset({False, True, 0, 1}):
        return instance_mask.astype(np.uint8)
    return (instance_mask == int(goal_instance_id)).astype(np.uint8)


def _get_original_goal_info(episode_path: str) -> tuple:
    goal_instance_id = _load_goal_instance_id(episode_path)

    images_dir = os.path.join(episode_path, "images")
    goal_img_idx = len(os.listdir(images_dir)) - 1

    goal_filepath = os.path.join(episode_path, "obs_g.npy")
    if os.path.exists(goal_filepath):
        obs_g = np.load(goal_filepath, allow_pickle=True)[()]
        instance_mask = obs_g.get("semantic_sensor", obs_g) if isinstance(obs_g, dict) else obs_g
    else:
        semantic_filepath = os.path.join(episode_path, f"images_sem/{goal_img_idx:05d}.npy")
        if not os.path.exists(semantic_filepath):
            raise FileNotFoundError(
                f"Could not find goal mask at {goal_filepath} or {semantic_filepath}"
            )
        semantic_mask = np.load(semantic_filepath, allow_pickle=True)
        instance_mask = semantic_mask == goal_instance_id

    goal_mask = _as_binary_goal_mask(instance_mask, goal_instance_id)
    return goal_img_idx, goal_mask, goal_instance_id


def _get_alt_goal_info(episode_path: str) -> tuple:
    goal_info_path = os.path.join(episode_path, "seen_but_unvisited_object_v2.npy")
    if not os.path.exists(goal_info_path):
        raise FileNotFoundError(f"Could not find alt-goal annotation: {goal_info_path}")

    goal_info = np.load(goal_info_path, allow_pickle=True)[()]
    goal_instance_id = int(goal_info["instance_id"])
    goal_img_idx = int(goal_info["image_id"])

    semantic_filepath = os.path.join(episode_path, f"images_sem/{goal_img_idx:05d}.npy")
    if not os.path.exists(semantic_filepath):
        raise FileNotFoundError(f"Could not find alternate-goal semantic mask: {semantic_filepath}")

    instance_mask = np.load(semantic_filepath, allow_pickle=True)
    goal_mask = (instance_mask == goal_instance_id).astype(np.uint8)
    return goal_img_idx, goal_mask, goal_instance_id


def _get_reverse_goal_info(episode_path: str) -> tuple:
    reverse_goal_path = os.path.join(episode_path, "reverse_goal.npy")
    if not os.path.exists(reverse_goal_path):
        raise FileNotFoundError(f"Could not find reverse goal metadata: {reverse_goal_path}")

    reverse_goal = np.load(reverse_goal_path, allow_pickle=True)[()]
    goal_instance_id = int(reverse_goal["instance_id"])

    images_sem_dir = Path(episode_path) / "images_sem"
    instance_img_paths = sorted(images_sem_dir.glob("*.npy"))
    if not instance_img_paths:
        raise FileNotFoundError(f"No semantic images found in {images_sem_dir}")

    best_img_idx = None
    best_mask = None
    best_area = 0
    for img_idx, instance_img_path in enumerate(instance_img_paths):
        instance_img = np.load(instance_img_path, allow_pickle=True)
        instance_mask = (instance_img == goal_instance_id).astype(np.uint8)
        mask_area = int(instance_mask.sum())
        if mask_area > best_area:
            best_img_idx = img_idx
            best_mask = instance_mask
            best_area = mask_area

    if best_img_idx is None:
        raise ValueError(
            f"Reverse goal instance {goal_instance_id} is not visible in {images_sem_dir}"
        )

    return best_img_idx, best_mask, goal_instance_id


def get_goal_info(episode_path: str, task_type: str = "original") -> tuple:
    """
    Get goal info from episode folder.
    
    Goal source depends on task type:
    - original / via_alt_goal: final episode goal mask.
    - alt_goal_v2: seen_but_unvisited_object_v2 annotation.
    - original_reverse: reverse_goal metadata, using the largest visible mask.
    
    Args:
        episode_path: Path to episode directory (contains images/, episode.npy, etc.)
        task_type: Task type for goal parsing.
    
    Returns:
        (goal_img_idx, goal_mask, goal_instance_id)
    """
    if task_type in {"original", "repeat", "via_alt_goal"}:
        return _get_original_goal_info(episode_path)
    if task_type == "alt_goal_v2":
        return _get_alt_goal_info(episode_path)
    if task_type in {"original_reverse", "reverse"}:
        return _get_reverse_goal_info(episode_path)

    raise ValueError(f"Unsupported task_type={task_type!r}")
