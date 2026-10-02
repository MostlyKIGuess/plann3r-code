"""CARE-style collision avoidance that bends predicted waypoints away from obstacles.

Depth is projected to top-down obstacle points, a repulsive direction rotates
the waypoints, and a heading gives the linear and angular velocity. Waypoints
are in the local robot frame, x forward and y left. task_setup.py calls
care_step after the controller when collision_avoidance.enabled=true, which is
off by default and for every reported result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_PARAMS = {
    "theta_clip": math.pi / 4.0,
    "theta_thres": math.pi / 6.0,
    "tau_z": 0.5,
    "depth_offset": -0.0,
    "vertical_offset": -0.05,
    "sample_step": 10,
    "min_obstacle_dist": 0.05,
    "v_forward": 0.15,
    "v_max": 0.2,
    "omega_max": 0.8,
    "cam_x_offset": 0.0,
    "cam_y_offset": 0.0,
    "cam_height": 0.0,
    "obstacle_min_height": 0.15,
    "obstacle_max_height": 1.8,
}


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def _merged_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return merged


def ConstructTopDownObstacleMap(depth, intrinsics, params=DEFAULT_PARAMS):
    """Project depth into local top-down obstacle points."""
    params = _merged_params(params)
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    height, width = depth.shape
    step = int(params.get("sample_step", 10))
    tau_z = float(params.get("tau_z", 1.0))
    depth_offset = float(params.get("depth_offset", 0.0))
    cam_x_off = float(params.get("cam_x_offset", 0.0))
    cam_y_off = float(params.get("cam_y_offset", 0.0))

    # Height-band filter: only points whose height above the floor falls in
    # [obstacle_min_height, obstacle_max_height] count as obstacles. Without
    # this the floor right in front of the camera (always within tau_z) is read
    # as a wall of obstacles and CARE perpetually steers around nothing. Set
    # obstacle_min_height <= 0 to disable. Requires a level RGBD sensor at a
    # known cam_height above the floor.
    cam_height = float(params.get("cam_height", 0.0))
    vertical_offset = float(params.get("vertical_offset", 0.0))
    obs_min_h = float(params.get("obstacle_min_height", 0.15))
    obs_max_h = float(params.get("obstacle_max_height", 1.8))
    use_height_filter = obs_min_h > 0.0 and cam_height > 0.0

    pts = []
    for v in range(0, height, step):
        for u in range(0, width, step):
            z = float(depth[v, u]) - depth_offset
            if z <= 0 or z > tau_z:
                continue

            if use_height_filter:
                # Camera-frame height (up positive) of the point, then height
                # above the floor. Floor -> ~0, ceiling -> large.
                y_up_cam = (cy - v) * z / fy
                height_above_floor = cam_height + y_up_cam + vertical_offset
                if height_above_floor < obs_min_h or height_above_floor > obs_max_h:
                    continue

            x_cam = (u - cx) * z / fx
            z_cam = z
            x_forward = z_cam + cam_x_off
            y_left = -x_cam + cam_y_off
            pts.append((x_forward, y_left))

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)

    pts = np.array(pts, dtype=float)
    bins = np.linspace(0.0, tau_z, 65)
    obstacles = []
    for i in range(64):
        lo, hi = bins[i], bins[i + 1]
        mask = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if not np.any(mask):
            continue
        selected = pts[mask]
        obstacles.append(selected[np.argmin(np.abs(selected[:, 1]))])

    if len(obstacles) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.array(obstacles, dtype=float)


def EstimateRepulsiveDirection(waypoints, obstacles, params=DEFAULT_PARAMS):
    """Estimate trajectory rotation from obstacle repulsive force."""
    params = _merged_params(params)
    if obstacles.shape[0] == 0:
        return 0.0, None, None

    wp_xy = np.asarray(waypoints)[:, :2]
    frep_all = np.zeros((wp_xy.shape[0], 2), dtype=float)
    eps = 1e-6
    min_obstacle_dist = float(params.get("min_obstacle_dist", 0.05))

    for k, pk in enumerate(wp_xy):
        vec = np.zeros(2, dtype=float)
        for obstacle in obstacles:
            diff = pk - obstacle
            dist = max(float(np.linalg.norm(diff)), min_obstacle_dist)
            vec += diff / (dist**3 + eps)
        frep_all[k] = vec

    k_star = int(np.argmax(np.linalg.norm(frep_all, axis=1)))
    theta_rep = math.atan2(frep_all[k_star, 1], frep_all[k_star, 0])
    theta_clip = float(params.get("theta_clip", math.pi / 4.0))
    return float(clamp(theta_rep, -theta_clip, theta_clip)), k_star, frep_all


def RotateTrajectory(waypoints, theta):
    """Rotate waypoint positions around the robot origin."""
    if waypoints is None or np.asarray(waypoints).size == 0:
        return waypoints

    wp = np.asarray(waypoints)
    if wp.ndim != 2 or wp.shape[1] < 2:
        raise ValueError(f"waypoints must be KxD with D>=2, got shape {wp.shape}")

    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    pos_rot = (rotation @ wp[:, :2].T).T
    if wp.shape[1] == 2:
        return pos_rot
    return np.concatenate([pos_rot, wp[:, 2:]], axis=1)


def ComputeDesiredHeading(adjusted_waypoints, k_star):
    """Return heading angle to the selected adjusted waypoint."""
    if k_star is None or adjusted_waypoints is None or adjusted_waypoints.shape[0] == 0:
        return 0.0
    p = adjusted_waypoints[k_star, :2]
    return math.atan2(p[1], p[0])


def MotionCommandFromHeading(theta_des, params=DEFAULT_PARAMS):
    """Convert heading into CARE velocity command."""
    params = _merged_params(params)
    theta_thres = float(params.get("theta_thres", math.pi / 6.0))
    v_forward = float(params.get("v_forward", 0.15))
    v_max = float(params.get("v_max", 0.2))
    omega_max = float(params.get("omega_max", 0.8))

    ang = float(theta_des)
    omega = math.copysign(min(abs(ang), omega_max), ang)
    if abs(ang) > theta_thres:
        return 0.0, omega
    return float(min(v_forward, v_max)), omega


def care_step(rgb, depth, waypoints, intrinsics, params=DEFAULT_PARAMS):
    """Run one CARE collision-avoidance adjustment."""
    params = _merged_params(params)
    waypoints = np.asarray(waypoints)
    obstacles = ConstructTopDownObstacleMap(depth, intrinsics, params)

    if obstacles.shape[0] == 0:
        k_idx = 1 if waypoints.shape[0] > 1 else 0
        theta_des = ComputeDesiredHeading(waypoints, k_idx)
        v, omega = MotionCommandFromHeading(theta_des, params)
        return {
            "v": v,
            "omega": omega,
            "adjusted_waypoints": waypoints,
            "theta_rot": 0.0,
            "k_star": None,
            "obstacles": obstacles,
        }

    theta_rot, k_star, frep_all = EstimateRepulsiveDirection(waypoints, obstacles, params)
    adjusted_waypoints = RotateTrajectory(waypoints, theta_rot)
    theta_des = ComputeDesiredHeading(adjusted_waypoints, k_star)
    v, omega = MotionCommandFromHeading(theta_des, params)
    return {
        "v": v,
        "omega": omega,
        "adjusted_waypoints": adjusted_waypoints,
        "theta_rot": theta_rot,
        "k_star": k_star,
        "obstacles": obstacles,
        "frep_all": frep_all,
    }
