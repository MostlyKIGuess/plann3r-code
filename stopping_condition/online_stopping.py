"""Decide online when the agent has reached the goal, without the oracle distance.

A stop needs three things for the middle frame of the last nine. Its localized
submap holds a frame less than map_frame_slack from the goal frame, at least n costmap pixels are below eps, and at
least n_depth of those pixels are closer than depth_m in base VGGT depth.
run_nav.py builds OnlineStoppingCondition when online_stopping.enabled=true,
which baseline/evaluate.sh sets for the no-oracle-paper mode.
"""

from collections import deque

import cv2
import numpy as np


class OnlineStoppingCondition:
    """Causal wrapper around the centered nine-frame stopping rule.

    A stop is decided for the middle frame of the last nine, once the four
    frames after it are observed, so the rule never looks ahead in time.
    """

    def __init__(self, model, eps=0.2, n=100, n_depth=100, depth_m=1.0, map_frame_slack=1):
        self.model = model
        self.eps = float(eps)
        self.n = int(n)
        self.n_depth = int(n_depth)
        self.depth_m = float(depth_m)
        self.map_frame_slack = int(map_frame_slack)
        self.window = deque(maxlen=9)

    def observe(self, step, rgb, costmap, localized_img_idxs, goal_frame_idx):
        # The rule was tuned on frames read with cv2.imread, so the depth model
        # keeps receiving BGR input.
        rgb_bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        self.window.append(
            {
                "step": int(step),
                "rgb": rgb_bgr,
                "costmap": np.asarray(costmap),
                "localized_img_idxs": [int(i) for i in localized_img_idxs],
            }
        )
        if len(self.window) < 9:
            return None

        samples = list(self.window)
        candidate = samples[4]
        near_goal_frame = any(
            abs(frame - int(goal_frame_idx)) < self.map_frame_slack
            for frame in candidate["localized_img_idxs"]
        )
        if not near_goal_frame:
            return None

        height, width = candidate["rgb"].shape[:2]
        costmap = cv2.resize(
            candidate["costmap"], (width, height), interpolation=cv2.INTER_NEAREST
        )
        low_cost_mask = costmap < self.eps
        low_cost_pixels = int(low_cost_mask.sum())
        if low_cost_pixels < self.n:
            return None

        depth = self.model.infer([sample["rgb"] for sample in samples])
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        close_depth_pixels = int((depth[low_cost_mask] < self.depth_m).sum())
        if close_depth_pixels < self.n_depth:
            return None

        return {
            "candidate_step": candidate["step"],
            "decision_step": int(step),
            "low_cost_pixels": low_cost_pixels,
            "close_depth_pixels": close_depth_pixels,
        }
