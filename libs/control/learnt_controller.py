"""GNM costmap controller that turns an RGB frame and a goal costmap into velocities.

The controller is the low-level half of VGGT-Nav. ObjRelLearntController builds
the GNM model from the vendored visualnav_transformer code, restores latest.pth
from the configured load_run, keeps a short history of frames and costmaps, and
predicts waypoints. It returns a linear and angular velocity from one waypoint.
Used by libs/experiments/task_setup.py and real_world/plann3r_realworld_server.py,
which pass it the Plann3r costmap.
"""

import sys
import numpy as np
import yaml
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import logging
import pprint

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", "Don't use ConvNormActivation directly, *")

# Add visualnav_transformer train folder to sys.path using absolute paths
from pathlib import Path
main_dir = Path(__file__).resolve().parents[2]
sys.path.append(f"{main_dir}/libs/control/visualnav_transformer/train")

logger = logging.getLogger("[Controller]") # logger level is explicitly set below by LOG_LEVEL

from libs.control.visualnav_transformer.train.vint_train.models.gnm.gnm import GNM
from libs.control.visualnav_transformer.train.vint_train.training.train_utils import get_goal_image, get_obs_image
from libs.control.visualnav_transformer.train.vint_train.data.data_utils import resize_and_aspect_crop
from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)


def restore_model_checkpoint(
    config: dict,
    model: torch.nn.Module,
    load_project_folder: str | Path,
) -> tuple[dict, int]:
    """Restore the controller without importing the training entry point."""
    checkpoint_path = Path(load_project_folder) / "latest.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Controller checkpoint does not exist: {checkpoint_path}")

    print(f"Loading model from {load_project_folder}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        weights_only=False,
    )
    if config["model_type"] == "nomad":
        model.load_state_dict(checkpoint, strict=False)
    else:
        loaded_model = checkpoint["model"]
        state_source = loaded_model.module if hasattr(loaded_model, "module") else loaded_model
        model.load_state_dict(state_source.state_dict(), strict=False)

    current_epoch = int(checkpoint.get("epoch", -1)) + 1
    return checkpoint, current_epoch

class ObjRelLearntController:
    """
    Object relative learnt controller, driven by a dense goal costmap.
    """
    def __init__(self, config, **kwargs):
        if type(config) == str:
            print(f"ObjRelLearntController: {config = }")
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            pprint.pprint(self.config, indent=2, width=8)

        elif type(config) == dict:
            self.config = config
        else:
            raise ValueError(f"config must be a filepath or a dict, not {type(config)}")

        expanded_load_run = os.path.expandvars(str(self.config["load_run"]))
        if "$" in expanded_load_run:
            raise ValueError(
                "Controller load_run contains an unresolved environment variable: "
                f"{expanded_load_run}"
            )
        self.config["load_run"] = expanded_load_run

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.goal_type = self.config["goal_type"]
        self.obs_type = self.config["obs_type"]
        self.image_size = self.config["image_size"]
        # `navmesh_costmap` is the loader's name for any dense costmap goal. The
        # ObjectReact image and object-mask goals are not part of this release.
        if self.goal_type != "navmesh_costmap":
            raise ValueError(
                f"Unsupported controller goal_type={self.goal_type!r}; expected navmesh_costmap"
            )

        self.use_vel_filter = self.config["use_vel_filter"]

        self.model = GNM(
            self.config["context_size"],
            self.config["len_traj_pred"],
            self.config["learn_angle"],
            self.config["obs_encoding_size"],
            self.config["goal_encoding_size"],
            goal_type=self.goal_type,
            obs_type=self.obs_type,
            goal_use_pl=self.config["goal_use_pl"],
            dims_segFt=self.config["dims_segFt"],
            goal_uses_context=self.config["goal_uses_context"],
            goal_uses_stacked_context=self.config["goal_uses_stacked_context"],
            use_mask_grad=self.config["use_mask_grad"],
            costmap_channels=self.config.get("costmap_channels", 1),
            costmap_history_size=self.config.get("costmap_history_size", None),
            **kwargs,
        )

        _ = restore_model_checkpoint(
            self.config,
            self.model,
            load_project_folder=self.config["load_run"],
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = ([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.transform = transforms.Compose(self.transform)

        # init params
        self.reset_params()

        self.waypoint_index = self.config["waypoint_index"]
        print("Learnt controller initialized!")

    def reset_params(self):
        """
        Reset the controller
        """
        self.image_history = []
        self.goal_history = []
        self.action_history = []
        self.controller_logs = []

        self.action_pred = None

        self.v_rollout = np.zeros(self.config["len_traj_pred"])
        self.w_rollout = np.zeros(self.config["len_traj_pred"])

    def maintain_history(self, curr, history, history_size=None):
        """
        Maintain history for context
        """
        if history_size is None:
            history_size = self.config["context_size"] + 1
        diff = len(history) - history_size
        if diff < 0:
            for _ in range(abs(diff)):
                history.append(curr)
        else:
            history.pop(0)
            history.append(curr)

    def encode_navmesh_costmap(self, goal_data):
        costmap = np.asarray(goal_data, dtype=np.float32)
        if costmap.ndim == 3:
            costmap = costmap[0]
        finite_mask = np.isfinite(costmap)
        if finite_mask.any():
            fill_value = float(costmap[finite_mask].max())
            costmap = np.where(finite_mask, costmap, fill_value).astype(np.float32)
        else:
            costmap = np.zeros_like(costmap, dtype=np.float32)

        normalization = self.config.get("costmap_normalization", "per_map_minmax")
        if normalization == "per_map_minmax":
            min_value = float(costmap.min())
            max_value = float(costmap.max())
            costmap = (costmap - min_value) / max(max_value - min_value, 1e-6)
        elif normalization != "none":
            raise ValueError(f"Unknown costmap_normalization: {normalization}")

        costmap_size = tuple(self.config.get("costmap_size", [16, 16]))
        costmap_tensor = torch.as_tensor(costmap, dtype=torch.float32)[None, None]
        costmap_tensor = F.interpolate(costmap_tensor, size=costmap_size, mode="area")
        goal_enc = costmap_tensor[0].cpu().numpy()
        return goal_enc

    def get_costmap_history_size(self):
        costmap_history_size = self.config.get("costmap_history_size", None)
        if costmap_history_size is not None:
            return int(costmap_history_size)
        if self.config["goal_uses_context"] or self.config["goal_uses_stacked_context"]:
            return self.config["context_size"] + 1
        return 1

    def filter_vel(self, action, win=5, self_update=True):
        self.action_history.append(action)
        if len(self.action_history) > win:
            self.action_history.pop(0)
        action = np.array(self.action_history).mean(axis=0)
        if self_update:
            self.action_history[-1] = action
        return action

    def ready_obs(self, rgb):
        obs_image = resize_and_aspect_crop(
            Image.fromarray(rgb), self.image_size)
        self.maintain_history(obs_image, self.image_history)
        obs_image = torch.as_tensor(
            torch.cat(self.image_history), dtype=torch.float32)
        obs_image, _ = get_obs_image(
            obs_image[None, ...], self.obs_type, self.transform, self.device)
        return obs_image

    def ready_goal(self, goal_data):
        goal_enc = self.encode_navmesh_costmap(goal_data)
        goal_enc = torch.as_tensor(goal_enc, dtype=torch.float32)
        costmap_history_size = self.get_costmap_history_size()
        if costmap_history_size > 1:
            self.maintain_history(goal_enc, self.goal_history, costmap_history_size)
            goal_enc = torch.cat(self.goal_history)
        goal_image = goal_enc[None, ...]

        goal_image, _ = get_goal_image(
            goal_image, self.goal_type, self.transform, self.device)
        return goal_image

    def predict(self, rgb, goal_data):
        """
        predict the linear velocity and angular velocity
        """
        v, w = 0, 0
        with torch.no_grad():
            obs_image = self.ready_obs(rgb)
            goal_image = self.ready_goal(goal_data)

            model_outputs = self.model(obs_image, goal_image)
            _, action_pred = model_outputs
            self.action_pred = action_pred[0].float().cpu().numpy()
            wp = self.action_pred[self.waypoint_index][:2]

            w_rollout = np.arctan2(self.action_pred[:, 1], self.action_pred[:, 0])
            w_rollout = np.insert(w_rollout, 0, 0)
            self.w_rollout = -(w_rollout[1:] - w_rollout[:-1])
            v_rollout = 0.2 * self.action_pred[:, 0]
            v_rollout = np.insert(v_rollout, 0, 0)
            self.v_rollout = v_rollout[1:] - v_rollout[:-1]

            w = np.arctan2(wp[-1], wp[-2])
            w = np.clip(w, -0.1, 0.1)
            v = min(wp[0]/100, 0.05)
            if self.use_vel_filter:
                v, w = self.filter_vel(
                    [v, w],
                    win=int(self.config.get("vel_filter_window", 5)),
                )

            logger.info(f"Predicted lin: {v:.2f} and ang: {w:.2f}")
            self.controller_logs.append({
                "action_pred": self.action_pred,
                "v_rollout": self.v_rollout,
                "w_rollout": self.w_rollout,
                })
        return v, -w
