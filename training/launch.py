# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Start Plann3r planner training from a Hydra config in training/config.

Adapted from VGGT training. It composes the config (nav_costmap by default)
with any key=value overrides and runs the Trainer. It is run from the training/
folder through torchrun, usually by training/run_nav_single_gpu.sh.

Usage:
    cd training && torchrun --standalone --nproc_per_node 1 launch.py --config nav_costmap [key=value ...]
"""

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="nav_costmap",
        help="Name of the config file (without .yaml extension, default: nav_costmap)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra-style config overrides, e.g. key=value",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


