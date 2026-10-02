#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

if [[ "$#" -eq 0 ]]; then
  exec zsh
fi

exec "$@"
