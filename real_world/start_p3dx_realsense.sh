#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

ROSARIA_PORT="${ROSARIA_PORT:-/dev/ttyUSB0}"
COLOR_WIDTH="${COLOR_WIDTH:-320}"
COLOR_HEIGHT="${COLOR_HEIGHT:-240}"
COLOR_FPS="${COLOR_FPS:-15}"
ENABLE_DEPTH="${ENABLE_DEPTH:-false}"
ALIGN_DEPTH="${ALIGN_DEPTH:-true}"

echo "Starting P3DX + RealSense stack"
echo "ROS_MASTER_URI=${ROS_MASTER_URI:-unset}"
echo "ROS_IP=${ROS_IP:-unset}"
echo "ROSARIA_PORT=${ROSARIA_PORT}"

if ! pgrep -f "roscore" >/dev/null 2>&1; then
  roscore &
  ROSCORE_PID=$!
  sleep 3
else
  ROSCORE_PID=""
fi

rosrun rosaria RosAria _port:="${ROSARIA_PORT}" &
ROSARIA_PID=$!

roslaunch realsense2_camera rs_camera.launch \
  color_width:="${COLOR_WIDTH}" \
  color_height:="${COLOR_HEIGHT}" \
  color_fps:="${COLOR_FPS}" \
  enable_depth:="${ENABLE_DEPTH}" \
  align_depth:="${ALIGN_DEPTH}" &
REALSENSE_PID=$!

cleanup() {
  echo "Stopping P3DX + RealSense stack"
  kill "${REALSENSE_PID}" "${ROSARIA_PID}" 2>/dev/null || true
  if [[ -n "${ROSCORE_PID}" ]]; then
    kill "${ROSCORE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait
