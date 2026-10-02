#!/usr/bin/env bash
# Start or attach to the Plann3r robot-side Docker container, then bring up a host-side
# tmux session whose panes docker-exec into it.
#
# Layout:
#   top-left     : roscore
#   top-right    : rosrun rosaria RosAria
#   bottom-right : roslaunch realsense2_camera rs_camera.launch
#   bottom-left  : Plann3r ROS client, map recorder, or teleop
#
# Usage:
#   real_world/start_robot_tmux.sh             # bring up container + tmux, attach
#   real_world/start_robot_tmux.sh detach      # bring up container + tmux, do not attach
#   real_world/start_robot_tmux.sh stop        # tear down tmux session and container
#
# Common overrides:
#   MODE=record real_world/start_robot_tmux.sh
#   MODE=teleop real_world/start_robot_tmux.sh
#   MODE=nav-live real_world/start_robot_tmux.sh  # live nav plus manual teleop override pane
#   MODE=record real_world/start_robot_tmux.sh  # recorder plus teleop pane
#   PLANN3R_SERVER_URL=http://<gpu-host>:8088 real_world/start_robot_tmux.sh  # dry run, attaches
#   DRY_RUN=0 PLANN3R_SERVER_URL=http://<gpu-host>:8088 real_world/start_robot_tmux.sh
#
# PLANN3R_SERVER_URL is required for MODE=nav and MODE=nav-live. See docs/real-world.md.
#   ROSARIA_CMD="rosrun rosaria RosAria _port:=/dev/ttyUSB0" real_world/start_robot_tmux.sh
#   CAMERA_CMD="roslaunch realsense2_camera rs_camera.launch color_width:=320 color_height:=240 color_fps:=15 align_depth:=true" real_world/start_robot_tmux.sh

set -euo pipefail

IMAGE="${IMAGE:-plann3r-rrc}"
CONTAINER="${CONTAINER:-plann3r-rrc}"
SESSION="${SESSION:-plann3r-rrc}"
MODE="${MODE:-nav}"
REQUESTED_MODE="$MODE"
NAV_TELEOP_DEFAULT=0
if [[ "$MODE" == "nav-live" ]]; then
  MODE="nav"
  DRY_RUN="${DRY_RUN:-0}"
  NAV_TELEOP_DEFAULT=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Host-side data root. This is mounted into the container as /data/plann3r_real.
# Keep the default user-writable; /data often requires root on robot laptops.
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/plann3r_real}"

RGB_TOPIC="${RGB_TOPIC:-/camera/color/image_raw}"
CMD_TOPIC="${CMD_TOPIC:-/RosAria/cmd_vel}"
ODOM_TOPIC="${ODOM_TOPIC:-/RosAria/pose}"
JOY_DEVICE="${JOY_DEVICE:-/dev/input/js0}"

HOST_IP="${HOST_IP:-$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')}"
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
HOST_IP="${HOST_IP:-127.0.0.1}"

ROSCORE_CMD="${ROSCORE_CMD:-roscore}"
ROSARIA_CMD="${ROSARIA_CMD:-rosrun rosaria RosAria _port:=/dev/ttyUSB0}"
CAMERA_CMD="${CAMERA_CMD:-roslaunch realsense2_camera rs_camera.launch color_width:=320 color_height:=240 color_fps:=15 align_depth:=true}"

SERVER_URL="${PLANN3R_SERVER_URL:-}"
NAV_HZ="${NAV_HZ:-5}"
MAX_V="${MAX_V:-0.20}"
MAX_W="${MAX_W:-0.60}"
LINEAR_SCALE="${LINEAR_SCALE:-3.0}"
ANGULAR_SCALE="${ANGULAR_SCALE:-3.0}"
ANGULAR_SIGN="${ANGULAR_SIGN:--1.0}"
MAX_FRAME_AGE="${MAX_FRAME_AGE:-0.75}"
MAX_RESPONSE_AGE="${MAX_RESPONSE_AGE:-1.50}"
EXECUTE_CMD_TIME="${EXECUTE_CMD_TIME:-0.35}"
GOAL_DISTANCE_THRESHOLD="${GOAL_DISTANCE_THRESHOLD:-1.00}"
MAX_ODOM_AGE="${MAX_ODOM_AGE:-0.75}"
REQUIRE_ODOM="${REQUIRE_ODOM:-1}"
CLIENT_VIS_EVERY="${CLIENT_VIS_EVERY:-0}"
RESET_MAP_FRAME="${RESET_MAP_FRAME:--1}"
TELEOP_TYPE="${TELEOP_TYPE:-both}"
TELEOP_MAX_V="${TELEOP_MAX_V:-0.20}"
TELEOP_MAX_W="${TELEOP_MAX_W:-0.75}"
TELEOP_LINEAR_AXIS="${TELEOP_LINEAR_AXIS:-1}"
TELEOP_ANGULAR_AXIS="${TELEOP_ANGULAR_AXIS:-3}"
TELEOP_DEADZONE="${TELEOP_DEADZONE:-0.08}"
TELEOP_ENABLE_BUTTON="${TELEOP_ENABLE_BUTTON:--1}"
TELEOP_CALIBRATE_SECONDS="${TELEOP_CALIBRATE_SECONDS:-0.75}"
TELEOP_INVERT_ANGULAR="${TELEOP_INVERT_ANGULAR:-1}"
NAV_TELEOP="${NAV_TELEOP:-$NAV_TELEOP_DEFAULT}"
NAV_TELEOP_IDLE_NO_PUBLISH="${NAV_TELEOP_IDLE_NO_PUBLISH:-1}"
DRY_RUN="${DRY_RUN:-1}"
RUN_NAME="${RUN_NAME:-nav_run_$(date +%Y%m%d_%H%M%S)}"
MAP_NAME="${MAP_NAME:-lab_map_$(date +%Y%m%d_%H%M%S)}"

ROS_SETUP="source /root/.rosrc"
CONTAINER_REPO="/workspace/plann3r"
CONTAINER_DATA="/data/plann3r_real"
CLIENT_VIS_DIR="${CLIENT_VIS_DIR:-$CONTAINER_DATA/$RUN_NAME/client_vis}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[start_robot_tmux.sh] missing dependency: $1" >&2
    exit 1
  }
}

need docker
need tmux

if [[ "${1:-}" != "stop" && "$MODE" == "nav" && -z "$SERVER_URL" && -z "${CLIENT_CMD:-}" ]]; then
  echo "[start_robot_tmux.sh] set PLANN3R_SERVER_URL=http://<gpu-host>:8088 for MODE=nav or MODE=nav-live" >&2
  exit 2
fi

if [[ "${1:-}" == "stop" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "[start_robot_tmux.sh] killed tmux session $SESSION"
  fi
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER" >/dev/null
    echo "[start_robot_tmux.sh] removed container $CONTAINER"
  fi
  exit 0
fi

mkdir -p "$DATA_DIR"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "[start_robot_tmux.sh] removing stale container $CONTAINER"
    docker rm -f "$CONTAINER" >/dev/null
  fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[start_robot_tmux.sh] image '$IMAGE' not found. Build it from the repo root:" >&2
    echo "    docker build -f real_world/Dockerfile -t $IMAGE ." >&2
    exit 1
  fi

echo "[start_robot_tmux.sh] starting container $CONTAINER from $IMAGE"
echo "[start_robot_tmux.sh] ROS_IP=$HOST_IP   SERVER_URL=$SERVER_URL"
echo "[start_robot_tmux.sh] MODE=$REQUESTED_MODE normalized=$MODE DRY_RUN=${DRY_RUN:-unset} NAV_TELEOP=$NAV_TELEOP MAX_RESPONSE_AGE=$MAX_RESPONSE_AGE EXECUTE_CMD_TIME=$EXECUTE_CMD_TIME GOAL_DISTANCE_THRESHOLD=$GOAL_DISTANCE_THRESHOLD CLIENT_VIS_EVERY=$CLIENT_VIS_EVERY"

  docker_args=(
    run -d --rm
    --name "$CONTAINER"
    --network host
    --privileged
    --uts host
    -v /dev:/dev
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "$REPO_ROOT:$CONTAINER_REPO"
    -v "$DATA_DIR:$CONTAINER_DATA"
    -e "DISPLAY=${DISPLAY:-:0}"
    -e "ROS_IP=$HOST_IP"
    -e "ROS_HOSTNAME=$HOST_IP"
    -e "ROS_MASTER_URI=http://$HOST_IP:11311"
    -e "PLANN3R_REPO=$CONTAINER_REPO"
  )
  if [[ -n "${DOCKER_GPU_ARGS:-}" ]]; then
    read -r -a gpu_args <<< "$DOCKER_GPU_ARGS"
    docker_args+=("${gpu_args[@]}")
  fi
  docker "${docker_args[@]}" "$IMAGE" sleep infinity >/dev/null

  for _ in $(seq 1 20); do
    if docker exec "$CONTAINER" true >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
else
  echo "[start_robot_tmux.sh] container $CONTAINER already running"
fi

EXEC="docker exec -it \
  -e ROS_IP=$HOST_IP \
  -e ROS_HOSTNAME=$HOST_IP \
  -e ROS_MASTER_URI=http://$HOST_IP:11311 \
  -e PLANN3R_REPO=$CONTAINER_REPO \
  $CONTAINER zsh -c"

if [[ "$TELEOP_TYPE" == "keyboard" ]]; then
  teleop_idle_arg=""
  if [[ "$MODE" == "nav" && "$NAV_TELEOP" != "0" && "$NAV_TELEOP" != "false" && "$NAV_TELEOP_IDLE_NO_PUBLISH" != "0" && "$NAV_TELEOP_IDLE_NO_PUBLISH" != "false" ]]; then
    teleop_idle_arg="--idle-no-publish"
  fi
  TELEOP_MODE_CMD="${TELEOP_CMD:-$CONTAINER_REPO/real_world/teleop keyboard --cmd-topic $CMD_TOPIC --max-v $TELEOP_MAX_V --max-w $TELEOP_MAX_W $teleop_idle_arg}"
elif [[ "$TELEOP_TYPE" == "joystick" || "$TELEOP_TYPE" == "both" ]]; then
  angular_invert_arg="--invert-angular"
  if [[ "$TELEOP_INVERT_ANGULAR" == "0" || "$TELEOP_INVERT_ANGULAR" == "false" ]]; then
    angular_invert_arg="--no-invert-angular"
  fi
  teleop_idle_arg=""
  if [[ "$MODE" == "nav" && "$NAV_TELEOP" != "0" && "$NAV_TELEOP" != "false" && "$NAV_TELEOP_IDLE_NO_PUBLISH" != "0" && "$NAV_TELEOP_IDLE_NO_PUBLISH" != "false" ]]; then
    teleop_idle_arg="--idle-no-publish"
  fi
  TELEOP_MODE_CMD="${TELEOP_CMD:-$CONTAINER_REPO/real_world/teleop $TELEOP_TYPE --device $JOY_DEVICE --cmd-topic $CMD_TOPIC --linear-axis $TELEOP_LINEAR_AXIS --angular-axis $TELEOP_ANGULAR_AXIS --deadzone $TELEOP_DEADZONE --max-v $TELEOP_MAX_V --max-w $TELEOP_MAX_W --enable-button $TELEOP_ENABLE_BUTTON --calibrate-seconds $TELEOP_CALIBRATE_SECONDS $angular_invert_arg $teleop_idle_arg}"
else
  echo "[start_robot_tmux.sh] unsupported TELEOP_TYPE=$TELEOP_TYPE. Use joystick, keyboard, or both." >&2
  exit 2
fi

if [[ "$MODE" == "record" ]]; then
  BOTTOM_LEFT_CMD="${RECORD_CMD:-python3 $CONTAINER_REPO/real_world/record_realsense_map.py --out-dir $CONTAINER_DATA/$MAP_NAME --rgb-topic $RGB_TOPIC --odom-topic $ODOM_TOPIC --cmd-topic $CMD_TOPIC --hz 1.5 --width 320 --height 240}"
elif [[ "$MODE" == "teleop" ]]; then
  BOTTOM_LEFT_CMD="$TELEOP_MODE_CMD"
elif [[ "$MODE" == "nav" ]]; then
  dry_run_arg=""
  if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "false" ]]; then
    dry_run_arg="--dry-run"
  fi
  require_odom_arg=""
  if [[ "$REQUIRE_ODOM" != "0" && "$REQUIRE_ODOM" != "false" ]]; then
    require_odom_arg="--require-odom"
  fi
  client_vis_arg=""
  if [[ "$CLIENT_VIS_EVERY" != "0" ]]; then
    client_vis_arg="--vis-dir $CLIENT_VIS_DIR --vis-every $CLIENT_VIS_EVERY"
  fi
  BOTTOM_LEFT_CMD="${CLIENT_CMD:-python3 $CONTAINER_REPO/real_world/plann3r_ros_client.py --server-url $SERVER_URL --rgb-topic $RGB_TOPIC --odom-topic $ODOM_TOPIC --cmd-topic $CMD_TOPIC --hz $NAV_HZ --max-v $MAX_V --max-w $MAX_W --linear-scale $LINEAR_SCALE --angular-scale $ANGULAR_SCALE --angular-sign $ANGULAR_SIGN --max-frame-age $MAX_FRAME_AGE --max-response-age $MAX_RESPONSE_AGE --execute-cmd-time $EXECUTE_CMD_TIME --goal-distance-threshold $GOAL_DISTANCE_THRESHOLD --max-odom-age $MAX_ODOM_AGE $require_odom_arg --reset-map-frame $RESET_MAP_FRAME $client_vis_arg $dry_run_arg --log-jsonl $CONTAINER_DATA/$RUN_NAME/client_log.jsonl}"
else
  echo "[start_robot_tmux.sh] unsupported MODE=$MODE. Use MODE=nav, MODE=nav-live, MODE=record, or MODE=teleop." >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start_robot_tmux.sh] tmux session '$SESSION' already exists; attaching"
  echo "[start_robot_tmux.sh] existing panes are not restarted; run '$0 stop' first to apply MODE/DRY_RUN changes"
else
  P0=$(tmux new-session -d -s "$SESSION" -n robot -x 220 -y 50 -P -F '#{pane_id}')
  tmux send-keys -t "$P0" "$EXEC '$ROS_SETUP && $ROSCORE_CMD'" C-m

  sleep 2

  P1=$(tmux split-window -h -t "$P0" -P -F '#{pane_id}')
  tmux send-keys -t "$P1" "$EXEC '$ROS_SETUP && $ROSARIA_CMD'" C-m

  P2=$(tmux split-window -v -t "$P1" -P -F '#{pane_id}')
  tmux send-keys -t "$P2" "$EXEC '$ROS_SETUP && $CAMERA_CMD'" C-m

  P3=$(tmux split-window -v -t "$P0" -P -F '#{pane_id}')
  tmux send-keys -t "$P3" "$EXEC '$ROS_SETUP && sleep 5 && cd $CONTAINER_REPO && $BOTTOM_LEFT_CMD'" C-m

  if [[ "$MODE" == "record" && "${RECORD_TELEOP:-1}" != "0" && "${RECORD_TELEOP:-1}" != "false" ]] || [[ "$MODE" == "nav" && "$NAV_TELEOP" != "0" && "$NAV_TELEOP" != "false" ]]; then
    P4=$(tmux split-window -h -t "$P3" -P -F '#{pane_id}')
    tmux send-keys -t "$P4" "$EXEC '$ROS_SETUP && sleep 5 && cd $CONTAINER_REPO && $TELEOP_MODE_CMD'" C-m
  fi

  tmux select-pane -t "$P0"
fi

if [[ "${1:-}" == "detach" ]]; then
  echo "[start_robot_tmux.sh] tmux session running; attach with: tmux attach -t $SESSION"
  echo "[start_robot_tmux.sh] tear down with: real_world/start_robot_tmux.sh stop"
  exit 0
fi

tmux attach -t "$SESSION"
