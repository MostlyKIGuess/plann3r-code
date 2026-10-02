# Real-world deployment

The robot records a traversal of the route, which becomes the map. At run time
it sends each camera frame and its odometry to a GPU server over HTTP. The
server runs Plann3r and the GNM controller and replies with a velocity command.
The code is in [`real_world/`](../real_world), and every value below is a code
default.

## Setup

| Machine | Needs | Code |
|---|---|---|
| Robot | ROS Noetic (`real_world/Dockerfile`), an RGB image topic, an odometry topic, a velocity command topic | `plann3r_ros_client.py`, `record_realsense_map.py` |
| Server | a CUDA GPU, the Pixi environment, the planner and controller checkpoints, the map with its propagation costmaps | `plann3r_realworld_server.py` |

| Topic or device | Default | Set with |
|---|---|---|
| Image | `/camera/color/image_raw` | `RGB_TOPIC` |
| Odometry | `/RosAria/pose` | `ODOM_TOPIC` |
| Command | `/RosAria/cmd_vel` | `CMD_TOPIC` |
| Base driver | `rosrun rosaria RosAria _port:=/dev/ttyUSB0` | `ROSARIA_CMD` |
| Camera driver | `roslaunch realsense2_camera rs_camera.launch color_width:=320 color_height:=240 color_fps:=15 align_depth:=true` | `CAMERA_CMD` |
| Joystick | `/dev/input/js0` | `JOY_DEVICE` |

Only the color stream is used. The server listens on port 8088 (`--port`).

## Recording the map

Drive the route by hand with `MODE=record`. The recorder saves 320x240 frames
at 1.5 Hz (`--hz`) with their odometry to `images/` and `frames.jsonl`. Stop it
with Ctrl-C so `map_meta.json` is written. Record with the same odometry source
used at run time, because the server localizes against it.

## Goal and propagation map

The goal is a frame index and a pixel (x, y) in that frame.
`build_plann3r_map.sh` builds the Plann3r propagation costmaps for that goal
with the same settings as the released maps (windows of 9 frames, stride 8)
and writes `vggt_propagation_costs.npy` and `vggt_propagation_costs_meta.json`
into the map folder.

## Server

For each request the server:

1. Localizes the query. It picks the map frame with the lowest
   `xy distance + 0.25 * |yaw difference|` relative to the start pose
   (`--odom-yaw-weight`). Without odometry it falls back to the closest 64x48
   grayscale thumbnail.
2. Takes 8 consecutive map frames around that frame as the submap
   (`--submap-size`).
3. Picks the anchor, the lowest propagation cost over the submap frames.
4. Predicts the query costmap with Plann3r and runs the GNM controller.
5. Clips the command to 0.20 m/s (`--max-v`) and 0.60 rad/s (`--max-w`).

The robot must start at the pose of the map's first frame with valid odometry,
or at the frame passed as `RESET_MAP_FRAME`.

## Controller

`--controller-config` selects the GNM controller config. The default,
`real_world/configs/gnm_gt_navmesh_costmap_history5.yaml`, expects a controller
trained with five stacked costmaps, set with `PLANN3R_REAL_CONTROLLER_RUN`. The
released controllers use `configs/controller/predicted_costmap.yaml`.

| Setting | Default | Config key |
|---|---|---|
| Costmap normalization | per-map min-max to [0, 1] | `costmap_normalization` |
| Costmap history | 5 | `costmap_history_size` |
| RGB context | 5 past frames at 85x64 | `context_size`, `image_size` |
| Waypoint used | last of 5 | `waypoint_index`, `len_traj_pred` |
| Velocity filter | off | `use_vel_filter`, `vel_filter_window` |

The velocity filter replaces each command by the mean of the last
`vel_filter_window` commands. It smooths the motion and delays turns.

The controller turns the chosen waypoint (forward, lateral) into a command with
fixed limits in `libs/control/learnt_controller.py`:

```text
w = -clip(arctan2(lateral, forward), -0.1, 0.1)
v = min(forward / 100, 0.05)
```

## Robot client

The client publishes each reply as

```text
linear.x  = clip(v * LINEAR_SCALE,                 -MAX_V, MAX_V)
angular.z = clip(w * ANGULAR_SCALE * ANGULAR_SIGN, -MAX_W, MAX_W)
```

holds it for `EXECUTE_CMD_TIME`, then publishes a stop and sends the next
frame. Set these as environment variables for `start_robot_tmux.sh`:

| Parameter | Variable | Default | Effect |
|---|---|---|---|
| Linear scale | `LINEAR_SCALE` | 3.0 | top speed is 0.05 x scale, 0.15 m/s |
| Angular scale | `ANGULAR_SCALE` | 3.0 | top turn rate is 0.1 x scale, 0.30 rad/s |
| Angular sign | `ANGULAR_SIGN` | -1.0 | flip if the base turns the wrong way |
| Max linear speed | `MAX_V` | 0.20 m/s | hard limit after scaling |
| Max angular speed | `MAX_W` | 0.60 rad/s | hard limit after scaling |
| Command hold | `EXECUTE_CMD_TIME` | 0.35 s | motion per cycle. Scale the speeds by the inverse factor to keep it |
| Loop rate | `NAV_HZ` | 5 Hz | upper bound on requests per second |
| HTTP timeout | `--timeout` | 2.0 s | slower replies stop the robot |
| Max frame age | `MAX_FRAME_AGE` | 0.75 s | older frames stop the robot |
| Max odometry age | `MAX_ODOM_AGE` | 0.75 s | older odometry stops the robot |
| Require odometry | `REQUIRE_ODOM` | on | no command before odometry arrives |
| Max reply age | `MAX_RESPONSE_AGE` | 1.50 s | replies older than this, from the image stamp, stop the robot |
| Goal distance | `GOAL_DISTANCE_THRESHOLD` | 1.00 m | stop when the odometry distance to the goal frame is within it |
| Dry run | `DRY_RUN` | on | print commands instead of publishing |

With the defaults one cycle moves at most 0.0525 m and 0.105 rad, close to the
0.05 m and 0.1 rad per simulator step. The goal distance is a straight line in
odometry, so set it above the odometry drift expected over the route.

The robot also stops on a reply that does not match the request, on any error,
and on shutdown. The code has no obstacle stop. Keep the base's emergency stop
in reach. In `MODE=nav-live` a teleop pane publishes to the same topic and
overrides the client while a key or stick is in use (0.20 m/s and 0.75 rad/s
maximum, deadzone 0.08).

## Running

| Variable | Machine | Meaning |
|---|---|---|
| `PLANN3R_ROOT` | server | release bundle root ([`setup.md`](setup.md)) |
| `PLANN3R_CKPT` | server | planner checkpoint, `$PLANN3R_ROOT/models/planner/checkpoint_best.pt` |
| `PLANN3R_REAL_CONTROLLER_RUN` | server | folder with the controller `latest.pth` |
| `PLANN3R_SERVER_URL` | robot | `http://<server-host>:8088` |
| `MODE` | robot | `record`, `teleop`, `nav` (dry run) or `nav-live` |
| `DATA_DIR` | robot | host folder for recorded maps, mounted as `/data/plann3r_real` (default `data/plann3r_real` in the repository) |

From the repository root:

```bash
# Robot: build the image and record the route
docker build -f real_world/Dockerfile -t plann3r-rrc .
MODE=record MAP_NAME=my_map real_world/start_robot_tmux.sh
real_world/start_robot_tmux.sh stop

# Copy $DATA_DIR/my_map from the robot to the server, then on the server:
export PLANN3R_ROOT=/path/to/plann3r-release
PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
pixi run bash real_world/build_plann3r_map.sh /path/to/my_map <goal_frame> <pixel_x> <pixel_y>

PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
PLANN3R_REAL_CONTROLLER_RUN=/path/to/controller_run \
pixi run python real_world/plann3r_realworld_server.py --map-dir /path/to/my_map

# Robot: dry run from the start pose, then drive
PLANN3R_SERVER_URL=http://<server-host>:8088 DRY_RUN=1 real_world/start_robot_tmux.sh
real_world/start_robot_tmux.sh stop
MODE=nav-live PLANN3R_SERVER_URL=http://<server-host>:8088 real_world/start_robot_tmux.sh
```

On a new robot, point the topic and driver variables at your drivers, check
the turn direction in a dry run, then set the scales and limits to what the
base can do safely.
