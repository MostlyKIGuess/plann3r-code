# Real-world deployment code

Code to run Plann3r navigation on a real robot. The robot records a traversal
and streams camera frames, and a GPU server runs Plann3r and the GNM controller
and returns velocity commands over HTTP. Every parameter and how to tune it is
in [`docs/real-world.md`](../docs/real-world.md).

## Files

| File | Runs on | What it does |
|---|---|---|
| `Dockerfile` | robot | ROS Noetic image with RosAria (built from source), `realsense2_camera`, tmux and the scripts below |
| `start_robot_tmux.sh` | robot host | starts the container and a tmux session with `roscore`, RosAria, the RealSense driver, and the recorder, client or teleop, chosen by `MODE` |
| `start_p3dx_realsense.sh` | robot container | starts `roscore`, RosAria and the RealSense driver without tmux |
| `start.sh` | robot container | image entrypoint, sources ROS and runs the given command or `zsh` |
| `record_realsense_map.py` | robot container | saves RGB frames, odometry and commands of a hand-driven traversal |
| `teleop`, `teleop_joystick.py`, `teleop_keyboard.py` | robot container | joystick and keyboard driving on `/RosAria/cmd_vel` |
| `plann3r_ros_client.py` | robot container | sends frames and odometry to the server and publishes the returned command |
| `build_plann3r_map.sh` | GPU | builds the Plann3r propagation costmaps of a recorded map |
| `plann3r_realworld_server.py` | GPU | HTTP server that localizes, predicts the query costmap and runs the controller |
| `configs/gnm_gt_navmesh_costmap_history5.yaml` | GPU | default GNM controller config, 5 stacked costmaps, velocity filter off |

## Expected hardware

- A differential-drive base that takes `geometry_msgs/Twist` and publishes
  `nav_msgs/Odometry`. The defaults are a Pioneer P3DX on RosAria at
  `/dev/ttyUSB0` with topics `/RosAria/cmd_vel` and `/RosAria/pose`
  (`start_robot_tmux.sh`).
- An RGB camera on a ROS image topic. The default is an Intel RealSense through
  `realsense2_camera`, color stream at 320x240 and 15 fps on
  `/camera/color/image_raw` (`start_robot_tmux.sh`). Only the color stream is
  used, and no script reads intrinsics.
- A robot-side machine with Docker, tmux and, for teleop, a joystick at
  `/dev/input/js0`. It needs no GPU and no torch.
- A CUDA GPU server with this repository's Pixi environment, reachable from the
  robot over HTTP on port 8088.

## Environment variables

| Variable | Machine | Meaning |
|---|---|---|
| `PLANN3R_CKPT` | GPU | Plann3r planner checkpoint, for example `$PLANN3R_ROOT/models/planner/checkpoint_best.pt` |
| `PLANN3R_REAL_CONTROLLER_RUN` | GPU | folder holding the GNM controller `latest.pth` |
| `PLANN3R_SERVER_URL` | robot | server address, `http://<gpu-host>:8088` |
| `DATA_DIR` | robot | host folder mounted as `/data/plann3r_real` (default `data/plann3r_real` in the repo) |
| `MODE` | robot | `record`, `teleop`, `nav` (dry run) or `nav-live` |

The default controller config expects a GNM trained with 5 stacked costmaps,
which is not one of the released controllers. A released controller needs its
own config through the server's `--controller-config`. The other launcher
variables, such as
`MAX_V`, `EXECUTE_CMD_TIME` and `GOAL_DISTANCE_THRESHOLD`, are listed with
their defaults in [`docs/real-world.md`](../docs/real-world.md#robot-client).

## Launch order

Run all commands from the repository root.

1. Build the robot image on the robot machine.

   ```bash
   docker build -f real_world/Dockerfile -t plann3r-rrc .
   ```

2. Record the traversal while driving with the joystick or keyboard. Stop the
   recorder with Ctrl-C when done, then tear down.

   ```bash
   MODE=record MAP_NAME=my_map real_world/start_robot_tmux.sh
   real_world/start_robot_tmux.sh stop
   ```

3. Copy `data/plann3r_real/my_map` to the GPU machine and build the propagation
   costmaps. The arguments after the map folder are the goal frame and the goal
   pixel x and y in the 320x240 image.

   ```bash
   PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
   pixi run bash real_world/build_plann3r_map.sh /data/plann3r_real/my_map <goal_frame> <pixel_x> <pixel_y>
   ```

4. Start the server on the GPU machine and keep it running.

   ```bash
   PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
   PLANN3R_REAL_CONTROLLER_RUN=/path/to/controller_run \
   pixi run python real_world/plann3r_realworld_server.py \
     --device cuda --map-dir /data/plann3r_real/my_map --port 8088 --retrieval odom
   ```

5. Put the robot at the start pose of the map and do a dry run. Commands are
   printed, not published.

   ```bash
   PLANN3R_SERVER_URL=http://<gpu-host>:8088 DRY_RUN=1 real_world/start_robot_tmux.sh
   ```

6. Stop the dry run and drive. `nav-live` publishes commands and opens a teleop
   pane for manual override. Keep the emergency stop within reach.

   ```bash
   real_world/start_robot_tmux.sh stop
   MODE=nav-live PLANN3R_SERVER_URL=http://<gpu-host>:8088 real_world/start_robot_tmux.sh
   ```

A running tmux session keeps its old settings, so run
`real_world/start_robot_tmux.sh stop` before changing any variable.
