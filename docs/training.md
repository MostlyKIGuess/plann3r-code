# Training

All commands assume the bundle root is set:

```bash
export PLANN3R_ROOT=/absolute/path/to/plann3r-release
```

## Planner data

Each planner sample contains:

- One query RGB image.
- Eight localized map RGB images.
- Query and map geometry used by the auxiliary heads.
- A goal-conditioned target costmap.
- Normalization metadata.

The released training root is expected at:

```text
$PLANN3R_ROOT/training/planner/vggtnav/
```

The subtrajectory folders use relative links into
`$PLANN3R_ROOT/training/planner/source_scenes/`. The released bundle contains
the referenced RGB, depth, and pointmap files once per scene. A bundle with
absolute links to its source machine is incomplete and must be replaced.

The planner bundle excludes rendered visualizations, navigation trajectory
files, metadata unused by the dataset loader, and waypoint targets unused by
the selected costmap configuration.

## Planner training

Train the selected Plann3r configuration:

```bash
cd "$PLANN3R_ROOT/plann3r-code"

DATA_ROOT="$PLANN3R_ROOT/training/planner/vggtnav" \
BASE_VGGT_MODEL_PATH="$PLANN3R_ROOT/models/vggt/model.pt" \
RUN_NAME=submap_normalized_costmaps_train_long \
LOG_DIR="$PLANN3R_ROOT/runs/planner" \
bash training/run_nav_single_gpu.sh 0
```

The selected model receives one query plus eight map images. Its MLP head
predicts a 16x16 query costmap. The paper checkpoint is selected by validation
loss and stored as `checkpoint_best.pt`.

## Controller data

The controller dataset is expected at:

```text
$PLANN3R_ROOT/training/controller/predicted_costmap/
```

This directory contains `data/`, `splits/`, and `source_episodes/`. Files under
`data/` use relative links to the RGB and trajectory inputs in
`source_episodes/`. The predicted costmaps remain under `data/`.

The controller bundle includes RGB and depth images. It excludes semantic
labels, pointmaps, navmesh costmaps, graph files, and rendered maps. The
selected loader opens RGB images, `traj_data.pkl`, `vggt_costmaps.npy`, and the
train/test trajectory lists. Depth is kept for related experiments.

Each trajectory supplies RGB observations, actions, and Plann3r-predicted
costmaps. The controller configuration uses:

```text
observation context: 5
costmap size: 16x16
costmap channels: 1
costmap normalization: per-map min-max
predicted waypoints: 5
learn angle: true
velocity filter: false
```

The historical field `goal_type: navmesh_costmap` is the tensor loader enum.
The selected controller was trained on Plann3r predictions, not on Habitat
navmesh costmaps.

## Controller training

The dataset paths in
`libs/control/visualnav_transformer/train/config/predicted_costmap.yaml` use
`PLANN3R_ROOT`. `train.py` expands the variable and raises an error if it is
missing.

```bash
cd "$PLANN3R_ROOT/plann3r-code/libs/control/visualnav_transformer/train"
pixi run python train.py -c config/predicted_costmap.yaml
```

Copy the selected `latest.pth` directory to:

```text
$PLANN3R_ROOT/models/controller/predicted_costmap/
```

The deployed model shape must match the training config. The released
checkpoint has a 20-value action head ([`method.md`](method.md#checkpoints)).
A ten-waypoint deployment config creates a 40-value head and is rejected during
checkpoint loading.
