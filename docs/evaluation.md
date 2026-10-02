# Evaluation

All commands run from the repository root:

```bash
export PLANN3R_ROOT=/absolute/path/to/plann3r-release
cd "$PLANN3R_ROOT/plann3r-code"
```

## Reported rows

Oracle localization, all four tasks:

```bash
GPU=0 \
ABLATIONS=paper \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

The same planner with the controller trained on ground-truth costmaps:

```bash
GPU=0 \
ABLATIONS=paper \
CONTROLLER_CONFIG_FILE=configs/controller/gt_trained_costmap.yaml \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

MegaLoc retrieval with the oracle 1 m stop:

```bash
GPU=0 \
ABLATIONS=megaloc \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

MegaLoc retrieval with inferred stopping:

```bash
GPU=0 \
ABLATIONS=no-oracle-paper \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

Every mode uses the planner `models/planner/checkpoint_best.pt`, the
predicted-costmap controller `models/controller/predicted_costmap/latest.pth`
unless overridden, the Plann3r propagation map costmaps
([`method.md`](method.md#map-files)), 300 steps, a 0.4 m
camera height during navigation, and a 1.31 m camera height in the mapping run.
The oracle and `megaloc` modes count an episode as a success when the geodesic
distance to the goal is below 1 m. The `no-oracle-paper` mode lets the agent run
until it stops itself, then scores the stop decision against the same 1 m
radius.

## Launcher options

| Variable | Default | Meaning |
|---|---|---|
| `GPU` | `0` | CUDA device index |
| `ABLATIONS` | `paper` | Space-separated [modes](#modes) |
| `TASKS` | all four | Any of `imitate reverse altgoal shortcut` |
| `CONTROLLER_CONFIG_FILE` | empty | Controller config to load instead of the default |
| `ORACLE_MODE` | `legacy` | Ground-truth localization rule, `legacy` (reported) or `odometry` |
| `MEGALOC_GOAL_LOCK` | `false` | Keep the MegaLoc submap on the goal frame once it is the top match (not used for reported results) |
| `PROP_MAP_TAG` | empty | Read propagation costmaps built by another planner checkpoint, `<name>_<tag>.npy` |
| `MAX_STEPS` | `300` | Episode step budget |
| `VISUALIZE` | `false` | `true` also writes per-step frames and a video |
| `SUMMARIZE` | `true` | Write the summary table after all runs |
| `RESULTS_ROOT` | `$PLANN3R_ROOT/runs/ablation_gt` | Output root |
| `DRY_RUN` | `false` | Print the resolved commands without running them |

[`setup.md`](setup.md#evaluation-paths) lists the path overrides. The resolved
settings of each run are saved in its `config.yaml`.

## Modes

| Mode | Planner checkpoint | Localization | Stop |
|---|---|---|---|
| `paper` | `checkpoint_best.pt` | ground-truth pose | oracle, 1 m |
| `megaloc` | `checkpoint_best.pt` | MegaLoc retrieval | oracle, 1 m |
| `no-oracle-paper` | `checkpoint_best.pt` | MegaLoc retrieval | inferred |
| `costmap_only` | `ablations/costmap_only.pt` | ground-truth pose | oracle, 1 m |
| `no_pointmap_loss` | `ablations/no_pointmap_loss.pt` | ground-truth pose | oracle, 1 m |
| `no_grad_loss` | `ablations/no_grad_loss.pt` | ground-truth pose | oracle, 1 m |
| `frozen_mlp_goal_token` | `ablations/frozen_mlp_goal_token.pt` | ground-truth pose | oracle, 1 m |

## Planner ablations

Every ablation navigates with the full planner's released propagation maps, so
only the online planner changes:

```bash
GPU=0 \
ABLATIONS="paper costmap_only no_pointmap_loss no_grad_loss frozen_mlp_goal_token" \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

Summaries are written to:

```text
$RESULTS_ROOT/ablation_summary.csv
$RESULTS_ROOT/ablation_summary.md
```

## Reproducibility

Checkpoint equality does not make two closed-loop runs identical. A small
numerical difference in one step changes the next observation, and the change
accumulates over up to 300 steps. PyTorch version, CUDA version, GPU
architecture, and the Habitat build all cause such differences. The evaluation
has no random sampling, so one machine and one software environment give the
same result on every run. Compare per-episode outcomes only within one machine
and environment, and report the GPU with the numbers.

## Planning cost benchmark (MARD)

`mard_benchmark/` computes the Mean Absolute Rank Difference between predicted
and ground-truth geodesic costmaps (paper Table 1). Ground truth comes from
Habitat NavMesh geodesic distances. Each costmap is rank-transformed within
the image, the lowest cost getting rank 0 and the highest rank 1. MARD is the
mean absolute rank difference over the selected pixels. Pixel selection is
either the lowest k percent of cost (cumulative) or the band between two cost
percentiles (slab), for k in 5, 15, 30, 50, and 100.

Run the four stages in order. The extractors write under
`$PLANN3R_ROOT/evaluation/mard`. The navmesh stage must finish first because
the other two read its anchor pixels.

```bash
cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
EPISODES="$PLANN3R_ROOT/plann3r-code/episodes_removing_blacklist.txt"
MARD="$PLANN3R_ROOT/evaluation/mard"

# 1. NavMesh geodesic ground truth and anchor pixels
PYTHONNOUSERSITE=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  pixi run python navmesh_extractor.py \
  --scene-list "$EPISODES" --workers 8 --export-stack --skip-existing \
  --output_dir "$MARD"

# 2. Euclidean distance between VGGT points
PYTHONNOUSERSITE=1 pixi run python euclidean_extractor.py \
  --scene-list "$EPISODES" --navmesh-root "$MARD" --output_dir "$MARD"

# 3. Plann3r predictions
PYTHONNOUSERSITE=1 pixi run python vggtnav_extractor.py \
  --scene-list "$EPISODES" --navmesh-root "$MARD" --output_dir "$MARD"

# 4. Scores
PYTHONNOUSERSITE=1 pixi run python compute_iou_costmaps.py --output_dir "$MARD"
```

`compute_iou_costmaps.py` writes `iou_overall.csv` (cumulative) and
`iou_overall_slabs.csv` (slab), plus per-trajectory and per-pair files. The
`mean_rank_mae` column is MARD on a 0 to 1 scale. Multiply by 10 to compare
with the paper, which reports ranks on a 0 to 10 scale. A slab with no pixels
in a costmap is skipped for that costmap.

ObjectReact costmaps are not produced by these stages. They come precomputed
from ObjectReact's own pipeline, one costmap per frame, and are read from
`--objectreact-root` (default `$PLANN3R_ROOT/evaluation/objectreact_costmaps`)
at the same query frames as the other methods. ObjectReact is skipped if that
directory does not exist.

## Output structure

```text
$RESULTS_ROOT/<mode>/<task>/
  <timestamp>.console.log
  <task-layout>/<run>/
    config.yaml
    metrics_summary.txt
    results_summary.csv
    <episode>_vggt_nav_topological_pixelwise/
      results.csv
      step_data/
```

`no-oracle-paper` runs also write `stopping_predictions.csv` and
`stopping_metrics.json` in the run folder. A nonzero exit from `run_nav.py`
stops the launcher, and so does a missing controller checkpoint. An episode
that fails during setup is recorded as a failure in the run output. After each
run the launcher checks that the saved `config.yaml` has the requested goal
method, map directory, and localizer.

## Visualization

`VISUALIZE=true` adds one PNG per navigation step (`frames/step_0000.png`,
`frames/step_0001.png`, and so on) and one `vggt_nav.mp4` to each episode
folder. Each frame contains:

- Current query RGB image.
- All eight localized submap images.
- Selected submap frame and goal pixel.
- Top-down map with mapping trajectory, executed trajectory, current pose, and goal.
- Plann3r query costmap with its finite range and minimum.
- Controller waypoint prediction in the robot frame.
- Step, distance, velocity, collision, localization, anchor, and stopping status.

The costmap white cross marks the minimum predicted value. The selected submap
image has a yellow border and the selected pixel is drawn on that image.
Rendering adds CPU image encoding and several GB of frames per task, so it is
off by default and no reported result used it.
