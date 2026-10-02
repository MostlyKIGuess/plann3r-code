# Setup

## Bundle root

All release paths are relative to one directory:

```bash
export PLANN3R_ROOT=/absolute/path/to/plann3r-release
```

The launchers and Hydra configs read `PLANN3R_ROOT` and stop with an error when
it is unset.

Clone the repository as `$PLANN3R_ROOT/plann3r-code`:

```bash
mkdir -p "$PLANN3R_ROOT"
git clone https://github.com/MostlyKIGuess/plann3r-code.git "$PLANN3R_ROOT/plann3r-code"
```

## Downloads

The Plann3r checkpoints are on [huggingface.co/MostlyK/plann3r](https://huggingface.co/MostlyK/plann3r)
under the same relative paths as `$PLANN3R_ROOT/models/`. VGGT and MegaLoc come
from their authors' releases. The ObjectReact
evaluation episodes are on
[huggingface.co/datasets/oravus/objectreact_hm3d_iin](https://huggingface.co/datasets/oravus/objectreact_hm3d_iin).
The other inputs are coming soon.

| Input | Destination | Source | Required for |
|---|---|---|---|
| Paper planner checkpoint | `$PLANN3R_ROOT/models/planner/checkpoint_best.pt` | Plann3r Hugging Face | Paper evaluation |
| Ablation planner checkpoints | `$PLANN3R_ROOT/models/planner/ablations/` | Plann3r Hugging Face | Planner ablations |
| Controllers | `$PLANN3R_ROOT/models/controller/{predicted_costmap,gt_trained}/latest.pth` | Plann3r Hugging Face | Paper evaluation |
| VGGT checkpoint | `$PLANN3R_ROOT/models/vggt/model.pt` | `model.pt` from [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | Planner training, map generation, inferred stopping |
| MegaLoc model and source | `$PLANN3R_ROOT/models/megaloc/` (`model.safetensors`, `source/`) | weights from [gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc), source from [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) | MegaLoc evaluation |
| Shortcut episodes | `$PLANN3R_ROOT/evaluation/datasets/object-rel-nav/maps_via_alt_goal/` | ObjectReact, `maps_via_alt_goal.zip` | Shortcut |
| Alt-goal metadata and semantic masks | `$PLANN3R_ROOT/evaluation/datasets/object-rel-nav/hm3d_iin_val/` | ObjectReact, `hm3d_iin_val.zip` | Alt-goal |
| Standard navigation episodes | `$PLANN3R_ROOT/evaluation/datasets/hm3d_navigation/hm3d_iin_val_320x240/` | coming soon | Imitate, reverse, alt-goal |
| HM3D val scenes | `$PLANN3R_ROOT/evaluation/datasets/hm3d_navigation/hm3d_v0.2/val/` | [HM3D](#hm3d) | Habitat simulation |
| Four task map directories with propagation maps | `$PLANN3R_ROOT/evaluation/maps/` | coming soon | Navigation evaluation |
| Planner training samples | `$PLANN3R_ROOT/training/planner/vggtnav/` and `source_scenes/` | coming soon | Planner training |
| Controller training samples | `$PLANN3R_ROOT/training/controller/predicted_costmap/` | coming soon | Controller training |

## HM3D

HM3D is available through the
[Habitat-Matterport 3D dataset page](https://aihabitat.org/datasets/hm3d/).
The annotation file
`hm3d_annotated_val_basis.scene_dataset_config.json` must be at the root of the
HM3D `val` directory.

Each alt-goal episode directory must contain:

```text
seen_but_unvisited_object.npy
seen_but_unvisited_object_v2.npy
images_sem/
  00000.npy
  ...
```

The semantic mask named by `seen_but_unvisited_object_v2.npy` is required. The
evaluator does not substitute a trajectory-frame pose when that mask is absent.

## Directory layout

The source repository:

```text
plann3r-code/
  baseline/                 Evaluation launcher and metric summaries
  configs/                  Hydra navigation and mapper configuration
  docs/                     Public setup and method documentation
  history/                  Earlier base-VGGT graph mapper, kept for reference and not imported
  libs/
    collision_avoidance/    CARE collision avoidance
    common/                 Goal lookup, simulator, and GPU memory helpers
    control/                GNM runtime and training code
    experiments/            Episode construction, Plann3r inference, and scoring
    localizer/              MegaLoc retrieval
    logger/                 Logging setup and LOG_LEVEL handling
    mapper/                 Plann3r propagation map building and loading
    simulation/             Habitat simulator setup
    visualizations/         Compact step frames and videos
  mard_benchmark/           Planning cost benchmark (MARD) against NavMesh costs
  stopping_condition/       Online inferred stopping
  training/                 Plann3r datasets, losses, heads, and launcher
  vggt/                     VGGT backbone source
  run_nav.py                Navigation entry point
  pixi.toml                 Environment and setup tasks
  pixi.lock                 Resolved package versions for the default environment
```

`libs/matcher` and MASt3R are absent because paper evaluation uses pose
localization or MegaLoc retrieval.

The artifact bundle:

```text
$PLANN3R_ROOT/
  plann3r-code/
  models/
    planner/
      checkpoint_best.pt
      ablations/
        costmap_only.pt
        no_pointmap_loss.pt
        no_grad_loss.pt
        frozen_mlp_goal_token.pt
    controller/
      predicted_costmap/
        latest.pth
    vggt/
      model.pt
    megaloc/
      model.safetensors
      source/
  evaluation/
    datasets/
      hm3d_navigation/
        hm3d_iin_val_320x240/
        hm3d_v0.2/val/
      object-rel-nav/
        maps_via_alt_goal/
        hm3d_iin_val/
    maps/
      hm3d_val_mapping_04ed325_commit_sg_habitat_vggt_costmaps/
      hm3d_val_mapping_original_reverse_vggt_multiview_w1/
      hm3d_val_mapping_alt_goal_v2_correct_vggt_multiview_w1/
  training/
    planner/
      source_scenes/
      vggtnav/
    controller/
      predicted_costmap/
        data/
        source_episodes/
        splits/
  cache/
    megaloc/
  runs/
```

Every task map directory contains one subdirectory per episode, with the
Plann3r propagation costmaps (`.npy` and `.json`)
([`method.md`](method.md#map-files)). The Shortcut costmaps sit in the episode
folders under `object-rel-nav/maps_via_alt_goal/`. Graph files
(`*.pkl.b2s`) from the earlier mapper are not needed.

## Install

```bash
cd "$PLANN3R_ROOT/plann3r-code"
pixi install
PYTHONNOUSERSITE=1 pixi run setup-habitat
```

`setup-habitat` checks out Habitat-Sim 0.2.4 and Habitat-Lab 0.2.4 under
`.dependencies/`. Habitat-Sim is built with Bullet and headless rendering.
Account-level Python packages are disabled during evaluation because mixing a
user-site `torch` with Pixi's `torchvision` produces binary errors.

The evaluator prepends `.pixi/envs/default/lib` to `LD_LIBRARY_PATH`. This is
needed when OpenCV requires a newer `libstdc++.so.6` than the host copy.

## CUDA versions

The default environment is:

```text
PyTorch 2.7.1
Torchvision 0.22.1
CUDA wheel 12.8
```

Every reported result was produced with this environment. Before evaluation,
the launcher creates a CUDA tensor and prints the PyTorch version, CUDA
version, GPU name, and compute capability. Closed-loop metrics also depend on
the GPU ([reproducibility](evaluation.md#reproducibility)).

## Evaluation paths

Setting `PLANN3R_ROOT` is enough when the downloaded bundle follows the
[directory layout](#directory-layout). These environment variables override
individual locations:

| Variable | Default |
|---|---|
| `PAPER_CHECKPOINT` | `$PLANN3R_ROOT/models/planner/checkpoint_best.pt` |
| `DATASETS` | `$PLANN3R_ROOT/evaluation/datasets` |
| `MAPS` | `$PLANN3R_ROOT/evaluation/maps` |
| `RESULTS_ROOT` | `$PLANN3R_ROOT/runs/ablation_gt` |
| `MEGALOC_CACHE_ROOT` | `$PLANN3R_ROOT/cache/megaloc` |
| `STOPPING_VGGT_CHECKPOINT` | `$PLANN3R_ROOT/models/vggt/model.pt` |
| `CONTROLLER_CONFIG_FILE` | Repository controller YAML selected by Hydra |

The controller YAML uses
`$PLANN3R_ROOT/models/controller/predicted_costmap`. The runtime expands this
environment variable and raises an error if it is unset.

## Direct Hydra paths

`baseline/evaluate.sh` supplies all navigation paths. Direct `run_nav.py` calls
must set these Hydra values:

```text
episodes_dir
hm3d_root_path
costmap_base_dir
episode_list_file
results_dirpath
vggtnav.checkpoint_path
controller.config_file
```

Map generation reads `configs/mapper/mapper_config.yaml`. Set
`vggtnav.checkpoint_path`, `scenes.base_dir`, and `scenes.base_out_dir` for a new machine.
Planner training uses `DATA_ROOT`, `BASE_VGGT_MODEL_PATH`, and `LOG_DIR`.
Controller training reads dataset paths from
`libs/control/visualnav_transformer/train/config/predicted_costmap.yaml`.

## Missing files

Evaluation does not repair a missing input. It does not download a model,
choose another checkpoint, search an alternate map directory, generate a
costmap, replace a missing semantic-instance mask with a frame pose,
redirect results to `/tmp`, or retry preprocessing.

Before navigation, the launcher stops with an error when a requested planner
checkpoint, the Habitat imports, the CUDA check, output-directory permissions,
or the MegaLoc files for the `megaloc` and `no-oracle-paper` modes fail.

The launcher then filters each episode list. It keeps an episode only when its
episode directory, propagation costmap file, and its JSON metadata exist, prints
the episode count, and skips a task with no remaining episodes. Episode
initialization checks scene files, costmap files, pose files, and task
annotations.

Map construction and training are explicit operations
([`method.md`](method.md#building-propagation-maps),
[`training.md`](training.md)). Copy their outputs to the paths above before
evaluation.
