# Plann3r

[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-3b6fd4)](https://plann3r.github.io/)
[![Project page](https://img.shields.io/badge/Project-Page-2451a6)](https://plann3r.github.io/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-ffd21e)](https://huggingface.co/MostlyK/plann3r)
[![Docs](https://img.shields.io/badge/Docs-site-1f2733)](https://mostlykiguess.github.io/plann3r-code/)

Code for "Plann3r: Predicting Planning Costs Grounded in 3D". Project page:
https://plann3r.github.io/.

Plann3r predicts a goal-conditioned geodesic costmap for visual navigation. At
each step, the planner receives the current query image and eight map images
selected by topological localization. A GNM controller converts the predicted
16x16 query costmap into a velocity command. The full pipeline is called
VGGT-Nav in the paper and in the code.

This repository contains the planner, the navigation evaluator, the controller
runtime and training code, the topological map generator, MegaLoc retrieval,
inferred stopping, and step-by-step navigation visualization.

## Release status

Data and weights are published separately from the source code.

- Plann3r planner, ablation and controller checkpoints:
  [huggingface.co/MostlyK/plann3r](https://huggingface.co/MostlyK/plann3r).
- VGGT-1B from [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) and
  MegaLoc from [gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc).
- Evaluation episodes from the ObjectReact benchmark:
  [huggingface.co/datasets/oravus/objectreact_hm3d_iin](https://huggingface.co/datasets/oravus/objectreact_hm3d_iin)
  (`evaluation/hm3d_iin_val.zip` and `evaluation/maps_via_alt_goal.zip`).
- Plann3r propagation maps, other map artifacts, and training samples: coming
  soon.

HM3D scene files are covered by the HM3D license and are not redistributed.
Obtain them through the official dataset process ([HM3D](docs/setup.md#hm3d)).

## Directory layout

All paths are relative to one bundle root, `$PLANN3R_ROOT`. Place the repository
and the downloaded artifacts under that root:

```text
$PLANN3R_ROOT/
  plann3r-code/     this repository
  models/           planner, controller, VGGT, and MegaLoc weights
  evaluation/       navigation episodes, HM3D scenes, and task maps
  training/         planner and controller training samples
  runs/             evaluation and training outputs
```

The full tree and every download are in [`docs/setup.md`](docs/setup.md#directory-layout).

## Quick start

Install the environment:

```bash
export PLANN3R_ROOT=/absolute/path/to/plann3r-release
mkdir -p "$PLANN3R_ROOT"
git clone https://github.com/MostlyKIGuess/plann3r-code.git "$PLANN3R_ROOT/plann3r-code"
cd "$PLANN3R_ROOT/plann3r-code"

pixi install
PYTHONNOUSERSITE=1 pixi run setup-habitat
```

Run the four navigation tasks after the model and evaluation archives are in
place:

```bash
cd "$PLANN3R_ROOT/plann3r-code"
GPU=0 \
ABLATIONS=paper \
TASKS="imitate reverse altgoal shortcut" \
bash baseline/evaluate.sh
```

The launcher does not generate maps or download replacement files.
[`docs/setup.md`](docs/setup.md#missing-files) lists what it checks and what
happens when an input is missing.

## Results

The paper's navigation results come from the commands in
[`docs/evaluation.md`](docs/evaluation.md#reported-rows): 300 steps, a 1 m
success radius, HM3D IIN-val episodes from the ObjectReact benchmark, Plann3r
propagation map costmaps, and the alt-goal protocol
([`docs/method.md`](docs/method.md#tasks)).

## Software environment

Every reported result was produced with the default Pixi environment
(`pixi.lock`, PyTorch 2.7.1, CUDA 12.8). Closed-loop results change with the
GPU even with identical inputs, so the paper states the GPU it used
([reproducibility](docs/evaluation.md#reproducibility),
[CUDA versions](docs/setup.md#cuda-versions)).

## Documentation

The docs are also published as a site at
https://mostlykiguess.github.io/plann3r-code/.

- [`docs/setup.md`](docs/setup.md): downloads, directory layout, installation, paths, and missing-file behavior.
- [`docs/evaluation.md`](docs/evaluation.md): evaluation commands, launcher options, planner ablations, MARD, and visualization.
- [`docs/method.md`](docs/method.md): the navigation protocol as implemented, checkpoints, and map artifacts.
- [`docs/training.md`](docs/training.md): planner and controller training.
- [`docs/real-world.md`](docs/real-world.md): running Plann3r on a real robot with the code in `real_world/`.

## Entry points

- `baseline/evaluate.sh` runs the reported evaluation modes.
- `run_nav.py` runs navigation from a Hydra configuration.
- `training/run_nav_single_gpu.sh` trains the Plann3r costmap model.
- `libs/control/visualnav_transformer/train/train.py` trains the GNM controller.
- `libs/mapper/create_vggt_prop_map.py` builds the Plann3r propagation map costmaps, and `baseline/build_prop_maps.sh` runs it per task.

## License

The code is released under the [MIT License](LICENSE). The VGGT backbone in
`vggt/` and the VGGT-derived training files are under the
[VGGT License](vggt/LICENSE.txt), and the GNM controller code in
`libs/control/visualnav_transformer/` keeps its own MIT License. The planner
checkpoints are fine-tuned from VGGT-1B, so the VGGT-1B license applies to them.
