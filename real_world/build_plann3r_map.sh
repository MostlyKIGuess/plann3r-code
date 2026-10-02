#!/usr/bin/env bash
# Build the Plann3r propagation costmaps for a recorded real-world traversal.
# Runs on the GPU machine from the repository root in the Pixi environment and
# writes vggt_propagation_costs.npy and vggt_propagation_costs_meta.json into
# MAP_DIR. See docs/real-world.md.
#
# Required environment: PLANN3R_CKPT (Plann3r planner checkpoint) and
#   PLANN3R_ROOT (release bundle root).
# Optional environment: PLANN3R_DEVICE (default cuda),
#   PLANN3R_COSTMAP_ACTIVATION (default gelu, as the released checkpoint),
#   PLANN3R_REPO (default: the parent of this script's folder).
#
#   PLANN3R_CKPT=$PLANN3R_ROOT/models/planner/checkpoint_best.pt \
#   pixi run bash real_world/build_plann3r_map.sh /data/plann3r_real/my_map 215 160 120
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "Usage: $0 MAP_DIR [GOAL_FRAME] [GOAL_PIXEL_X] [GOAL_PIXEL_Y]"
  echo "Example: $0 /data/plann3r_real/my_map 120 160 120"
  exit 2
fi

if [[ -z "${PLANN3R_CKPT:-}" ]]; then
  echo "Set PLANN3R_CKPT to the Plann3r planner checkpoint." >&2
  exit 2
fi
# The mapper config resolves paths from PLANN3R_ROOT, and writing the metadata
# JSON fails without it.
if [[ -z "${PLANN3R_ROOT:-}" ]]; then
  echo "Set PLANN3R_ROOT to the release bundle root." >&2
  exit 2
fi

MAP_DIR="$(realpath "$1")"
GOAL_FRAME="${2:-}"
GOAL_PIXEL_X="${3:-160}"
GOAL_PIXEL_Y="${4:-120}"

if [[ ! -d "${MAP_DIR}/images" ]]; then
  echo "Expected images/ under MAP_DIR: ${MAP_DIR}" >&2
  exit 1
fi

if [[ -z "${GOAL_FRAME}" ]]; then
  if [[ -f "${MAP_DIR}/goal.json" ]]; then
    GOAL_FRAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_idx"])' "${MAP_DIR}/goal.json")"
    GOAL_PIXEL_X="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pixel_x"])' "${MAP_DIR}/goal.json")"
    GOAL_PIXEL_Y="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pixel_y"])' "${MAP_DIR}/goal.json")"
  else
    GOAL_FRAME="$(find "${MAP_DIR}/images" -maxdepth 1 -type f | wc -l)"
    GOAL_FRAME="$((GOAL_FRAME - 1))"
  fi
fi

REPO_ROOT="${PLANN3R_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAP_PARENT="$(dirname "${MAP_DIR}")"
MAP_NAME="$(basename "${MAP_DIR}")"
PLANN3R_DEVICE="${PLANN3R_DEVICE:-cuda}"
PLANN3R_COSTMAP_ACTIVATION="${PLANN3R_COSTMAP_ACTIVATION:-gelu}"

cd "${REPO_ROOT}"

# Same propagation settings as the released simulator maps
# (configs/mapper/mapper_config.yaml): window 9, stride 8, multi_query.
python -m libs.mapper.create_vggt_prop_map \
  scenes.base_dir="${MAP_PARENT}" \
  scenes.multi_scene=false \
  scenes.scene_name="${MAP_NAME}" \
  scenes.base_out_dir=null \
  image.width=320 \
  image.height=240 \
  goal.mode=config \
  goal.image_idx="${GOAL_FRAME}" \
  goal.pixel_x="${GOAL_PIXEL_X}" \
  goal.pixel_y="${GOAL_PIXEL_Y}" \
  prop_map.overwrite=true \
  prop_map.window_size=9 \
  prop_map.stride=8 \
  prop_map.save_filename=vggt_propagation_costs.npy \
  prop_map.metadata_filename=vggt_propagation_costs_meta.json \
  vggtnav.checkpoint_path="${PLANN3R_CKPT}" \
  vggtnav.device="${PLANN3R_DEVICE}" \
  vggtnav.costmap_activation="${PLANN3R_COSTMAP_ACTIVATION}"
