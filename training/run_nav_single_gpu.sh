#!/usr/bin/env bash
set -euo pipefail

# Single-GPU launcher for VGGTNav training.
# Usage:
#   ./run_nav_single_gpu.sh 0
#   GPU_INDEX=1 ./run_nav_single_gpu.sh
#   ./run_nav_single_gpu.sh 0 logging.log_freq=10 max_epochs=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGGT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_INDEX="${GPU_INDEX:-0}"
if [[ "${#}" -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
  GPU_INDEX="${1}"
  shift
fi

PIXI_BIN="${PIXI_BIN:-}"
if [[ -z "${PIXI_BIN}" ]]; then
  PIXI_BIN="$(command -v pixi || true)"
fi
if [[ -z "${PIXI_BIN}" ]]; then
  echo "Error: pixi was not found in PATH. Set PIXI_BIN explicitly."
  exit 1
fi

FASTVGGT_MANIFEST="${FASTVGGT_MANIFEST:-${VGGT_ROOT}}"
if [[ ! -f "${FASTVGGT_MANIFEST}/pixi.toml" ]]; then
  echo "Error: pixi.toml not found under ${FASTVGGT_MANIFEST}."
  exit 1
fi

if [[ -z "${PLANN3R_ROOT:-}" ]]; then
  echo "Error: set PLANN3R_ROOT to the release bundle root."
  exit 1
fi
DATA_ROOT="${DATA_ROOT:-${PLANN3R_ROOT}/training/planner/vggtnav}"
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Error: DATA_ROOT directory does not exist: ${DATA_ROOT}"
  exit 1
fi

BASE_VGGT_MODEL_PATH="${BASE_VGGT_MODEL_PATH:-${PLANN3R_ROOT}/models/vggt/model.pt}"
if [[ ! -f "${BASE_VGGT_MODEL_PATH}" ]]; then
  echo "Error: BASE_VGGT_MODEL_PATH does not exist: ${BASE_VGGT_MODEL_PATH}"
  exit 1
fi

RUN_NAME="${RUN_NAME:-nav_all_scenes_1gpu}"
CACHE_ROOT="${CACHE_ROOT:-${PLANN3R_ROOT}/cache/planner-training}"
TMP_DIR="${TMP_DIR:-${CACHE_ROOT}/tmp}"
XDG_CACHE_DIR="${XDG_CACHE_DIR:-${CACHE_ROOT}/xdg-cache}"
PIXI_CACHE_DIR_VALUE="${PIXI_CACHE_DIR_VALUE:-${CACHE_ROOT}/pixi-cache}"
HF_HOME_DIR="${HF_HOME_DIR:-${CACHE_ROOT}/hf-home}"
TORCH_HOME_DIR="${TORCH_HOME_DIR:-${CACHE_ROOT}/torch-home}"
LOG_DIR="${LOG_DIR:-${PLANN3R_ROOT}/runs/planner/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-${LOG_DIR}/ckpts}"

NPROC_PER_NODE=1
MASTER_PORT="${MASTER_PORT:-29511}"

# Memory-safe defaults validated on this setup.
IMG_SIZE="${IMG_SIZE:-224}"
MAX_IMG_PER_GPU="${MAX_IMG_PER_GPU:-54}"
TRAIN_SUBMAP_IMAGES="${TRAIN_SUBMAP_IMAGES:-8}"
VAL_SUBMAP_IMAGES="${VAL_SUBMAP_IMAGES:-8}"
TRAIN_TOTAL_IMAGES=$((TRAIN_SUBMAP_IMAGES + 1))
TRAIN_IMG_NUMS="${TRAIN_IMG_NUMS:-[${TRAIN_TOTAL_IMAGES},${TRAIN_TOTAL_IMAGES}]}"
VAL_IMG_NUMS="${VAL_IMG_NUMS:-[2,6]}"
LOG_FREQ="${LOG_FREQ:-1}"

mkdir -p "${TMP_DIR}" "${XDG_CACHE_DIR}" "${PIXI_CACHE_DIR_VALUE}" "${HF_HOME_DIR}" "${TORCH_HOME_DIR}" "${CKPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TMPDIR="${TMP_DIR}"
export PYTHONPATH="${VGGT_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_DIR}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR_VALUE}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"
export TORCH_HOME="${TORCH_HOME_DIR}"
# Keep ~/.local site-packages out of the way; a user-site torch shadows the env one.
export PYTHONNOUSERSITE=1
# Fork-based dataloader workers deadlock if the intra-op threadpool is live at fork time.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CMD=(
  "${PIXI_BIN}" run --manifest-path "${FASTVGGT_MANIFEST}"
  torchrun --standalone --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}"
  launch.py --config nav_costmap
  "data.train.dataset.dataset_configs.0.VGGTNAV_DIR=${DATA_ROOT}"
  "data.val.dataset.dataset_configs.0.VGGTNAV_DIR=${DATA_ROOT}"
  "checkpoint.resume_checkpoint_path=${BASE_VGGT_MODEL_PATH}"
  "exp_name=${RUN_NAME}"
  "logging.log_dir=${LOG_DIR}"
  "checkpoint.save_dir=${CKPT_DIR}"
  "img_size=${IMG_SIZE}"
  "max_img_per_gpu=${MAX_IMG_PER_GPU}"
  "train_submap_images=${TRAIN_SUBMAP_IMAGES}"
  "val_submap_images=${VAL_SUBMAP_IMAGES}"
  "data.train.common_config.img_nums=${TRAIN_IMG_NUMS}"
  "data.val.common_config.img_nums=${VAL_IMG_NUMS}"
  "logging.log_freq=${LOG_FREQ}"
)

# Allow passing extra Hydra overrides as script args.
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Launching VGGTNav single-GPU training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with command:"
printf ' %q' "${CMD[@]}"
echo

cd "${SCRIPT_DIR}"
"${CMD[@]}"
