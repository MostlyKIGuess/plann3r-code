#!/usr/bin/env bash
# VGGT-Nav paper evaluation plus opt-in training ablations and inferred stopping.
#
# Standard rows hold localization at oracle (GT), so their only variable is the
# planner checkpoint. Two modes use the paper checkpoint with MegaLoc retrieval:
# megaloc keeps the oracle 1 m stop, and no-oracle-paper adds inferred stopping.
set -euo pipefail

# Prevent account-level Python packages from mixing with the pinned Pixi
# environment. A user-site torch paired with Pixi's torchvision lacks the
# compiled torchvision operators required during import.
export PYTHONNOUSERSITE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

PIXI_LIBRARY_DIR="$REPO/.pixi/envs/default/lib"
if [ -d "$PIXI_LIBRARY_DIR" ]; then
  export LD_LIBRARY_PATH="$PIXI_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

GPU="${GPU:-0}"
if [ -z "${PLANN3R_ROOT:-}" ]; then
  echo "ERROR: set PLANN3R_ROOT to the release bundle root" >&2
  exit 2
fi
export PLANN3R_ROOT
TASKS="${TASKS:-imitate reverse altgoal shortcut}"
MAX_STEPS="${MAX_STEPS:-300}"
# false: metrics and per-step CSVs only, as for every reported result.
VISUALIZE="${VISUALIZE:-false}"
RESULTS_ROOT="${RESULTS_ROOT:-$PLANN3R_ROOT/runs/ablation_gt}"
FILTER_DIR="${FILTER_DIR:-$RESULTS_ROOT/episode_lists}"
CONTROLLER="${CONTROLLER:-vggt_nav_baseline}"
CONTROLLER_CONFIG_FILE="${CONTROLLER_CONFIG_FILE:-}"
SUMMARIZE="${SUMMARIZE:-true}"
MEGALOC_CACHE_ROOT="${MEGALOC_CACHE_ROOT:-$PLANN3R_ROOT/cache/megaloc}"
NO_ORACLE_GOAL_THRESHOLD="${NO_ORACLE_GOAL_THRESHOLD:-0.2}"
STOPPING_SUCCESS_DISTANCE="${STOPPING_SUCCESS_DISTANCE:-1.0}"
STOPPING_N="${STOPPING_N:-100}"
STOPPING_N_DEPTH="${STOPPING_N_DEPTH:-100}"
STOPPING_EPS="${STOPPING_EPS:-0.2}"
STOPPING_DEPTH_M="${STOPPING_DEPTH_M:-1.0}"
STOPPING_MAP_FRAME_SLACK="${STOPPING_MAP_FRAME_SLACK:-1}"
STOPPING_VGGT_CHECKPOINT="${STOPPING_VGGT_CHECKPOINT:-$PLANN3R_ROOT/models/vggt/model.pt}"
DRY_RUN="${DRY_RUN:-false}"
# Ground-truth localization rule for oracle rows. `legacy` ranks map frames by
# position and produced the reported results. `odometry` adds rotation angle.
ORACLE_MODE="${ORACLE_MODE:-legacy}"
# Keep the MegaLoc submap on the goal frame once it is the top-1 match. See
# configs/localizer/megaloc.yaml.
MEGALOC_GOAL_LOCK="${MEGALOC_GOAL_LOCK:-false}"
if [ "$MEGALOC_GOAL_LOCK" != "true" ] && [ "$MEGALOC_GOAL_LOCK" != "false" ]; then
  echo "ERROR: MEGALOC_GOAL_LOCK must be true or false, got: $MEGALOC_GOAL_LOCK" >&2
  exit 2
fi
if [ "$ORACLE_MODE" != "legacy" ] && [ "$ORACLE_MODE" != "odometry" ]; then
  echo "ERROR: ORACLE_MODE must be legacy or odometry, got: $ORACLE_MODE" >&2
  exit 2
fi

# Mode name -> planner checkpoint. `paper` is the full planner and is the
# reference row for the ablations.
PAPER_CHECKPOINT="${PAPER_CHECKPOINT:-$PLANN3R_ROOT/models/planner/checkpoint_best.pt}"

declare -A ABLATION_CKPT=(
  [paper]="$PAPER_CHECKPOINT"
  [megaloc]="$PAPER_CHECKPOINT"
  [no-oracle-paper]="$PAPER_CHECKPOINT"
  [costmap_only]="$PLANN3R_ROOT/models/planner/ablations/costmap_only.pt"
  [no_pointmap_loss]="$PLANN3R_ROOT/models/planner/ablations/no_pointmap_loss.pt"
  [no_grad_loss]="$PLANN3R_ROOT/models/planner/ablations/no_grad_loss.pt"
  [frozen_mlp_goal_token]="$PLANN3R_ROOT/models/planner/ablations/frozen_mlp_goal_token.pt"
)

ABLATIONS="${ABLATIONS:-paper}"

if [[ " $ABLATIONS " == *" no-oracle-paper "* ]] || [[ " $ABLATIONS " == *" megaloc "* ]]; then
  for artifact in \
    "$PLANN3R_ROOT/models/megaloc/model.safetensors" \
    "$PLANN3R_ROOT/models/megaloc/source/megaloc_model.py"; do
    if [ ! -f "$artifact" ]; then
      echo "Missing no-oracle artifact: $artifact" >&2
      exit 2
    fi
  done
fi

if ! mkdir -p "$RESULTS_ROOT"; then
  echo "ERROR: cannot create RESULTS_ROOT=$RESULTS_ROOT" >&2
  exit 2
fi
if [ ! -w "$RESULTS_ROOT" ]; then
  echo "ERROR: RESULTS_ROOT is not writable: $RESULTS_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULTS_ROOT" "$FILTER_DIR"

DATASETS="${DATASETS:-$PLANN3R_ROOT/evaluation/datasets}"
MAPS="${MAPS:-$PLANN3R_ROOT/evaluation/maps}"
STANDARD_EPISODES="$DATASETS/hm3d_navigation/hm3d_iin_val_320x240"
STANDARD_LIST="${STANDARD_LIST:-$REPO/episodes_removing_blacklist.txt}"
# Optional tag for propagation costmaps built by another planner checkpoint,
# stored beside the released files as <name>_<tag>.npy and <name>_<tag>.json.
PROP_MAP_TAG="${PROP_MAP_TAG:-}"
require_dir() {
  local requested="$1"
  local context="$2"
  if [ ! -d "$requested" ]; then
    echo "ERROR: $context map directory is missing: $requested" >&2
    exit 2
  fi
  echo "$requested"
}

assert_run_config() {
  local output_dir="$1"
  local expected_goal="$2"
  local expected_map="$3"
  local expected_localizer_name="$4"
  local expected_retrieval="$5"
  local expected_costmap_file="$6"
  local expected_costmap_meta_file="$7"

  local run_config
  run_config="$(find "$output_dir" -type f -name config.yaml -printf '%T@|%p\n' | sort -t'|' -k1,1nr | head -n 1 | cut -d'|' -f2-)"
  if [ -z "$run_config" ] || [ ! -f "$run_config" ]; then
    echo "ERROR: Could not locate config.yaml in $output_dir" >&2
    return 1
  fi

  python3 - "$run_config" "$expected_goal" "$expected_map" "$expected_localizer_name" "$expected_retrieval" "$expected_costmap_file" "$expected_costmap_meta_file" <<'PY'
import os
import sys
from pathlib import Path
import yaml

(
    cfg_path,
    expected_goal,
    expected_map,
    expected_localizer_name,
    expected_retrieval,
    expected_costmap_file,
    expected_costmap_meta_file,
) = sys.argv[1:8]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

errors = []
goal_position = cfg.get("goal_position_method")
if goal_position != expected_goal:
    errors.append(f"goal_position_method mismatch: expected={expected_goal} got={goal_position}")

# The run must have read the Plann3r propagation costmaps and their sidecar.
for key, expected in (
    ("costmap_filename", expected_costmap_file),
    ("costmap_metadata_filename", expected_costmap_meta_file),
):
    if str(cfg.get(key)) != expected:
        errors.append(f"{key} mismatch: expected={expected} got={cfg.get(key)}")

configured_map = str(cfg.get("costmap_base_dir", ""))
expected_map = str(expected_map)
if configured_map != expected_map:
    configured_norm = os.path.normpath(configured_map)
    expected_norm = os.path.normpath(expected_map)
    # Allow equivalent spellings/relative expansions while still rejecting real
    # mismatches.
    if configured_norm != expected_norm:
        errors.append(
            f"costmap_base_dir mismatch: expected={expected_map} got={configured_map}"
        )

if configured_map != expected_map:
    try:
        configured_resolved = str(Path(configured_map).resolve())
        expected_resolved = str(Path(expected_map).resolve())
    except Exception:
        configured_resolved = configured_norm
        expected_resolved = expected_norm
    if configured_resolved != expected_resolved:
        if all([configured_resolved != expected_resolved, configured_norm != expected_norm]):
            errors.append(
                f"costmap_base_dir resolved mismatch: expected={expected_map} got={configured_map}"
            )

localizer = cfg.get("localizer", {})
# configs/localizer/megaloc.yaml extends the topological localizer config, so
# its resolved Hydra config intentionally has name=topological, retrieval=megaloc.
if expected_localizer_name == "megaloc":
    expected_retrieval = "megaloc"
resolved_expected_name = (
    "topological" if expected_retrieval == "megaloc" else expected_localizer_name
)
if localizer.get("name") != resolved_expected_name:
    errors.append(
        f"localizer.name mismatch: expected={resolved_expected_name} "
        f"got={localizer.get('name')}"
    )
if expected_retrieval:
    if localizer.get("retrieval") != expected_retrieval:
      errors.append(
          f"localizer.retrieval mismatch: expected={expected_retrieval} got={localizer.get('retrieval')}"
      )

if errors:
    print("ABLATION_RUN_CONFIG_MISMATCH")
    for line in errors:
        print(line)
    raise SystemExit(1)
print("ABLATION_RUN_CONFIG_OK")
PY
}

for ablation in $ABLATIONS; do
  checkpoint="${ABLATION_CKPT[$ablation]:-}"
  if [ -z "$checkpoint" ]; then
    echo "Unknown ablation: $ablation (known: ${!ABLATION_CKPT[*]})" >&2
    exit 2
  fi
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for $ablation: $checkpoint" >&2
    exit 2
  fi
done

if ! habitat_check="$(pixi run python -c 'import habitat_sim; import habitat' 2>&1)"; then
  echo "Importing Habitat-Sim and Habitat-Lab failed:" >&2
  echo "$habitat_check" | tail -5 >&2
  echo "If they are not installed, run: pixi run setup-habitat" >&2
  exit 2
fi

if [ "$DRY_RUN" != "true" ]; then
  if ! cuda_preflight="$(CUDA_VISIBLE_DEVICES="$GPU" pixi run python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; value = torch.ones(1, device="cuda") + 1; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)} value={value.item():.0f}")' 2>&1)"; then
    echo "CUDA preflight failed before episode processing:" >&2
    echo "$cuda_preflight" >&2
    exit 2
  fi
  printf '  %-18s %s\n' "CUDA preflight" "$cuda_preflight"
fi

# run_nav.py reports a missing controller per episode and keeps going, which
# turns every episode into a failure. Stop here instead.
# YAML values are read with sed, so drop trailing comments and spaces.
yaml_value() {
  sed -n "s/^$1:[[:space:]]*//p" "$2" | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//'
}
controller_config="${CONTROLLER_CONFIG_FILE:-$(yaml_value config_file "$REPO/configs/controller/$CONTROLLER.yaml")}"
controller_run="$(yaml_value load_run "$REPO/$controller_config")"
controller_run="${controller_run//\$PLANN3R_ROOT/$PLANN3R_ROOT}"
if [ ! -f "$controller_run/latest.pth" ]; then
  echo "ERROR: controller checkpoint missing: $controller_run/latest.pth (from $controller_config)" >&2
  exit 2
fi

mkdir -p "$RESULTS_ROOT" "$FILTER_DIR"

task_config() {
  # Alt goal always scores against the annotated semantic object
  # (task_setup.py _set_alt_goal). The other tasks use the agent position at
  # the goal frame.
  GOAL_POSITION_METHOD="trajectory"
  case "$1" in
    imitate)
      TASK_TYPE="original"; REVERSE="false"; EPISODES_DIR="$STANDARD_EPISODES"
      COSTMAP_DIR="$MAPS/hm3d_val_mapping_04ed325_commit_sg_habitat_vggt_costmaps"
      EPISODE_LIST="$STANDARD_LIST"
      PROP_COSTMAP_FILE="vggt_propagation_costs.npy"
      PROP_COSTMAP_META_FILE="vggt_propagation_costs_meta.json" ;;
    reverse)
      TASK_TYPE="original"; REVERSE="true"; EPISODES_DIR="$STANDARD_EPISODES"
      COSTMAP_DIR="$MAPS/hm3d_val_mapping_original_reverse_vggt_multiview_w1"
      EPISODE_LIST="$REPO/episode_lists/reverse_with_reverse_goal_31.txt"
      PROP_COSTMAP_FILE="vggt_propagation_costs_reverse.npy"
      PROP_COSTMAP_META_FILE="vggt_propagation_costs_reverse.json" ;;
    altgoal)
      TASK_TYPE="alt_goal_v2"; REVERSE="false"; EPISODES_DIR="$STANDARD_EPISODES"
      GOAL_POSITION_METHOD="semantic_instance"
      COSTMAP_DIR="$MAPS/hm3d_val_mapping_alt_goal_v2_correct_vggt_multiview_w1"
      EPISODE_LIST="$STANDARD_LIST"
      PROP_COSTMAP_FILE="vggt_propagation_costs_alt_goal.npy"
      PROP_COSTMAP_META_FILE="vggt_propagation_costs_alt_goal.json" ;;
    shortcut)
      TASK_TYPE="via_alt_goal"; REVERSE="false"
      EPISODES_DIR="$DATASETS/object-rel-nav/maps_via_alt_goal"
      # The shortcut propagation maps live alongside the episodes, not under $MAPS.
      COSTMAP_DIR="$DATASETS/object-rel-nav/maps_via_alt_goal"
      EPISODE_LIST="${SHORTCUT_LIST:-$REPO/episode_lists/shortcut_via_alt_goal_mapped_31.txt}"
      PROP_COSTMAP_FILE="vggt_propagation_costs_via_alt_goal.npy"
      PROP_COSTMAP_META_FILE="vggt_propagation_costs_meta_via_alt_goal.json" ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac

  # Plann3r propagation map costmaps written by
  # libs/mapper/create_vggt_prop_map.py, or the tagged files of another
  # planner checkpoint (baseline/build_prop_maps.sh).
  COSTMAP_FILE="$PROP_COSTMAP_FILE"
  COSTMAP_META_FILE="$PROP_COSTMAP_META_FILE"
  if [ -n "$PROP_MAP_TAG" ]; then
    COSTMAP_FILE="${PROP_COSTMAP_FILE%.npy}_${PROP_MAP_TAG}.npy"
    COSTMAP_META_FILE="${PROP_COSTMAP_META_FILE%.json}_${PROP_MAP_TAG}.json"
  fi
}

# An episode is ready when its episode folder, its propagation costmap file
# and that file's JSON sidecar exist.
ready_list() {
  local source_list="$1" episodes_dir="$2"
  local costmap_dir="$3" output_list="$4"
  : > "$output_list"
  while IFS= read -r episode || [ -n "$episode" ]; do
    episode="${episode%$'\r'}"
    [ -z "$episode" ] && continue
    [ ! -d "$episodes_dir/$episode" ] && continue
    if [ -f "$costmap_dir/$episode/$COSTMAP_FILE" ] && \
       [ -f "$costmap_dir/$episode/$COSTMAP_META_FILE" ]; then
      echo "$episode" >> "$output_list"
    fi
  done < "$source_list"
}

pretty_stream() {
  awk '
    function emit(line) { print line; fflush() }
    /Episode [0-9]+\/[0-9]+:/ {
      line=$0; sub(/^.*Episode /, "Episode ", line); sub(/\r.*$/, "", line)
      emit(""); emit(line); next
    }
    /\[\[RunNav\]\]\[INFO\] - Episode .*: / {
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " line); next
    }
    /VGGTNav costmaps:/ {
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " line); next
    }
    /Results will be saved to:/ {
      line=$0; sub(/^.*Results will be saved to: /, "", line); sub(/\r.*$/, "", line)
      emit("  Run results       " line); next
    }
    /Final Results Summary/ { emit(""); emit("RESULTS"); next }
    /(Total Episodes|Successful|Failed|Exceeded Steps|Stuck|Success Rate|SPL|SSPL):/ {
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " line); next
    }
    /(Results summary|Metrics summary) saved to:/ {
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " line); next
    }
    /Saved VGGT-Nav compact video:/ {
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " line); next
    }
    /\]\[(WARNING|ERROR)\] - / {
      level=($0 ~ /\]\[ERROR\] - /) ? "ERROR" : "WARNING"
      line=$0; sub(/^.*\] - /, "", line); sub(/\r.*$/, "", line)
      emit("  " level ": " line); next
    }
    /^WARN:/ { emit("  " $0); next }
  '
}

for task in $TASKS; do
  task_config "$task" || continue

  for ablation in $ABLATIONS; do
    checkpoint="${ABLATION_CKPT[$ablation]}"

    COSTMAP_DIR="$(require_dir "$COSTMAP_DIR" "$ablation $task")"

    CONTROLLER_ARGS=("controller=$CONTROLLER")
    if [ -n "$CONTROLLER_CONFIG_FILE" ]; then
      CONTROLLER_ARGS+=("controller.config_file=$CONTROLLER_CONFIG_FILE")
      printf '  %-18s %s\n' "Controller" "override:$CONTROLLER -> $CONTROLLER_CONFIG_FILE"
    else
      printf '  %-18s %s\n' "Controller" "$CONTROLLER"
    fi

    if [ "$ablation" = "no-oracle-paper" ]; then
      LOCALIZER_LABEL="MegaLoc + inferred stopping"
      RUN_EXP_NAME="${ablation}_${task}"
      GOAL_THRESHOLD="$NO_ORACLE_GOAL_THRESHOLD"
      COMPACT_VIS="$VISUALIZE"
      SAVE_RAW="true"
      SAVE_RAW_COSTMAP="true"
      LOCALIZER_ARGS=(
        "localizer=megaloc"
        "localizer.megaloc_cache_root=$MEGALOC_CACHE_ROOT"
      )
      STOPPING_ARGS=(
        "++online_stopping.enabled=true"
        "++online_stopping.eps=$STOPPING_EPS"
        "++online_stopping.N=$STOPPING_N"
        "++online_stopping.N_depth=$STOPPING_N_DEPTH"
        "++online_stopping.depth_m=$STOPPING_DEPTH_M"
        "++online_stopping.map_frame_slack=$STOPPING_MAP_FRAME_SLACK"
        "++online_stopping.success_distance=$STOPPING_SUCCESS_DISTANCE"
        "++online_stopping.vggt_checkpoint=$STOPPING_VGGT_CHECKPOINT"
      )
    elif [ "$ablation" = "megaloc" ]; then
      LOCALIZER_LABEL="MegaLoc + oracle stopping, goal lock $MEGALOC_GOAL_LOCK"
      RUN_EXP_NAME="${ablation}_${task}"
      GOAL_THRESHOLD="1.0"
      COMPACT_VIS="$VISUALIZE"
      SAVE_RAW="false"
      SAVE_RAW_COSTMAP="false"
      LOCALIZER_ARGS=(
        "localizer=megaloc"
        "localizer.megaloc_cache_root=$MEGALOC_CACHE_ROOT"
        "localizer.megaloc_goal_lock=$MEGALOC_GOAL_LOCK"
      )
      STOPPING_ARGS=()
    else
      LOCALIZER_LABEL="oracle (GT, $ORACLE_MODE)"
      RUN_EXP_NAME="${ablation}_oracle_${task}"
      GOAL_THRESHOLD="1.0"
      COMPACT_VIS="$VISUALIZE"
      SAVE_RAW="false"
      SAVE_RAW_COSTMAP="false"
      LOCALIZER_ARGS=(
        "localizer=topological"
        "localizer.retrieval=oracle"
        "localizer.use_gt_localization=true"
        "localizer.oracle_mode=$ORACLE_MODE"
      )
      STOPPING_ARGS=()
    fi

    # Pass the task's goal method explicitly so the saved config records it.
    STOPPING_ARGS+=( "goal_position_method=$GOAL_POSITION_METHOD" )

    # Named per ablation so concurrent invocations sharing RESULTS_ROOT never
    # truncate a list another run is reading.
    filtered_list="$FILTER_DIR/${task}_${ablation}_ready.txt"
    ready_list "$EPISODE_LIST" "$EPISODES_DIR" "$COSTMAP_DIR" "$filtered_list"
    if [ ! -s "$filtered_list" ]; then
      echo "Skipping ablation=$ablation task=$task: no compatible episodes" >&2
      continue
    fi

    output_dir="$RESULTS_ROOT/$ablation/$task"
    mkdir -p "$output_dir"
    run_stamp="$(date -u +%Y%m%d-%H%M%S)"
    console_log="$output_dir/${run_stamp}.console.log"
    episode_count="$(wc -l < "$filtered_list" | tr -d ' ')"

    printf '\n%s\n' "========================================================================"
    printf 'ABLATION EVALUATION\n'
    printf '%s\n' "------------------------------------------------------------------------"
    printf '  %-18s %s\n' "Ablation" "$ablation"
    printf '  %-18s %s\n' "Checkpoint" "$checkpoint"
    printf '  %-18s %s\n' "Localizer" "$LOCALIZER_LABEL"
    printf '  %-18s %s\n' "Task" "$task"
    printf '  %-18s %s\n' "Goal method" "$GOAL_POSITION_METHOD"
    printf '  %-18s %s\n' "Episodes" "$episode_count"
    printf '  %-18s %s\n' "Max steps" "$MAX_STEPS"
    printf '  %-18s %s\n' "Visualizations" "$VISUALIZE"
    printf '  %-18s %s\n' "Episode data" "$EPISODES_DIR"
    printf '  %-18s %s\n' "Episode list" "$filtered_list"
    printf '  %-18s %s\n' "Map artifacts" "$COSTMAP_DIR"
    printf '  %-18s %s\n' "Output parent" "$output_dir"
    printf '  %-18s %s\n' "Full console log" "$console_log"
    printf '%s\n' "------------------------------------------------------------------------"

    NAV_CMD=(pixi run python run_nav.py \
      "experiment=vggt_nav_baseline" \
      "goal_source=topological_pixelwise" \
      "task_type=$TASK_TYPE" \
      "reverse=$REVERSE" \
      "episodes_dir=$EPISODES_DIR" \
      "hm3d_root_path=$DATASETS/hm3d_navigation/hm3d_v0.2/val" \
      "costmap_base_dir=$COSTMAP_DIR" \
      "costmap_filename=$COSTMAP_FILE" \
      "costmap_metadata_filename=$COSTMAP_META_FILE" \
      "episode_list_file=$filtered_list" \
      "multi_episode=true" \
      "max_steps=$MAX_STEPS" \
      "goal_distance_threshold=$GOAL_THRESHOLD" \
      "results_dirpath=$output_dir" \
      "exp_name=$RUN_EXP_NAME" \
      "pts3d_source=none" \
      "vggtnav.checkpoint_path=$checkpoint" \
      "visualization.vggt_nav_compact.enabled=$COMPACT_VIS" \
      "visualization.save_raw_data.enabled=$SAVE_RAW" \
      "visualization.save_raw_data.unnormalized_costmaps=$SAVE_RAW_COSTMAP" \
      "visualization.render_visualizations.enabled=false" \
      "${CONTROLLER_ARGS[@]}" \
      "${LOCALIZER_ARGS[@]}" \
      "${STOPPING_ARGS[@]}")

    if [ "$DRY_RUN" = "true" ]; then
      printf '  %-18s %s\n' "Status" "DRY_RUN=true (skipping execution)"
      printf '  %-18s %s\n' "CUDA_VISIBLE_DEVICES" "$GPU"
      printf '  %-18s %s\n' "Command" ""
      printf '  %s\n' "CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=\"${PYTHONWARNINGS:-ignore::FutureWarning}\" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${NAV_CMD[*]}"
      printf '\n'
      continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONNOUSERSITE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${NAV_CMD[@]}" \
      2>&1 | tee "$console_log" | pretty_stream
    run_status=${PIPESTATUS[0]}
    if [ "$run_status" -ne 0 ]; then
      printf '  ERROR: run exited with status %s; see %s\n' "$run_status" "$console_log" >&2
      exit "$run_status"
    fi

    expected_retrieval=""
    if printf '%s\n' "${LOCALIZER_ARGS[@]}" | grep -q '^localizer\.retrieval='; then
      expected_retrieval="$(printf '%s\n' "${LOCALIZER_ARGS[@]}" | awk -F= '/^localizer\.retrieval=/{print $2}')"
    fi
    if ! assert_run_config "$output_dir" "$GOAL_POSITION_METHOD" "$COSTMAP_DIR" "${LOCALIZER_ARGS[0]#*=}" "$expected_retrieval" "$COSTMAP_FILE" "$COSTMAP_META_FILE"; then
      printf '  ERROR: run completed but saved config mismatched requested settings. See %s\n' "$output_dir" >&2
      exit 1
    fi

    if [ "$ablation" = "no-oracle-paper" ]; then
      run_config="$(find "$output_dir" -type f -name config.yaml -printf '%T@|%p\n' | sort -t'|' -k1,1nr | head -n 1 | cut -d'|' -f2-)"
      run_dir="${run_config%/config.yaml}"
      stopping_log="$output_dir/${run_stamp}.stopping.log"
      printf '  %-18s %s\n' "Stopping input" "$run_dir"
      printf '  %-18s %s\n' "Stopping log" "$stopping_log"
      pixi run python stopping_condition/summarize_online_stopping.py \
        "$run_dir" \
        "--success-distance=$STOPPING_SUCCESS_DISTANCE" \
        2>&1 | tee "$stopping_log"
      stopping_status=${PIPESTATUS[0]}
      if [ "$stopping_status" -ne 0 ]; then
        printf '  ERROR: stopping evaluation exited with status %s; see %s\n' "$stopping_status" "$stopping_log" >&2
        exit "$stopping_status"
      fi
    fi
  done
done

# Concurrent invocations sharing RESULTS_ROOT would each overwrite the summary
# with a partial view, so set SUMMARIZE=false and run summarize_ablation.py once
# after the last job exits.
if [ "$SUMMARIZE" = "true" ]; then
  pixi run python "$REPO/baseline/summarize_ablation.py" --root "$RESULTS_ROOT"
fi
