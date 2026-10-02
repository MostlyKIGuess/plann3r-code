#!/usr/bin/env bash
# Build Plann3r propagation map costmaps for one task with a given planner
# checkpoint, using the mapper settings stored in the released maps' metadata
# (multi_query, window 9, stride 8). Each ablation row needs its own planner's
# maps, because Plann3r builds the map costmaps as well as the query costmaps.
#
# Output goes beside the released files as <name>_<tag>.npy and <name>_<tag>.json,
# which evaluate.sh reads with PROP_MAP_TAG=<tag>. With checkpoint_best.pt and an
# empty tag this rebuilds the released files, so set OUT_ROOT to write elsewhere.
#
# Usage: build_prop_maps.sh <tag> <checkpoint> <task> [episode]
#   task: imitate | reverse | shortcut | altgoal
set -euo pipefail
TAG="$1"
CKPT="$2"
TASK="$3"
ONLY_EPISODE="${4:-}"
: "${PLANN3R_ROOT:?set PLANN3R_ROOT to the release bundle root}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-}"
EVAL="$PLANN3R_ROOT/evaluation"
IIN="$EVAL/datasets/hm3d_navigation/hm3d_iin_val_320x240"
cd "$REPO"
export PYTHONNOUSERSITE=1

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 2
fi

SUFFIX=""
[ -n "$TAG" ] && SUFFIX="_$TAG"

run_mapper() {
  pixi run --frozen python -m libs.mapper.create_vggt_prop_map \
    "vggtnav.checkpoint_path=$CKPT" \
    prop_map.multi_query=true \
    prop_map.batched_exact=false \
    prop_map.overwrite=true \
    "$@"
}

case "$TASK" in
  imitate)
    MAPS="$EVAL/maps/hm3d_val_mapping_04ed325_commit_sg_habitat_vggt_costmaps"
    LIST="$REPO/episodes_removing_blacklist.txt"
    BASE="$IIN"; TASK_TYPE=original
    FILE=vggt_propagation_costs; META=vggt_propagation_costs_meta ;;
  reverse)
    MAPS="$EVAL/maps/hm3d_val_mapping_original_reverse_vggt_multiview_w1"
    LIST="$REPO/episode_lists/reverse_with_reverse_goal_31.txt"
    BASE="$IIN"; TASK_TYPE=original_reverse
    FILE=vggt_propagation_costs_reverse; META=vggt_propagation_costs_reverse ;;
  shortcut)
    MAPS="$EVAL/datasets/object-rel-nav/maps_via_alt_goal"
    LIST="$REPO/episode_lists/shortcut_via_alt_goal_mapped_31.txt"
    BASE="$MAPS"; TASK_TYPE=via_alt_goal
    FILE=vggt_propagation_costs_via_alt_goal; META=vggt_propagation_costs_meta_via_alt_goal ;;
  altgoal)
    MAPS="$EVAL/maps/hm3d_val_mapping_alt_goal_v2_correct_vggt_multiview_w1"
    LIST="$REPO/episodes_removing_blacklist.txt"
    BASE="$IIN"; TASK_TYPE=original
    FILE=vggt_propagation_costs_alt_goal; META=vggt_propagation_costs_alt_goal ;;
  *) echo "ERROR: unknown task $TASK" >&2; exit 2 ;;
esac

OUT="${OUT_ROOT:+$OUT_ROOT/$TASK}"
OUT="${OUT:-$MAPS}"
mkdir -p "$OUT"
EPISODES="$OUT/.episodes_$TASK$SUFFIX.txt"

if [ -n "$ONLY_EPISODE" ]; then
  echo "$ONLY_EPISODE" > "$EPISODES"
else
  # Only episodes with released propagation maps, so the episode set matches.
  : > "$EPISODES"
  while IFS= read -r episode || [ -n "$episode" ]; do
    episode="${episode%$'\r'}"
    [ -z "$episode" ] && continue
    [ -f "$MAPS/$episode/$FILE.npy" ] && echo "$episode" >> "$EPISODES"
  done < "$LIST"
fi
echo "$(date -u +%FT%TZ) ${TAG:-release} $TASK: $(wc -l < "$EPISODES") episodes -> $OUT"

if [ "$TASK" = "altgoal" ]; then
  # Alt goal is built per episode from the goal frame and pixel stored in the
  # released metadata.
  while IFS= read -r episode; do
    read -r goal_idx pixel_x pixel_y < <(pixi run --frozen python -c \
      "import json, sys; m = json.load(open(sys.argv[1])); print(m['goal_img_idx'], m['goal_pixel'][0], m['goal_pixel'][1])" \
      "$MAPS/$episode/$META.json")
    run_mapper \
      "scenes.base_dir=$BASE" "scenes.base_out_dir=$OUT" \
      scenes.multi_scene=false "scenes.scene_name=$episode" \
      goal.mode=config "goal.task_type=$TASK_TYPE" \
      "goal.image_idx=$goal_idx" "goal.pixel_x=$pixel_x" "goal.pixel_y=$pixel_y" \
      "prop_map.save_filename=$FILE$SUFFIX.npy" "prop_map.metadata_filename=$META$SUFFIX.json"
  done < "$EPISODES"
else
  run_mapper \
    "scenes.base_dir=$BASE" "scenes.base_out_dir=$OUT" \
    scenes.multi_scene=true "scenes.scene_list_file=$EPISODES" \
    goal.mode=episode "goal.task_type=$TASK_TYPE" \
    "prop_map.save_filename=$FILE$SUFFIX.npy" "prop_map.metadata_filename=$META$SUFFIX.json"
fi

built=$(find "$OUT" -mindepth 2 -maxdepth 2 -name "$FILE$SUFFIX.npy" | wc -l)
expected=$(wc -l < "$EPISODES")
echo "$(date -u +%FT%TZ) ${TAG:-release} $TASK done: $built of $expected files"
if [ "$built" -lt "$expected" ]; then
  echo "ERROR: $((expected - built)) episodes have no map file" >&2
  exit 1
fi
