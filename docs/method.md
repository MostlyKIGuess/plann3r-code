# Method

What the evaluator does at each step, the checkpoints it loads, and the map
files it reads.

## One navigation step

1. Habitat renders a 320x240 RGB query image (120 degree horizontal field of view).
2. The localizer picks eight map images near the agent, from the ground-truth
   pose (oracle) or from MegaLoc retrieval.
3. The Plann3r propagation map costmaps of those eight images give the submap
   anchor: the pixel Plann3r is asked to reach.
4. Plann3r takes the query, the eight map images and the anchor, and predicts a
   16x16 costmap for the query.
5. The GNM controller takes the recent RGB frames and that costmap and predicts
   five waypoints, which become a velocity command.

## Localization

Oracle localization takes the eight map frames nearest to the agent position.
`ORACLE_MODE=legacy` is the default and the setting of every reported result.

MegaLoc retrieval replaces the pose lookup. On alt goal, oracle localization
and MegaLoc both consider only map frames up to the frame where the goal object
was seen, because that frame is part of the task.

## Tasks

| Task | `task_type` | Goal |
|---|---|---|
| Imitate | `original` | end of the mapping trajectory |
| Reverse | `original`, `reverse=true` | start of the mapping trajectory |
| Alt goal | `alt_goal_v2` | the annotated object (`seen_but_unvisited_object_v2.npy`), snapped to the NavMesh |
| Shortcut | `via_alt_goal` | end of the via-alt-goal trajectory |

Success means the final geodesic distance to the goal is within the success
radius (1 m, widened for alt-goal starts farther than 5 m). SPL weights success
by path efficiency, and SSPL also credits progress on failed episodes.

## Checkpoints

| Component | Path | SHA256 |
|---|---|---|
| Plann3r | `models/planner/checkpoint_best.pt` | `7a6658edbb44371909e05cda8d73c777a5a7b5af9e6900891896223cda7c2bbe` |
| Controller | `models/controller/predicted_costmap/latest.pth` | `10c6d3e4fbc0ab95a779053c484478c5ab931e24d84f75f00dd133a9425c80d5` |
| GT-trained controller | `models/controller/gt_trained/latest.pth` | `02e26ab97aeaec71fdd2c4891d5d85363ee89380f9347960b053d8acd040e970` |
| Ablation planners | `models/planner/ablations/*.pt` | listed on the Hugging Face model page |
| VGGT | `models/vggt/model.pt` | used for map building and inferred stopping |
| MegaLoc | `models/megaloc/` | only for MegaLoc modes |

## Map files

The evaluator reads one propagation costmap array and its JSON metadata per
episode. Imitate and Shortcut take the goal frame from the metadata. Reverse
and Alt goal set their goals from the trajectory and the alt-goal annotation.
No graph file is read.

| Task | Directory | Costmaps | Metadata |
|---|---|---|---|
| Imitate | `evaluation/maps/hm3d_val_mapping_04ed325_commit_sg_habitat_vggt_costmaps/` | `vggt_propagation_costs.npy` | `vggt_propagation_costs_meta.json` |
| Alt goal | `evaluation/maps/hm3d_val_mapping_alt_goal_v2_correct_vggt_multiview_w1/` | `vggt_propagation_costs_alt_goal.npy` | `vggt_propagation_costs_alt_goal.json` |
| Shortcut | `evaluation/datasets/object-rel-nav/maps_via_alt_goal/` | `vggt_propagation_costs_via_alt_goal.npy` | `vggt_propagation_costs_meta_via_alt_goal.json` |
| Reverse | `evaluation/maps/hm3d_val_mapping_original_reverse_vggt_multiview_w1/` | `vggt_propagation_costs_reverse.npy` | `vggt_propagation_costs_reverse.json` |

## Building propagation maps

`libs/mapper/create_vggt_prop_map.py` builds the map costmaps with Plann3r. It
starts at the goal frame, predicts costmaps for windows of 9 frames, and moves
on with a stride of 8, so neighbouring windows share one frame. The
lowest-cost patch of the shared frame becomes the next window's goal, and its
cost is added to the next window. `baseline/build_prop_maps.sh` runs it with the
settings of the released maps:

```bash
bash baseline/build_prop_maps.sh "" "$PLANN3R_ROOT/models/planner/checkpoint_best.pt" imitate
```

With an empty tag this overwrites the released files in place. Set `OUT_ROOT`
to write elsewhere, or pass a tag to write `<name>_<tag>.npy` beside them,
which `evaluate.sh` reads with `PROP_MAP_TAG=<tag>`.
