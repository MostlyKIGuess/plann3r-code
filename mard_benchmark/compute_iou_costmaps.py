"""Score each method's costmaps against the NavMesh ground truth with MARD.

This is the last of the four MARD stages (paper Table 1) and needs
navmesh_extractor.py, euclidean_extractor.py and vggtnav_extractor.py to have
run first. It writes the iou_*.csv files, where mean_rank_mae is MARD. Precomputed
ObjectReact costmaps (<scene>/costmaps_npy/<frame>.npy) are scored only when
--objectreact-root exists, and only at the other methods' query frames.
PLANN3R_ROOT must be set.

Usage:
    cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
    PYTHONNOUSERSITE=1 pixi run python compute_iou_costmaps.py --output_dir "$PLANN3R_ROOT/evaluation/mard"
"""

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np


def _plann3r_root() -> Path:
    root = os.environ.get("PLANN3R_ROOT")
    if not root:
        raise RuntimeError(
            "PLANN3R_ROOT is not set; export it to the Plann3r release bundle root."
        )
    return Path(root)


DEFAULT_BENCHMARKS_DIR = _plann3r_root() / "evaluation/mard"
DEFAULT_OBJECTREACT_ROOT = _plann3r_root() / "evaluation/objectreact_costmaps"
K_VALUES = [5, 15, 30, 50, 100]
SLAB_BINS = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 100)]
TARGET_SHAPE = (16, 16)

# Per-scene subdirectories holding subtraj_*.npy costmaps, keyed by method label.
VGGTNAV_VARIANTS = [
    ("vggtnav_costmaps", "vggtnav"),
    ("vggtnav_lora_costmaps", "vggtnav_lora"),
    ("vggtnav_no_aux_loss_costmaps", "vggtnav_no_aux_loss"),
    ("vggtnav_frozen_global_anchor_mlp_costmaps", "vggtnav_frozen_global_anchor_mlp"),
    ("vggtnav_no_pointmap_loss_costmaps", "vggtnav_no_pointmap_loss"),
    ("vggtnav_no_grad_loss_costmaps", "vggtnav_no_grad_loss"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IoU between costmap variants and navmesh costmaps."
    )
    parser.add_argument(
        "benchmarks_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_BENCHMARKS_DIR,
        help="Root directory containing per-scene benchmark folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write CSV outputs (defaults to benchmarks_dir).",
    )
    parser.add_argument(
        "--objectreact-root",
        type=Path,
        default=DEFAULT_OBJECTREACT_ROOT,
        help="Root of precomputed ObjectReact costmaps (<scene>/costmaps_npy/<frame>.npy). "
        "ObjectReact is scored only if this directory exists.",
    )
    return parser.parse_args()


def load_costmap(path: Path) -> np.ndarray:
    return np.load(path)


def resize_costmap(costmap: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = target_shape
    finite = np.isfinite(costmap)
    if not np.any(finite):
        resized = np.full((h, w), np.inf, dtype=np.float32)
        return resized, np.zeros((h, w), dtype=bool)

    max_finite = float(np.max(costmap[finite]))
    filled = costmap.copy()
    filled[~finite] = max_finite

    resized = cv2.resize(filled, (w, h), interpolation=cv2.INTER_AREA)
    mask_float = finite.astype(np.float32)
    resized_mask = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_NEAREST)
    resized_mask = resized_mask > 0.5

    return resized.astype(np.float32), resized_mask


def local_normalize(costmap: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(costmap) & valid_mask
    if not np.any(finite):
        normalized = np.zeros_like(costmap, dtype=np.float32)
        normalized[~finite] = np.nan
        return normalized

    min_val = float(np.min(costmap[finite]))
    max_val = float(np.max(costmap[finite]))
    if max_val - min_val < 1e-8:
        normalized = np.zeros_like(costmap, dtype=np.float32)
        normalized[~finite] = np.nan
        return normalized

    normalized = (costmap - min_val) / (max_val - min_val)
    normalized = normalized.astype(np.float32)
    normalized[~finite] = np.nan
    return normalized


def prepare_costmap(
    costmap: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    resized, valid_mask = resize_costmap(costmap, target_shape)
    normalized = local_normalize(resized, valid_mask)
    return normalized, valid_mask


def rank_transform(costmap: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(costmap) & valid_mask
    ranks = np.full_like(costmap, np.nan, dtype=np.float32)
    if not np.any(finite):
        return ranks

    values = costmap[finite]
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = sorted_vals.size
    ranked_values = np.empty_like(values, dtype=np.float32)

    start = 0
    while start < n:
        end = start
        while end + 1 < n and sorted_vals[end + 1] == sorted_vals[start]:
            end += 1
        avg_rank = (start + end) / 2.0
        ranked_values[order[start : end + 1]] = avg_rank
        start = end + 1

    if n > 1:
        ranked_values /= float(n - 1)
    else:
        ranked_values[:] = 0.0

    ranks[finite] = ranked_values
    return ranks


def slab_mask_from_ranks(
    rank_map: np.ndarray,
    valid_mask: np.ndarray,
    k_low: float,
    k_high: float,
) -> np.ndarray | None:
    finite = np.isfinite(rank_map) & valid_mask
    if not np.any(finite):
        return None

    low = k_low / 100.0
    high = k_high / 100.0
    if k_high >= 100:
        mask = (rank_map >= low) & (rank_map <= 1.0)
    else:
        mask = (rank_map >= low) & (rank_map < high)

    mask &= finite
    if not np.any(mask):
        return None
    return mask


def compute_metrics_from_masks(
    a_mask: np.ndarray | None,
    b_mask: np.ndarray | None,
    a_norm: np.ndarray,
    b_norm: np.ndarray,
    a_rank: np.ndarray,
    b_rank: np.ndarray,
) -> tuple[float, float, float]:
    if a_mask is None or b_mask is None:
        return float("nan"), float("nan"), float("nan")

    intersection = np.logical_and(a_mask, b_mask).sum()
    union_mask = np.logical_or(a_mask, b_mask)
    union = union_mask.sum()
    if union == 0:
        iou = float("nan")
    else:
        iou = float(intersection) / float(union)

    finite_norm = np.isfinite(a_norm) & np.isfinite(b_norm)
    valid_mae = union_mask & finite_norm
    if not np.any(valid_mae):
        mae = float("nan")
    else:
        diff = np.abs(a_norm - b_norm)
        mae = float(np.nanmean(diff[valid_mae]))

    finite_rank = np.isfinite(a_rank) & np.isfinite(b_rank)
    valid_rank = union_mask & finite_rank
    if not np.any(valid_rank):
        rank_mae = float("nan")
    else:
        diff_rank = np.abs(a_rank - b_rank)
        rank_mae = float(np.nanmean(diff_rank[valid_rank]))

    return iou, mae, rank_mae


def compute_k_metrics_from_prepared(
    a_norm: np.ndarray,
    a_valid: np.ndarray,
    b_norm: np.ndarray,
    b_valid: np.ndarray,
    a_rank: np.ndarray,
    b_rank: np.ndarray,
    k: float,
) -> tuple[float, float, float]:
    a_mask = bottom_k_count_mask(a_norm, k, a_valid)
    b_mask = bottom_k_count_mask(b_norm, k, b_valid)
    return compute_metrics_from_masks(a_mask, b_mask, a_norm, b_norm, a_rank, b_rank)


def compute_slab_metrics_from_prepared(
    a_norm: np.ndarray,
    a_valid: np.ndarray,
    b_norm: np.ndarray,
    b_valid: np.ndarray,
    a_rank: np.ndarray,
    b_rank: np.ndarray,
    k_low: float,
    k_high: float,
) -> tuple[float, float, float]:
    a_mask = slab_mask_from_ranks(a_rank, a_valid, k_low, k_high)
    b_mask = slab_mask_from_ranks(b_rank, b_valid, k_low, k_high)
    return compute_metrics_from_masks(a_mask, b_mask, a_norm, b_norm, a_rank, b_rank)


def bottom_k_count_mask(
    costmap: np.ndarray,
    k: float,
    valid_mask: np.ndarray,
) -> np.ndarray | None:
    flat = costmap.ravel()
    valid_flat = valid_mask.ravel()
    valid_indices = np.flatnonzero(valid_flat & np.isfinite(flat))
    if valid_indices.size == 0:
        return None

    n_top = int(valid_indices.size * k / 100.0)
    if n_top <= 0:
        return None
    if n_top >= valid_indices.size:
        return valid_mask.copy()

    values = flat[valid_indices]
    top_idx = np.argpartition(values, n_top)[:n_top]
    mask = np.zeros_like(valid_mask, dtype=bool)
    mask.ravel()[valid_indices[top_idx]] = True
    return mask


def compute_iou(
    a: np.ndarray,
    b: np.ndarray,
    k: float,
    target_shape: tuple[int, int] = TARGET_SHAPE,
) -> float:
    a_norm, a_valid = prepare_costmap(a, target_shape)
    b_norm, b_valid = prepare_costmap(b, target_shape)

    a_mask = bottom_k_count_mask(a_norm, k, a_valid)
    b_mask = bottom_k_count_mask(b_norm, k, b_valid)
    if a_mask is None or b_mask is None:
        return float("nan")

    intersection = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return float("nan")
    return float(intersection) / float(union)


def compute_masked_mae(
    a: np.ndarray,
    b: np.ndarray,
    k: float,
    target_shape: tuple[int, int] = TARGET_SHAPE,
) -> float:
    a_norm, a_valid = prepare_costmap(a, target_shape)
    b_norm, b_valid = prepare_costmap(b, target_shape)

    a_mask = bottom_k_count_mask(a_norm, k, a_valid)
    b_mask = bottom_k_count_mask(b_norm, k, b_valid)
    if a_mask is None or b_mask is None:
        return float("nan")

    union_mask = np.logical_or(a_mask, b_mask)
    finite_mask = np.isfinite(a_norm) & np.isfinite(b_norm)
    valid_mask = union_mask & finite_mask
    if not np.any(valid_mask):
        return float("nan")

    diff = np.abs(a_norm - b_norm)
    return float(np.nanmean(diff[valid_mask]))


def compute_rank_mae(
    a: np.ndarray,
    b: np.ndarray,
    k: float,
    target_shape: tuple[int, int] = TARGET_SHAPE,
) -> float:
    a_norm, a_valid = prepare_costmap(a, target_shape)
    b_norm, b_valid = prepare_costmap(b, target_shape)

    a_mask = bottom_k_count_mask(a_norm, k, a_valid)
    b_mask = bottom_k_count_mask(b_norm, k, b_valid)
    if a_mask is None or b_mask is None:
        return float("nan")

    union_mask = np.logical_or(a_mask, b_mask)
    a_rank = rank_transform(a_norm, a_valid)
    b_rank = rank_transform(b_norm, b_valid)
    finite_mask = np.isfinite(a_rank) & np.isfinite(b_rank)
    valid_mask = union_mask & finite_mask
    if not np.any(valid_mask):
        return float("nan")

    diff = np.abs(a_rank - b_rank)
    return float(np.nanmean(diff[valid_mask]))


def parse_subtraj_name(path: Path) -> tuple[int, int, int] | None:
    name = path.stem
    parts = name.split("_")
    if len(parts) < 6:
        return None
    try:
        if parts[0] != "subtraj" or parts[2] != "st" or parts[4] != "end":
            return None
        sub_idx = int(parts[1])
        start_idx = int(parts[3])
        end_idx = int(parts[5])
        return sub_idx, start_idx, end_idx
    except (ValueError, IndexError):
        return None


def iter_scene_dirs(benchmarks_dir: Path) -> list[Path]:
    return sorted([p for p in benchmarks_dir.iterdir() if p.is_dir()])


def navmesh_arrays_dir(scene_dir: Path) -> Path:
    arrays_dir = scene_dir / "navmesh_costmaps" / "arrays"
    if arrays_dir.is_dir():
        return arrays_dir
    return scene_dir / "navmesh_costmaps"


def load_navmesh_costmap(arrays_dir: Path, frame_idx: int) -> np.ndarray | None:
    path = arrays_dir / f"{frame_idx:05d}.npy"
    if not path.exists():
        return None
    return load_costmap(path)


def collect_vggtnav_pairs(scene_dir: Path) -> list[dict]:
    pairs = []
    for subdir_name, method in VGGTNAV_VARIANTS:
        vggt_dir = scene_dir / subdir_name
        if not vggt_dir.is_dir():
            continue

        for path in sorted(vggt_dir.glob("subtraj_*.npy")):
            parsed = parse_subtraj_name(path)
            if parsed is None:
                continue
            sub_idx, start_idx, end_idx = parsed
            query_idx = (start_idx + end_idx) // 2
            pairs.append({
                "method": method,
                "scene": scene_dir.name,
                "sub_idx": sub_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "query_idx": query_idx,
                "costmap_path": path,
            })
    return pairs


def collect_euclidean_pairs(scene_dir: Path) -> list[dict]:
    pairs = []
    euclid_dir = scene_dir / "euclidean_costmap"
    if not euclid_dir.is_dir():
        return pairs

    for path in sorted(euclid_dir.glob("subtraj_*.npy")):
        parsed = parse_subtraj_name(path)
        if parsed is None:
            continue
        sub_idx, start_idx, end_idx = parsed
        query_idx = (start_idx + end_idx) // 2
        pairs.append({
            "method": "euclidean",
            "scene": scene_dir.name,
            "sub_idx": sub_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "query_idx": query_idx,
            "costmap_path": path,
        })
    return pairs


def collect_mast3r_pairs(scene_dir: Path) -> list[dict]:
    pairs = []
    mast3r_dir = scene_dir / "mast3r_nav_costmaps"
    if not mast3r_dir.is_dir():
        return pairs

    for path in sorted(mast3r_dir.glob("*.npy")):
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        if frame_idx % 9 != 4:
            continue
        sub_idx = frame_idx // 9
        start_idx = sub_idx * 9
        end_idx = start_idx + 8
        pairs.append({
            "method": "mast3rnav",
            "scene": scene_dir.name,
            "sub_idx": sub_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "query_idx": frame_idx,
            "costmap_path": path,
        })
    return pairs


def collect_objectreact_pairs(
    scene_dir: Path,
    objectreact_root: Path,
    reference_pairs: list[dict],
) -> list[dict]:
    # ObjectReact writes one costmap per frame, named by the same frame index as the
    # navmesh arrays. Keep only the query frames of the other methods' subtraj windows
    # so every method is scored on identical pairs.
    objectreact_dir = objectreact_root / scene_dir.name / "costmaps_npy"
    if not objectreact_dir.is_dir():
        raise FileNotFoundError(f"ObjectReact costmaps missing for scene: {objectreact_dir}")

    windows = sorted({
        (entry["sub_idx"], entry["start_idx"], entry["end_idx"], entry["query_idx"])
        for entry in reference_pairs
    })
    pairs = []
    for sub_idx, start_idx, end_idx, query_idx in windows:
        path = objectreact_dir / f"{query_idx:05d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"ObjectReact costmap missing for query frame: {path}")
        pairs.append({
            "method": "objectreact",
            "scene": scene_dir.name,
            "sub_idx": sub_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "query_idx": query_idx,
            "costmap_path": path,
        })
    return pairs


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    benchmarks_dir = args.benchmarks_dir
    output_dir = args.output_dir or benchmarks_dir

    if not benchmarks_dir.exists() or not benchmarks_dir.is_dir():
        raise ValueError(f"benchmarks_dir not found: {benchmarks_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    include_objectreact = args.objectreact_root.is_dir()

    per_pair_rows = []
    per_pair_slab_rows = []
    per_scene_accum: dict[tuple[str, str, int], list[float]] = {}
    per_scene_mae_accum: dict[tuple[str, str, int], list[float]] = {}
    per_scene_rank_mae_accum: dict[tuple[str, str, int], list[float]] = {}
    per_scene_slab_accum: dict[tuple[str, str, int, int], list[float]] = {}
    per_scene_slab_mae_accum: dict[tuple[str, str, int, int], list[float]] = {}
    per_scene_slab_rank_mae_accum: dict[tuple[str, str, int, int], list[float]] = {}
    overall_accum: dict[tuple[str, int], list[float]] = {}
    overall_mae_accum: dict[tuple[str, int], list[float]] = {}
    overall_rank_mae_accum: dict[tuple[str, int], list[float]] = {}
    overall_slab_accum: dict[tuple[str, int, int], list[float]] = {}
    overall_slab_mae_accum: dict[tuple[str, int, int], list[float]] = {}
    overall_slab_rank_mae_accum: dict[tuple[str, int, int], list[float]] = {}

    for scene_dir in iter_scene_dirs(benchmarks_dir):
        arrays_dir = navmesh_arrays_dir(scene_dir)
        if not arrays_dir.exists():
            continue

        pairs = []
        pairs.extend(collect_vggtnav_pairs(scene_dir))
        pairs.extend(collect_euclidean_pairs(scene_dir))
        pairs.extend(collect_mast3r_pairs(scene_dir))
        if include_objectreact and pairs:
            pairs.extend(collect_objectreact_pairs(scene_dir, args.objectreact_root, pairs))

        for entry in pairs:
            costmap_path = entry["costmap_path"]
            navmesh = load_navmesh_costmap(arrays_dir, entry["query_idx"])
            if navmesh is None:
                continue
            costmap = load_costmap(costmap_path)

            cost_norm, cost_valid = prepare_costmap(costmap, TARGET_SHAPE)
            nav_norm, nav_valid = prepare_costmap(navmesh, TARGET_SHAPE)
            cost_rank = rank_transform(cost_norm, cost_valid)
            nav_rank = rank_transform(nav_norm, nav_valid)

            for k in K_VALUES:
                iou, mae, rank_mae = compute_k_metrics_from_prepared(
                    cost_norm,
                    cost_valid,
                    nav_norm,
                    nav_valid,
                    cost_rank,
                    nav_rank,
                    k,
                )
                per_pair_rows.append([
                    entry["scene"],
                    entry["method"],
                    k,
                    entry["sub_idx"],
                    entry["start_idx"],
                    entry["end_idx"],
                    entry["query_idx"],
                    iou,
                    mae,
                    rank_mae,
                ])
                if math.isfinite(iou):
                    per_scene_accum.setdefault((entry["scene"], entry["method"], k), []).append(iou)
                    overall_accum.setdefault((entry["method"], k), []).append(iou)
                if math.isfinite(mae):
                    per_scene_mae_accum.setdefault((entry["scene"], entry["method"], k), []).append(mae)
                    overall_mae_accum.setdefault((entry["method"], k), []).append(mae)
                if math.isfinite(rank_mae):
                    per_scene_rank_mae_accum.setdefault((entry["scene"], entry["method"], k), []).append(rank_mae)
                    overall_rank_mae_accum.setdefault((entry["method"], k), []).append(rank_mae)

            for k_low, k_high in SLAB_BINS:
                slab_iou, slab_mae, slab_rank_mae = compute_slab_metrics_from_prepared(
                    cost_norm,
                    cost_valid,
                    nav_norm,
                    nav_valid,
                    cost_rank,
                    nav_rank,
                    k_low,
                    k_high,
                )
                per_pair_slab_rows.append([
                    entry["scene"],
                    entry["method"],
                    k_low,
                    k_high,
                    entry["sub_idx"],
                    entry["start_idx"],
                    entry["end_idx"],
                    entry["query_idx"],
                    slab_iou,
                    slab_mae,
                    slab_rank_mae,
                ])
                if math.isfinite(slab_iou):
                    per_scene_slab_accum.setdefault(
                        (entry["scene"], entry["method"], k_low, k_high), []
                    ).append(slab_iou)
                    overall_slab_accum.setdefault((entry["method"], k_low, k_high), []).append(slab_iou)
                if math.isfinite(slab_mae):
                    per_scene_slab_mae_accum.setdefault(
                        (entry["scene"], entry["method"], k_low, k_high), []
                    ).append(slab_mae)
                    overall_slab_mae_accum.setdefault((entry["method"], k_low, k_high), []).append(slab_mae)
                if math.isfinite(slab_rank_mae):
                    per_scene_slab_rank_mae_accum.setdefault(
                        (entry["scene"], entry["method"], k_low, k_high), []
                    ).append(slab_rank_mae)
                    overall_slab_rank_mae_accum.setdefault((entry["method"], k_low, k_high), []).append(slab_rank_mae)

    per_pair_path = output_dir / "iou_per_pair.csv"
    write_csv(
        per_pair_path,
        [
            "scene",
            "method",
            "k",
            "subtraj_index",
            "start_index",
            "end_index",
            "query_index",
            "iou",
            "mae",
            "rank_mae",
        ],
        per_pair_rows,
    )

    per_scene_rows = []
    all_scene_keys = (
        set(per_scene_accum.keys())
        | set(per_scene_mae_accum.keys())
        | set(per_scene_rank_mae_accum.keys())
    )
    for (scene, method, k) in sorted(all_scene_keys):
        values = per_scene_accum.get((scene, method, k), [])
        mae_values = per_scene_mae_accum.get((scene, method, k), [])
        rank_mae_values = per_scene_rank_mae_accum.get((scene, method, k), [])
        mean_iou = mean(values) if values else float("nan")
        median_iou = median(values) if values else float("nan")
        mean_mae = mean(mae_values) if mae_values else float("nan")
        median_mae = median(mae_values) if mae_values else float("nan")
        mean_rank_mae = mean(rank_mae_values) if rank_mae_values else float("nan")
        median_rank_mae = median(rank_mae_values) if rank_mae_values else float("nan")
        per_scene_rows.append([
            scene,
            method,
            k,
            mean_iou,
            median_iou,
            len(values),
            mean_mae,
            median_mae,
            len(mae_values),
            mean_rank_mae,
            median_rank_mae,
            len(rank_mae_values),
        ])
    per_scene_path = output_dir / "iou_per_trajectory.csv"
    write_csv(
        per_scene_path,
        [
            "scene",
            "method",
            "k",
            "mean_iou",
            "median_iou",
            "count_iou",
            "mean_mae",
            "median_mae",
            "count_mae",
            "mean_rank_mae",
            "median_rank_mae",
            "count_rank_mae",
        ],
        per_scene_rows,
    )

    overall_rows = []
    all_overall_keys = (
        set(overall_accum.keys())
        | set(overall_mae_accum.keys())
        | set(overall_rank_mae_accum.keys())
    )
    for (method, k) in sorted(all_overall_keys):
        values = overall_accum.get((method, k), [])
        mae_values = overall_mae_accum.get((method, k), [])
        rank_mae_values = overall_rank_mae_accum.get((method, k), [])
        mean_iou = mean(values) if values else float("nan")
        median_iou = median(values) if values else float("nan")
        mean_mae = mean(mae_values) if mae_values else float("nan")
        median_mae = median(mae_values) if mae_values else float("nan")
        mean_rank_mae = mean(rank_mae_values) if rank_mae_values else float("nan")
        median_rank_mae = median(rank_mae_values) if rank_mae_values else float("nan")
        overall_rows.append([
            method,
            k,
            mean_iou,
            median_iou,
            len(values),
            mean_mae,
            median_mae,
            len(mae_values),
            mean_rank_mae,
            median_rank_mae,
            len(rank_mae_values),
        ])
    overall_path = output_dir / "iou_overall.csv"
    write_csv(
        overall_path,
        [
            "method",
            "k",
            "mean_iou",
            "median_iou",
            "count_iou",
            "mean_mae",
            "median_mae",
            "count_mae",
            "mean_rank_mae",
            "median_rank_mae",
            "count_rank_mae",
        ],
        overall_rows,
    )

    print(f"Wrote {per_pair_path}")
    print(f"Wrote {per_scene_path}")
    print(f"Wrote {overall_path}")

    per_pair_slab_path = output_dir / "iou_per_pair_slabs.csv"
    write_csv(
        per_pair_slab_path,
        [
            "scene",
            "method",
            "k_low",
            "k_high",
            "subtraj_index",
            "start_index",
            "end_index",
            "query_index",
            "iou",
            "mae",
            "rank_mae",
        ],
        per_pair_slab_rows,
    )

    per_scene_slab_rows = []
    all_slab_scene_keys = (
        set(per_scene_slab_accum.keys())
        | set(per_scene_slab_mae_accum.keys())
        | set(per_scene_slab_rank_mae_accum.keys())
    )
    for (scene, method, k_low, k_high) in sorted(all_slab_scene_keys):
        values = per_scene_slab_accum.get((scene, method, k_low, k_high), [])
        mae_values = per_scene_slab_mae_accum.get((scene, method, k_low, k_high), [])
        rank_mae_values = per_scene_slab_rank_mae_accum.get((scene, method, k_low, k_high), [])
        mean_iou = mean(values) if values else float("nan")
        median_iou = median(values) if values else float("nan")
        mean_mae = mean(mae_values) if mae_values else float("nan")
        median_mae = median(mae_values) if mae_values else float("nan")
        mean_rank_mae = mean(rank_mae_values) if rank_mae_values else float("nan")
        median_rank_mae = median(rank_mae_values) if rank_mae_values else float("nan")
        per_scene_slab_rows.append([
            scene,
            method,
            k_low,
            k_high,
            mean_iou,
            median_iou,
            len(values),
            mean_mae,
            median_mae,
            len(mae_values),
            mean_rank_mae,
            median_rank_mae,
            len(rank_mae_values),
        ])
    per_scene_slab_path = output_dir / "iou_per_trajectory_slabs.csv"
    write_csv(
        per_scene_slab_path,
        [
            "scene",
            "method",
            "k_low",
            "k_high",
            "mean_iou",
            "median_iou",
            "count_iou",
            "mean_mae",
            "median_mae",
            "count_mae",
            "mean_rank_mae",
            "median_rank_mae",
            "count_rank_mae",
        ],
        per_scene_slab_rows,
    )

    overall_slab_rows = []
    all_slab_overall_keys = (
        set(overall_slab_accum.keys())
        | set(overall_slab_mae_accum.keys())
        | set(overall_slab_rank_mae_accum.keys())
    )
    for (method, k_low, k_high) in sorted(all_slab_overall_keys):
        values = overall_slab_accum.get((method, k_low, k_high), [])
        mae_values = overall_slab_mae_accum.get((method, k_low, k_high), [])
        rank_mae_values = overall_slab_rank_mae_accum.get((method, k_low, k_high), [])
        mean_iou = mean(values) if values else float("nan")
        median_iou = median(values) if values else float("nan")
        mean_mae = mean(mae_values) if mae_values else float("nan")
        median_mae = median(mae_values) if mae_values else float("nan")
        mean_rank_mae = mean(rank_mae_values) if rank_mae_values else float("nan")
        median_rank_mae = median(rank_mae_values) if rank_mae_values else float("nan")
        overall_slab_rows.append([
            method,
            k_low,
            k_high,
            mean_iou,
            median_iou,
            len(values),
            mean_mae,
            median_mae,
            len(mae_values),
            mean_rank_mae,
            median_rank_mae,
            len(rank_mae_values),
        ])
    overall_slab_path = output_dir / "iou_overall_slabs.csv"
    write_csv(
        overall_slab_path,
        [
            "method",
            "k_low",
            "k_high",
            "mean_iou",
            "median_iou",
            "count_iou",
            "mean_mae",
            "median_mae",
            "count_mae",
            "mean_rank_mae",
            "median_rank_mae",
            "count_rank_mae",
        ],
        overall_slab_rows,
    )

    print(f"Wrote {per_pair_slab_path}")
    print(f"Wrote {per_scene_slab_path}")
    print(f"Wrote {overall_slab_path}")


if __name__ == "__main__":
    main()
