"""Compute ground-truth NavMesh geodesic costmaps for every episode in the MARD list.

This is the first MARD stage. The other extractors read its costmaps to pick
the anchor pixel, the lowest-cost pixel in each submap, so it must finish first.
The per-frame work is in gt_mesh_generator.py, run in parallel over episodes.
PLANN3R_ROOT must be set, since the default parent_dir and --scene-root are
under it. Add --dry-run to list the work without running it.

Usage:
    cd "$PLANN3R_ROOT/plann3r-code/mard_benchmark"
    PYTHONNOUSERSITE=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet pixi run python navmesh_extractor.py \
      --scene-list "$PLANN3R_ROOT/plann3r-code/episodes_removing_blacklist.txt" \
      --workers 8 --export-stack --skip-existing --output_dir "$PLANN3R_ROOT/evaluation/mard"
"""

import argparse
import logging
import os
import shutil
import csv
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
from natsort import natsorted

from gt_mesh_generator import (
    create_sim_and_load_navmesh,
    init_worker,
    load_goal_pixel,
    run_session_parallel,
)


def _plann3r_root() -> Path:
    root = os.environ.get("PLANN3R_ROOT")
    if not root:
        raise RuntimeError(
            "PLANN3R_ROOT is not set; export it to the Plann3r release bundle root."
        )
    return Path(root)


DEFAULT_PARENT_DIR = (
    _plann3r_root() / "evaluation/datasets/hm3d_navigation/hm3d_iin_val_320x240"
)
DEFAULT_SCENE_ROOT = (
    _plann3r_root() / "evaluation/datasets/hm3d_navigation/hm3d_v0.2/val"
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute navmesh costmaps for each scene and optionally export a "
            "stacked navmesh_costmaps.npy per scene."
        )
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_PARENT_DIR,
        help="Parent directory containing scene subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Optional output root directory; default writes into each episode dir.",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=None,
        help="Optional scene list file; if omitted, uses all episodes in parent_dir.",
    )
    parser.add_argument(
        "--episode-list",
        type=Path,
        default=None,
        help="Optional text file with episode directory paths (one per line).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel processes for navmesh geodesics.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=DEFAULT_SCENE_ROOT,
        help="Root directory with HM3D scene folders (e.g., 00023-<scene>).",
    )
    parser.add_argument(
        "--export-stack",
        action="store_true",
        help="Save a stacked navmesh_costmaps.npy per scene.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="navmesh_costmaps.npy",
        help="Filename for stacked output (used with --export-stack).",
    )
    parser.add_argument(
        "--stack-only",
        action="store_true",
        help="Only keep the stacked navmesh_costmaps.npy (no arrays or pngs).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already have navmesh_costmaps.npy.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV log path (defaults to <parent_dir>/navmesh_extractor_log.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grouping info and exit without processing scenes.",
    )
    parser.add_argument(
        "--split-into",
        type=int,
        default=4,
        help="Number of split files to generate from the episode list.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Directory to write split episode lists.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Write split files and exit without processing scenes.",
    )
    return parser.parse_args()


def load_scene_names(scene_list_path: Path) -> set:
    if not scene_list_path.exists():
        raise ValueError(f"Scene list file not found: {scene_list_path}")

    lines = scene_list_path.read_text().splitlines()
    return {
        Path(line.strip()).name
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


def load_episode_paths(episode_list_path: Path) -> list[Path]:
    if not episode_list_path.exists():
        raise ValueError(f"Episode list file not found: {episode_list_path}")

    base_dir = episode_list_path.parent
    lines = episode_list_path.read_text().splitlines()
    episode_paths: list[Path] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        episode_paths.append(path)
    return episode_paths


def resolve_scene_folder(scene_root: Path, episode_dir: Path) -> Path | None:
    scene_id = episode_dir.name.split("_")[0]
    if not scene_id:
        return None
    matches = sorted(scene_root.glob(f"*-{scene_id}"))
    if not matches:
        return None
    return matches[0]


def find_navmesh_path(scene_root: Path, episode_dir: Path) -> Path | None:
    scene_folder = resolve_scene_folder(scene_root, episode_dir)
    if scene_folder is None:
        return None
    candidates = list(scene_folder.glob("*.basis.navmesh"))
    if not candidates:
        return None
    return candidates[0]


def find_glb_path(scene_root: Path, episode_dir: Path) -> Path | None:
    scene_folder = resolve_scene_folder(scene_root, episode_dir)
    if scene_folder is None:
        return None
    candidates = list(scene_folder.glob("*.glb"))
    if not candidates:
        return None
    return candidates[0]


def load_costmaps_from_arrays(arrays_dir: Path) -> np.ndarray:
    array_files = natsorted([p for p in arrays_dir.glob("*.npy") if p.is_file()])
    if not array_files:
        raise FileNotFoundError(f"No .npy files found in {arrays_dir}")

    costmaps = []
    for array_file in array_files:
        costmap = np.load(array_file)
        costmaps.append(costmap)

    return np.stack(costmaps, axis=0)


def resolve_images_dir(episode_dir: Path) -> Path:
    for name in (
        "images_fov90",
        "images",
        "images_downsampled_fov120",
        "images_downsampled",
        "rgb",
        "color",
    ):
        candidate = episode_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No images directory found in {episode_dir}")


def count_frames_from_images(images_dir: Path) -> int:
    count = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        count += len(list(images_dir.glob(ext)))
    return count


def is_complete_navmesh_stack(stacked_path: Path, expected_frames: int) -> tuple[bool, str]:
    if not stacked_path.exists():
        return False, "missing stacked output"
    try:
        costmaps = np.load(stacked_path)
    except Exception as exc:
        return False, f"failed to load stacked output: {exc}"
    if costmaps.ndim != 3:
        return False, f"unexpected stacked shape: {costmaps.shape}"
    if expected_frames > 0 and costmaps.shape[0] != expected_frames:
        return False, f"frame count mismatch: {costmaps.shape[0]} != {expected_frames}"
    valid_per_frame = np.isfinite(costmaps).reshape(costmaps.shape[0], -1).any(axis=1)
    if not np.all(valid_per_frame):
        return False, "one or more frames have no finite costs"
    return True, "complete"


def iter_scene_dirs(parent_dir: Path, allowed: set[str]) -> Iterable[Path]:
    scene_dirs = sorted([p for p in parent_dir.iterdir() if p.is_dir()])
    return [p for p in scene_dirs if p.name in allowed]


class SceneEntry(NamedTuple):
    scene_dir: Path
    navmesh_path: Path
    glb_path: Path | None
    output_dir: Path
    stacked_path: Path


def distribute_groups(
    groups: list[tuple[Path, list[SceneEntry]]],
    num_parts: int,
) -> list[list[SceneEntry]]:
    buckets: list[list[SceneEntry]] = [[] for _ in range(num_parts)]
    sizes = [0 for _ in range(num_parts)]

    sorted_groups = sorted(groups, key=lambda item: len(item[1]), reverse=True)
    for _, entries in sorted_groups:
        idx = sizes.index(min(sizes))
        buckets[idx].extend(entries)
        sizes[idx] += len(entries)

    return buckets


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    parent_dir = args.parent_dir

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise ValueError(f"Parent directory not found: {parent_dir}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_list is None:
        if args.episode_list is not None:
            scene_dirs = [p for p in load_episode_paths(args.episode_list) if p.is_dir()]
        else:
            scene_dirs = sorted([p for p in parent_dir.iterdir() if p.is_dir()])
    else:
        allowed_scene_names = load_scene_names(args.scene_list)
        if not allowed_scene_names:
            raise ValueError(f"Scene list is empty: {args.scene_list}")
        scene_dirs = list(iter_scene_dirs(parent_dir, allowed_scene_names))
    if not scene_dirs:
        raise ValueError(
            "No matching scene directories found under: "
            f"{parent_dir} (scene list: {args.scene_list})"
        )

    if args.stack_only and not args.export_stack:
        raise ValueError("--stack-only requires --export-stack")

    log_path = args.log_path or (parent_dir / "navmesh_extractor_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with open(log_path, "w", newline="", buffering=1) as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["episode", "status", "reason", "time_sec"])
        log_file.flush()

        entries: list[SceneEntry] = []

        for scene_dir in scene_dirs:
            start_time = time.perf_counter()
            navmesh_path = find_navmesh_path(args.scene_root, scene_dir)
            if navmesh_path is None:
                reason = f"no navmesh under {args.scene_root}"
                logger.warning(
                    "No navmesh found for episode %s under %s",
                    scene_dir.name,
                    args.scene_root,
                )
                elapsed = time.perf_counter() - start_time
                writer.writerow([scene_dir.name, "skipped", reason, f"{elapsed:.2f}"])
                log_file.flush()
                continue

            try:
                load_goal_pixel(scene_dir)
            except Exception as exc:
                reason = f"no goal: {exc}"
                logger.warning(
                    "No goal found for episode %s: %s",
                    scene_dir.name,
                    exc,
                )
                elapsed = time.perf_counter() - start_time
                writer.writerow([scene_dir.name, "skipped", reason, f"{elapsed:.2f}"])
                log_file.flush()
                continue

            # --output_dir is an output ROOT: give each episode its own
            # subdirectory, mirroring the in-place layout. Without this every
            # episode writes to the same folder and overwrites the previous one.
            base_dir = (
                args.output_dir / scene_dir.name
                if args.output_dir is not None
                else scene_dir
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            scene_out_dir = base_dir / "navmesh_costmaps"
            stacked_path = base_dir / args.output_name

            if args.skip_existing:
                try:
                    images_dir = resolve_images_dir(scene_dir)
                    expected_frames = count_frames_from_images(images_dir)
                except Exception as exc:
                    expected_frames = 0
                    logger.warning("Failed to resolve images for %s: %s", scene_dir.name, exc)

                complete, reason = is_complete_navmesh_stack(stacked_path, expected_frames)
                if complete:
                    logger.info("Skipping %s: %s", scene_dir.name, reason)
                    elapsed = time.perf_counter() - start_time
                    writer.writerow([scene_dir.name, "skipped", reason, f"{elapsed:.2f}"])
                    log_file.flush()
                    continue

            glb_path = find_glb_path(args.scene_root, scene_dir)
            entries.append(
                SceneEntry(
                    scene_dir=scene_dir,
                    navmesh_path=navmesh_path,
                    glb_path=glb_path,
                    output_dir=scene_out_dir,
                    stacked_path=stacked_path,
                )
            )

        grouped: dict[Path, list[SceneEntry]] = defaultdict(list)
        for entry in entries:
            key = entry.glb_path if entry.glb_path is not None else entry.navmesh_path
            grouped[key].append(entry)

        final_groups: list[tuple[Path, list[SceneEntry]]] = []
        for group_key, group_entries in grouped.items():
            navmesh_paths = {entry.navmesh_path for entry in group_entries}
            if len(navmesh_paths) != 1:
                logger.warning(
                    "GLB group %s has multiple navmesh paths; splitting by navmesh",
                    group_key,
                )
                subgroups: dict[Path, list[SceneEntry]] = defaultdict(list)
                for entry in group_entries:
                    subgroups[entry.navmesh_path].append(entry)
                for navmesh_path, sub_entries in subgroups.items():
                    final_groups.append((navmesh_path, sub_entries))
                continue

            final_groups.append((group_key, group_entries))

        if args.split_output_dir is not None:
            if args.split_into < 1:
                raise ValueError("--split-into must be >= 1")
            args.split_output_dir.mkdir(parents=True, exist_ok=True)
            buckets = distribute_groups(final_groups, args.split_into)
            for idx, bucket in enumerate(buckets, start=1):
                out_path = args.split_output_dir / f"episodes_part_{idx}.txt"
                with open(out_path, "w") as handle:
                    for entry in bucket:
                        handle.write(f"{entry.scene_dir}\n")
                logger.info("Wrote %s (%d episodes)", out_path, len(bucket))
            if args.split_only:
                return

        if args.dry_run:
            logger.info("Dry run: %d groups", len(final_groups))
            for group_key, group_entries in final_groups:
                navmesh_path = next(iter({entry.navmesh_path for entry in group_entries}))
                glb_label = group_key if group_key.suffix == ".glb" else None
                logger.info(
                    "Group: episodes=%d | glb=%s | navmesh=%s",
                    len(group_entries),
                    glb_label or "unknown",
                    navmesh_path,
                )
                for entry in group_entries:
                    logger.info("  %s", entry.scene_dir.name)
            return

        for group_key, group_entries in final_groups:
            navmesh_path = next(iter({entry.navmesh_path for entry in group_entries}))
            glb_label = group_key if group_key.suffix == ".glb" else None
            logger.info(
                "Processing group: episodes=%d | glb=%s | navmesh=%s",
                len(group_entries),
                glb_label or "unknown",
                navmesh_path,
            )

            try:
                main_sim, main_navmesh = create_sim_and_load_navmesh(navmesh_path)
            except Exception as exc:
                for entry in group_entries:
                    reason = f"navmesh load failed: {exc}"
                    writer.writerow([entry.scene_dir.name, "failed", reason, "0.00"])
                log_file.flush()
                continue

            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(str(navmesh_path),),
            ) as executor:
                for entry in group_entries:
                    start_time = time.perf_counter()
                    try:
                        run_session_parallel(
                            session_folder=entry.scene_dir,
                            navmesh_path=navmesh_path,
                            output_dir=entry.output_dir,
                            num_workers=args.workers,
                            save_arrays=True,
                            save_pngs=not args.stack_only,
                            executor=executor,
                            main_sim=main_sim,
                            main_navmesh=main_navmesh,
                        )
                    except Exception as exc:
                        reason = f"run failed: {exc}"
                        logger.warning("Failed %s: %s", entry.scene_dir.name, exc)
                        elapsed = time.perf_counter() - start_time
                        writer.writerow([entry.scene_dir.name, "failed", reason, f"{elapsed:.2f}"])
                        log_file.flush()
                        continue

                    if args.export_stack:
                        arrays_dir = entry.output_dir / "arrays"
                        try:
                            costmaps = load_costmaps_from_arrays(arrays_dir)
                        except Exception as exc:
                            reason = f"stacking failed: {exc}"
                            logger.warning("Failed to stack %s: %s", entry.scene_dir.name, exc)
                            elapsed = time.perf_counter() - start_time
                            writer.writerow(
                                [entry.scene_dir.name, "failed", reason, f"{elapsed:.2f}"]
                            )
                            log_file.flush()
                            continue

                        entry.stacked_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(entry.stacked_path, costmaps)
                        logger.info(
                            "Saved %s: %s shape=%s",
                            entry.scene_dir.name,
                            entry.stacked_path.name,
                            costmaps.shape,
                        )

                        if args.stack_only:
                            if arrays_dir.exists():
                                shutil.rmtree(arrays_dir)
                            for png_path in entry.output_dir.glob("*.png"):
                                png_path.unlink()

                    elapsed = time.perf_counter() - start_time
                    writer.writerow([entry.scene_dir.name, "success", "", f"{elapsed:.2f}"])
                    log_file.flush()
                    processed += 1

    if processed == 0:
        logger.warning("No scenes processed. See log: %s", log_path)


if __name__ == "__main__":
    main()
