"""Write one composite frame per step and an MP4 video for a VGGT-Nav episode.

Each frame shows the query image, the localized submap with the goal anchor,
the predicted costmap, the top-down path and the controller waypoints. run_nav.py
creates it when visualization.vggt_nav_compact.enabled=true.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from habitat.utils.visualizations import maps


class VggtNavCompactVisualizer:
    """Write one composite RGB frame per step and stream the same frames to MP4."""

    def __init__(
        self,
        episode_results_path: Path,
        sim: Any,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        map_positions: np.ndarray,
        goal_frame_index: int = -1,
        width: int = 1280,
        height: int = 720,
        fps: float = 5.0,
        topdown_meters_per_pixel: float = 0.05,
        save_frames: bool = True,
    ) -> None:
        self.output_dir = Path(episode_results_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sim = sim
        self.start_position = np.asarray(start_position)
        self.goal_position = np.asarray(goal_position)
        self.map_positions = np.asarray(map_positions)
        self.goal_frame_index = int(goal_frame_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.save_frames = bool(save_frames)
        self.video_path = self.output_dir / "vggt_nav.mp4"
        self.frames_dir = self.output_dir / "frames"
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.video_writer: Optional[cv2.VideoWriter] = None

        height_level = float(self.start_position[1])
        topdown = maps.get_topdown_map(
            sim.pathfinder,
            height_level,
            meters_per_pixel=float(topdown_meters_per_pixel),
        )
        colors = np.array(
            [[35, 35, 35], [110, 110, 110], [235, 235, 235]], dtype=np.uint8
        )
        self.topdown_base = colors[np.clip(topdown, 0, 2)]
        self.topdown_dims = self.topdown_base.shape[:2]

    @staticmethod
    def _title(panel: np.ndarray, text: str) -> None:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 24, 28), -1)
        cv2.putText(
            panel, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (245, 245, 245), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _fit(image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        image_height, image_width = image.shape[:2]
        scale = min(width / image_width, height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        offset_x = (width - resized_width) // 2
        offset_y = (height - resized_height) // 2
        canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        return canvas, scale, offset_x, offset_y

    def _query_panel(
        self,
        query_rgb: np.ndarray,
        width: int,
        height: int,
        anchor_pixel: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        query_bgr = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
        content, scale, offset_x, offset_y = self._fit(query_bgr, width, height - 30)
        panel = np.vstack([np.zeros((30, width, 3), dtype=np.uint8), content])
        if anchor_pixel is not None:
            u, v = float(anchor_pixel[0]), float(anchor_pixel[1])
            marker = (
                offset_x + int(round(u * scale)),
                30 + offset_y + int(round(v * scale)),
            )
            cv2.circle(panel, marker, 9, (0, 220, 255), 3, cv2.LINE_AA)
            self._title(panel, "QUERY | yellow = selected subgoal")
        else:
            self._title(panel, "QUERY")
        return panel

    def _submap_panel(
        self,
        images_rgb: Sequence[np.ndarray],
        indices: Sequence[int],
        anchor_index: int,
        anchor_pixel: Tuple[float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        panel = np.full((height, width, 3), 18, dtype=np.uint8)
        self._title(panel, "LOCALIZED SUBMAP | yellow = selected subgoal")
        count = max(1, len(images_rgb))
        rows = 1 if count <= 4 else 2
        columns = int(np.ceil(count / rows))
        tile_width = width // columns
        tile_height = (height - 30) // rows

        for item, image_rgb in enumerate(images_rgb):
            row, column = divmod(item, columns)
            x0 = column * tile_width
            y0 = 30 + row * tile_height
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            tile, scale, offset_x, offset_y = self._fit(
                image_bgr, tile_width - 4, tile_height - 4
            )
            selected = item < len(indices) and int(indices[item]) == int(anchor_index)
            if selected:
                u, v = float(anchor_pixel[0]), float(anchor_pixel[1])
                marker = (
                    x0 + 2 + offset_x + int(round(u * scale)),
                    y0 + 2 + offset_y + int(round(v * scale)),
                )
                cv2.circle(panel, marker, 8, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.rectangle(
                    panel, (x0 + 1, y0 + 1),
                    (x0 + tile_width - 2, y0 + tile_height - 2),
                    (0, 220, 255), 3,
                )
            panel[y0 + 2:y0 + tile_height - 2, x0 + 2:x0 + tile_width - 2] = tile
            if selected:
                cv2.rectangle(
                    panel, (x0 + 1, y0 + 1),
                    (x0 + tile_width - 2, y0 + tile_height - 2),
                    (0, 220, 255), 3,
                )
                cv2.circle(panel, marker, 8, (0, 220, 255), 2, cv2.LINE_AA)
            label = str(indices[item]) if item < len(indices) else "?"
            cv2.putText(
                panel, label, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 220, 255) if selected else (255, 255, 255),
                1, cv2.LINE_AA,
            )
        return panel

    def _to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        row, column = maps.to_grid(
            float(position[2]), float(position[0]), self.topdown_dims,
            pathfinder=self.sim.pathfinder,
        )
        return int(row), int(column)

    def _topdown_panel(
        self,
        trajectory_history: List[Any],
        localized_indices: Sequence[int],
        anchor_index: int,
        anchor_world_position: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> np.ndarray:
        topdown = self.topdown_base.copy()
        map_grid_path = [self._to_grid(position) for position in self.map_positions]
        for first, second in zip(map_grid_path[:-1], map_grid_path[1:]):
            cv2.line(
                topdown,
                (first[1], first[0]),
                (second[1], second[0]),
                (255, 180, 20),
                2,
            )
        positions = [np.asarray(state.position) for state in trajectory_history]
        grid_path = [self._to_grid(position) for position in positions]
        for first, second in zip(grid_path[:-1], grid_path[1:]):
            cv2.line(topdown, (first[1], first[0]), (second[1], second[0]), (220, 40, 220), 2)

        for index in localized_indices:
            if 0 <= int(index) < len(self.map_positions):
                row, column = self._to_grid(self.map_positions[int(index)])
                cv2.circle(topdown, (column, row), 4, (255, 200, 0), -1)

        if anchor_world_position is not None:
            row, column = self._to_grid(np.asarray(anchor_world_position))
            cv2.circle(topdown, (column, row), 8, (0, 220, 255), 3)
        elif 0 <= int(anchor_index) < len(self.map_positions):
            row, column = self._to_grid(self.map_positions[int(anchor_index)])
            cv2.circle(topdown, (column, row), 8, (0, 220, 255), 3)

        start_row, start_column = self._to_grid(self.start_position)
        goal_row, goal_column = self._to_grid(self.goal_position)
        cv2.circle(topdown, (start_column, start_row), 6, (40, 40, 230), -1)
        if 0 <= self.goal_frame_index < len(self.map_positions):
            frame_row, frame_column = self._to_grid(
                self.map_positions[self.goal_frame_index]
            )
            cv2.circle(topdown, (frame_column, frame_row), 9, (245, 245, 245), 3)
        cv2.drawMarker(
            topdown, (goal_column, goal_row), (40, 220, 40),
            cv2.MARKER_TILTED_CROSS, 20, 3,
        )
        if grid_path:
            row, column = grid_path[-1]
            cv2.circle(topdown, (column, row), 7, (255, 255, 255), -1)
            cv2.circle(topdown, (column, row), 5, (255, 100, 20), -1)

        content, _, _, _ = self._fit(topdown, width, height - 30)
        panel = np.vstack([np.zeros((30, width, 3), dtype=np.uint8), content])
        self._title(
            panel,
            "TOP DOWN | green object | white goal frame | cyan map | yellow subgoal | magenta nav",
        )
        return panel

    def _costmap_panel(
        self,
        costmap: np.ndarray,
        width: int,
        height: int,
        title: str,
    ) -> np.ndarray:
        values = np.asarray(costmap, dtype=np.float32)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        if finite.size and float(finite.max()) > float(finite.min()):
            normalized = (values - float(finite.min())) / (float(finite.max()) - float(finite.min()))
        else:
            normalized = np.zeros_like(values)
        color = cv2.applyColorMap(
            np.clip(normalized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        color = cv2.resize(color, (width, height - 30), interpolation=cv2.INTER_NEAREST)
        panel = np.vstack([np.zeros((30, width, 3), dtype=np.uint8), color])
        if finite.size:
            minimum_flat_index = int(np.nanargmin(np.where(finite_mask, values, np.nan)))
            minimum_row, minimum_column = np.unravel_index(minimum_flat_index, values.shape)
            marker_x = int(round((minimum_column + 0.5) * width / values.shape[1]))
            marker_y = 30 + int(round((minimum_row + 0.5) * (height - 30) / values.shape[0]))
            cv2.drawMarker(
                panel, (marker_x, marker_y), (255, 255, 255),
                cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA,
            )
            title = f"{title} | {float(finite.min()):.2f} to {float(finite.max()):.2f}"
        self._title(panel, title)
        return panel

    def _status_bar(
        self,
        step: int,
        distance_to_goal: float,
        velocity: float,
        angular_velocity: float,
        collided: bool,
        localized_count: int,
        closest_index: int,
        anchor_index: int,
        inferred_stop: bool,
    ) -> np.ndarray:
        bar = np.full((34, self.width, 3), (20, 24, 28), dtype=np.uint8)
        stop_state = "STOP" if inferred_stop else "run"
        collision_state = "collision" if collided else "clear"
        text = (
            f"step {step:03d} | distance {distance_to_goal:.2f} m | "
            f"v {velocity:+.3f} | w {angular_velocity:+.3f} | {collision_state} | "
            f"localized {localized_count} | closest {closest_index} | "
            f"anchor {anchor_index} | {stop_state}"
        )
        cv2.putText(
            bar, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (245, 245, 245), 1, cv2.LINE_AA,
        )
        return bar

    def _waypoint_panel(self, waypoints: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
        panel = np.full((height, width, 3), 24, dtype=np.uint8)
        self._title(panel, "GNM WAYPOINTS | robot frame")
        origin = np.array([width // 2, height - 35], dtype=np.float32)
        cv2.line(panel, (0, int(origin[1])), (width, int(origin[1])), (70, 70, 70), 1)
        cv2.line(panel, (int(origin[0]), 30), (int(origin[0]), height), (70, 70, 70), 1)
        cv2.circle(panel, tuple(origin.astype(int)), 6, (255, 255, 255), -1)
        if waypoints is None:
            return panel
        points = np.asarray(waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            return panel
        points = points[:, :2]
        extent = max(1.0, float(np.max(np.abs(points))) * 1.2)
        scale = min((height - 80) / extent, (width / 2 - 25) / extent)
        pixels = np.stack(
            [origin[0] - points[:, 1] * scale, origin[1] - points[:, 0] * scale], axis=1
        ).astype(np.int32)
        path = np.vstack([origin.astype(np.int32), pixels])
        cv2.polylines(panel, [path], False, (255, 180, 30), 3, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(panel, tuple(pixel), 5, (255, 180, 30), -1)
        cv2.circle(panel, tuple(pixels[-1]), 9, (0, 220, 255), 3)
        return panel

    def render(
        self,
        step: int,
        query_rgb: np.ndarray,
        submap_images_rgb: Sequence[np.ndarray],
        submap_indices: Sequence[int],
        anchor_index: int,
        anchor_pixel: Tuple[float, float],
        costmap: np.ndarray,
        costmap_title: str,
        anchor_world_position: Optional[np.ndarray],
        waypoints: Optional[np.ndarray],
        trajectory_history: List[Any],
        distance_to_goal: float,
        velocity: float,
        angular_velocity: float,
        collided: bool,
        closest_index: int,
        inferred_stop: bool,
    ) -> Path:
        query_width = 384
        top_height = 288
        topdown_width = 480
        costmap_width = 320
        content_height = self.height - 34
        bottom_height = content_height - top_height

        canvas = np.full((self.height, self.width, 3), 12, dtype=np.uint8)
        query_anchor_pixel = (
            anchor_pixel
            if int(anchor_index) < 0 and anchor_pixel[0] >= 0 and anchor_pixel[1] >= 0
            else None
        )
        canvas[:top_height, :query_width] = self._query_panel(
            query_rgb, query_width, top_height, query_anchor_pixel
        )
        canvas[:top_height, query_width:] = self._submap_panel(
            submap_images_rgb, submap_indices, anchor_index, anchor_pixel,
            self.width - query_width, top_height,
        )
        canvas[top_height:content_height, :topdown_width] = self._topdown_panel(
            trajectory_history, submap_indices, anchor_index,
            anchor_world_position, topdown_width, bottom_height
        )
        canvas[top_height:content_height, topdown_width:topdown_width + costmap_width] = self._costmap_panel(
            costmap, costmap_width, bottom_height, costmap_title
        )
        canvas[top_height:content_height, topdown_width + costmap_width:] = self._waypoint_panel(
            waypoints, self.width - topdown_width - costmap_width, bottom_height
        )
        canvas[content_height:] = self._status_bar(
            step=step,
            distance_to_goal=distance_to_goal,
            velocity=velocity,
            angular_velocity=angular_velocity,
            collided=collided,
            localized_count=len(submap_indices),
            closest_index=closest_index,
            anchor_index=anchor_index,
            inferred_stop=inferred_stop,
        )

        frame_path = self.frames_dir / f"step_{step:04d}.png"
        if self.save_frames:
            saved = cv2.imwrite(
                str(frame_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1]
            )
            if not saved:
                raise OSError(f"Could not write visualization frame: {frame_path}")
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), fourcc, self.fps, (self.width, self.height)
            )
            if not self.video_writer.isOpened():
                raise RuntimeError(f"Could not open compact video writer: {self.video_path}")
        self.video_writer.write(canvas)
        return frame_path

    def close(self) -> Path:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        return self.video_path
