"""Load the Plann3r propagation map costmaps of one episode.

create_vggt_prop_map.py writes an (N, 16, 16) .npy array of map costmaps and a
JSON sidecar with the goal frame and pixel. task_setup.py reads both at run time.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def resolve_costmap_metadata_path(costmap_path: Path, metadata_path: Optional[Path]) -> Path:
	"""Return the sidecar path, defaulting to `<stem>_meta.json` beside the .npy file."""
	if metadata_path is None:
		return costmap_path.with_name(f"{costmap_path.stem}_meta.json")
	return Path(metadata_path)


def load_costmap_metadata(costmap_path: Path, metadata_path: Optional[Path] = None) -> dict:
	"""Read the JSON sidecar of a propagation costmap file.

	The goal frame comes from this metadata, so a missing file is an error.
	"""
	costmap_path = Path(costmap_path)
	metadata_path = resolve_costmap_metadata_path(costmap_path, metadata_path)
	if not metadata_path.is_file():
		raise FileNotFoundError(
			f"Costmap metadata not found for {costmap_path}: {metadata_path}"
		)
	with open(metadata_path, "r") as metadata_file:
		return json.load(metadata_file)


@dataclass
class CostmapData:
	costmaps: np.ndarray  # shape: (N_images, H, W)
	metadata: dict

	@staticmethod
	def from_file(costmap_path: Path, metadata_path: Optional[Path] = None) -> "CostmapData":
		"""Load a propagation costmap .npy file and its JSON sidecar.

		The sidecar is `metadata_path` when given, else `<stem>_meta.json` next to
		the .npy file.
		"""
		costmap_path = Path(costmap_path)
		if costmap_path.suffix != ".npy":
			raise ValueError(
				f"Expected a propagation costmap .npy file, got: {costmap_path}"
			)
		costmaps = np.load(costmap_path)
		metadata = load_costmap_metadata(costmap_path, metadata_path)
		return CostmapData(costmaps=costmaps, metadata=metadata)

	def get_costmap(self) -> np.ndarray:
		return self.costmaps

	def get_metadata(self) -> dict:
		return self.metadata
