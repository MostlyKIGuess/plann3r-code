"""MegaLoc global-descriptor retrieval for topological navigation.

The online input is RGB only. Map descriptors are computed once and cached on
disk. Retrieval returns a contiguous window around the top-1 MegaLoc frame so
the downstream policy receives the same kind of local submap as the oracle.
Used by libs/experiments/task_setup.py when localizer.retrieval=megaloc.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


logger = logging.getLogger("[MegaLoc]")


class MegaLocRetriever:
    def __init__(
        self,
        map_image_paths: Sequence[str | Path],
        model_source: str | Path,
        weights_path: str | Path,
        cache_root: str | Path,
        device: str | torch.device = "cuda",
        top_k: int = 8,
        batch_size: int = 16,
    ) -> None:
        if not map_image_paths:
            raise ValueError("MegaLoc requires at least one map image")

        self.map_image_paths = [Path(path) for path in map_image_paths]
        self.model_source = Path(model_source)
        self.weights_path = Path(weights_path)
        self.device = torch.device(device)
        self.top_k = min(max(1, int(top_k)), len(self.map_image_paths))
        self.batch_size = max(1, int(batch_size))
        self.cache_root = Path(cache_root).expanduser()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.Resize(size=[322, 322], antialias=True),
            ]
        )
        self.locked_frame_idx: int | None = None
        self.model = self._load_model()
        self.model.eval().to(self.device)
        self.map_descriptors = self._load_or_build_map_descriptors()

    def _load_model(self) -> torch.nn.Module:
        model_file = self.model_source / "megaloc_model.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"MegaLoc model source not found: {model_file}")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"MegaLoc weights not found: {self.weights_path}")

        spec = importlib.util.spec_from_file_location("baseline_megaloc_model", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load MegaLoc module from: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.MegaLoc()
        model.load_state_dict(load_file(str(self.weights_path), device="cpu"))
        return model

    def _cache_path(self) -> Path:
        image_fingerprint = []
        for path in self.map_image_paths:
            stat = path.stat()
            image_fingerprint.append(
                [str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)]
            )
        payload = {
            "weights": str(self.weights_path.resolve()),
            "weights_mtime_ns": int(self.weights_path.stat().st_mtime_ns),
            "images": image_fingerprint,
            "preprocess": "imagenet_resize_322_v1",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        return self.cache_root / f"map_descriptors_{digest}.npz"

    def _encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        descriptors = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.transform(image) for image in images[start : start + self.batch_size]]
            ).to(self.device)
            with torch.inference_mode():
                encoded = self.model(batch)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
            descriptors.append(encoded.cpu())
        return torch.cat(descriptors, dim=0)

    def _load_or_build_map_descriptors(self) -> torch.Tensor:
        cache_path = self._cache_path()
        if cache_path.is_file():
            with np.load(cache_path) as cached:
                descriptors = cached["descriptors"]
            logger.info("Loaded %d MegaLoc descriptors from %s", len(descriptors), cache_path)
            return torch.from_numpy(descriptors).to(self.device)

        images = []
        for path in self.map_image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        descriptors = self._encode(images).numpy().astype(np.float16)
        temporary_path = cache_path.with_suffix(".tmp")
        with open(temporary_path, "wb") as handle:
            np.savez_compressed(handle, descriptors=descriptors)
        temporary_path.replace(cache_path)
        logger.info("Cached %d MegaLoc descriptors at %s", len(descriptors), cache_path)
        return torch.from_numpy(descriptors).to(self.device)

    def _window(self, center_idx: int, frame_count: int) -> list[int]:
        window_size = min(self.top_k, frame_count)
        start = center_idx - window_size // 2
        start = max(0, min(start, frame_count - window_size))
        return list(range(start, start + window_size))

    def retrieve(
        self,
        query_rgb: np.ndarray,
        last_frame_idx: int | None = None,
        lock_frame_idx: int | None = None,
    ) -> tuple[list[int], int]:
        """Return a contiguous map window around the top-1 MegaLoc frame.

        last_frame_idx limits candidates to frames up to that index, the same
        candidate set that oracle localization uses on alt-goal. When
        lock_frame_idx is set and becomes the top-1 frame, every later call in
        the episode returns the window around that frame.
        """
        if self.locked_frame_idx is not None:
            frame_count = self.locked_frame_idx + 1
            return self._window(self.locked_frame_idx, frame_count), self.locked_frame_idx

        query = Image.fromarray(np.asarray(query_rgb, dtype=np.uint8)).convert("RGB")
        query_descriptor = self._encode([query]).to(self.device)[0]
        frame_count = len(self.map_image_paths)
        if last_frame_idx is not None:
            if not 0 <= last_frame_idx < frame_count:
                raise ValueError(
                    f"last_frame_idx={last_frame_idx} is outside [0, {frame_count})"
                )
            frame_count = last_frame_idx + 1
        scores = self.map_descriptors[:frame_count].float() @ query_descriptor.float()
        closest_idx = int(torch.argmax(scores).item())

        if lock_frame_idx is not None and closest_idx == lock_frame_idx:
            self.locked_frame_idx = lock_frame_idx
            logger.info(
                "MegaLoc locked to goal frame %d at score %.4f",
                lock_frame_idx,
                float(scores[closest_idx].item()),
            )

        localized_img_idxs = self._window(closest_idx, frame_count)
        logger.debug(
            "MegaLoc top-1=%d score=%.4f window=%s",
            closest_idx,
            float(scores[closest_idx].item()),
            localized_img_idxs,
        )
        return localized_img_idxs, closest_idx
