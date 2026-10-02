"""Load the Plann3r VGGTNav checkpoint and predict goal costmaps from images.

The model takes a query image and its submap frames, with the goal given as an
anchor pixel in one frame, and returns a 16x16 patch costmap that is normalized
and upsampled for the controller. The multi and batched_exact variants decode
many frames at once for the propagation maps. Used by task_setup.py during
navigation, by libs/mapper/create_vggt_prop_map.py and by
mard_benchmark/vggtnav_extractor.py. Loading fails on a checkpoint whose
costmap head is missing or whose trained tensors the model does not define.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
import cv2


def _ensure_vggt_on_path() -> None:
    """Try to locate the VGGTNav repo and add it to sys.path for import."""
    candidates = []

    env_root = (
        os.environ.get("VGGTNAV_ROOT")
        or os.environ.get("VGGT_ROOT")
        or os.environ.get("VGGT_PATH")
    )
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root)
    candidates.append(repo_root.parent / "VGGTNav")
    candidates.append(repo_root.parent / "vggt")
    candidates.append(repo_root.parent.parent / "VGGTNav")
    candidates.append(repo_root.parent.parent / "vggt")

    for base in candidates:
        if not base.exists():
            continue

        if (base / "vggt").is_dir():
            path_to_add = base
        elif base.name == "vggt" and (base / "__init__.py").exists():
            path_to_add = base.parent
        else:
            continue

        if str(path_to_add) not in sys.path:
            sys.path.append(str(path_to_add))
        return


try:
    from vggt.models.vggt_nav import VGGTNav
except ModuleNotFoundError as exc:
    if exc.name != "vggt":
        raise
    _ensure_vggt_on_path()
    from vggt.models.vggt_nav import VGGTNav


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip a leading 'module.' prefix from state_dict keys if present."""
    if all(key.startswith("module.") for key in state_dict.keys() if isinstance(key, str)):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_vggtnav_model(cfg: dict, device: str):
    """Load a VGGTNav model from checkpoint and move it to the target device."""
    if cfg is None:
        raise ValueError("VGGTNav config is required to load the model.")

    checkpoint_path = Path(cfg.get("checkpoint_path", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VGGTNav checkpoint not found: {checkpoint_path}")

    model = VGGTNav(
        img_size=int(cfg.get("img_size", 224)),
        patch_size=int(cfg.get("patch_size", 14)),
        init_costmap_from_depth=bool(cfg.get("init_costmap_from_depth", False)),
        costmap_activation=cfg.get("costmap_activation", "gelu"),
        costmap_head_type=cfg.get("costmap_head_type", "mlp"),
        costmap_mlp_ratio=float(cfg.get("costmap_mlp_ratio", 4.0)),
        costmap_mlp_drop=float(cfg.get("costmap_mlp_drop", 0.0)),
        costmap_mlp_layer_idx=int(cfg.get("costmap_mlp_layer_idx", -1)),
        enable_point_aux=bool(cfg.get("enable_point_aux", False)),
        train_point_head=bool(cfg.get("train_point_head", False)),
    )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    # strict=False is needed because training-only heads are saved but never
    # built here. Fail on anything else it would hide: a costmap head left
    # random-initialized, or trained tensors the inference model has no module for.
    missing = [
        k for k in getattr(load_result, "missing_keys", []) if k.startswith("costmap_head.")
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is missing costmap_head.* weights, which would "
            f"leave the costmap head random-initialized. First missing keys: {missing[:5]}"
        )
    training_only_heads = ("point_head.", "camera_head.", "track_head.")
    dropped = [
        k for k in getattr(load_result, "unexpected_keys", []) if not k.startswith(training_only_heads)
    ]
    if dropped:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' has trained tensors the inference model does not "
            f"define, so they would be ignored. First keys: {dropped[:5]}"
        )

    model = model.to(device)
    model.eval()

    return model


def compute_resize_params_wh(width: int, height: int, target_size: int, patch_size: int) -> Tuple[int, int, float, int, int]:
    """Resize/pad params for a width x height image. Returns (w, h, scale, pad_left, pad_top)."""
    scale = float(target_size) / float(max(width, height))
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized_w = min(resized_w, target_size)
    resized_h = min(resized_h, target_size)
    resized_w = max(patch_size, int(round(resized_w / patch_size)) * patch_size)
    resized_h = max(patch_size, int(round(resized_h / patch_size)) * patch_size)
    resized_w = min(resized_w, target_size)
    resized_h = min(resized_h, target_size)
    pad_left = (target_size - resized_w) // 2
    pad_top = (target_size - resized_h) // 2
    return resized_w, resized_h, scale, pad_left, pad_top


def _compute_resize_params(image: Image.Image, target_size: int, patch_size: int) -> Tuple[int, int, float, int, int]:
    width, height = image.size
    return compute_resize_params_wh(width, height, target_size, patch_size)


def preprocess_rgb_images(
    images: List[np.ndarray],
    target_size: int = 224,
    patch_size: int = 14,
) -> Tuple[torch.Tensor, List[dict]]:
    """Preprocess a list of RGB images for VGGTNav inference.

    Returns:
        images: torch.Tensor [N, 3, target_size, target_size] in [0, 1]
        metadata: List of transform metadata for each image
    """
    if len(images) == 0:
        raise ValueError("At least one image is required for VGGTNav inference.")

    to_tensor = TF.ToTensor()
    processed = []
    metadata = []

    for img in images:
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.convert("RGB")

        resized_w, resized_h, scale, pad_left, pad_top = _compute_resize_params(
            pil_img, target_size, patch_size
        )

        resized_img = pil_img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        square_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        square_img.paste(resized_img, (pad_left, pad_top))

        tensor_img = to_tensor(square_img)
        processed.append(tensor_img)
        metadata.append({
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "resized_w": resized_w,
            "resized_h": resized_h,
            "orig_w": pil_img.width,
            "orig_h": pil_img.height,
        })

    images_tensor = torch.stack(processed)
    return images_tensor, metadata


def compute_anchor_patch_idx(goal_uv_resized: np.ndarray, out_w: int, out_h: int, patch_size: int) -> int:
    u = int(np.clip(np.floor(goal_uv_resized[0]), 0, out_w - 1))
    v = int(np.clip(np.floor(goal_uv_resized[1]), 0, out_h - 1))
    patch_w = max(1, out_w // patch_size)
    patch_x = min(u // patch_size, patch_w - 1)
    patch_y = min(v // patch_size, max(1, out_h // patch_size) - 1)
    return int(patch_y * patch_w + patch_x)


def transform_goal_pixel(goal_pixel: Tuple[int, int], metadata: dict) -> np.ndarray:
    px, py = int(goal_pixel[0]), int(goal_pixel[1])
    u = px * metadata["scale"] + metadata["pad_left"]
    v = py * metadata["scale"] + metadata["pad_top"]
    return np.array([u, v], dtype=np.float32)


def predict_vggtnav_costmap(
    model: torch.nn.Module,
    query_image: np.ndarray,
    submap_images: List[np.ndarray],
    anchor_frame_index: int,
    anchor_pixel: Tuple[int, int],
    img_size: int = 224,
    patch_size: int = 14,
    normalize: bool = True,
    upsample_size: int = 60,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run VGGTNav inference and return a normalized 60x60 costmap plus the raw 16x16 costmap."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    images = [query_image] + list(submap_images)
    image_tensor, transform_meta = preprocess_rgb_images(images, target_size=img_size, patch_size=patch_size)
    image_tensor = image_tensor.to(device)

    anchor_meta = transform_meta[anchor_frame_index]
    goal_uv_resized = transform_goal_pixel(anchor_pixel, anchor_meta)

    out_h = img_size
    out_w = img_size
    anchor_patch_idx = compute_anchor_patch_idx(goal_uv_resized, out_w, out_h, patch_size)
    sequence_len = len(images)
    anchor_idx_arr = np.full((sequence_len,), -1, dtype=np.int64)
    anchor_idx_arr[anchor_frame_index] = anchor_patch_idx

    anchor_idx_tensor = torch.from_numpy(anchor_idx_arr).long().to(device).unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor, anchor_patch_idx=anchor_idx_tensor)

    costmap = outputs["costmap"]
    if costmap.ndim == 5:
        costmap = costmap[0, 0, :, :, 0]
    elif costmap.ndim == 4:
        costmap = costmap[0, :, :, 0]
    else:
        raise ValueError(f"Unexpected costmap output shape: {costmap.shape}")

    raw_costmap = costmap.detach().cpu().float().numpy()
    upsampled = _normalize_and_upsample_costmap(raw_costmap, upsample_size, normalize)
    return upsampled, raw_costmap


def _build_slot_anchor_tensor(
    anchor_slot: int,
    anchor_pixel: Tuple[int, int],
    transform_meta: List[dict],
    img_size: int,
    patch_size: int,
) -> Optional[np.ndarray]:
    """Build a per-slot anchor vector of length S, -1 everywhere except `anchor_slot`.

    Returns None when `anchor_slot` is negative (no goal conditioning). The [S] form is
    reshaped to [1, S] by the callers, never the [B] form -- see `Aggregator`, where a
    length-B vector is hard-bound to slot 0 instead of following the frames.
    """
    num_slots = len(transform_meta)
    if anchor_slot < 0:
        return None
    if anchor_slot >= num_slots:
        raise ValueError(f"anchor_slot={anchor_slot} out of range for {num_slots} frames")

    goal_uv_resized = transform_goal_pixel(anchor_pixel, transform_meta[anchor_slot])
    patch_idx = compute_anchor_patch_idx(goal_uv_resized, img_size, img_size, patch_size)

    anchor = np.full((num_slots,), -1, dtype=np.int64)
    anchor[anchor_slot] = patch_idx
    return anchor


def predict_vggtnav_costmaps_multi(
    model: torch.nn.Module,
    frame_images: List[np.ndarray],
    anchor_slot: int,
    anchor_pixel: Tuple[int, int],
    query_slots: Optional[List[int]] = None,
    img_size: int = 224,
    patch_size: int = 14,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Decode costmaps for many frames from a SINGLE aggregator pass.

    `frame_images` is already in slot order -- this function never reorders it. Slot 0
    is whatever the caller put first, and that choice matters: slot 0 receives a distinct
    camera/register token, and the costmap head was only ever trained on slot-0 tokens.
    Costmaps for slots 1..S-1 are therefore an approximation of what the per-query path
    would produce. Ordering policy belongs to the caller, which knows which frame's
    costmap carries downstream consequences.

    Args:
        frame_images: window frames in slot order.
        anchor_slot: slot holding the goal/anchor frame, or -1 for no conditioning.
        anchor_pixel: goal pixel, in the coordinates of `frame_images[anchor_slot]`.
        query_slots: slots to decode, in the desired output order. None means all slots.

    Returns:
        Raw [grid, grid] float32 costmaps, one per entry of `query_slots`, **in
        `query_slots` order** -- callers building a dict must zip, not assume sorted.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_tensor, transform_meta = preprocess_rgb_images(
        frame_images, target_size=img_size, patch_size=patch_size
    )
    image_tensor = image_tensor.to(device).unsqueeze(0)

    anchor = _build_slot_anchor_tensor(anchor_slot, anchor_pixel, transform_meta, img_size, patch_size)
    anchor_tensor = None
    if anchor is not None:
        anchor_tensor = torch.from_numpy(anchor).long().to(device).unsqueeze(0)

    selection = "all" if query_slots is None else list(query_slots)

    with torch.no_grad():
        outputs = model(image_tensor, anchor_patch_idx=anchor_tensor, query_frame_idx=selection)

    costmap = outputs["costmap"]
    if costmap.ndim != 5:
        raise ValueError(f"Unexpected costmap output shape: {tuple(costmap.shape)}")

    costmap = costmap[0, :, :, :, 0].detach().cpu().float().numpy()
    return [costmap[k].astype(np.float32) for k in range(costmap.shape[0])]


def predict_vggtnav_costmaps_batched_exact(
    model: torch.nn.Module,
    frame_images: List[np.ndarray],
    anchor_slot: int,
    anchor_pixel: Tuple[int, int],
    query_slots: Optional[List[int]] = None,
    batch_chunk: int = 3,
    img_size: int = 224,
    patch_size: int = 14,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Exact reference: one row per query, each with its own query in slot 0.

    Mathematically identical to calling `predict_vggtnav_costmap` once per query frame --
    every row still puts its query at slot 0 and its submap frames in temporal order --
    but preprocessing happens once and rows are batched along B. Use it as the plumbing
    control for the multi-query harness, and as an exact-but-faster fallback.

    `batch_chunk` bounds B: B rows of S frames is B*S images through the patch embed plus
    B copies of the global-attention activations, so a full B=S=9 is a real memory step up.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_slots = len(frame_images)
    slots = list(range(num_slots)) if query_slots is None else list(query_slots)

    image_tensor, transform_meta = preprocess_rgb_images(
        frame_images, target_size=img_size, patch_size=patch_size
    )
    image_tensor = image_tensor.to(device)

    slot_anchor = _build_slot_anchor_tensor(
        anchor_slot, anchor_pixel, transform_meta, img_size, patch_size
    )

    rows: List[List[int]] = []
    for q in slots:
        # Same ordering the per-query path builds: [query] + others in temporal order.
        rows.append([q] + [i for i in range(num_slots) if i != q])

    results: List[np.ndarray] = []
    for start in range(0, len(rows), max(1, batch_chunk)):
        chunk = rows[start : start + max(1, batch_chunk)]
        order = torch.tensor(chunk, dtype=torch.long, device=device)  # [b, S]
        batch_images = image_tensor[order]  # [b, S, 3, H, W]

        anchor_tensor = None
        if slot_anchor is not None:
            # Permute the slot-order anchor vector into each row's ordering.
            row_anchor = np.stack([slot_anchor[np.asarray(r)] for r in chunk], axis=0)
            anchor_tensor = torch.from_numpy(row_anchor).long().to(device)

        with torch.no_grad():
            outputs = model(batch_images, anchor_patch_idx=anchor_tensor, query_frame_idx=0)

        costmap = outputs["costmap"][:, 0, :, :, 0].detach().cpu().float().numpy()
        results.extend(costmap[b].astype(np.float32) for b in range(costmap.shape[0]))

    return results


def _normalize_and_upsample_costmap(costmap: np.ndarray, upsample_size: int, normalize: bool) -> np.ndarray:
    if normalize:
        if np.isfinite(costmap).any():
            finite = costmap[np.isfinite(costmap)]
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
        else:
            min_val, max_val = 0.0, 0.0

        if max_val > min_val:
            costmap = (costmap - min_val) / (max_val - min_val)
        else:
            costmap = np.zeros_like(costmap, dtype=np.float32)

    costmap = np.nan_to_num(costmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    upsampled = cv2.resize(costmap, (upsample_size, upsample_size), interpolation=cv2.INTER_CUBIC)
    return upsampled.astype(np.float32)
