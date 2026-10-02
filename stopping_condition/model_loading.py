"""Base VGGT depth model for the online stopping rule.

VGGTMetricModel loads base VGGT from a local checkpoint or a Hugging Face name,
runs it on a nine-frame window resized to 518x518, and returns the depth of the
center frame at the input size. run_nav.py loads it once when
online_stopping.enabled=true and passes it to OnlineStoppingCondition.
"""

import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import sys


class VGGTMetricModel(nn.Module):
    def __init__(self, pretrained_model="facebook/VGGT-1B"):
        super().__init__()
        # Lazy import VGGT dependencies
        sys.path.append(str(Path(__file__).parent.parent))
        from vggt.models.vggt import VGGT
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading VGGT model from {pretrained_model}...")
        checkpoint_path = Path(pretrained_model)
        if checkpoint_path.is_file():
            self.vggt_model = VGGT()
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self.vggt_model.load_state_dict(state_dict, strict=True)
        else:
            self.vggt_model = VGGT.from_pretrained(pretrained_model)
        self.vggt_model = self.vggt_model.to(self.device).eval()

    @torch.inference_mode()
    def infer(self, rgb_sequence: list):
        """
        :param rgb_sequence: List of 9 RGB numpy arrays (H, W, 3) representing the context window.
        :return: Depth map of the center frame as a numpy array.
        """
        assert len(rgb_sequence) == 9, "VGGT expects exactly 9 frames for this pipeline."
        
        # Determine target size based on first image
        h, w, c = rgb_sequence[0].shape
        
        # Convert HWC to CHW tensor
        tensors = []
        for img in rgb_sequence:
            tensors.append(torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.device)) # Shape: (3, H, W)
        
        # Stack to (S, 3, H, W)
        image_tensor_4d = torch.stack(tensors, dim=0).float() / 255.0
        
        # VGGT expects images whose dimensions are multiples of 14 (default 518x518).
        # We resize the sequence here before adding the batch dimension.
        target_size = (518, 518)
        image_tensor_4d = F.interpolate(image_tensor_4d, size=target_size, mode='bilinear', align_corners=False)
        
        # Add batch dim -> (1, S, 3, H_target, W_target)
        image_tensor = image_tensor_4d.unsqueeze(0)
        
        # Run inference
        predictions = self.vggt_model(image_tensor)
        
        # Extract depth maps [B, S, H, W, 1]
        depth_maps = predictions["depth"]
        
        # Get the center frame's depth (index 4 out of 0-8)
        center_depth = depth_maps[0, 4, ..., 0]  # Shape: (H, W)
        
        # Resize if necessary
        dh, dw = center_depth.shape
        if dh != h or dw != w:
            center_depth = center_depth.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            center_depth = F.interpolate(center_depth, size=(h, w), mode='nearest')
            center_depth = center_depth.squeeze(0).squeeze(0)
            
        return center_depth.cpu().numpy()
