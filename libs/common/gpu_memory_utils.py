"""Clear CUDA memory during navigation.

run_nav.py calls clear_gpu_cache between steps.
"""

import torch


def clear_gpu_cache():
    """Clear the GPU cache to free unreferenced memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
