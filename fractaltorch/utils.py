import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for python, numpy and torch and enable deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Make cudnn deterministic where possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # Newer PyTorch: enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def estimate_parameters_bytes(model, dtype=torch.float32):
    """Estimate total bytes required for model parameters for a given dtype."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_elems = 0
    for p in model.parameters():
        total_elems += p.numel()
    return total_elems * bytes_per_elem


def move_model_to_gpu_with_vram_limit(model, device=None, max_vram_bytes=5 * 1024 ** 3):
    """
    Move a PyTorch model to GPU within a maximum weight VRAM budget.
    Strategy:
      - Estimate parameter bytes in float32. If <= limit, move as float32.
      - Otherwise convert model weights to float16 (half) and move. If still too large, raise.

    Returns a tuple (model, dtype_used, estimated_bytes).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if device.type != 'cuda':
        # no GPU available, just keep on CPU
        return model.to(device), torch.float32, estimate_parameters_bytes(model, torch.float32)

    # estimate as float32
    est32 = estimate_parameters_bytes(model, torch.float32)
    if est32 <= max_vram_bytes:
        model = model.to(device, dtype=torch.float32)
        return model, torch.float32, est32

    # try float16
    est16 = estimate_parameters_bytes(model, torch.float16)
    if est16 <= max_vram_bytes:
        # convert to half then move
        model.half()
        model = model.to(device)
        return model, torch.float16, est16

    # As a fallback, attempt parameter offloading (not implemented): raise with guidance
    raise RuntimeError(f"Model parameters exceed VRAM limit: float32={est32} bytes, float16={est16} bytes, limit={max_vram_bytes} bytes.\nConsider reducing model size or implementing parameter sharding/offloading.")


__all__ = ["set_seed", "estimate_parameters_bytes", "move_model_to_gpu_with_vram_limit"]
