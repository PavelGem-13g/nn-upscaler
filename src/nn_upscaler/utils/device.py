"""Hardware acceleration utilities supporting Apple Silicon (MPS), CUDA, and CPU."""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Detect and return the best available compute device.

    Args:
        preferred: Optional explicit device string ('mps', 'cuda', 'cpu').

    Returns:
        torch.device instance.
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device | None = None) -> str:
    """Return human-readable device description."""
    if device is None:
        device = get_device()

    dev_type = device.type
    if dev_type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if dev_type == "mps":
        return "Apple Silicon (MPS - Metal Performance Shaders)"
    return "CPU"


def clear_device_cache(device: torch.device | None = None) -> None:
    """Safely flush memory cache for the active device."""
    if device is None:
        device = get_device()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
