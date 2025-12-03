"""CUDA device management utilities."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: Prefer CUDA if available

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info() -> None:
    """Print information about available devices."""
    print("\nDevice Information:")
    print("-" * 40)

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")
    else:
        print("CUDA available: No")
        print("Using CPU")

    print("-" * 40)


def get_memory_usage() -> dict:
    """Get current GPU memory usage.

    Returns:
        Dict with allocated and cached memory in GB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "cached": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "cached": torch.cuda.memory_reserved() / 1024**3,
    }


def clear_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
