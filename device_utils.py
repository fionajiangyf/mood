"""
Device Utilities Module

This module provides cross-platform device detection and management for PyTorch.
It automatically selects the best available device (CUDA, MPS, or CPU) and provides
utility functions for memory management across different backends.
"""

import gc
import os
import platform

# Enable MPS fallback to CPU for unsupported operations (must be set before importing torch)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


def _is_mps_fully_available() -> bool:
    """
    Check if MPS is truly available and functional.
    This checks both that MPS is built and that it's actually usable.
    Also checks that we're running native ARM, not under Rosetta emulation.
    """
    # Check if running under Rosetta 2 emulation (x86_64 on ARM Mac)
    # MPS doesn't work properly under Rosetta
    import subprocess
    try:
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        arch = result.stdout.strip()
        if arch == 'x86_64' and platform.system() == 'Darwin':
            # Running under Rosetta emulation - MPS won't work
            return False
    except Exception:
        pass
    
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        return False
    
    # Try to actually use MPS to verify it works
    try:
        test_tensor = torch.zeros(1, device="mps")
        del test_tensor
        return True
    except Exception:
        return False


def get_device() -> str:
    """
    Get the best available compute device.
    
    Returns:
        str: "cuda" if NVIDIA GPU available, "mps" if Apple Silicon, otherwise "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif _is_mps_fully_available():
        return "mps"
    else:
        return "cpu"


def get_accelerator() -> str:
    """
    Get the accelerator string for PyTorch Lightning Trainer.
    
    Returns:
        str: "gpu" for CUDA, "mps" for Apple Silicon, "cpu" for CPU
    """
    if torch.cuda.is_available():
        return "gpu"
    elif _is_mps_fully_available():
        return "mps"
    else:
        return "cpu"


def get_devices_for_trainer():
    """
    Get the devices argument for PyTorch Lightning Trainer.
    
    Returns:
        list or str: [0] for GPU, 1 for MPS, "auto" for CPU
    """
    if torch.cuda.is_available():
        return [0]
    elif _is_mps_fully_available():
        return 1
    else:
        return "auto"


def clear_memory():
    """
    Clear GPU/MPS memory cache and run garbage collection.
    Works across CUDA, MPS, and CPU backends.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but we can synchronize
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except AttributeError:
            # Older PyTorch versions may not have these methods
            pass


def to_device(tensor_or_model, device: str = None):
    """
    Move a tensor or model to the specified device.
    
    Args:
        tensor_or_model: PyTorch tensor or model
        device: Target device (if None, uses get_device())
        
    Returns:
        The tensor or model on the target device
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


# Set default device globally
DEVICE = get_device()
ACCELERATOR = get_accelerator()

