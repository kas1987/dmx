"""Standalone CUDA kernel loader — no torch C++ ABI dependency.

Loads the pre-compiled dmx_cuda.so/.dll via ctypes. Works with any
torch version since there's no C++ extension ABI coupling.

The kernel is compiled from kernel/dmx_standalone.cu using nvcc only
(no torch headers needed). It exposes a C API:
  - dmx_bfp_decompress(exponents, mantissas, output, n, group_size, m_bits, stream)
  - dmx_version()
  - dmx_cuda_available()
"""

import ctypes
import os
import sys
import platform
from pathlib import Path
from typing import Optional

import torch

_lib: Optional[ctypes.CDLL] = None
_lib_checked = False


def _find_library() -> Optional[Path]:
    """Find the pre-compiled dmx_cuda shared library."""
    if platform.system() == "Windows":
        lib_name = "dmx_cuda.dll"
    else:
        lib_name = "dmx_cuda.so"

    # Search order:
    # 1. Same directory as this file (installed in package)
    # 2. Package site-packages root
    # 3. Adjacent to dmx_cli.py
    candidates = [
        Path(__file__).parent / lib_name,
        Path(__file__).parent.parent / lib_name,
    ]

    # Also check next to dmx_cli.py
    try:
        import dmx_cli
        cli_dir = Path(dmx_cli.__file__).parent
        candidates.append(cli_dir / lib_name)
        candidates.append(cli_dir / "dmx" / lib_name)
    except ImportError:
        pass

    # Check site-packages directly
    for sp in sys.path:
        candidates.append(Path(sp) / lib_name)

    for p in candidates:
        if p.exists():
            return p

    return None


def get_native_lib():
    """Load the standalone CUDA library. Returns ctypes.CDLL or None."""
    global _lib, _lib_checked
    if _lib_checked:
        return _lib
    _lib_checked = True

    path = _find_library()
    if path is None:
        return None

    try:
        _lib = ctypes.CDLL(str(path))

        # Set up function signatures
        _lib.dmx_bfp_decompress.restype = ctypes.c_int
        _lib.dmx_bfp_decompress.argtypes = [
            ctypes.c_void_p,  # d_exponents
            ctypes.c_void_p,  # d_mantissas
            ctypes.c_void_p,  # d_output
            ctypes.c_int,     # num_elements
            ctypes.c_int,     # group_size
            ctypes.c_int,     # mantissa_bits
            ctypes.c_void_p,  # stream
        ]

        _lib.dmx_version.restype = ctypes.c_int
        _lib.dmx_version.argtypes = []

        _lib.dmx_cuda_available.restype = ctypes.c_int
        _lib.dmx_cuda_available.argtypes = []

        return _lib

    except OSError as e:
        # Library found but can't load (missing CUDA runtime, etc.)
        return None


def bfp_decompress(exponents: torch.Tensor, mantissas: torch.Tensor,
                   group_size: int, mantissa_bits: int) -> torch.Tensor:
    """Decompress BFP data using the native CUDA kernel.

    Drop-in replacement for dmx_cuda_v2.bfp_decompress() but loaded
    via ctypes instead of torch C++ extension (no ABI coupling).

    Args:
        exponents: uint8 tensor on CUDA [n_groups]
        mantissas: uint8 tensor on CUDA [num_elements]
        group_size: BFP group size (32)
        mantissa_bits: mantissa precision (1-7)

    Returns:
        int16 tensor on CUDA [num_elements] (reinterpret as float16)
    """
    lib = get_native_lib()
    if lib is None:
        raise RuntimeError("dmx_cuda native library not found")

    assert exponents.is_cuda and mantissas.is_cuda, "Tensors must be on CUDA"
    assert exponents.dtype == torch.uint8 and mantissas.dtype == torch.uint8

    n = mantissas.numel()
    output = torch.empty(n, dtype=torch.int16, device=mantissas.device)

    # Get raw data pointers
    exp_ptr = exponents.data_ptr()
    mant_ptr = mantissas.data_ptr()
    out_ptr = output.data_ptr()

    # Get current CUDA stream
    stream = torch.cuda.current_stream().cuda_stream

    err = lib.dmx_bfp_decompress(
        ctypes.c_void_p(exp_ptr),
        ctypes.c_void_p(mant_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int(n),
        ctypes.c_int(group_size),
        ctypes.c_int(mantissa_bits),
        ctypes.c_void_p(stream),
    )

    if err != 0:
        raise RuntimeError(f"dmx_bfp_decompress failed with CUDA error {err}")

    return output.view(torch.float16)
