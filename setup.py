"""Build script for dmx-compress with optional CUDA kernels.

When CUDA + nvcc are available, builds the native BFP decompress kernel
into the wheel. When they're not (CPU-only machines), builds a pure
Python wheel — the runtime falls back to vectorized torch ops.

Users never need to compile anything. The pre-built wheel from PyPI
includes the native kernel for their platform.
"""
import os
import sys
from setuptools import setup

# Try to build CUDA extension. If CUDA/nvcc aren't available, skip silently.
ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel")
        cuda_source = os.path.join(kernel_dir, "dmx_kernels_v2.cu")

        if os.path.exists(cuda_source):
            ext_modules.append(
                CUDAExtension(
                    name="dmx_cuda_v2",
                    sources=[cuda_source],
                )
            )
            cmdclass["build_ext"] = BuildExtension
            print(f"[dmx-compress] Building with CUDA kernel: {cuda_source}")
        else:
            print(f"[dmx-compress] CUDA source not found at {cuda_source}, skipping kernel build")
    else:
        print("[dmx-compress] CUDA not available, building pure Python wheel")
except ImportError:
    print("[dmx-compress] PyTorch not installed, building pure Python wheel")

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
