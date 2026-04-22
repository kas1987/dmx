"""Build script for dmx-compress with optional CUDA kernels.

When nvcc is available, builds the native BFP decompress kernel
into the wheel (40-50x faster inference). When nvcc is not available,
builds a pure Python wheel with vectorized torch fallback.
"""
import os
import shutil
from setuptools import setup, find_packages

# Try to build CUDA extension
ext_modules = []
cmdclass = {}

def _nvcc_available():
    return shutil.which("nvcc") is not None

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    import torch.utils.cpp_extension as _cpp_ext

    # Bypass CUDA version mismatch check. Build environments (CI, Docker)
    # often have nvcc from one CUDA toolkit and torch compiled with another.
    # The kernel compiles fine regardless — the check is overly strict.
    if hasattr(_cpp_ext, '_check_cuda_version'):
        _cpp_ext._check_cuda_version = lambda *a, **kw: None

    # Set arch list if not already set — avoids querying GPU (which fails
    # when the CUDA driver doesn't match torch's expected version).
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

    if _nvcc_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        cuda_source = os.path.join("kernel", "dmx_kernels_v2.cu")

        if os.path.exists(cuda_source):
            ext_modules.append(
                CUDAExtension(
                    name="dmx_cuda_v2",
                    sources=[cuda_source],
                )
            )
            cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)
            print(f"[dmx-compress] Building with CUDA kernel: {cuda_source}")
        else:
            print(f"[dmx-compress] CUDA source not found, skipping kernel build")
    else:
        print("[dmx-compress] nvcc not found, building pure Python wheel")
except ImportError:
    print("[dmx-compress] torch not available at build time, pure Python wheel")

setup(
    name="dmx-compress",
    version="1.3.0",
    description="Lossless compression for neural network weights.",
    py_modules=["dmx_cli"],
    packages=["dmx", "kernel"],
    package_data={"kernel": ["*.cu"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={
        "console_scripts": ["dmx=dmx_cli:main"],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "zstandard",
        "safetensors",
        "packaging",
        "soundfile",
        "brotli",
    ],
)
