"""Build hook for CUDA kernels. All package metadata is in pyproject.toml.

When nvcc is available, builds the native BFP decompress kernel
into the wheel (40-50x faster inference). When nvcc is not available,
builds a pure Python wheel with vectorized torch fallback.
"""
import os
import shutil
from setuptools import setup

ext_modules = []
cmdclass = {}

def _nvcc_available():
    return shutil.which("nvcc") is not None

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    import torch.utils.cpp_extension as _cpp_ext

    # Bypass CUDA version mismatch check for CI/Docker builds.
    if hasattr(_cpp_ext, '_check_cuda_version'):
        _cpp_ext._check_cuda_version = lambda *a, **kw: None

    # Set arch list to avoid querying GPU at build time.
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
except ImportError:
    pass

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
