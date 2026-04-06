"""
Build and load DMX CUDA kernels via torch.utils.cpp_extension.

Usage:
    from kernel.build import dmx_cuda
    # Then: dmx_cuda.quantize_int16(tensor, scale)
    #        dmx_cuda.delta_compute_i16(base, target, scale)
    #        dmx_cuda.bfp_compress(fp16_tensor, group_size, mantissa_bits)
    #        etc.
"""
import os
import torch
from torch.utils.cpp_extension import load

_kernel_dir = os.path.dirname(os.path.abspath(__file__))
_source = os.path.join(_kernel_dir, "dmx_kernels_v2.cu")

dmx_cuda = None

def get_kernels():
    """Load CUDA kernels, compiling on first use."""
    global dmx_cuda
    if dmx_cuda is None:
        if not torch.cuda.is_available():
            raise RuntimeError("DMX CUDA kernels require a CUDA-capable GPU")
        dmx_cuda = load(
            name="dmx_cuda_v2",
            sources=[_source],
            verbose=False,
        )
    return dmx_cuda
