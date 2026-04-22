"""Build hook for dmx-compress. All metadata is in pyproject.toml.

The CUDA kernel (dmx_cuda.so/.dll) is compiled separately by CI
using nvcc directly, then included as package data. No torch build
dependency, no C++ ABI coupling.

For local development with nvcc available:
    nvcc -shared -o dmx/dmx_cuda.so kernel/dmx_standalone.cu -O2 --compiler-options -fPIC
    pip install -e .
"""
from setuptools import setup
setup()
