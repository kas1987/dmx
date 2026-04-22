"""Build hook for dmx-compress. All metadata is in pyproject.toml.

When dmx/dmx_cuda.so or dmx/dmx_cuda.dll is present (compiled by CI),
forces a platform-specific wheel so pip installs the right binary for
each OS. Otherwise produces a pure Python wheel (fallback).
"""
import os
from setuptools import setup, Distribution


class BinaryDistribution(Distribution):
    """Force platform-specific wheel when pre-compiled CUDA binary is present."""
    def has_ext_modules(self):
        return True


# Detect if the CUDA binary exists in the package
dmx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dmx')
has_binary = (
    os.path.exists(os.path.join(dmx_dir, 'dmx_cuda.so')) or
    os.path.exists(os.path.join(dmx_dir, 'dmx_cuda.dll'))
)

setup(
    distclass=BinaryDistribution if has_binary else Distribution,
)
