"""Build hook for standalone CUDA kernel. All package metadata is in pyproject.toml.

Compiles kernel/dmx_standalone.cu into dmx_cuda.so/.dll using nvcc directly.
No torch C++ headers — the kernel is loaded via ctypes at runtime, so it
works with ANY torch version. Zero ABI coupling.

Build requires only: nvcc (CUDA toolkit). Does NOT require torch at build time.
"""
import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools import Extension


class CUDABuildExt(_build_ext):
    """Custom build_ext that compiles .cu files with nvcc directly."""

    def build_extensions(self):
        for ext in self.extensions:
            self._build_cuda(ext)

    def _build_cuda(self, ext):
        source = ext.sources[0]
        build_dir = Path(self.build_lib)

        if platform.system() == "Windows":
            output_name = "dmx_cuda.dll"
        else:
            output_name = "dmx_cuda.so"

        # Output goes into the dmx/ package directory
        output_dir = build_dir / "dmx"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        # Also copy to source tree for editable installs
        src_output = Path("dmx") / output_name

        # Determine CUDA architectures
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")
        gencode_flags = []
        for arch in arch_list.replace(" ", "").split(";"):
            if arch:
                compute = arch.replace(".", "")
                gencode_flags.extend([
                    f"-gencode=arch=compute_{compute},code=sm_{compute}",
                ])

        # Build command
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("[dmx-compress] nvcc not found — skipping CUDA kernel build")
            return

        if platform.system() == "Windows":
            cmd = [
                nvcc, "-shared", "-o", str(output_path),
                source, "-O2",
            ] + gencode_flags
        else:
            cmd = [
                nvcc, "-shared", "-o", str(output_path),
                source, "-O2", "--compiler-options", "-fPIC",
            ] + gencode_flags

        print(f"[dmx-compress] Building standalone CUDA kernel:")
        print(f"  {' '.join(cmd)}")

        try:
            subprocess.check_call(cmd)
            print(f"[dmx-compress] Built: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

            # Copy to source for editable installs
            if not src_output.exists() or src_output.stat().st_size != output_path.stat().st_size:
                shutil.copy2(str(output_path), str(src_output))

        except subprocess.CalledProcessError as e:
            print(f"[dmx-compress] nvcc compilation failed (exit {e.returncode})")
            print(f"[dmx-compress] Falling back to pure Python (slow path)")
        except Exception as e:
            print(f"[dmx-compress] Build error: {e}")


def _nvcc_available():
    return shutil.which("nvcc") is not None


ext_modules = []
cmdclass = {}

if _nvcc_available() or os.environ.get("FORCE_CUDA", "0") == "1":
    cuda_source = os.path.join("kernel", "dmx_standalone.cu")
    if os.path.exists(cuda_source):
        ext_modules.append(Extension(
            name="dmx.dmx_cuda",
            sources=[cuda_source],
        ))
        cmdclass["build_ext"] = CUDABuildExt
        print(f"[dmx-compress] Will build standalone CUDA kernel: {cuda_source}")
else:
    print("[dmx-compress] nvcc not found — pure Python wheel")

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
