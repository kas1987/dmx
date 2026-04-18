"""
Lossless delta compression for neural network weight tensors.

Encodes the bitwise difference between two tensors (base, target) using
integer subtraction on the raw bit representation, byte-plane transposition,
and per-plane zstd compression.  Decoding is the exact reverse.

Guarantees: torch.equal(decode(base, encode(base, target)), target)
for float16, bfloat16, and float32 tensors (including inf/nan).
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import torch
import zstandard as zstd

# ── constants ────────────────────────────────────────────────────────────────

MAGIC = b"DMXL"
VERSION = 1

_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

_DTYPE_TO_INT = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
}

_DTYPE_TO_NP_UINT = {
    torch.float16: np.uint16,
    torch.bfloat16: np.uint16,
    torch.float32: np.uint32,
}

ZSTD_LEVEL = 19


# ── helpers ──────────────────────────────────────────────────────────────────

def _bitcast_to_int(tensor: torch.Tensor) -> np.ndarray:
    """Bitcast a float tensor to a flat numpy unsigned-int array."""
    int_dtype = _DTYPE_TO_INT[tensor.dtype]
    np_uint = _DTYPE_TO_NP_UINT[tensor.dtype]
    # view() is zero-copy bit reinterpretation
    int_tensor = tensor.contiguous().view(int_dtype)
    # Convert signed int16/int32 numpy array to unsigned for wrapping arithmetic
    return int_tensor.numpy().ravel().view(np_uint)


def _byte_planes(arr: np.ndarray) -> list[np.ndarray]:
    """Split an unsigned integer array into per-byte planes."""
    raw = arr.tobytes()
    n = arr.size
    nbytes = arr.itemsize  # 2 for uint16, 4 for uint32
    # Reshape so each element occupies one row of `nbytes` columns
    mat = np.frombuffer(raw, dtype=np.uint8).reshape(n, nbytes)
    # Transpose: one plane per byte position (column-major extraction)
    return [np.ascontiguousarray(mat[:, i]) for i in range(nbytes)]


def _merge_byte_planes(planes: list[np.ndarray], n: int, np_uint) -> np.ndarray:
    """Inverse of _byte_planes: reassemble unsigned-int array from byte planes."""
    nbytes = len(planes)
    mat = np.column_stack(planes)  # (n, nbytes)
    return np.frombuffer(mat.tobytes(), dtype=np_uint)


# ── public API ───────────────────────────────────────────────────────────────

def delta_lossless_encode(base: torch.Tensor, target: torch.Tensor) -> bytes:
    """Compute a lossless compressed delta from *base* to *target*.

    Parameters
    ----------
    base, target : torch.Tensor
        Must have identical shape and dtype.
        dtype must be float16, bfloat16, or float32.

    Returns
    -------
    bytes
        Compressed delta payload.  Feed to :func:`delta_lossless_decode`
        together with the same *base* to recover *target* bit-exactly.
    """
    if base.shape != target.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {target.shape}")
    if base.dtype != target.dtype:
        raise ValueError(f"Dtype mismatch: {base.dtype} vs {target.dtype}")
    if base.dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype {base.dtype}")

    dtype_code = _DTYPE_TO_CODE[base.dtype]
    np_uint = _DTYPE_TO_NP_UINT[base.dtype]

    # 1. Bitcast to unsigned int
    base_int = _bitcast_to_int(base)
    target_int = _bitcast_to_int(target)

    # 2. Wrapping integer subtraction (numpy unsigned naturally wraps)
    delta_int = (target_int.astype(np_uint) - base_int.astype(np_uint)).astype(np_uint)

    # 3. Byte-plane transposition
    planes = _byte_planes(delta_int)

    # 4. Compress each plane with zstd
    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed_planes = [compressor.compress(p.tobytes()) for p in planes]

    # 5. Pack into wire format
    shape = base.shape
    parts: list[bytes] = []
    parts.append(MAGIC)
    parts.append(struct.pack("B", VERSION))
    parts.append(struct.pack("B", dtype_code))
    parts.append(struct.pack("<H", len(shape)))
    for dim in shape:
        parts.append(struct.pack("<I", dim))
    parts.append(struct.pack("B", len(compressed_planes)))
    for cp in compressed_planes:
        parts.append(struct.pack("<I", len(cp)))
        parts.append(cp)

    return b"".join(parts)


def delta_lossless_decode(base: torch.Tensor, delta_bytes: bytes) -> torch.Tensor:
    """Reconstruct *target* from *base* and the encoded delta.

    Parameters
    ----------
    base : torch.Tensor
        The same base tensor used during encoding.
    delta_bytes : bytes
        Output of :func:`delta_lossless_encode`.

    Returns
    -------
    torch.Tensor
        Reconstructed target, bit-exact match to the original.
    """
    buf = memoryview(delta_bytes)
    off = 0

    # Magic
    if bytes(buf[off:off + 4]) != MAGIC:
        raise ValueError("Bad magic")
    off += 4

    # Version
    version = struct.unpack_from("B", buf, off)[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version {version}")
    off += 1

    # Dtype
    dtype_code = struct.unpack_from("B", buf, off)[0]
    off += 1
    float_dtype = _CODE_TO_DTYPE[dtype_code]
    int_dtype = _DTYPE_TO_INT[float_dtype]
    np_uint = _DTYPE_TO_NP_UINT[float_dtype]

    # Shape
    shape_len = struct.unpack_from("<H", buf, off)[0]
    off += 2
    shape = []
    for _ in range(shape_len):
        shape.append(struct.unpack_from("<I", buf, off)[0])
        off += 4
    shape = tuple(shape)

    # Validate base
    if base.shape != shape:
        raise ValueError(f"Base shape {base.shape} != encoded shape {shape}")

    # Number of planes
    num_planes = struct.unpack_from("B", buf, off)[0]
    off += 1

    # Decompress planes
    decompressor = zstd.ZstdDecompressor()
    planes: list[np.ndarray] = []
    n = int(np.prod(shape))
    for _ in range(num_planes):
        plane_len = struct.unpack_from("<I", buf, off)[0]
        off += 4
        plane_data = decompressor.decompress(bytes(buf[off:off + plane_len]))
        off += plane_len
        planes.append(np.frombuffer(plane_data, dtype=np.uint8))

    # Merge byte planes back to delta int array
    delta_int = _merge_byte_planes(planes, n, np_uint)

    # Bitcast base to unsigned int
    base_int = _bitcast_to_int(base)

    # Wrapping addition to recover target
    target_int = (base_int.astype(np_uint) + delta_int.astype(np_uint)).astype(np_uint)

    # Convert back to signed int for torch, then bitcast to float
    if np_uint == np.uint16:
        signed = target_int.view(np.int16)
    else:
        signed = target_int.view(np.int32)

    result_tensor = torch.from_numpy(signed.copy()).view(int_dtype).reshape(shape).view(float_dtype)
    return result_tensor


# ── verification ─────────────────────────────────────────────────────────────

def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Check bitwise equality (handles NaN correctly)."""
    int_dtype = _DTYPE_TO_INT[a.dtype]
    a_int = a.contiguous().view(int_dtype)
    b_int = b.contiguous().view(int_dtype)
    return torch.equal(a_int, b_int)


def verify_lossless():
    """Run 24+ test cases and print pass/fail for each."""
    torch.manual_seed(42)
    results = []

    def run_case(name: str, base: torch.Tensor, target: torch.Tensor):
        try:
            delta_bytes = delta_lossless_encode(base, target)
            reconstructed = delta_lossless_decode(base, delta_bytes)
            ok = _bitwise_equal(target, reconstructed)
            status = "PASS" if ok else "FAIL"
            results.append((name, status, len(delta_bytes)))
            print(f"  {status}  {name:50s}  delta={len(delta_bytes):>8d} bytes")
        except Exception as e:
            results.append((name, "FAIL", 0))
            print(f"  FAIL  {name:50s}  ERROR: {e}")

    print("=" * 80)
    print("Lossless Delta Verification")
    print("=" * 80)

    # --- 10 random FP32 cases ---
    for i in range(10):
        base = torch.randn(256, 256, dtype=torch.float32)
        target = base + torch.randn_like(base) * 0.01
        run_case(f"fp32_random_{i:02d}", base, target)

    # --- 10 random FP16 cases ---
    for i in range(10):
        base = torch.randn(256, 256, dtype=torch.float16)
        target = base + torch.randn(256, 256, dtype=torch.float16) * 0.01
        run_case(f"fp16_random_{i:02d}", base, target)

    # --- Edge cases ---
    # Zeros
    base_z = torch.zeros(64, 64, dtype=torch.float32)
    run_case("fp32_zeros", base_z, base_z.clone())

    # Identical tensors
    base_id = torch.randn(128, dtype=torch.float16)
    run_case("fp16_identical", base_id, base_id.clone())

    # Inf values
    base_inf = torch.tensor([1.0, -1.0, float('inf'), float('-inf')], dtype=torch.float32)
    target_inf = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0], dtype=torch.float32)
    run_case("fp32_inf", base_inf, target_inf)

    # NaN values (must use bitwise check)
    base_nan = torch.tensor([float('nan'), 0.0, float('nan'), 1.0], dtype=torch.float32)
    target_nan = torch.tensor([0.0, float('nan'), float('nan'), float('nan')], dtype=torch.float32)
    run_case("fp32_nan", base_nan, target_nan)

    # BFloat16
    base_bf = torch.randn(128, 128, dtype=torch.bfloat16)
    target_bf = base_bf + torch.randn(128, 128, dtype=torch.bfloat16) * 0.01
    run_case("bf16_random", base_bf, target_bf)

    # Mixed inf/nan fp16
    base_mix = torch.tensor([float('inf'), float('nan'), 0.0, -0.0], dtype=torch.float16)
    target_mix = torch.tensor([float('nan'), float('inf'), -0.0, 0.0], dtype=torch.float16)
    run_case("fp16_inf_nan_mix", base_mix, target_mix)

    print("=" * 80)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    verify_lossless()
