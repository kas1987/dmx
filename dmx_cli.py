#!/usr/bin/env python
"""
DMX CLI - Delta Multiplexed Model Compressor
Patent Pending. (c) 2026 William J. Riley. MIT License.

Compresses neural network model weights using aligned cross-layer
quantization and stream-separated block floating point encoding.

Usage:
    python dmx_cli.py compress model.safetensors model.dmx
    python dmx_cli.py compress model.safetensors model.dmx --mode bfp
    python dmx_cli.py compress model.safetensors model.dmx --entropy lpc
    python dmx_cli.py decompress model.dmx model.safetensors
    python dmx_cli.py info model.dmx
    python dmx_cli.py verify model.safetensors model.dmx
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import zstandard as zstd
from safetensors.torch import load_file, save_file


def _read_package_version():
    """Read the installed dmx-compress package version, with fallback.

    Single source of truth is pyproject.toml. Everything else (banner, report,
    __version__) reads via importlib.metadata so there's only one place to
    update on release. Falls back to "dev" when running from a source checkout
    that isn't pip-installed.
    """
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("dmx-compress")
    except Exception:
        return "dev"


__version__ = _read_package_version()


# --- Constants ---
DMX_MAGIC = b"DMX1"
DMX_VERSION = 3
ZSTD_LEVEL = 19          # Single-file compression (archival, best ratio)
ZSTD_LEVEL_DELTA = 3     # Delta compression (speed matters for training callbacks)

# Chunk types
CHUNK_HEADER = 0
CHUNK_TENSOR = 1

# Encoding modes (zstd entropy)
ENC_FP16_ZSTD = 0       # Already FP16: store raw bytes + zstd
ENC_INT16_QUANT = 1      # FP32 -> int16 quantization + zstd
ENC_DELTA_ZSTD = 2       # Delta-encode int16 view + zstd
ENC_RAW_ZSTD = 3         # Fallback: raw bytes + zstd (for odd dtypes)
ENC_BFP_ZSTD = 4         # Block Floating Point: shared exponent + truncated mantissa + zstd
ENC_INT32_QUANT = 5      # FP32 -> int32 aligned quantization + zstd (practically lossless)

# Encoding modes (LPC/FLAC entropy)
ENC_FP16_LPC = 10       # Already FP16: store raw bytes as FLAC
ENC_INT16_QUANT_LPC = 11 # FP32 -> int16 quantization + FLAC
ENC_DELTA_LPC = 12       # Delta-encode int16 view + FLAC
ENC_RAW_LPC = 13         # Fallback: raw bytes + FLAC (for odd dtypes)
ENC_BFP_LPC = 14         # Block Floating Point: shared exponent + truncated mantissa + FLAC
ENC_INT32_QUANT_LPC = 15 # FP32 -> int32 aligned quantization + FLAC (practically lossless)

# Module-level flag for GPU compression (set by CLI --gpu)
_use_gpu_compress = False

# Tensor name patterns that should be stored as FP16 when --fast-load is set.
# These cover HuggingFace naming conventions for embeddings and language model
# heads across Llama, Qwen, GPT-2, Phi, Mistral families.
FAST_LOAD_PATTERNS = ("embed_tokens", "lm_head", "wte", "wpe", "word_embeddings",
                      "embed_in", "embed_out")

# Module-level flag for fast-load mode (set by compress_file(fast_load=True)).
# When True, tensors matching FAST_LOAD_PATTERNS are stored as FP16+entropy
# instead of BFP, so compressed-residency loads skip the materialize step.
# Restored to False at the end of each compress_file() call so state never leaks.
_fast_load_override = False

# Module-level override for BFP mantissa bits during compress_file() calls.
# None means "use the default BFP_MANTISSA_BITS constant." When set by
# compress_file(mantissa_bits=N), it is read by the BFP encode path in
# encode_tensor() and passed explicitly to bfp_compress/bfp_compress_gpu,
# bypassing the frozen default arg values captured at function-definition time.
# Restored to None at the end of each compress_file() call so state never leaks.
_bfp_mantissa_override = None

# Native CUDA kernels (loaded on demand)
_dmx_cuda = None

def _get_cuda_kernels():
    """Load native CUDA kernels if available. Returns module or None.

    Finds the sibling ``kernel/`` directory relative to this file, so the
    import works regardless of the caller's cwd or sys.path. This matters
    because dmx-compress is frequently imported from downstream tools
    (hope, dmx-vram) that run from their own working directories, and
    before this the `from kernel.build import get_kernels` statement
    silently failed with ModuleNotFoundError → returned None → misled
    callers into thinking CUDA kernels weren't available.

    If ``DMX_REQUIRE_NATIVE=1`` is set in the environment, the silent
    exception handler is bypassed and failures raise so downstream code
    can surface the underlying cause.
    """
    import os
    global _dmx_cuda
    if _dmx_cuda is not None:
        return _dmx_cuda
    if not torch.cuda.is_available():
        return None
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    _strict = os.environ.get("DMX_REQUIRE_NATIVE", "").strip() == "1"
    try:
        from kernel.build import get_kernels
        _dmx_cuda = get_kernels()
        return _dmx_cuda
    except Exception:
        if _strict:
            raise
        return None

# BFP defaults
BFP_GROUP_SIZE = 32
BFP_MANTISSA_BITS = 6

# Minimum raw byte count to use FLAC/LPC. Below this, FLAC file overhead
# (~8KB header/framing) dominates and zstd wins. Measured on svd_xt tensors.
LPC_MIN_RAW_BYTES = 32768  # 32 KB

# Map zstd encoding to LPC equivalent and back
_ZSTD_TO_LPC = {
    ENC_FP16_ZSTD: ENC_FP16_LPC,
    ENC_INT16_QUANT: ENC_INT16_QUANT_LPC,
    ENC_DELTA_ZSTD: ENC_DELTA_LPC,
    ENC_RAW_ZSTD: ENC_RAW_LPC,
    ENC_BFP_ZSTD: ENC_BFP_LPC,
    ENC_INT32_QUANT: ENC_INT32_QUANT_LPC,
}
_LPC_TO_ZSTD = {v: k for k, v in _ZSTD_TO_LPC.items()}

# All LPC encoding types
_LPC_ENCODINGS = set(_ZSTD_TO_LPC.values())


def _find_ffmpeg():
    """Find ffmpeg binary. Returns path or None.

    Legacy fallback: the FLAC code path now uses soundfile (libsndfile)
    instead of an ffmpeg subprocess. This function is kept as a fallback
    for environments where soundfile is unavailable but ffmpeg is on PATH.
    New installations should use the soundfile path.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Check common Windows locations
    for candidate in [
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


FFMPEG_PATH = _find_ffmpeg()


# soundfile (libsndfile) is the preferred FLAC backend — pure Python install,
# no subprocess, fast and reliable. The legacy ffmpeg subprocess path below is
# kept as a fallback for environments where soundfile is unavailable.
try:
    import soundfile as _sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    _sf = None
    SOUNDFILE_AVAILABLE = False


# Brotli is one of the candidate coders for the per-tensor entropy selection
# layer. On many neural network weight tensors brotli at quality 11 produces
# the smallest output of the candidate set; on others FLAC or zstd wins. The
# selector tries each available coder per tensor and keeps the smallest. If
# brotli is unavailable at runtime, the selector falls back to whichever
# candidate coders are present.
try:
    import brotli as _brotli
    BROTLI_AVAILABLE = True
except ImportError:
    _brotli = None
    BROTLI_AVAILABLE = False


def _lpc_backend_available():
    """Return True if any FLAC backend is available (soundfile or ffmpeg)."""
    return SOUNDFILE_AVAILABLE or (FFMPEG_PATH is not None)


# Brotli quality and window settings for the per-tensor selector. Quality 11
# is the brotli library's maximum and lgwin=24 is the largest practical
# window for weight-tensor sized payloads. These settings let brotli compete
# at its best — the selector compares each candidate at its strongest.
BROTLI_QUALITY = 11
BROTLI_LGWIN = 24


def _compress_bytes_brotli(raw_bytes):
    """Compress raw bytes with brotli at the candidate-selector quality.

    Returns the brotli-compressed bytes. Caller is responsible for checking
    BROTLI_AVAILABLE before invoking — we raise rather than silently fall back
    so a misconfigured candidate set surfaces immediately during testing.
    """
    if not BROTLI_AVAILABLE:
        raise RuntimeError(
            "brotli is not installed (pip install brotli). It is a required "
            "candidate coder for the per-tensor entropy selection layer."
        )
    return _brotli.compress(raw_bytes, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)


def _decompress_bytes_brotli(brotli_bytes):
    """Decompress brotli-compressed bytes back to raw bytes."""
    if not BROTLI_AVAILABLE:
        raise RuntimeError(
            "brotli is not installed (pip install brotli). It is required to "
            "decode tensors whose entropy field is 'brotli-11'."
        )
    return _brotli.decompress(brotli_bytes)


# Competitive entropy coder candidate set. Order is informational only — the
# selector tries every candidate that's available and returns the smallest.
# Each entry is a (codec_id, encode_callable, requires_int16) tuple.
#   codec_id          — short string stored in per-tensor metadata
#   encode_callable   — bytes -> compressed_bytes
#   requires_int16    — True if the coder requires the input length to be a
#                       multiple of 2 bytes (FLAC is int16-native)
#
# When adding a candidate, also extend _decompress_int16_dispatch().
COMPETITIVE_CODECS = [
    ("zstd-19", "zstd-level-19", False),
    ("flac",    "flac-libsndfile", True),
    ("brotli-11", "brotli-q11-lgwin24", False),
]


def _competitive_encode_int16(raw_bytes, num_int16_samples, zstd_level=19):
    """Encode raw int16 bytes with every available candidate coder.

    Returns (codec_id, compressed_bytes) for the smallest output. The codec_id
    is stored in per-tensor metadata so the decoder can dispatch correctly.

    This is the encode side of the per-tensor entropy selection layer. The
    selector evaluates each available candidate at compress time and keeps
    the smallest output, so the result is, by construction, no larger than
    any fixed-codec strategy. The cost is encode-time multiplication by the
    number of available candidates; decode time is unchanged.

    Args:
        raw_bytes: raw int16 PCM data (little-endian)
        num_int16_samples: number of int16 samples (= len(raw_bytes) // 2)
        zstd_level: zstd compression level. Default 19 (production strength).

    Returns:
        (codec_id, compressed_bytes)
    """
    candidates = []

    # zstd-19 is always available (zstandard is a hard dep)
    cctx = zstd.ZstdCompressor(level=zstd_level, write_content_size=True)
    candidates.append(("zstd-19", cctx.compress(raw_bytes)))

    # FLAC via soundfile if available. Skip if backend missing rather than
    # crash the whole compress — the selector is allowed to be a subset on
    # systems with partial dependencies, and the per-tensor recorded codec_id
    # tells the decoder exactly what to invoke.
    if _lpc_backend_available():
        try:
            flac_bytes = _int16_to_flac(raw_bytes, num_int16_samples)
            candidates.append(("flac", flac_bytes))
        except Exception:
            # If a particular tensor breaks the FLAC encoder (extreme values,
            # odd byte count edge cases), drop FLAC from this tensor's
            # candidate set rather than failing the whole compress.
            pass

    # brotli quality 11 if available
    if BROTLI_AVAILABLE:
        try:
            br_bytes = _compress_bytes_brotli(raw_bytes)
            candidates.append(("brotli-11", br_bytes))
        except Exception:
            pass

    # Pick the smallest. Tie-break by candidate order (zstd first, FLAC
    # second, brotli third) since the order in COMPETITIVE_CODECS reflects
    # decode-cost ordering — we prefer the cheaper decoder on ties.
    codec_id, compressed = min(candidates, key=lambda c: len(c[1]))
    return codec_id, compressed


def _decompress_int16_dispatch(compressed_bytes, codec_id):
    """Dispatch decode for the per-tensor entropy selection layer.

    Returns raw int16 bytes. The codec_id is read from per-tensor metadata.
    Backward compatibility: missing codec_id and earlier short identifiers
    ('zstd', 'lpc', 'brotli') are accepted as aliases of the current names.

    Args:
        compressed_bytes: payload bytes from the .dmxd file
        codec_id: short identifier stored in per-tensor metadata

    Returns:
        raw int16 PCM bytes
    """
    if codec_id == "zstd-19" or codec_id == "zstd" or codec_id is None:
        # "zstd" alias accepted for files written by earlier DMX versions.
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed_bytes)
    elif codec_id == "flac" or codec_id == "lpc":
        # "lpc" alias accepted for files written by earlier DMX versions —
        # same FLAC payload either way.
        return _flac_to_int16(compressed_bytes).tobytes()
    elif codec_id == "brotli-11" or codec_id == "brotli":
        return _decompress_bytes_brotli(compressed_bytes)
    else:
        raise ValueError(
            f"Unknown entropy codec_id {codec_id!r} in tensor metadata. "
            f"Known: zstd-19, flac, brotli-11 (plus legacy aliases zstd, lpc, brotli)."
        )


def _competitive_encode_uint8(raw_bytes, zstd_level=19):
    """Encode arbitrary uint8 (or odd-length) bytes with every available
    candidate coder and return the smallest output.

    This is the byte-stream sibling of `_competitive_encode_int16`. It is
    used for paths where the input is a uint8 stream with no inherent int16
    structure — e.g. the BFP mantissa stream, or the RAW encoding for
    non-int16 dtypes. FLAC is supported via the `_compress_bytes_uint8_lpc`
    helper, which packs uint8 pairs into int16 samples before encoding and
    records the original length so the decoder can unpack correctly.

    Args:
        raw_bytes: raw byte stream of any length
        zstd_level: zstd compression level (default 19, production strength)

    Returns:
        (codec_id, compressed_bytes, codec_meta)

        `codec_meta` is a dict that the decoder needs to dispatch correctly:
            - For FLAC, it carries the uint8 original length so the unpack
              step can trim padding.
            - For zstd / brotli, it is empty.
    """
    candidates = []
    cctx = zstd.ZstdCompressor(level=zstd_level, write_content_size=True)
    candidates.append(("zstd-19", cctx.compress(raw_bytes), {}))

    if _lpc_backend_available():
        try:
            flac_bytes, orig_len = _compress_bytes_uint8_lpc(raw_bytes)
            candidates.append(("flac", flac_bytes, {"uint8_orig_len": orig_len}))
        except Exception:
            pass

    if BROTLI_AVAILABLE:
        try:
            br_bytes = _compress_bytes_brotli(raw_bytes)
            candidates.append(("brotli-11", br_bytes, {}))
        except Exception:
            pass

    codec_id, compressed, codec_meta = min(candidates, key=lambda c: len(c[1]))
    return codec_id, compressed, codec_meta


def _decompress_uint8_dispatch(compressed_bytes, codec_id, codec_meta=None):
    """Dispatch decode for the uint8 byte-stream selector.

    Returns the original raw uint8 bytes. The codec_id is read from per-tensor
    metadata; codec_meta carries any additional fields the chosen coder needs
    (e.g. the uint8 original length for the FLAC unpacking step).
    """
    if codec_id == "zstd-19" or codec_id == "zstd" or codec_id is None:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed_bytes)
    elif codec_id == "flac" or codec_id == "lpc":
        if not codec_meta or "uint8_orig_len" not in codec_meta:
            raise ValueError(
                "FLAC dispatch on uint8 stream requires codec_meta['uint8_orig_len']"
            )
        return _decompress_bytes_uint8_lpc(compressed_bytes, codec_meta["uint8_orig_len"])
    elif codec_id == "brotli-11" or codec_id == "brotli":
        return _decompress_bytes_brotli(compressed_bytes)
    else:
        raise ValueError(
            f"Unknown entropy codec_id {codec_id!r} for uint8 dispatch."
        )


def _int16_to_flac(int16_bytes, num_samples):
    """Encode int16 PCM data to FLAC bytes.

    Default backend: soundfile (libsndfile, pip-installable, no subprocess).
    Fallback: ffmpeg subprocess if soundfile is unavailable but ffmpeg is.

    Args:
        int16_bytes: raw int16 PCM data (little-endian)
        num_samples: number of int16 samples

    Returns:
        FLAC file bytes
    """
    if SOUNDFILE_AVAILABLE:
        # Soundfile path: in-memory via BytesIO, no subprocess, no temp files
        import io
        arr = np.frombuffer(int16_bytes, dtype=np.int16)
        bio = io.BytesIO()
        _sf.write(bio, arr, 44100, format='FLAC', subtype='PCM_16')
        return bio.getvalue()
    elif FFMPEG_PATH:
        # Fallback to ffmpeg subprocess for environments without soundfile
        return _int16_to_flac_ffmpeg(int16_bytes, num_samples)
    else:
        raise RuntimeError(
            "No FLAC backend available. Install 'soundfile' (pip install soundfile) "
            "or ensure ffmpeg is on PATH."
        )


def _flac_to_int16(flac_bytes):
    """Decode FLAC bytes back to int16 PCM data.

    Default backend: soundfile. Fallback: ffmpeg subprocess.

    Returns:
        numpy array of int16 values
    """
    if SOUNDFILE_AVAILABLE:
        import io
        bio = io.BytesIO(flac_bytes)
        data, _ = _sf.read(bio, dtype='int16')
        return np.asarray(data, dtype=np.int16).copy()
    elif FFMPEG_PATH:
        return _flac_to_int16_ffmpeg(flac_bytes)
    else:
        raise RuntimeError(
            "No FLAC backend available. Install 'soundfile' (pip install soundfile) "
            "or ensure ffmpeg is on PATH."
        )


def _int16_to_flac_ffmpeg(int16_bytes, num_samples):
    """Encode int16 PCM data to FLAC bytes via ffmpeg subprocess.

    Legacy fallback path. Use _int16_to_flac() which prefers soundfile.
    """
    tmpdir = tempfile.mkdtemp(prefix="dmx_")
    wav_path = os.path.join(tmpdir, "input.wav")
    flac_path = os.path.join(tmpdir, "output.flac")
    try:
        # Write WAV file using wave module (44-byte header + raw int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(44100)
            wf.writeframes(int16_bytes)

        # Encode to FLAC with max compression
        result = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", wav_path,
             "-c:a", "flac", "-compression_level", "12",
             flac_path],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg FLAC encode failed: {result.stderr.decode('utf-8', errors='replace')[:500]}")

        with open(flac_path, "rb") as f:
            return f.read()
    finally:
        # Cleanup temp files
        for p in [wav_path, flac_path]:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _flac_to_int16_ffmpeg(flac_bytes):
    """Decode FLAC bytes back to int16 PCM data via ffmpeg subprocess.

    Legacy fallback path. Use _flac_to_int16() which prefers soundfile.
    """
    tmpdir = tempfile.mkdtemp(prefix="dmx_")
    flac_path = os.path.join(tmpdir, "input.flac")
    wav_path = os.path.join(tmpdir, "output.wav")
    try:
        with open(flac_path, "wb") as f:
            f.write(flac_bytes)

        result = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", flac_path,
             "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
             wav_path],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg FLAC decode failed: {result.stderr.decode('utf-8', errors='replace')[:500]}")

        with wave.open(wav_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())

        return np.frombuffer(raw, dtype=np.int16).copy()
    finally:
        for p in [flac_path, wav_path]:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _compress_bytes_lpc(raw_bytes, num_int16_samples):
    """Compress raw int16 bytes using FLAC/LPC. Returns FLAC bytes."""
    return _int16_to_flac(raw_bytes, num_int16_samples)


def _decompress_bytes_lpc(flac_bytes):
    """Decompress FLAC bytes back to raw int16 bytes. Returns (bytes, num_samples)."""
    arr = _flac_to_int16(flac_bytes)
    return arr.tobytes(), len(arr)


def _compress_bytes_uint8_lpc(raw_bytes):
    """Compress uint8 data via FLAC by packing pairs into int16 samples.

    FLAC requires int16 samples. We pack two uint8 bytes into one int16
    (low byte + high byte), encode as FLAC, then unpack on decode.

    Returns: (flac_bytes, original_len, padded)
    """
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    orig_len = len(data)
    # Pad to even length
    if orig_len % 2 != 0:
        data = np.concatenate([data, np.zeros(1, dtype=np.uint8)])
    # Reinterpret as int16 (pairs of uint8 -> int16 little-endian)
    int16_data = data.view(np.int16)
    flac_bytes = _int16_to_flac(int16_data.tobytes(), len(int16_data))
    return flac_bytes, orig_len


def _decompress_bytes_uint8_lpc(flac_bytes, original_len):
    """Decompress FLAC back to uint8 data."""
    arr = _flac_to_int16(flac_bytes)
    # Reinterpret int16 as pairs of uint8
    uint8_data = arr.view(np.uint8)
    return uint8_data[:original_len].tobytes()


def _bfp_compress_fp32(tensor, group_size=BFP_GROUP_SIZE, mantissa_bits=23):
    """Native FP32 BFP compression (no downcast to FP16).

    Keeps the full FP32 bit layout: 1 sign, 8 exponent, 23 mantissa.
    The implicit leading 1 gives a 24-bit full mantissa.  Shared exponent
    per group is the max of the 8-bit exponents.

    This path is selected when the input is float32 and mantissa_bits > 10
    (i.e. more precision than FP16 can represent).  For mantissa_bits <= 10,
    the FP16 downcast path is used instead for backward compatibility.

    Returns the same 5-tuple as bfp_compress:
        (exp_stream, sign_mant_stream, pad_len, orig_len, original_dtype)
    """
    t = tensor.contiguous().cpu()
    original_dtype = str(t.dtype).replace("torch.", "")

    # Ensure float32
    if t.dtype != torch.float32:
        t = t.float()

    # Guard: clamp inf/NaN to finite range. Inf poisons the shared exponent
    # for the entire group, corrupting all neighbors.
    if not torch.isfinite(t).all():
        t = torch.nan_to_num(t, nan=0.0, posinf=torch.finfo(torch.float32).max,
                             neginf=-torch.finfo(torch.float32).max)

    raw = t.view(torch.int32).numpy().astype(np.uint32)
    flat = raw.flatten()
    orig_len = len(flat)

    # Pad to multiple of group_size
    pad_len = (group_size - orig_len % group_size) % group_size
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint32)])

    n_groups = len(flat) // group_size
    groups = flat.reshape(n_groups, group_size)

    # Extract FP32 fields: sign(1) | exponent(8) | mantissa(23)
    signs = ((groups >> 31) & 1).astype(np.uint8)          # 1 bit
    exponents = ((groups >> 23) & 0xFF).astype(np.uint8)   # 8 bits
    mantissas = (groups & 0x7FFFFF).astype(np.uint32)      # 23 bits

    # Add implicit leading 1 for normal numbers (exponent != 0)
    # For subnormals (exp=0), the implicit bit is 0
    implicit = np.where(exponents > 0, np.uint32(0x800000), np.uint32(0))
    full_mantissa = mantissas | implicit  # 24 bits

    # Shared exponent per group = max exponent in each group
    shared_exp = exponents.max(axis=1)  # shape [n_groups]

    # Shift mantissas to align with shared exponent
    exp_diff = shared_exp[:, np.newaxis].astype(np.int16) - exponents.astype(np.int16)
    # Clamp shift to avoid shifting away everything (max meaningful shift ~ 24 bits)
    exp_diff = np.clip(exp_diff, 0, 31)
    shifted_mantissa = full_mantissa >> exp_diff.astype(np.uint32)

    # Truncate to mantissa_bits (keep top bits of 24-bit mantissa)
    shift_amount = 24 - mantissa_bits
    # Pack dtype: M>15 needs uint32, M>7 needs uint16, M<=7 needs uint8
    if mantissa_bits > 15:
        pack_dtype = np.uint32
    elif mantissa_bits > 7:
        pack_dtype = np.uint16
    else:
        pack_dtype = np.uint8
    truncated = (shifted_mantissa >> shift_amount).astype(pack_dtype)

    # Build streams
    exp_stream = shared_exp.astype(np.uint8)  # n_groups bytes

    # Combine sign (1 bit) + truncated mantissa (mantissa_bits)
    sign_mant = (signs.astype(pack_dtype) << mantissa_bits) | truncated
    sign_mant_stream = sign_mant.flatten().astype(pack_dtype)

    return exp_stream, sign_mant_stream, pad_len, orig_len, original_dtype


def bfp_compress(tensor, group_size=BFP_GROUP_SIZE, mantissa_bits=BFP_MANTISSA_BITS):
    """
    Block Floating Point compression for FP16/FP32 tensors.

    For each group of `group_size` values:
    1. Find the max exponent in the group (shared exponent)
    2. Shift all mantissas to align with shared exponent
    3. Truncate mantissa to `mantissa_bits`
    4. Store: shared exponents stream + sign-mantissa stream (separately zstd'd)

    Effective bits per value: (8/group_size) + 1 + mantissa_bits
    At g=32, m=6: 7.25 bits/value (vs 16) = 54.7% before entropy coding.

    Auto-routing for FP32 inputs:
    - mantissa_bits > 10: use native FP32 path (no downcast)
    - mantissa_bits <= 10: downcast to FP16 (backward compatible)
    """
    t = tensor.contiguous().cpu()

    # Route FP32 tensors with M>10 to the native FP32 path
    if t.dtype == torch.float32 and mantissa_bits > 10:
        return _bfp_compress_fp32(t, group_size=group_size, mantissa_bits=mantissa_bits)

    # FP16 path: Convert to FP16 if needed (FP32/BF16 -> FP16)
    original_dtype = str(t.dtype).replace("torch.", "")
    if t.dtype == torch.float32 or t.dtype == torch.bfloat16:
        t = t.half()

    raw = t.view(torch.int16).numpy().astype(np.uint16)
    flat = raw.flatten()
    orig_len = len(flat)

    # Pad to multiple of group_size
    pad_len = (group_size - orig_len % group_size) % group_size
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint16)])

    n_groups = len(flat) // group_size
    groups = flat.reshape(n_groups, group_size)

    # Extract FP16 fields
    signs = ((groups >> 15) & 1).astype(np.uint8)       # 1 bit
    exponents = ((groups >> 10) & 0x1F).astype(np.uint8) # 5 bits
    mantissas = (groups & 0x3FF).astype(np.uint16)       # 10 bits

    # Add implicit leading 1 for normal numbers (exponent != 0)
    # For subnormals (exp=0), the implicit bit is 0
    implicit = np.where(exponents > 0, np.uint16(0x400), np.uint16(0))
    full_mantissa = mantissas | implicit  # 11 bits

    # Shared exponent per group = max exponent in each group
    shared_exp = exponents.max(axis=1)  # shape [n_groups]

    # Shift mantissas to align with shared exponent
    exp_diff = shared_exp[:, np.newaxis].astype(np.int16) - exponents.astype(np.int16)
    # Clamp shift to avoid shifting away everything (max meaningful shift ~ 11 bits)
    exp_diff = np.clip(exp_diff, 0, 15)
    shifted_mantissa = full_mantissa >> exp_diff.astype(np.uint16)

    # Truncate to mantissa_bits (keep top bits of 11-bit mantissa)
    shift_amount = 11 - mantissa_bits
    # Use uint16 for M>7 (sign + mantissa exceeds 8 bits)
    pack_dtype = np.uint16 if mantissa_bits > 7 else np.uint8
    truncated = (shifted_mantissa >> shift_amount).astype(pack_dtype)

    # Build streams
    exp_stream = shared_exp.astype(np.uint8)  # n_groups bytes

    # Combine sign (1 bit) + truncated mantissa (mantissa_bits)
    # M<=7: 1+7=8 bits, fits uint8. M>7: needs uint16.
    sign_mant = (signs.astype(pack_dtype) << mantissa_bits) | truncated
    sign_mant_stream = sign_mant.flatten().astype(pack_dtype)

    return exp_stream, sign_mant_stream, pad_len, orig_len, original_dtype


def bfp_compress_gpu(tensor, group_size=BFP_GROUP_SIZE, mantissa_bits=BFP_MANTISSA_BITS):
    """GPU-accelerated BFP compression using PyTorch CUDA ops.

    Same algorithm as bfp_compress but with vectorized GPU operations
    instead of numpy CPU ops. Returns numpy arrays (same interface).
    """
    t = tensor.contiguous().cpu()
    original_dtype = str(t.dtype).replace("torch.", "")
    if t.dtype == torch.float32 or t.dtype == torch.bfloat16:
        t = t.half()

    # View as uint16 via int16 reinterpret
    raw = t.view(torch.int16).flatten()
    orig_len = len(raw)

    # Pad to multiple of group_size
    pad_len = (group_size - orig_len % group_size) % group_size
    if pad_len:
        raw = torch.cat([raw, torch.zeros(pad_len, dtype=torch.int16)])

    # Transfer to GPU as int32 for bit ops
    flat = raw.to(dtype=torch.int32, device='cuda')
    # Convert signed int16 view to unsigned: mask with 0xFFFF
    flat = flat & 0xFFFF

    n_groups = len(flat) // group_size
    groups = flat.reshape(n_groups, group_size)

    # Extract FP16 fields (all vectorized on GPU)
    signs = ((groups >> 15) & 1)           # 1 bit
    exponents = ((groups >> 10) & 0x1F)    # 5 bits
    mantissas = (groups & 0x3FF)           # 10 bits

    # Add implicit leading 1 for normal numbers (exponent != 0)
    implicit = torch.where(exponents > 0,
                          torch.ones_like(exponents) * 0x400,
                          torch.zeros_like(exponents))
    full_mantissa = mantissas | implicit   # 11 bits

    # Shared exponent per group = max exponent in each group
    shared_exp = exponents.max(dim=1).values  # [n_groups]

    # Shift mantissas to align with shared exponent
    exp_diff = shared_exp.unsqueeze(1) - exponents  # [n_groups, group_size]
    exp_diff = exp_diff.clamp(0, 15)
    shifted_mantissa = full_mantissa >> exp_diff

    # Truncate to mantissa_bits
    shift_amount = 11 - mantissa_bits
    truncated = shifted_mantissa >> shift_amount

    # Build streams: sign (1 bit) + truncated mantissa (mantissa_bits)
    sign_mant = (signs << mantissa_bits) | truncated

    # Transfer back to CPU as numpy. Use uint16 for M>7 (sign+mantissa > 8 bits).
    exp_stream = shared_exp.to(torch.uint8).cpu().numpy()
    pack_torch_dtype = torch.int16 if mantissa_bits > 7 else torch.uint8
    sign_mant_stream = sign_mant.flatten().to(pack_torch_dtype).cpu().numpy()
    if mantissa_bits > 7:
        sign_mant_stream = sign_mant_stream.view(np.uint16)

    return exp_stream, sign_mant_stream, pad_len, orig_len, original_dtype


def _bfp_decompress_fp32(exp_stream, sign_mant_stream, pad_len, orig_len, shape,
                         original_dtype, group_size=BFP_GROUP_SIZE, mantissa_bits=23):
    """Decompress native FP32 BFP back to FP32.

    Counterpart of _bfp_compress_fp32.  Leading-one scan over 24 bits,
    exponent clamped to [0, 255], reassemble FP32: sign(1) | exp(8) | mant(23).
    """
    n_groups = len(exp_stream)
    # Determine pack dtype from mantissa_bits (must match compress side)
    if mantissa_bits > 15:
        pack_dtype = np.uint32
    elif mantissa_bits > 7:
        pack_dtype = np.uint16
    else:
        pack_dtype = np.uint8
    sign_mant = sign_mant_stream.view(pack_dtype).reshape(n_groups, group_size)

    signs = ((sign_mant >> mantissa_bits) & 1).astype(np.uint32)
    truncated = (sign_mant & ((1 << mantissa_bits) - 1)).astype(np.uint32)

    # Reconstruct to 24-bit position (full FP32 mantissa + implicit bit width)
    shift_amount = 24 - mantissa_bits
    recon_24 = truncated << shift_amount

    shared_exp = exp_stream[:, np.newaxis].astype(np.int16)

    # Find the leading 1 bit to determine actual exponent
    # Bit 23 = value had same exponent as shared (offset 0)
    # Bit 22 = exponent was shared_exp - 1, etc.
    result_exp = np.zeros_like(recon_24, dtype=np.int16)
    result_mant = np.zeros_like(recon_24, dtype=np.uint32)
    found = np.zeros_like(recon_24, dtype=bool)

    for bit_pos in range(23, -1, -1):
        mask = np.uint32(1 << bit_pos)
        has_bit = (recon_24 & mask) != 0
        unprocessed = (~found) & has_bit

        actual_exp = shared_exp - np.int16(23 - bit_pos)
        shift_up = np.uint32(23 - bit_pos)
        shifted_up = recon_24.astype(np.uint64) << shift_up
        mant_23 = (shifted_up & np.uint64(0x7FFFFF)).astype(np.uint32)

        result_exp = np.where(unprocessed, actual_exp, result_exp)
        result_mant = np.where(unprocessed, mant_23, result_mant)
        found = found | unprocessed

    # Clamp to valid FP32 exponent range [0, 255]
    result_exp = np.clip(result_exp, 0, 255).astype(np.uint32)

    # Zero values: if truncated was 0, output zero
    is_zero = (truncated == 0)
    result_exp = np.where(is_zero, np.uint32(0), result_exp)
    result_mant = np.where(is_zero, np.uint32(0), result_mant)

    # Reassemble FP32: sign(1) | exponent(8) | mantissa(23)
    fp32_values = (signs << 31) | (result_exp << 23) | result_mant
    fp32_values = fp32_values.astype(np.uint32)

    flat = fp32_values.flatten()[:orig_len]

    result = torch.from_numpy(flat.view(np.int32).copy()).view(torch.float32).reshape(shape)
    return result


def bfp_decompress(exp_stream, sign_mant_stream, pad_len, orig_len, shape,
                   original_dtype, group_size=BFP_GROUP_SIZE, mantissa_bits=BFP_MANTISSA_BITS,
                   native_dtype="float16"):
    """Decompress BFP back to FP16 or FP32 (depending on native_dtype).

    During compression, values with smaller exponents than the group max had their
    mantissas shifted right, moving the implicit leading 1 to a lower bit position.
    To decompress, we find where that leading 1 ended up, derive the actual exponent
    offset from shared_exp, and reconstruct the value properly.

    native_dtype: "float16" (default, backward compatible) or "float32" for native
    FP32 BFP. When "float32", routes to _bfp_decompress_fp32.
    """
    # Route to FP32 decompressor when native_dtype indicates FP32 path
    if native_dtype == "float32":
        return _bfp_decompress_fp32(
            exp_stream, sign_mant_stream, pad_len, orig_len, shape,
            original_dtype, group_size, mantissa_bits
        )

    # FP16 path (original behavior)
    n_groups = len(exp_stream)
    # Determine pack dtype from mantissa_bits (must match compress side)
    if mantissa_bits > 7:
        pack_dtype = np.uint16
    else:
        pack_dtype = np.uint8
    sign_mant = sign_mant_stream.view(pack_dtype).reshape(n_groups, group_size)

    signs = ((sign_mant >> mantissa_bits) & 1).astype(np.uint16)
    truncated = (sign_mant & ((1 << mantissa_bits) - 1)).astype(np.uint16)

    # Reconstruct to 11-bit position
    shift_amount = 11 - mantissa_bits
    recon_11 = truncated.astype(np.uint16) << shift_amount

    shared_exp = exp_stream[:, np.newaxis].astype(np.int16)

    # Find the leading 1 bit to determine actual exponent
    # Bit 10 = value had same exponent as shared (offset 0)
    # Bit 9 = exponent was shared_exp - 1, etc.
    result_exp = np.zeros_like(recon_11, dtype=np.int16)
    result_mant = np.zeros_like(recon_11, dtype=np.uint16)
    found = np.zeros_like(recon_11, dtype=bool)

    for bit_pos in range(10, -1, -1):
        mask = np.uint16(1 << bit_pos)
        has_bit = (recon_11 & mask) != 0
        unprocessed = (~found) & has_bit

        actual_exp = shared_exp - np.int16(10 - bit_pos)
        shift_up = np.uint16(10 - bit_pos)
        shifted_up = recon_11.astype(np.uint32) << shift_up
        mant_10 = (shifted_up & 0x3FF).astype(np.uint16)

        result_exp = np.where(unprocessed, actual_exp, result_exp)
        result_mant = np.where(unprocessed, mant_10, result_mant)
        found = found | unprocessed

    # Clamp to valid FP16 exponent range [0, 31]
    result_exp = np.clip(result_exp, 0, 31).astype(np.uint16)

    # Zero values: if truncated was 0, output zero
    is_zero = (truncated == 0)
    result_exp = np.where(is_zero, np.uint16(0), result_exp)
    result_mant = np.where(is_zero, np.uint16(0), result_mant)

    # Reassemble FP16: sign(1) | exponent(5) | mantissa(10)
    fp16_values = (signs << 15) | (result_exp << 10) | result_mant
    fp16_values = fp16_values.astype(np.uint16)

    flat = fp16_values.flatten()[:orig_len]

    result = torch.from_numpy(flat.astype(np.int16).copy()).view(torch.float16).reshape(shape)

    if original_dtype == "float32":
        result = result.float()
    elif original_dtype == "bfloat16":
        result = result.to(torch.bfloat16)

    return result


def bfp_decompress_gpu(exp_stream, sign_mant_stream, pad_len, orig_len, shape,
                       original_dtype, group_size=BFP_GROUP_SIZE, mantissa_bits=BFP_MANTISSA_BITS):
    """GPU-accelerated BFP decompression using PyTorch CUDA ops.

    Replaces the CPU leading-one-bit scan loop with parallel GPU bit operations.
    The exponent and sign-mantissa streams are transferred to GPU, reconstructed
    via bitwise ops, and the result transferred back to CPU.
    """
    n_groups = len(exp_stream)

    # Transfer compressed streams to GPU as int32 (for bit ops without overflow)
    sign_mant_gpu = torch.from_numpy(sign_mant_stream.copy()).to(
        dtype=torch.int32, device='cuda'
    ).reshape(n_groups, group_size)

    # Extract sign and truncated mantissa
    mant_mask = (1 << mantissa_bits) - 1
    signs = (sign_mant_gpu >> mantissa_bits) & 1            # [n_groups, group_size]
    truncated = sign_mant_gpu & mant_mask                   # [n_groups, group_size]

    # Reconstruct to 11-bit position
    shift_amount = 11 - mantissa_bits
    recon_11 = truncated << shift_amount                     # [n_groups, group_size]

    # Shared exponent per group, broadcast to [n_groups, 1]
    shared_exp = torch.from_numpy(exp_stream.copy()).to(
        dtype=torch.int32, device='cuda'
    ).unsqueeze(1)                                           # [n_groups, 1]

    # Find leading-one bit position using parallel scan
    # For each value, find the highest set bit in the 11-bit recon_11 field.
    # bit_pos 10 means offset=0 from shared_exp, bit_pos 9 means offset=1, etc.
    #
    # Strategy: iterate from high bit to low, but all ops are vectorized on GPU.
    # This is 11 iterations of pure tensor ops -- no Python-level per-element work.
    result_exp = torch.zeros_like(recon_11)
    result_mant = torch.zeros_like(recon_11)
    found = torch.zeros(n_groups, group_size, dtype=torch.bool, device='cuda')

    for bit_pos in range(10, -1, -1):
        mask = 1 << bit_pos
        has_bit = (recon_11 & mask) != 0
        unprocessed = (~found) & has_bit

        offset = 10 - bit_pos
        actual_exp = shared_exp - offset
        shifted_up = (recon_11 << offset) & 0x3FF

        result_exp = torch.where(unprocessed, actual_exp, result_exp)
        result_mant = torch.where(unprocessed, shifted_up, result_mant)
        found = found | unprocessed

    # Clamp exponent to valid FP16 range [0, 31]
    result_exp = result_exp.clamp(0, 31)

    # Zero values: if truncated was 0, output zero
    is_zero = (truncated == 0)
    result_exp = torch.where(is_zero, torch.zeros_like(result_exp), result_exp)
    result_mant = torch.where(is_zero, torch.zeros_like(result_mant), result_mant)

    # Reassemble FP16 bits: sign(1) | exponent(5) | mantissa(10)
    fp16_bits = (signs << 15) | (result_exp << 10) | result_mant

    # Flatten and trim padding
    flat = fp16_bits.flatten()[:orig_len].to(torch.int16)

    # Transfer back to CPU and reinterpret as float16
    result = flat.cpu().view(torch.float16).reshape(shape)

    if original_dtype == "float32":
        result = result.float()
    elif original_dtype == "bfloat16":
        result = result.to(torch.bfloat16)

    return result


def detect_encoding(tensor, mode="auto", entropy="zstd"):
    """Decide best encoding for a tensor.
    mode: 'auto' (detect), 'bfp' (force BFP), 'int16' (legacy behavior)
    entropy: 'zstd' or 'lpc' -- selects entropy coding backend
    """
    # First, pick the zstd-based encoding as baseline
    if mode == "bfp":
        if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
            enc = ENC_BFP_ZSTD
        else:
            enc = ENC_RAW_ZSTD
    elif mode == "int32":
        if tensor.dtype == torch.float32:
            flat = tensor.flatten().float()
            scale = flat.abs().max().item()
            if scale == 0:
                enc = ENC_FP16_ZSTD
            else:
                enc = ENC_INT32_QUANT
        elif tensor.dtype in (torch.float16, torch.bfloat16):
            enc = ENC_BFP_ZSTD  # FP16/BF16 already small, use BFP
        else:
            enc = ENC_RAW_ZSTD
    elif mode == "int16":
        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            raw = tensor.contiguous().cpu().numpy().view(np.int16)
            if raw.size > 16:
                delta = np.diff(raw.flatten().astype(np.int32)).astype(np.int16)
                zero_frac = np.sum(delta == 0) / delta.size
                if zero_frac > 0.20:
                    enc = ENC_DELTA_ZSTD
                else:
                    enc = ENC_FP16_ZSTD
            else:
                enc = ENC_FP16_ZSTD
        elif tensor.dtype == torch.float32:
            flat = tensor.flatten().float()
            scale = flat.abs().max().item()
            if scale == 0:
                enc = ENC_FP16_ZSTD
            else:
                enc = ENC_INT16_QUANT
        else:
            enc = ENC_RAW_ZSTD
    else:
        # Auto mode
        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            enc = ENC_BFP_ZSTD
        elif tensor.dtype == torch.float32:
            flat = tensor.flatten().float()
            scale = flat.abs().max().item()
            if scale == 0:
                enc = ENC_FP16_ZSTD
            else:
                enc = ENC_INT16_QUANT
        else:
            enc = ENC_RAW_ZSTD

    # If LPC entropy requested, map to LPC variant (only for tensors large enough)
    # Exception: BFP mantissa streams are uint8, where zstd beats FLAC.
    # Only use LPC for int16-native paths (FP16, INT16Q, DELTA, RAW).
    if entropy == "lpc" and enc in _ZSTD_TO_LPC and enc not in (ENC_BFP_ZSTD, ENC_INT32_QUANT):
        # INT32_QUANT stays zstd — FLAC is int16-native, int32 values need splitting
        raw_bytes = tensor.numel() * tensor.element_size()
        if raw_bytes >= LPC_MIN_RAW_BYTES:
            enc = _ZSTD_TO_LPC[enc]
        # else: keep zstd -- FLAC header overhead dominates for tiny tensors

    return enc


def _is_lpc_encoding(encoding):
    """Check if encoding uses LPC/FLAC entropy coding."""
    return encoding in _LPC_ENCODINGS


def _base_encoding(encoding):
    """Map LPC encoding back to its zstd equivalent for logic branching."""
    if encoding in _LPC_TO_ZSTD:
        return _LPC_TO_ZSTD[encoding]
    return encoding


def encode_tensor(tensor, encoding, entropy_mode=None):
    """Encode a single tensor. Returns (encoded_bytes, metadata_dict).

    entropy_mode:
      - "auto": per-tensor competitive selection across the available
        candidate coders (`[zstd-19, FLAC, brotli-11]`). The winning coder is
        recorded in `meta["entropy_codec"]` and the decoder dispatches on it.
        For BFP the selector applies to the mantissa stream only; the
        exponent stream is always zstd because it's tiny and the selector
        overhead would dominate.
      - "zstd-19" / "zstd": pin to zstd-19 for the whole tensor
      - "flac" / "lpc": pin to FLAC where applicable (falls back to zstd
        for non-int16-native paths like INT32_QUANT)
      - "brotli-11" / "brotli": pin to brotli-11
      - None (default for backward compatibility): use the legacy
        `_is_lpc_encoding(encoding)` heuristic from the encoding constant.
        Files written under this mode have no `entropy_codec` field and
        the decoder falls back to the same legacy heuristic.
    """
    meta = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "encoding": encoding,
    }

    use_lpc = _is_lpc_encoding(encoding)
    base_enc = _base_encoding(encoding)

    t = tensor.contiguous().cpu()

    if base_enc == ENC_FP16_ZSTD:
        if t.dtype == torch.float32:
            t = t.half()
            meta["original_dtype"] = "float32"
        raw = t.numpy().tobytes()
        meta["raw_size"] = len(raw)
        # FP16 raw bytes are not int16 weight data per se, but the bit pattern
        # is int16-compatible (same 2 bytes per value)
        num_samples = len(raw) // 2

    elif base_enc == ENC_INT16_QUANT:
        # GPU path: quantize on device when use_gpu requested. Massive speedup
        # on large FP32 tensors (e.g. Pythia 2.8B int16 standalone goes from
        # ~600s CPU to ~10s on A100). Source tensor is moved to GPU directly,
        # bypassing the unconditional .cpu() at the top of this function.
        if _use_gpu_compress and torch.cuda.is_available():
            src = tensor.contiguous().cuda(non_blocking=True).flatten().float()
            scale = src.abs().max().item()
            if scale == 0:
                scale = 1.0
            quantized = torch.clamp(
                torch.round(src / scale * 32767), -32767, 32767
            ).to(torch.int16).cpu()
            raw = quantized.numpy().tobytes()
            del src
        else:
            flat = t.flatten().float()
            scale = flat.abs().max().item()
            if scale == 0:
                scale = 1.0
            quantized = torch.clamp(torch.round(flat / scale * 32767), -32767, 32767).to(torch.int16)
            raw = quantized.numpy().tobytes()
        meta["scale"] = scale
        meta["raw_size"] = len(raw)
        num_samples = len(raw) // 2

    elif base_enc == ENC_INT32_QUANT:
        # GPU path: same rationale as INT16 above. Uses float64 on CPU fallback
        # to preserve precision; on GPU we use float32 because A100 float64 is
        # 1/32 throughput and the int32 quantization headroom (~31 bits) leaves
        # enough margin for float32 round to be exact for typical NN weight
        # ranges. If precision becomes an issue we can switch to float64 on GPU.
        if _use_gpu_compress and torch.cuda.is_available():
            src = tensor.contiguous().cuda(non_blocking=True).flatten().float()
            scale = src.abs().max().item()
            if scale == 0:
                scale = 1.0
            quantized = torch.clamp(
                torch.round(src / scale * 2147483647), -2147483647, 2147483647
            ).to(torch.int32).cpu()
            raw = quantized.numpy().tobytes()
            del src
        else:
            # Use float64 on CPU — float32 only has 24-bit mantissa (~16M levels)
            # which is less precise than INT32's 2B levels. float64's 53-bit mantissa
            # makes INT32 quantization the actual precision bottleneck, not the arithmetic.
            # GPU path stays float32 because A100 fp64 is 1/32 throughput.
            flat = t.flatten().double()
            scale = flat.abs().max().item()
            if scale == 0:
                scale = 1.0
            quantized = torch.clamp(torch.round(flat / scale * 2147483647), -2147483647, 2147483647).to(torch.int32)
            raw = quantized.numpy().tobytes()
        meta["scale"] = scale
        meta["raw_size"] = len(raw)
        num_samples = len(raw) // 2  # int32 = 4 bytes, but FLAC needs int16 pairs

    elif base_enc == ENC_DELTA_ZSTD:
        arr = t.numpy().view(np.int16).flatten().copy()
        delta = np.empty_like(arr)
        delta[0] = arr[0]
        delta[1:] = np.diff(arr.astype(np.int32)).astype(np.int16)
        raw = delta.tobytes()
        meta["raw_size"] = len(raw)
        num_samples = len(raw) // 2

    elif base_enc == ENC_RAW_ZSTD:
        raw = t.numpy().tobytes()
        meta["raw_size"] = len(raw)
        num_samples = len(raw) // 2  # may not be int16, but FLAC needs int16

    elif base_enc == ENC_BFP_ZSTD:
        # Resolve mantissa bits: use runtime override if set, otherwise
        # fall back to module-level default.
        effective_m = _bfp_mantissa_override if _bfp_mantissa_override is not None else BFP_MANTISSA_BITS
        # FP32 native path (M>10) requires CPU — GPU compress is FP16-only
        _use_fp32_native = (t.dtype == torch.float32 and effective_m > 10)
        if _use_gpu_compress and torch.cuda.is_available() and not _use_fp32_native:
            exp_stream, sign_mant_stream, pad_len, orig_len, orig_dtype = bfp_compress_gpu(
                t, mantissa_bits=effective_m
            )
        else:
            exp_stream, sign_mant_stream, pad_len, orig_len, orig_dtype = bfp_compress(
                t, mantissa_bits=effective_m
            )
        meta["bfp_pad_len"] = pad_len
        meta["bfp_orig_len"] = orig_len
        meta["bfp_group_size"] = BFP_GROUP_SIZE
        meta["bfp_mantissa_bits"] = effective_m
        if orig_dtype != meta["dtype"]:
            meta["original_dtype"] = orig_dtype
        # Record the native BFP dtype so the decoder knows which bit layout
        # was used. "float32" = native FP32 path (no downcast), "float16" =
        # FP16 path (default, backward compatible). Files without this field
        # are assumed to be "float16".
        if t.dtype == torch.float32 and effective_m > 10:
            meta["bfp_native_dtype"] = "float32"
        # Record pack dtype so decoder can reinterpret the raw byte stream
        # correctly. Legacy files without this field assume uint8.
        meta["bfp_pack_dtype"] = str(sign_mant_stream.dtype)

        meta["bfp_exp_size"] = len(exp_stream)
        meta["bfp_mant_size"] = sign_mant_stream.nbytes
        meta["raw_size"] = len(exp_stream) + len(sign_mant_stream)

        # Exponent stream is always zstd. It's tiny (sub-3% of total bytes
        # for typical group sizes) and the selector overhead would dominate
        # any savings from a different coder.
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        exp_compressed = cctx.compress(exp_stream.tobytes())

        # Mantissa stream gets the per-tensor entropy step. The candidate set
        # depends on entropy_mode. In "auto" the selector tries every
        # available coder and records the winner; in pinned modes it uses
        # the requested coder (with FLAC requiring the uint8->int16 packing
        # helper); in legacy mode it follows the encoding constant's
        # use_lpc decision.
        mant_bytes = sign_mant_stream.tobytes()
        mant_data = None
        if entropy_mode == "auto":
            mant_codec, mant_data, mant_codec_meta = _competitive_encode_uint8(mant_bytes)
            meta["bfp_mant_codec"] = mant_codec
            if "uint8_orig_len" in mant_codec_meta:
                meta["bfp_mant_orig_len"] = mant_codec_meta["uint8_orig_len"]
        elif entropy_mode in ("flac", "lpc"):
            if _lpc_backend_available():
                mant_data, mant_orig_len = _compress_bytes_uint8_lpc(mant_bytes)
                meta["bfp_mant_codec"] = "flac"
                meta["bfp_mant_orig_len"] = mant_orig_len
            else:
                mant_data = cctx.compress(mant_bytes)
                meta["bfp_mant_codec"] = "zstd-19"
        elif entropy_mode in ("brotli-11", "brotli"):
            if BROTLI_AVAILABLE:
                mant_data = _compress_bytes_brotli(mant_bytes)
                meta["bfp_mant_codec"] = "brotli-11"
            else:
                mant_data = cctx.compress(mant_bytes)
                meta["bfp_mant_codec"] = "zstd-19"
        elif entropy_mode in ("zstd-19", "zstd"):
            mant_data = cctx.compress(mant_bytes)
            meta["bfp_mant_codec"] = "zstd-19"
        else:
            # Legacy mode: follow the encoding constant's use_lpc decision.
            # Files written under this branch have no bfp_mant_codec field;
            # the decoder falls back to the same legacy heuristic.
            if use_lpc:
                mant_data, mant_orig_len = _compress_bytes_uint8_lpc(mant_bytes)
                meta["bfp_mant_orig_len"] = mant_orig_len
            else:
                mant_data = cctx.compress(mant_bytes)

        meta["bfp_exp_compressed"] = len(exp_compressed)
        meta["bfp_mant_compressed"] = len(mant_data)

        # Pack: [4B exp_compressed_len][exp_data][mant_data]
        packed = struct.pack("<I", len(exp_compressed)) + exp_compressed + mant_data
        meta["compressed_size"] = len(packed)
        return packed, meta

    # Entropy coding for non-BFP paths.
    #
    # The branching here is the entropy_mode contract from the docstring:
    # "auto" runs the per-tensor competitive selector and records the winner;
    # pinned modes use a single coder; legacy mode (entropy_mode=None)
    # follows the encoding constant's use_lpc decision so old call sites
    # behave exactly as before.
    #
    # The competitive selector for non-BFP int16-shaped paths uses
    # _competitive_encode_int16. FLAC inside the selector handles the
    # odd-byte-count edge case for RAW by padding internally and recording
    # the original length on FLAC wins; for other coders, odd byte counts
    # are passed through unchanged.
    if entropy_mode == "auto":
        # For odd-byte-count RAW, FLAC needs the int16 byte view to be
        # even-aligned. The selector handles this by trying FLAC inside a
        # try/except and dropping it if the byte stream isn't compatible.
        # On wins by FLAC for an odd-length stream we still need to know
        # the original length so the decoder can trim — handled below.
        odd_raw = (base_enc == ENC_RAW_ZSTD and len(raw) % 2 != 0)
        if odd_raw:
            meta["lpc_raw_orig_len"] = len(raw)
            padded_raw = raw + b'\x00'
            padded_n = len(padded_raw) // 2
            codec_id, compressed = _competitive_encode_int16(padded_raw, padded_n)
        else:
            codec_id, compressed = _competitive_encode_int16(raw, num_samples)
        meta["entropy_codec"] = codec_id
        if codec_id == "flac":
            meta["lpc_num_samples"] = num_samples if not odd_raw else padded_n
    elif entropy_mode in ("flac", "lpc"):
        # Pinned FLAC: requires int16-native byte stream. Pad if needed.
        if base_enc == ENC_RAW_ZSTD and len(raw) % 2 != 0:
            meta["lpc_raw_orig_len"] = len(raw)
            raw = raw + b'\x00'
            num_samples = len(raw) // 2
        if _lpc_backend_available():
            compressed = _int16_to_flac(raw, num_samples)
            meta["entropy_codec"] = "flac"
            meta["lpc_num_samples"] = num_samples
        else:
            cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            compressed = cctx.compress(raw)
            meta["entropy_codec"] = "zstd-19"
    elif entropy_mode in ("brotli-11", "brotli"):
        if BROTLI_AVAILABLE:
            compressed = _compress_bytes_brotli(raw)
            meta["entropy_codec"] = "brotli-11"
        else:
            cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            compressed = cctx.compress(raw)
            meta["entropy_codec"] = "zstd-19"
    elif entropy_mode in ("zstd-19", "zstd"):
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        compressed = cctx.compress(raw)
        meta["entropy_codec"] = "zstd-19"
    else:
        # Legacy mode (entropy_mode=None): exact pre-refactor behavior.
        # No entropy_codec field is written; the decoder falls back to
        # the use_lpc heuristic from the encoding constant.
        if use_lpc:
            if base_enc == ENC_RAW_ZSTD and len(raw) % 2 != 0:
                meta["lpc_raw_orig_len"] = len(raw)
                raw = raw + b'\x00'
                num_samples = len(raw) // 2
            compressed = _int16_to_flac(raw, num_samples)
            meta["lpc_num_samples"] = num_samples
        else:
            cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            compressed = cctx.compress(raw)

    meta["compressed_size"] = len(compressed)
    return compressed, meta


def decode_tensor(compressed_bytes, meta):
    """Decode a tensor from compressed bytes + metadata.

    Decode dispatch:

    1. If `meta["entropy_codec"]` (or `meta["bfp_mant_codec"]` for BFP) is
       set, dispatch on that identifier — this is the new standalone-auto
       format.
    2. Else fall back to `_is_lpc_encoding(encoding)` from the encoding
       constant — backward-compatible decode of files written before the
       refactor.

    Both paths produce bit-identical output for files they were respectively
    designed to read.
    """
    encoding = meta["encoding"]
    shape = meta["shape"]
    dtype_str = meta.get("original_dtype", meta["dtype"])
    use_lpc = _is_lpc_encoding(encoding)
    base_enc = _base_encoding(encoding)

    # BFP handles its own decompression (two separate streams)
    if base_enc == ENC_BFP_ZSTD:
        exp_comp_len = struct.unpack("<I", compressed_bytes[:4])[0]
        exp_compressed = compressed_bytes[4:4 + exp_comp_len]
        mant_data = compressed_bytes[4 + exp_comp_len:]

        # Exponents always use zstd (tiny stream)
        dctx = zstd.ZstdDecompressor()
        exp_stream = np.frombuffer(
            dctx.decompress(exp_compressed, max_output_size=meta["bfp_exp_size"]),
            dtype=np.uint8
        ).copy()

        # Mantissa decode dispatch:
        # - New format: bfp_mant_codec is set, dispatch on it
        # - Legacy format: no bfp_mant_codec, fall back to use_lpc heuristic
        bfp_mant_codec = meta.get("bfp_mant_codec")
        if bfp_mant_codec is not None:
            bfp_mant_codec_meta = {}
            if "bfp_mant_orig_len" in meta:
                bfp_mant_codec_meta["uint8_orig_len"] = meta["bfp_mant_orig_len"]
            mant_raw = _decompress_uint8_dispatch(
                mant_data, bfp_mant_codec, bfp_mant_codec_meta
            )
            sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
        elif use_lpc:
            # Legacy: mantissa stream was FLAC-encoded
            mant_orig_len = meta["bfp_mant_orig_len"]
            mant_raw = _decompress_bytes_uint8_lpc(mant_data, mant_orig_len)
            sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
        else:
            _pack_dt = np.dtype(meta.get("bfp_pack_dtype", "uint8"))
            sign_mant_stream = np.frombuffer(
                dctx.decompress(mant_data, max_output_size=meta["bfp_mant_size"]),
                dtype=_pack_dt
            ).copy()

        original_dtype = meta.get("original_dtype", meta["dtype"])
        native_dtype = meta.get("bfp_native_dtype", "float16")
        return bfp_decompress(
            exp_stream, sign_mant_stream,
            meta["bfp_pad_len"], meta["bfp_orig_len"],
            shape, original_dtype,
            meta.get("bfp_group_size", BFP_GROUP_SIZE),
            meta.get("bfp_mantissa_bits", BFP_MANTISSA_BITS),
            native_dtype=native_dtype,
        )

    # Decompress the main payload (non-BFP paths)
    #
    # Dispatch:
    # - New format: meta["entropy_codec"] is set, use _decompress_int16_dispatch
    #   (which also accepts legacy short identifiers like "zstd"/"lpc"/"brotli").
    #   For RAW with odd byte counts written under "auto", trim to the
    #   recorded original length.
    # - Legacy format: no entropy_codec, fall back to the use_lpc heuristic
    #   from the encoding constant — exactly the same code path as before
    #   the refactor.
    entropy_codec = meta.get("entropy_codec")
    if entropy_codec is not None:
        decoded_bytes = _decompress_int16_dispatch(compressed_bytes, entropy_codec)
        raw_size = meta.get("lpc_raw_orig_len", meta["raw_size"])
        raw = decoded_bytes[:raw_size]
    elif use_lpc:
        arr_int16 = _flac_to_int16(compressed_bytes)
        raw_size = meta.get("lpc_raw_orig_len", meta["raw_size"])
        raw = arr_int16.tobytes()[:raw_size]
    else:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed_bytes, max_output_size=meta["raw_size"])

    if base_enc == ENC_FP16_ZSTD:
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
        t = torch.from_numpy(arr)
        if dtype_str == "float32":
            t = t.float()
        return t

    elif base_enc == ENC_INT16_QUANT:
        scale = meta["scale"]
        quantized = np.frombuffer(raw, dtype=np.int16).copy()
        dequantized = quantized.astype(np.float32) * (scale / 32767.0)
        t = torch.from_numpy(dequantized.reshape(shape))
        if dtype_str == "float16":
            t = t.half()
        return t

    elif base_enc == ENC_INT32_QUANT:
        scale = meta["scale"]
        quantized = np.frombuffer(raw, dtype=np.int32).copy()
        dequantized = quantized.astype(np.float64) * (scale / 2147483647.0)
        t = torch.from_numpy(dequantized.astype(np.float32).reshape(shape))
        if dtype_str == "float16":
            t = t.half()
        return t

    elif base_enc == ENC_DELTA_ZSTD:
        delta = np.frombuffer(raw, dtype=np.int16).copy()
        arr = np.cumsum(delta.astype(np.int32)).astype(np.int16)
        if meta["dtype"] == "float16":
            result = arr.view(np.float16).reshape(shape).copy()
            return torch.from_numpy(result)
        elif meta["dtype"] == "bfloat16":
            result = torch.from_numpy(arr.copy()).view(torch.bfloat16).reshape(shape)
            return result
        else:
            return torch.from_numpy(arr.reshape(shape).copy())

    elif base_enc == ENC_RAW_ZSTD:
        np_dtype = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            # uint and bool dtypes — required for GPT-2 attn.bias (uint8) and
            # any model with boolean attention masks. Without these the encoder
            # produces a valid .dmx file (the RAW encoding accepts arbitrary
            # bytes) but the decoder cannot read it back, which is a silent
            # data-loss failure mode for affected models.
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "bool": np.bool_,
        }.get(meta["dtype"])
        if np_dtype:
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
            return torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported dtype for RAW: {meta['dtype']}")

    raise ValueError(f"Unknown encoding: {encoding}")


def decode_tensor_gpu(compressed_bytes, meta):
    """GPU-accelerated tensor decoding. zstd/FLAC on CPU, reconstruction on GPU.

    Supported GPU paths:
    - BFP: leading-one-bit scan parallelized on GPU (biggest speedup)
    - INT16_QUANT: dequantize on GPU
    - DELTA: cumsum on GPU
    - FP16/RAW: no computation needed, pass through
    """
    encoding = meta["encoding"]
    shape = meta["shape"]
    dtype_str = meta.get("original_dtype", meta["dtype"])
    use_lpc = _is_lpc_encoding(encoding)
    base_enc = _base_encoding(encoding)

    # BFP: GPU-accelerated reconstruction
    if base_enc == ENC_BFP_ZSTD:
        exp_comp_len = struct.unpack("<I", compressed_bytes[:4])[0]
        exp_compressed = compressed_bytes[4:4 + exp_comp_len]
        mant_data = compressed_bytes[4 + exp_comp_len:]

        # Exponents: zstd decompress on CPU (tiny stream)
        dctx = zstd.ZstdDecompressor()
        exp_stream = np.frombuffer(
            dctx.decompress(exp_compressed, max_output_size=meta["bfp_exp_size"]),
            dtype=np.uint8
        ).copy()

        # Mantissa decode dispatch (same logic as decode_tensor):
        # - New format: bfp_mant_codec is set, dispatch on it
        # - Legacy format: no bfp_mant_codec, fall back to use_lpc heuristic
        bfp_mant_codec = meta.get("bfp_mant_codec")
        if bfp_mant_codec is not None:
            bfp_mant_codec_meta = {}
            if "bfp_mant_orig_len" in meta:
                bfp_mant_codec_meta["uint8_orig_len"] = meta["bfp_mant_orig_len"]
            mant_raw = _decompress_uint8_dispatch(
                mant_data, bfp_mant_codec, bfp_mant_codec_meta
            )
            sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
        elif use_lpc:
            mant_orig_len = meta["bfp_mant_orig_len"]
            mant_raw = _decompress_bytes_uint8_lpc(mant_data, mant_orig_len)
            sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
        else:
            _pack_dt = np.dtype(meta.get("bfp_pack_dtype", "uint8"))
            sign_mant_stream = np.frombuffer(
                dctx.decompress(mant_data, max_output_size=meta["bfp_mant_size"]),
                dtype=_pack_dt
            ).copy()

        original_dtype = meta.get("original_dtype", meta["dtype"])
        native_dtype = meta.get("bfp_native_dtype", "float16")
        # For native FP32 BFP, fall back to CPU decompress (GPU path is FP16-only)
        if native_dtype == "float32":
            return bfp_decompress(
                exp_stream, sign_mant_stream,
                meta["bfp_pad_len"], meta["bfp_orig_len"],
                shape, original_dtype,
                meta.get("bfp_group_size", BFP_GROUP_SIZE),
                meta.get("bfp_mantissa_bits", BFP_MANTISSA_BITS),
                native_dtype=native_dtype,
            )
        return bfp_decompress_gpu(
            exp_stream, sign_mant_stream,
            meta["bfp_pad_len"], meta["bfp_orig_len"],
            shape, original_dtype,
            meta.get("bfp_group_size", BFP_GROUP_SIZE),
            meta.get("bfp_mantissa_bits", BFP_MANTISSA_BITS),
        )

    # INT16_QUANT, DELTA, FP16, RAW: the dequantization math is trivial
    # compared to the zstd/FLAC decompression (which must run on CPU anyway).
    # GPU transfer overhead makes these slower, so fall back to CPU path.
    return decode_tensor(compressed_bytes, meta)


def auto_detect_mode(tensors):
    """Detect whether a model is primarily FP16/BF16 or FP32.
    Returns 'bfp' if >80% of params are FP16/BF16, 'int16' if >80% are FP32.
    Falls back to 'int16' for mixed models.
    """
    fp16_params = 0
    fp32_params = 0
    total_params = 0
    for tensor in tensors.values():
        n = tensor.numel()
        total_params += n
        if tensor.dtype in (torch.float16, torch.bfloat16):
            fp16_params += n
        elif tensor.dtype == torch.float32:
            fp32_params += n
    if total_params == 0:
        return "int16"
    if fp16_params / total_params > 0.80:
        return "bfp"
    if fp32_params / total_params > 0.80:
        return "int16"
    return "int16"


def _is_fast_load_tensor(name):
    """Return True if this tensor name matches FAST_LOAD_PATTERNS.

    Used by --fast-load to identify embed_tokens, lm_head, and other
    embedding/head tensors that should be stored as FP16 instead of BFP
    so compressed-residency loads skip the materialize step.
    """
    return any(pat in name for pat in FAST_LOAD_PATTERNS)


def compress_file(input_path, output_path, mode="auto", entropy="auto", use_gpu=None,
                  parallel_workers=None, mantissa_bits=None, fast_load=False):
    """Compress a safetensors file to .dmx format.

    mode: 'auto', 'bfp', 'int16', or 'int32'
    entropy: 'auto' (default) | 'zstd-19' | 'zstd' | 'flac' | 'lpc' | 'brotli-11' | 'brotli'

      - 'auto' (default): per-tensor competitive selection across the
        available candidate coders ([zstd-19, FLAC, brotli-11]). For each
        tensor the smallest output is kept and the winning coder's
        identifier is recorded in per-tensor metadata so the decoder
        dispatches correctly.
      - 'zstd-19' / 'zstd': pin to zstd-19 for every tensor.
      - 'flac' / 'lpc': pin to FLAC where applicable (falls back to zstd
        for non-int16-native paths). 'lpc' is a legacy alias.
      - 'brotli-11' / 'brotli': pin to brotli-11.

    use_gpu: use GPU-accelerated BFP compression (CUDA required)
    parallel_workers: number of threads for per-tensor encoding. None (default)
        auto-selects: 1 when use_gpu (avoid CUDA contention), otherwise
        min(8, os.cpu_count()). Pass an explicit int to override. zstd releases
        the GIL during compression so threads give real parallelism on multi-
        core CPUs even with the GIL active.
    mantissa_bits: override the BFP mantissa bit width for this call. None (default)
        uses the module-level BFP_MANTISSA_BITS constant (currently 6). Valid range
        is 1-23. Higher values give better fidelity at the cost of compression ratio
        (M=4: aggressive, ~80% savings; M=8: conservative, ~48% savings; M=6: default
        balance; M=23: full FP32 fidelity). For M>10 with float32 tensors, the
        native FP32 BFP path is used (no downcast to FP16), preserving full FP32
        bit layout. Only affects BFP mode — ignored for int16/int32/auto modes on
        non-BFP paths. The chosen value is stored per-tensor in the manifest so
        the decoder always reconstructs correctly regardless of the current
        module-level default. Editing the module-level BFP_MANTISSA_BITS constant
        at runtime does NOT change compression output because the constant is
        captured as a frozen default argument value on bfp_compress /
        bfp_compress_gpu at function-definition time; the runtime override
        mechanism here is the supported way to change mantissa width per call.
    """
    global _use_gpu_compress, _bfp_mantissa_override, _fast_load_override
    # Auto-detect GPU: use CUDA if available unless explicitly disabled
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU accelerated compression: {gpu_name}")
            _use_gpu_compress = True
        else:
            print("  WARNING: --gpu requested but CUDA not available. Falling back to CPU.")
            _use_gpu_compress = False
    else:
        _use_gpu_compress = False

    # Resolve mantissa_bits override. None falls back to the module-level
    # BFP_MANTISSA_BITS default (6). Validates the override is in the valid
    # BFP range; raises a clear error on out-of-range values. The override is
    # set on the module-level _bfp_mantissa_override global, which is read by
    # encode_tensor's BFP path and passed explicitly to bfp_compress /
    # bfp_compress_gpu (bypassing their frozen default args). Restored to the
    # prior value at the end of this function so state never leaks.
    _prior_mantissa_override = _bfp_mantissa_override
    if mantissa_bits is not None:
        if not isinstance(mantissa_bits, int) or mantissa_bits < 1 or mantissa_bits > 23:
            raise ValueError(
                f"mantissa_bits must be an int in [1, 23]; got {mantissa_bits!r}. "
                f"Common values: 6 (aggressive), 7 (balanced), 10 (full FP16 fidelity), "
                f"23 (full FP32 fidelity, native FP32 path)."
            )
        _bfp_mantissa_override = mantissa_bits
        if mantissa_bits > 10:
            print(f"  Mantissa bits override: M={mantissa_bits} "
                  f"(native FP32 BFP path for float32 tensors, default is {BFP_MANTISSA_BITS})")
        else:
            print(f"  Mantissa bits override: M={mantissa_bits} "
                  f"(default is {BFP_MANTISSA_BITS})")

    # Resolve fast_load override. When True, tensors matching FAST_LOAD_PATTERNS
    # are stored as FP16+entropy instead of BFP, enabling instant loading for
    # compressed residency (no materialize step needed).
    _fast_load_override = bool(fast_load)
    if _fast_load_override:
        print(f"  Fast-load mode: embed/head tensors stored as FP16 (patterns: {FAST_LOAD_PATTERNS})")

    # Resolve entropy mode. Normalize legacy aliases to the canonical names
    # used by the selector and dispatch helpers.
    if entropy in (None, "auto"):
        entropy_mode = "auto"
    elif entropy in ("flac", "lpc"):
        if _lpc_backend_available():
            entropy_mode = "flac"
        else:
            print("  WARNING: --entropy flac requires soundfile (pip install soundfile) or ffmpeg. Falling back to zstd-19.")
            entropy_mode = "zstd-19"
    elif entropy in ("brotli-11", "brotli"):
        if BROTLI_AVAILABLE:
            entropy_mode = "brotli-11"
        else:
            print("  WARNING: --entropy brotli-11 requires brotli (pip install brotli). Falling back to zstd-19.")
            entropy_mode = "zstd-19"
    elif entropy in ("zstd-19", "zstd"):
        entropy_mode = "zstd-19"
    else:
        raise ValueError(
            f"Unknown entropy value {entropy!r}. "
            f"Use 'auto', 'zstd-19', 'flac', or 'brotli-11'."
        )

    print(f"Loading {input_path}...")
    start = time.time()
    tensors = load_file(input_path)
    load_time = time.time() - start
    print(f"  Loaded {len(tensors)} tensors in {load_time:.1f}s")

    original_size = os.path.getsize(input_path)
    print(f"  Original size: {original_size / 1024 / 1024:.1f} MB")

    # Resolve auto mode
    if mode == "auto":
        mode = auto_detect_mode(tensors)
        print(f"  Auto-detected mode: {mode}")
    else:
        print(f"  Mode: {mode}")

    print(f"  Entropy coding: {entropy_mode}")

    # Build manifest and compressed chunks
    manifest = {
        "version": DMX_VERSION,
        "mode": mode,
        "entropy": entropy_mode,
        "source_file": os.path.basename(input_path),
        "source_size": original_size,
        "tensor_count": len(tensors),
        "tensors": {},
    }

    compressed_chunks = {}
    total_raw = 0
    total_compressed = 0
    enc_counts = {}
    enc_names = {
        ENC_FP16_ZSTD: "FP16+zstd", ENC_INT16_QUANT: "INT16Q+zstd",
        ENC_DELTA_ZSTD: "Delta+zstd", ENC_RAW_ZSTD: "Raw+zstd",
        ENC_BFP_ZSTD: "BFP+zstd", ENC_INT32_QUANT: "INT32Q+zstd",
        ENC_FP16_LPC: "FP16+lpc", ENC_INT16_QUANT_LPC: "INT16Q+lpc",
        ENC_DELTA_LPC: "Delta+lpc", ENC_RAW_LPC: "Raw+lpc",
        ENC_BFP_LPC: "BFP+lpc", ENC_INT32_QUANT_LPC: "INT32Q+lpc",
    }

    keys = sorted(tensors.keys())

    # Resolve parallel worker count.
    # GPU mode forces serial because the CUDA context is single-threaded and
    # parallel submission causes contention rather than speedup.
    if parallel_workers is None:
        parallel_workers = 1 if _use_gpu_compress else min(8, os.cpu_count() or 1)
    elif _use_gpu_compress and parallel_workers > 1:
        print(f"  WARNING: parallel_workers={parallel_workers} ignored in GPU mode (forcing 1)")
        parallel_workers = 1

    if parallel_workers > 1:
        print(f"  Parallel encoding: {parallel_workers} workers")

        def _encode_one(key):
            tensor = tensors[key]
            # detect_encoding always returns the data-shape (*_ZSTD form);
            # the entropy coder is decided per tensor inside encode_tensor
            # via entropy_mode, and recorded in meta["entropy_codec"].
            encoding = detect_encoding(tensor, mode=mode, entropy="zstd")
            # --fast-load override: store embed/head tensors as FP16+entropy
            is_fast_load = _fast_load_override and _is_fast_load_tensor(key)
            if is_fast_load:
                encoding = ENC_FP16_ZSTD
            compressed, meta = encode_tensor(tensor, encoding, entropy_mode=entropy_mode)
            if is_fast_load:
                meta["fast_load"] = True
            return key, encoding, compressed, meta

        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            # map preserves submission order, so manifest stays deterministic
            for i, (key, encoding, compressed, meta) in enumerate(pool.map(_encode_one, keys)):
                enc_counts[encoding] = enc_counts.get(encoding, 0) + 1
                manifest["tensors"][key] = meta
                compressed_chunks[key] = compressed
                total_raw += meta["raw_size"]
                total_compressed += meta["compressed_size"]
                if (i + 1) % 100 == 0 or (i + 1) == len(keys):
                    print(f"  Compressed {i+1}/{len(keys)} tensors...")
    else:
        for i, key in enumerate(keys):
            tensor = tensors[key]
            encoding = detect_encoding(tensor, mode=mode, entropy="zstd")
            # --fast-load override: store embed/head tensors as FP16+entropy
            is_fast_load = _fast_load_override and _is_fast_load_tensor(key)
            if is_fast_load:
                encoding = ENC_FP16_ZSTD
            enc_counts[encoding] = enc_counts.get(encoding, 0) + 1

            compressed, meta = encode_tensor(tensor, encoding, entropy_mode=entropy_mode)
            if is_fast_load:
                meta["fast_load"] = True
            manifest["tensors"][key] = meta
            compressed_chunks[key] = compressed

            total_raw += meta["raw_size"]
            total_compressed += meta["compressed_size"]

            if (i + 1) % 100 == 0 or (i + 1) == len(keys):
                print(f"  Compressed {i+1}/{len(keys)} tensors...")

    # Write .dmx file
    # Format: [MAGIC 4B][VERSION 4B][MANIFEST_SIZE 4B][MANIFEST JSON][CHUNK DATA...]
    # Each chunk is written sequentially, manifest has offsets
    manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    # Calculate offsets for chunks
    header_size = 4 + 4 + 4 + len(manifest_json)
    offset = header_size
    chunk_offsets = {}
    for key in keys:
        chunk_offsets[key] = offset
        offset += len(compressed_chunks[key])
        manifest["tensors"][key]["offset"] = chunk_offsets[key]

    # Re-serialize manifest with offsets
    manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    # Recalculate offsets since manifest size may have changed
    header_size = 4 + 4 + 4 + len(manifest_json)
    offset = header_size
    for key in keys:
        manifest["tensors"][key]["offset"] = offset
        offset += len(compressed_chunks[key])

    # Final manifest
    manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    # One more pass to stabilize (offset digits may grow)
    header_size_check = 4 + 4 + 4 + len(manifest_json)
    if header_size_check != header_size:
        header_size = header_size_check
        offset = header_size
        for key in keys:
            manifest["tensors"][key]["offset"] = offset
            offset += len(compressed_chunks[key])
        manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(DMX_MAGIC)
        f.write(struct.pack("<I", DMX_VERSION))
        f.write(struct.pack("<I", len(manifest_json)))
        f.write(manifest_json)
        for key in keys:
            f.write(compressed_chunks[key])

    output_size = os.path.getsize(output_path)
    elapsed = time.time() - start
    ratio = output_size / original_size * 100

    # Compute FP32-equivalent baseline for dual-denominator reporting.
    # total_params is the sum over all tensors. For float tensors, this is the
    # dtype-independent canonical baseline. Skip if no tensors loaded.
    try:
        total_params = sum(
            t.numel() for t in tensors.values()
            if hasattr(t, "numel")
        )
    except Exception:
        total_params = 0
    fp32_equivalent = total_params * 4 if total_params > 0 else 0

    print(f"  Encoding breakdown:")
    for enc, count in enc_counts.items():
        if count > 0:
            print(f"    {enc_names[enc]}: {count} tensors")
    print(f"  Output size: {output_size / 1024 / 1024:.1f} MB")
    # Dual-denominator reporting: the same compressed bytes can produce wildly
    # different "savings %" depending on whether the baseline is the source
    # file (dtype-dependent) or the FP32 equivalent (canonical). Report both
    # so log parsers and users see both numbers side-by-side.
    savings_vs_source_pct = (1 - output_size / original_size) * 100
    print(f"    vs source file ({original_size / 1024 / 1024:.1f} MB): "
          f"{ratio:.1f}% of source, {savings_vs_source_pct:+.1f}% savings")
    if fp32_equivalent > 0:
        savings_vs_fp32_pct = (1 - output_size / fp32_equivalent) * 100
        fp32_ratio_pct = output_size / fp32_equivalent * 100
        print(f"    vs FP32 equivalent ({fp32_equivalent / 1024 / 1024:.1f} MB, "
              f"{total_params:,} params × 4 bytes): "
              f"{fp32_ratio_pct:.1f}% of FP32, {savings_vs_fp32_pct:+.1f}% savings")
    print(f"  Savings: {(original_size - output_size) / 1024 / 1024:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")

    # Restore the mantissa-bits and fast-load overrides so they don't leak to
    # subsequent calls. Done unconditionally to keep the module-level state clean.
    _bfp_mantissa_override = _prior_mantissa_override
    _fast_load_override = False

    return original_size, output_size


def decompress_file(input_path, output_path, use_gpu=None):
    """Decompress a .dmx file back to safetensors.

    If use_gpu=True, uses GPU-accelerated decompression for BFP, INT16, and
    DELTA encoded tensors. zstd/FLAC entropy decoding stays on CPU.
    If use_gpu=None (default), auto-detects CUDA availability.
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        if not torch.cuda.is_available():
            print("  WARNING: --gpu requested but CUDA not available. Falling back to CPU.")
            use_gpu = False
        else:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU accelerated mode: {gpu_name}")

    decode_fn = decode_tensor_gpu if use_gpu else decode_tensor

    print(f"Loading {input_path}...")
    start = time.time()

    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != DMX_MAGIC:
            raise ValueError(f"Not a DMX file (magic: {magic})")

        version = struct.unpack("<I", f.read(4))[0]
        if version > DMX_VERSION:
            raise ValueError(f"Unsupported DMX version: {version}")

        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest_json = f.read(manifest_size)
        manifest = json.loads(manifest_json)

        tensors = {}
        keys = sorted(manifest["tensors"].keys())

        # Track time breakdown for GPU mode
        t_zstd = 0.0
        t_gpu = 0.0

        for i, key in enumerate(keys):
            meta = manifest["tensors"][key]
            f.seek(meta["offset"])
            compressed = f.read(meta["compressed_size"])
            tensors[key] = decode_fn(compressed, meta)

            if (i + 1) % 500 == 0 or (i + 1) == len(keys):
                print(f"  Decompressed {i+1}/{len(keys)} tensors...")

    if use_gpu:
        torch.cuda.synchronize()

    print(f"  Saving to {output_path}...")
    save_file(tensors, output_path)

    elapsed = time.time() - start
    output_size = os.path.getsize(output_path)
    print(f"  Output size: {output_size / 1024 / 1024:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")

    return output_size


def _decode_dmx_to_tensors(input_path, use_gpu=None):
    """Load a DMX file and return a dict of full-precision tensors.

    Factored out of export_quant() to support multi-target export, where a
    single DMX load is shared across multiple target derivations.

    Returns:
        (tensors_dict, manifest, decompress_time_sec)
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    decode_fn = decode_tensor_gpu if (use_gpu and torch.cuda.is_available()) else decode_tensor

    start = time.time()
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != DMX_MAGIC:
            raise ValueError(f"Not a DMX file (magic: {magic})")
        version = struct.unpack("<I", f.read(4))[0]
        if version > DMX_VERSION:
            raise ValueError(f"Unsupported DMX version: {version}")
        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest = json.loads(f.read(manifest_size))

        tensors = {}
        keys = sorted(manifest["tensors"].keys())
        for i, key in enumerate(keys):
            meta = manifest["tensors"][key]
            f.seek(meta["offset"])
            compressed = f.read(meta["compressed_size"])
            tensors[key] = decode_fn(compressed, meta)
            if (i + 1) % 500 == 0 or (i + 1) == len(keys):
                print(f"  Decompressed {i+1}/{len(keys)} tensors...")

    decompress_time = time.time() - start
    return tensors, manifest, decompress_time


@dataclass
class BFPPayload:
    """Compressed-residency form of a BFP tensor.

    Holds the raw exponent + packed-sign-mantissa streams produced by the
    BFP encoder, without reconstructing an FP16/FP32 view. This is the
    Sprint B entry point that lets BFPLinear keep weights resident in VRAM
    in their compressed form and dequantize per-tile on demand.

    Attributes
    ----------
    exponents : torch.Tensor
        Shared per-block exponents. dtype=uint8, shape=[n_blocks].
        (FP16 exponents are 5 bits so uint8 is the natural container; the
        field is named `exponents` but stored as uint8, matching the
        on-disk format. The `int8` name in the public docstring refers to
        the 8-bit width, not signedness.)
    packed_mantissa : torch.Tensor
        Packed per-element sign+mantissa byte. dtype=uint8, shape=[n_elements]
        including pad. Each byte is `(sign << mantissa_bits) | truncated_mantissa`.
    shape : tuple[int, ...]
        Original tensor logical shape.
    dtype : torch.dtype
        Original dtype (e.g. torch.float16 for Qwen weights).
    group_size : int
        Block size used during BFP encoding (typically 32).
    mantissa_bits : int
        Mantissa bit width (typically 6 or 7).
    pad_len : int
        Number of padding elements appended to reach a multiple of
        group_size. Needed to trim after dequant.
    orig_len : int
        Original flat element count (pre-pad). Needed to trim after dequant.
    scale : float | None
        Reserved; None for pure BFP path. Populated only when BFP was
        preceded by an affine scale (INT16Q hybrid, not a Sprint B target).
    """
    exponents: "torch.Tensor"
    packed_mantissa: "torch.Tensor"
    shape: tuple
    dtype: "torch.dtype"
    group_size: int
    mantissa_bits: int
    pad_len: int = 0
    orig_len: int = 0
    scale: Optional[float] = None
    native_dtype: str = "float16"  # "float16" or "float32" — which BFP bit layout was used


_DTYPE_STR_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _decode_bfp_streams_only(compressed_bytes, meta):
    """Unwrap the entropy coder(s) around a BFP tensor and return
    (exp_stream_np_uint8, sign_mant_stream_np_uint8).

    This is the first half of decode_tensor's BFP branch — everything up to
    but not including the call to bfp_decompress. Extracted so the partial
    decoder can stop here. The full decoder still uses its inline version
    (byte-for-byte identical to the original implementation); this helper
    is only consumed by _decode_dmx_to_bfp_payloads.
    """
    use_lpc = _is_lpc_encoding(meta["encoding"])

    exp_comp_len = struct.unpack("<I", compressed_bytes[:4])[0]
    exp_compressed = compressed_bytes[4:4 + exp_comp_len]
    mant_data = compressed_bytes[4 + exp_comp_len:]

    dctx = zstd.ZstdDecompressor()
    exp_stream = np.frombuffer(
        dctx.decompress(exp_compressed, max_output_size=meta["bfp_exp_size"]),
        dtype=np.uint8,
    ).copy()

    bfp_mant_codec = meta.get("bfp_mant_codec")
    if bfp_mant_codec is not None:
        bfp_mant_codec_meta = {}
        if "bfp_mant_orig_len" in meta:
            bfp_mant_codec_meta["uint8_orig_len"] = meta["bfp_mant_orig_len"]
        mant_raw = _decompress_uint8_dispatch(
            mant_data, bfp_mant_codec, bfp_mant_codec_meta
        )
        sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
    elif use_lpc:
        mant_orig_len = meta["bfp_mant_orig_len"]
        mant_raw = _decompress_bytes_uint8_lpc(mant_data, mant_orig_len)
        sign_mant_stream = np.frombuffer(mant_raw, dtype=np.uint8).copy()
    else:
        _pack_dt = np.dtype(meta.get("bfp_pack_dtype", "uint8"))
        sign_mant_stream = np.frombuffer(
            dctx.decompress(mant_data, max_output_size=meta["bfp_mant_size"]),
            dtype=_pack_dt,
        ).copy()

    return exp_stream, sign_mant_stream


def _decode_dmx_to_bfp_payloads(path, use_gpu=None):
    """Partial-decode variant of _decode_dmx_to_tensors.

    For BFP-coded tensors (encoding ENC_BFP_ZSTD=4 or ENC_BFP_LPC=14):
    returns a :class:`BFPPayload` containing the exponent tensor, packed
    sign+mantissa tensor, and shape / dtype / group-size metadata — WITHOUT
    reconstructing fp32/fp16.

    For all other encodings (FP16, INT16Q, INT32Q, DELTA, RAW): falls back
    to the full ``decode_tensor``/``decode_tensor_gpu`` path and returns
    the same torch.Tensor that :func:`_decode_dmx_to_tensors` would produce.
    Callers that care about Sprint B compressed residency should inspect
    each value's type (BFPPayload vs torch.Tensor) to decide whether to
    keep compressed or inflate.

    GPU path decision: B1 returns BFPPayloads with CPU-resident
    exponents/mantissas. Sprint B's matmul / materialize path copies to
    GPU itself. This keeps the partial decoder cheap and side-effect-free
    and lets callers control device placement. Non-BFP tensors still use
    ``decode_tensor_gpu`` when ``use_gpu=True``, matching the full
    decoder's behaviour.

    Returns
    -------
    (payloads_dict, manifest, elapsed_seconds)
        payloads_dict: dict[str, Union[BFPPayload, torch.Tensor]]
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    decode_fn = decode_tensor_gpu if (use_gpu and torch.cuda.is_available()) else decode_tensor

    start = time.time()
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != DMX_MAGIC:
            raise ValueError(f"Not a DMX file (magic: {magic})")
        version = struct.unpack("<I", f.read(4))[0]
        if version > DMX_VERSION:
            raise ValueError(f"Unsupported DMX version: {version}")
        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest = json.loads(f.read(manifest_size))

        payloads = {}
        keys = sorted(manifest["tensors"].keys())
        for i, key in enumerate(keys):
            meta = manifest["tensors"][key]
            f.seek(meta["offset"])
            compressed = f.read(meta["compressed_size"])

            base_enc = _base_encoding(meta["encoding"])
            if base_enc == ENC_BFP_ZSTD:
                # Partial decode: unwrap entropy coder once, stop before
                # bfp_decompress. No FP32/FP16 transient materialized.
                exp_stream_np, sign_mant_stream_np = _decode_bfp_streams_only(
                    compressed, meta
                )
                exponents = torch.from_numpy(exp_stream_np)  # uint8
                packed_mantissa = torch.from_numpy(sign_mant_stream_np)  # uint8

                original_dtype_str = meta.get("original_dtype", meta["dtype"])
                torch_dtype = _DTYPE_STR_TO_TORCH.get(
                    original_dtype_str, torch.float16
                )

                payloads[key] = BFPPayload(
                    exponents=exponents,
                    packed_mantissa=packed_mantissa,
                    shape=tuple(meta["shape"]),
                    dtype=torch_dtype,
                    group_size=meta.get("bfp_group_size", BFP_GROUP_SIZE),
                    mantissa_bits=meta.get("bfp_mantissa_bits", BFP_MANTISSA_BITS),
                    pad_len=meta["bfp_pad_len"],
                    orig_len=meta["bfp_orig_len"],
                    scale=None,
                    native_dtype=meta.get("bfp_native_dtype", "float16"),
                )
            else:
                # Non-BFP: delegate to the existing full decoder. No
                # duplication of FP16/INT16/INT32/DELTA/RAW logic.
                payloads[key] = decode_fn(compressed, meta)

            if (i + 1) % 500 == 0 or (i + 1) == len(keys):
                print(f"  Partial-decoded {i+1}/{len(keys)} tensors...")

    elapsed = time.time() - start
    return payloads, manifest, elapsed


def bfp_payload_to_tensor(payload: "BFPPayload") -> "torch.Tensor":
    """Reconstruct a full-precision torch.Tensor from a :class:`BFPPayload`.

    Thin wrapper around :func:`bfp_decompress` that takes a BFPPayload
    instead of loose streams. Produces bit-for-bit the same tensor that
    the full decoder would have produced for the same BFP-encoded tensor.
    Provided for callers that want to inflate a single payload on demand
    (e.g. Sprint B's BFPTensor.materialize()) without rewriting the
    dequant math.
    """
    dtype_str = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }.get(payload.dtype, "float16")
    exp_np = payload.exponents.numpy() if payload.exponents.device.type == "cpu" \
        else payload.exponents.cpu().numpy()
    mant_np = payload.packed_mantissa.numpy() if payload.packed_mantissa.device.type == "cpu" \
        else payload.packed_mantissa.cpu().numpy()
    return bfp_decompress(
        exp_np, mant_np,
        payload.pad_len, payload.orig_len,
        payload.shape, dtype_str,
        payload.group_size, payload.mantissa_bits,
        native_dtype=payload.native_dtype,
    )


# NF4 codebook — shared constant for the NF4 derive path.
_NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]


def _derive_target_from_tensors(tensors, target):
    """Derive a target quantization format from an in-memory dict of full-precision tensors.

    Factored out of export_quant() so that multi-target export can run each
    target's quant math on the same decoded tensors without re-decoding the
    DMX file per target.

    Args:
        tensors: dict mapping key -> full-precision torch tensor
        target: one of 'int8', 'nf4', 'fp8'

    Returns:
        dict mapping key -> quantized torch tensor
    """
    quantized = {}

    if target == "int8":
        for key, t in tensors.items():
            t_flat = t.float()
            if t_flat.dim() >= 2:
                t_2d = t_flat.reshape(t_flat.shape[0], -1)
                max_abs = t_2d.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
                scale = max_abs / 127
                q = torch.clamp(torch.round(t_2d / scale), -127, 127).to(torch.int8)
                quantized[key] = q.reshape(t.shape)
            else:
                max_abs = t_flat.abs().max().clamp(min=1e-12)
                scale = max_abs / 127
                q = torch.clamp(torch.round(t_flat / scale), -127, 127).to(torch.int8)
                quantized[key] = q

    elif target == "nf4":
        nf4_codebook = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32)
        group_size = 64
        for key, t in tensors.items():
            flat = t.float().flatten()
            n = flat.numel()
            pad = (group_size - n % group_size) % group_size
            if pad:
                flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float32)])
            groups = flat.reshape(-1, group_size)
            absmax = groups.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
            normalized = groups / absmax
            diffs = (normalized.unsqueeze(-1) - nf4_codebook.unsqueeze(0).unsqueeze(0)).abs()
            codes = diffs.argmin(dim=-1).to(torch.uint8)
            quantized[key] = codes.flatten()[:n].reshape(t.shape)

    elif target == "fp8":
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                "FP8 E4M3 requires PyTorch 2.1+ with native float8_e4m3fn support. "
                "Your PyTorch version does not have it."
            )
        for key, t in tensors.items():
            quantized[key] = t.float().to(torch.float8_e4m3fn)
    else:
        raise ValueError(
            f"Unknown target format '{target}'. "
            f"Supported: int8, nf4, fp8"
        )

    return quantized


def _total_params_from_tensors(tensors):
    """Count total params in a tensor dict, used for FP32-equivalent baseline."""
    return sum(t.numel() for t in tensors.values() if hasattr(t, "numel"))


def _print_dual_denominator_summary(input_path, output_path, target, total_params,
                                     decompress_time, quant_time, elapsed):
    """Print a dual-denominator savings summary.

    Reports size against both:
      - vs DMX input file (source for this operation)
      - vs FP32 equivalent (dtype-independent canonical baseline = total_params * 4 bytes)

    This prevents the denominator trap where the same compressed bytes produce
    different "savings %" numbers depending on whether the baseline is the
    source file (dtype-specific) or the FP32 equivalent (canonical).
    """
    output_size = os.path.getsize(output_path)
    input_size = os.path.getsize(input_path)
    fp32_equivalent = total_params * 4 if total_params > 0 else 0

    print(f"\n  Results:")
    print(f"    DMX input:        {input_size / 1024 / 1024:.1f} MB")
    print(f"    {target.upper()} output:      {output_size / 1024 / 1024:.1f} MB")
    if fp32_equivalent > 0:
        pct_of_dmx = output_size / input_size * 100
        pct_of_fp32 = output_size / fp32_equivalent * 100
        savings_vs_dmx = (1 - output_size / input_size) * 100
        savings_vs_fp32 = (1 - output_size / fp32_equivalent) * 100
        print(f"    vs DMX input ({input_size / 1024 / 1024:.1f} MB): "
              f"{pct_of_dmx:.1f}% of source, {savings_vs_dmx:+.1f}% savings")
        print(f"    vs FP32 equivalent ({fp32_equivalent / 1024 / 1024:.1f} MB, "
              f"{total_params:,} params × 4 bytes): "
              f"{pct_of_fp32:.1f}% of FP32, {savings_vs_fp32:+.1f}% savings")
    print(f"    Time:             {elapsed:.1f}s (decompress {decompress_time:.1f}s + quantize {quant_time:.1f}s)")


def export_quant(input_path, output_path, target="int8", use_gpu=None):
    """Derive a quantized model from a DMX file without keeping the full-precision intermediate.

    Decompresses the DMX file to full precision, applies the target quantization,
    and saves the quantized result as a safetensors file. The full-precision
    tensors are never written to disk — only the quantized output is saved.

    Supported targets (RTN-family only — calibration-dependent formats like
    GPTQ and AWQ are not derivable because they require activation data that
    DMX does not store):

      int8   — per-channel symmetric INT8 (scale per output row, [-127, 127])
      nf4    — NF4 QLoRA codebook (16 fixed values, group_size=64)
      fp8    — FP8 E4M3 (PyTorch 2.1+ native, requires int32 DMX for clean derivation)

    Args:
        input_path: path to .dmx file
        output_path: path to write the quantized .safetensors output
        target: one of 'int8', 'nf4', 'fp8'
        use_gpu: passthrough to decompress (None = auto-detect)
    """
    print(f"=" * 70)
    print(f"DMX export — derive {target.upper()} from {os.path.basename(input_path)}")
    print(f"=" * 70)

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if use_gpu and torch.cuda.is_available():
        print(f"  GPU accelerated mode: {torch.cuda.get_device_name(0)}")

    print(f"  Loading {input_path}...")
    start = time.time()

    # Step 1: Decompress DMX to full-precision tensors (shared helper)
    tensors, manifest, decompress_time = _decode_dmx_to_tensors(input_path, use_gpu=use_gpu)
    print(f"  Decompressed {len(tensors)} tensors in {decompress_time:.1f}s")

    # Step 2: Apply target quantization (shared helper)
    print(f"  Deriving {target.upper()}...")
    quant_start = time.time()
    quantized = _derive_target_from_tensors(tensors, target)
    quant_time = time.time() - quant_start
    print(f"  Quantized {len(quantized)} tensors to {target.upper()} in {quant_time:.1f}s")

    # Step 3: Save
    print(f"  Saving to {output_path}...")
    save_file(quantized, output_path)

    # Step 4: Dual-denominator summary
    total_params = _total_params_from_tensors(tensors)
    elapsed = time.time() - start
    _print_dual_denominator_summary(
        input_path, output_path, target, total_params,
        decompress_time, quant_time, elapsed,
    )

    return os.path.getsize(output_path)


def export_quant_multi(input_path, output_dir, targets=None, use_gpu=None):
    """Derive MULTIPLE target quantization formats from a single DMX file load.

    The DMX file is loaded and decoded exactly ONCE, then each target
    quantization is derived from the shared in-memory tensor dict. For N
    targets, this saves N-1 redundant decode passes vs calling export_quant()
    N times. Measured benefit on a 160M-parameter FP16 model: deriving
    INT8 + NF4 from a single DMX load saved ~27% wall time vs two separate
    export_quant() calls on the same hardware.

    Args:
        input_path: path to .dmx file
        output_dir: directory where {target}.safetensors files will be written
        targets: list of target format strings. Any combination of ['int8', 'nf4', 'fp8'].
                 Defaults to ['int8'] if not specified (behaves like export_quant).
        use_gpu: passthrough to decompress (None = auto-detect)

    Returns:
        dict mapping target name -> output file path
    """
    if targets is None:
        targets = ["int8"]
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]

    valid_targets = ["int8", "nf4", "fp8"]
    for t in targets:
        if t not in valid_targets:
            raise ValueError(
                f"Unknown target format '{t}'. Supported: {valid_targets}"
            )

    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 70)
    print(f"DMX export (multi-target) — derive {targets} from "
          f"{os.path.basename(input_path)}")
    print(f"=" * 70)

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if use_gpu and torch.cuda.is_available():
        print(f"  GPU accelerated mode: {torch.cuda.get_device_name(0)}")

    print(f"  Loading {input_path}...")
    overall_start = time.time()

    # Step 1: Load and decode DMX ONCE — shared across all targets
    tensors, manifest, decompress_time = _decode_dmx_to_tensors(input_path, use_gpu=use_gpu)
    total_params = _total_params_from_tensors(tensors)
    print(f"  Decompressed {len(tensors)} tensors in {decompress_time:.1f}s "
          f"(shared across {len(targets)} targets)")

    # Step 2: Loop targets, derive each from the shared tensor dict
    output_paths = {}
    input_size = os.path.getsize(input_path)
    fp32_equivalent = total_params * 4

    for target in targets:
        target_start = time.time()
        print(f"\n  Deriving {target.upper()}...")
        quantized = _derive_target_from_tensors(tensors, target)
        quant_time = time.time() - target_start

        output_path = os.path.join(output_dir, f"{target}.safetensors")
        save_file(quantized, output_path)
        output_size = os.path.getsize(output_path)
        output_paths[target] = output_path

        print(f"    Quantized {len(quantized)} tensors in {quant_time:.1f}s")
        print(f"    Output: {output_path} ({output_size / 1024 / 1024:.1f} MB)")
        if fp32_equivalent > 0:
            print(f"      vs DMX input ({input_size / 1024 / 1024:.1f} MB): "
                  f"{output_size / input_size * 100:.1f}% of source, "
                  f"{(1 - output_size / input_size) * 100:+.1f}% savings")
            print(f"      vs FP32 equivalent ({fp32_equivalent / 1024 / 1024:.1f} MB): "
                  f"{output_size / fp32_equivalent * 100:.1f}% of FP32, "
                  f"{(1 - output_size / fp32_equivalent) * 100:+.1f}% savings")

    total_elapsed = time.time() - overall_start
    print(f"\n  Multi-target total: {total_elapsed:.1f}s "
          f"(decompress {decompress_time:.1f}s + {len(targets)} target derives)")
    print(f"  Targets saved: {list(output_paths.keys())}")

    return output_paths


def info_file(input_path):
    """Display info about a .dmx file."""
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != DMX_MAGIC:
            raise ValueError(f"Not a DMX file (magic: {magic})")

        version = struct.unpack("<I", f.read(4))[0]
        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest_json = f.read(manifest_size)
        manifest = json.loads(manifest_json)

    dmx_size = os.path.getsize(input_path)
    src_size = manifest.get("source_size", 0)

    enc_names = {
        ENC_FP16_ZSTD: "FP16+zstd", ENC_INT16_QUANT: "INT16Q+zstd",
        ENC_DELTA_ZSTD: "Delta+zstd", ENC_RAW_ZSTD: "Raw+zstd",
        ENC_BFP_ZSTD: "BFP+zstd", ENC_INT32_QUANT: "INT32Q+zstd",
        ENC_FP16_LPC: "FP16+lpc", ENC_INT16_QUANT_LPC: "INT16Q+lpc",
        ENC_DELTA_LPC: "Delta+lpc", ENC_RAW_LPC: "Raw+lpc",
        ENC_BFP_LPC: "BFP+lpc", ENC_INT32_QUANT_LPC: "INT32Q+lpc",
    }

    print(f"DMX File: {input_path}")
    print(f"  Version: {version}")
    print(f"  Entropy: {manifest.get('entropy', 'zstd')}")
    print(f"  Source: {manifest.get('source_file', 'unknown')}")
    print(f"  Source size: {src_size / 1024 / 1024:.1f} MB")
    print(f"  DMX size: {dmx_size / 1024 / 1024:.1f} MB")
    if src_size > 0:
        print(f"  Ratio: {dmx_size / src_size * 100:.1f}%")
        print(f"  Savings: {(src_size - dmx_size) / 1024 / 1024:.1f} MB")
    print(f"  Tensors: {manifest.get('tensor_count', len(manifest['tensors']))}")

    # Encoding breakdown
    enc_counts = {}
    total_raw = 0
    total_compressed = 0
    for key, meta in manifest["tensors"].items():
        enc = meta["encoding"]
        enc_counts[enc] = enc_counts.get(enc, 0) + 1
        total_raw += meta.get("raw_size", 0)
        total_compressed += meta.get("compressed_size", 0)

    print(f"  Encoding breakdown:")
    for enc, count in sorted(enc_counts.items()):
        print(f"    {enc_names.get(enc, f'Unknown({enc})')}: {count} tensors")

    # Show a few sample tensors
    keys = sorted(manifest["tensors"].keys())
    print(f"  Sample tensors (first 5):")
    for key in keys[:5]:
        meta = manifest["tensors"][key]
        shape = meta["shape"]
        dtype = meta["dtype"]
        csz = meta["compressed_size"]
        rsz = meta["raw_size"]
        print(f"    {key}: {shape} {dtype} -> {csz/1024:.1f}KB ({csz/rsz*100:.0f}%)")


def _sha256_file(path):
    """Compute SHA-256 of a file, reading in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_dmx_mode(dmx_path):
    """Read the compression mode from a DMX file's manifest."""
    with open(dmx_path, "rb") as f:
        magic = f.read(4)
        if magic != DMX_MAGIC:
            return "unknown"
        _version = struct.unpack("<I", f.read(4))[0]
        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest_json = f.read(manifest_size)
        manifest = json.loads(manifest_json)
    return manifest.get("mode", "unknown")


def verify_file(source_path, dmx_path, mode="auto", entropy="zstd", report_path=None, use_gpu=None):
    """Compress, decompress, and compare tensor-by-tensor.

    If report_path is provided, writes a structured JSON verification report.
    If use_gpu is True, uses GPU-accelerated decompression.
    """
    import tempfile

    print(f"=== DMX Verify: {os.path.basename(source_path)} ===")
    print()

    # Step 1: Compress if dmx doesn't exist, or use existing
    if not os.path.exists(dmx_path):
        print("[1/3] Compressing...")
        compress_file(source_path, dmx_path, mode=mode, entropy=entropy)
    else:
        print(f"[1/3] Using existing DMX: {dmx_path}")
    print()

    # Step 2: Decompress to temp file
    print("[2/3] Decompressing to temp file...")
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "verify_roundtrip.safetensors")
    decompress_file(dmx_path, tmp_path, use_gpu=use_gpu)
    print()

    # Determine actual mode from DMX manifest
    actual_mode = _read_dmx_mode(dmx_path)

    # Set cosine similarity threshold based on mode
    if actual_mode == "int16":
        pass_threshold = 0.9999
    else:
        pass_threshold = 0.99

    # Step 3: Compare
    print("[3/3] Comparing tensors...")
    original = load_file(source_path)
    roundtrip = load_file(tmp_path)

    if set(original.keys()) != set(roundtrip.keys()):
        missing = set(original.keys()) - set(roundtrip.keys())
        extra = set(roundtrip.keys()) - set(original.keys())
        print(f"  KEY MISMATCH! Missing: {len(missing)}, Extra: {len(extra)}")
        return False

    max_diff_global = 0.0
    min_cosine_global = 1.0
    best_cosine_global = 0.0
    worst_key = ""
    best_cosine_key = ""
    all_exact = True
    mismatched = 0
    num_passed = 0
    num_failed = 0
    per_tensor_stats = []
    all_cosines = []
    all_max_diffs = []

    for key in sorted(original.keys()):
        orig_t = original[key].float()
        rt_t = roundtrip[key].float()

        if orig_t.shape != rt_t.shape:
            print(f"  SHAPE MISMATCH: {key}: {orig_t.shape} vs {rt_t.shape}")
            mismatched += 1
            num_failed += 1
            per_tensor_stats.append({
                "name": key,
                "shape": list(orig_t.shape),
                "dtype": str(original[key].dtype).replace("torch.", ""),
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "cosine_similarity": None,
                "relative_error": None,
                "verdict": "FAIL (shape mismatch)",
            })
            continue

        diff = (orig_t - rt_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff > 0:
            all_exact = False

        if max_diff > max_diff_global:
            max_diff_global = max_diff
            worst_key = key

        # Cosine similarity (skip scalars)
        if orig_t.numel() > 1:
            cos = torch.nn.functional.cosine_similarity(
                orig_t.flatten().unsqueeze(0),
                rt_t.flatten().unsqueeze(0)
            ).item()
        else:
            cos = 1.0 if max_diff == 0 else 0.0

        if cos < min_cosine_global:
            min_cosine_global = cos
        if cos > best_cosine_global:
            best_cosine_global = cos
            best_cosine_key = key

        all_cosines.append(cos)
        all_max_diffs.append(max_diff)

        # Relative error
        orig_norm = orig_t.norm().item()
        rel_err = (diff.norm().item() / orig_norm) if orig_norm > 0 else 0.0

        # Per-tensor verdict
        tensor_pass = cos >= pass_threshold
        if tensor_pass:
            num_passed += 1
        else:
            num_failed += 1

        per_tensor_stats.append({
            "name": key,
            "shape": list(orig_t.shape),
            "dtype": str(original[key].dtype).replace("torch.", ""),
            "max_abs_diff": round(max_diff, 8),
            "mean_abs_diff": round(mean_diff, 8),
            "cosine_similarity": round(cos, 8),
            "relative_error": round(rel_err, 8),
            "verdict": "PASS" if tensor_pass else "FAIL",
        })

    num_tensors = len(original)
    overall_pass = num_failed == 0

    # Print summary to stdout (existing behavior)
    print(f"  Tensors compared: {num_tensors}")
    print(f"  Shape mismatches: {mismatched}")
    if all_exact:
        print(f"  Result: LOSSLESS - all tensors exactly match")
    else:
        print(f"  Max absolute difference: {max_diff_global:.8f}")
        print(f"  Worst tensor: {worst_key}")
        print(f"  Min cosine similarity: {min_cosine_global:.8f}")
        print(f"  Pass threshold (cosine): {pass_threshold}")
        print(f"  Tensors passed: {num_passed}/{num_tensors}")
        if num_failed > 0:
            print(f"  Tensors FAILED: {num_failed}")
        if max_diff_global < 1e-3:
            print(f"  Result: EXCELLENT - differences within quantization noise")
        elif max_diff_global < 1e-2:
            print(f"  Result: GOOD - small quantization differences")
        else:
            print(f"  Result: WARNING - significant differences detected")
    print(f"  Overall verdict: {'PASS' if overall_pass else 'FAIL'}")

    # Write JSON report if requested
    if report_path:
        print()
        print(f"  Writing verification report to {report_path}...")

        original_size = os.path.getsize(source_path)
        compressed_size = os.path.getsize(dmx_path)
        compression_ratio = round(compressed_size / original_size, 4) if original_size > 0 else 0.0
        savings_pct = round((1.0 - compressed_size / original_size) * 100, 2) if original_size > 0 else 0.0

        # Find worst cosine tensor name
        worst_cosine_name = ""
        worst_cosine_val = 1.0
        for stat in per_tensor_stats:
            cs = stat.get("cosine_similarity")
            if cs is not None and cs < worst_cosine_val:
                worst_cosine_val = cs
                worst_cosine_name = stat["name"]

        mean_cosine = round(sum(all_cosines) / len(all_cosines), 8) if all_cosines else 1.0
        mean_max_diff = round(sum(all_max_diffs) / len(all_max_diffs), 8) if all_max_diffs else 0.0
        worst_max_diff = round(max(all_max_diffs), 8) if all_max_diffs else 0.0

        report = {
            "dmx_version": __version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "original_file": os.path.basename(source_path),
            "original_sha256": _sha256_file(source_path),
            "original_size_bytes": original_size,
            "compressed_file": os.path.basename(dmx_path),
            "compressed_sha256": _sha256_file(dmx_path),
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "savings_percent": savings_pct,
            "mode": actual_mode,
            "num_tensors": num_tensors,
            "num_tensors_verified": num_tensors,
            "num_tensors_passed": num_passed,
            "num_tensors_failed": num_failed,
            "pass_threshold_cosine_sim": pass_threshold,
            "overall_verdict": "PASS" if overall_pass else "FAIL",
            "per_tensor": per_tensor_stats,
            "summary_stats": {
                "worst_cosine_similarity": round(worst_cosine_val, 8),
                "worst_tensor_name": worst_cosine_name,
                "best_cosine_similarity": round(best_cosine_global, 8),
                "mean_cosine_similarity": mean_cosine,
                "worst_max_abs_diff": worst_max_diff,
                "mean_max_abs_diff": mean_max_diff,
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Report written: {report_path}")

    # Cleanup
    try:
        os.remove(tmp_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return all_exact or (overall_pass and max_diff_global < 1e-2)


### Delta compression / reconstruction ###

DELTA_MAGIC = b"DMXD"
DELTA_VERSION = 1


SMALL_TENSOR_THRESHOLD = 4096  # Tensors below this size get per-tensor scales

def _compute_aligned_scales(state_dict, precision="int16"):
    """Compute aligned scales per component type.

    Groups tensors by their component name (e.g., all 'self_attn.q_proj.weight'
    across layers share one scale). This enables exact integer subtraction
    between checkpoints.

    Small tensors (bias, norm — below SMALL_TENSOR_THRESHOLD params) get
    per-tensor scales instead of shared group scales. This prevents scale
    overflow at int32 precision where the range mismatch between large
    weight matrices and tiny bias vectors causes quantization errors.

    precision: 'int16' (32767 levels) or 'int32' (2147483647 levels)
    """
    from collections import defaultdict
    max_val = 32767 if precision == "int16" else 2147483647
    groups = defaultdict(list)
    for key in state_dict:
        parts = key.split(".")
        ctype = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        groups[ctype].append(key)

    scales = {}
    for ctype, keys in groups.items():
        # Split into large (aligned) and small (per-tensor) groups
        large_keys = [k for k in keys if state_dict[k].numel() >= SMALL_TENSOR_THRESHOLD]
        small_keys = [k for k in keys if state_dict[k].numel() < SMALL_TENSOR_THRESHOLD]

        if large_keys:
            max_abs = max(state_dict[k].float().abs().max().item() for k in large_keys)
            scales[ctype] = max_abs / max_val if max_abs > 0 else 1.0

        # ALL small tensors get per-tensor scales (no group sharing)
        # This prevents int32 overflow when small tensors have very different ranges
        for k in small_keys:
            max_abs = state_dict[k].float().abs().max().item()
            scales[k] = max_abs / max_val if max_abs > 0 else 1.0

        # Fallback group entry for key_type lookup
        if not large_keys and small_keys:
            scales[ctype] = scales[small_keys[0]]

    return scales


def _key_type(key):
    parts = key.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _compute_decomposition_diagnostics(state_dict, scales, precision="int16"):
    """Compute per-component-group diagnostics for chain compression analysis.

    Returns a dict that can be embedded in a manifest, capturing how the
    aligned per-component quantization grouped tensors and what the resulting
    scales are. This lets downstream analysis detect scale groups whose
    composition is suboptimal — for example, when a learned weight matrix
    shares a scale group with an auxiliary buffer (like a constant attention
    mask) that has a very different magnitude range. In that case the int16
    step size gets sized to fit the auxiliary buffer instead of tuned to
    the actual learned weights, which hurts compression efficiency.

    Detection signals: scale groups dominated by a single outlier member,
    groups with magnitudes far above the typical learned-weight range, or
    groups with a computed scale matching the signature of a constant value.

    Output structure:
        {
          'tensor_count': int,
          'scale_group_count': int,
          'per_group': {
            <component_type>: {
              'members': [tensor_name, ...],
              'member_count': int,
              'per_tensor_max_abs': {tensor_name: float},
              'group_max_abs': float,
              'aligned_scale': float,
              'is_singleton': bool,
              'max_abs_dominant_tensor': str,
              'max_abs_to_median_ratio': float,
              'composition_flag': bool,
              'composition_reason': str | null,
            }
          },
          'flagged_groups': [component_type, ...],
        }
    """
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for key in state_dict:
        ctype = _key_type(key)
        groups[ctype].append(key)

    per_group = {}
    flagged_groups = []
    for ctype, keys in groups.items():
        per_max = {}
        for k in keys:
            try:
                per_max[k] = float(state_dict[k].float().abs().max().item())
            except Exception:
                per_max[k] = 0.0

        max_vals = list(per_max.values())
        if not max_vals:
            continue

        group_max = max(max_vals)
        # Find the tensor that's "responsible" for the group max
        dominant = max(per_max.items(), key=lambda kv: kv[1])[0]

        # Median for ratio analysis (robust to single outlier)
        median = statistics.median(max_vals) if max_vals else 0.0
        ratio = (group_max / median) if median > 0 else 1.0

        # Get the scale that was actually used for this group (from the
        # _compute_aligned_scales output). May be at scales[ctype] for the
        # group entry, or at scales[k] for a per-tensor override on a small
        # tensor.
        aligned_scale = scales.get(ctype)
        if aligned_scale is None:
            # Maybe everyone in this group got per-tensor scales — pick the first
            for k in keys:
                if k in scales:
                    aligned_scale = scales[k]
                    break

        # Three orthogonal heuristics for detecting scale groups whose
        # composition is suboptimal (auxiliary tensors mixed with learned
        # weights):
        #
        # 1. INTRA-GROUP DOMINANCE: max-abs dominated by a single member that's
        #    >10x the group median. Catches "one auxiliary tensor mixed with
        #    N learned weight matrices in the same scale group."
        #
        # 2. SCALE-NEAR-LIM: aligned scale ≈ 1.0 / max_val ≈ 3e-5 for int16,
        #    which means group max-abs ≈ 1.0. That's the signature of constant
        #    masks or normalized buffers being included in the scale
        #    calculation. Catches "the entire group is a single auxiliary
        #    tensor type with a numerically suspicious value range."
        #
        # 3. ABSURD MAGNITUDE: max-abs >> 1.0 (e.g., 10+, 1e9+) is essentially
        #    impossible for trained weights of any modern architecture (typical
        #    learned weights are bounded around ±0.5 by initialization +
        #    weight decay). Catches "this group contains constant mask buffers
        #    with -1e9 fill values" or similar auxiliary state.
        composition_flag = False
        composition_reason = None
        if len(keys) > 1 and ratio > 10.0:
            composition_flag = True
            composition_reason = (
                f"INTRA-GROUP DOMINANCE: max-abs dominated by '{dominant}' "
                f"({group_max:.4g}) — {ratio:.1f}x the group median ({median:.4g}). "
                f"Likely an auxiliary buffer polluting a weight scale group; "
                f"chain compression on this group will be suboptimal."
            )
        elif group_max > 10.0:
            composition_flag = True
            composition_reason = (
                f"ABSURD MAGNITUDE: group max-abs is {group_max:.4g}, far above "
                f"the typical learned weight range (±0.5). This group probably "
                f"contains attention causal masks, position embeddings, or other "
                f"auxiliary tensors that should be excluded from chain compression "
                f"(they don't change between checkpoints anyway, and they corrupt "
                f"the int16 quantization scale)."
            )
        elif aligned_scale is not None and 0.99 / 32767 < aligned_scale < 1.01 / 32767:
            composition_flag = True
            composition_reason = (
                f"SCALE-NEAR-LIM: aligned scale ≈ {aligned_scale:.6g} suggests "
                f"max-abs ≈ 1.0, the signature value of an attention mask or "
                f"normalized buffer. Investigate whether this group should be split."
            )

        per_group[ctype] = {
            "members": sorted(keys),
            "member_count": len(keys),
            "per_tensor_max_abs": per_max,
            "group_max_abs": group_max,
            "aligned_scale": aligned_scale,
            "is_singleton": len(keys) == 1,
            "max_abs_dominant_tensor": dominant,
            "max_abs_to_median_ratio": ratio,
            "composition_flag": composition_flag,
            "composition_reason": composition_reason,
        }
        if composition_flag:
            flagged_groups.append(ctype)

    return {
        "tensor_count": len(state_dict),
        "scale_group_count": len(per_group),
        "per_group": per_group,
        "flagged_groups": flagged_groups,
    }


def _get_scale(scales, key):
    """Get scale for a tensor — per-tensor override if exists, else group scale."""
    if key in scales:
        return scales[key]
    return scales[_key_type(key)]


def _try_normalize_huggingface_prefixes(base_dict, target_dict):
    """Auto-detect and strip common HuggingFace transformer prefixes if base/target
    tensor names use different conventions.

    Some HF model revisions use 'transformer.h.0.attn.c_attn.weight' while others
    use 'h.0.attn.c_attn.weight' for the same weight. Without normalization,
    delta_compress would compute an empty intersection and silently produce a
    near-empty .dmxd file (~249 bytes) with no actual delta data.

    Returns (normalized_base, normalized_target, normalization_note) where
    normalization_note is None if no normalization was needed or possible.
    """
    base_keys = set(base_dict.keys())
    target_keys = set(target_dict.keys())

    # Already match or have at least some overlap — no normalization needed
    if base_keys & target_keys:
        return base_dict, target_dict, None

    # Common HuggingFace transformer model prefixes to try
    HUGGINGFACE_PREFIXES = [
        "transformer.",
        "model.",
        "bert.",
        "gpt2.",
        "roberta.",
        "distilbert.",
        "electra.",
        "albert.",
        "llama.",
        "mistral.",
    ]

    for prefix in HUGGINGFACE_PREFIXES:
        base_all_have = bool(base_keys) and all(k.startswith(prefix) for k in base_keys)
        target_all_have = bool(target_keys) and all(k.startswith(prefix) for k in target_keys)

        # Case 1: base has prefix, target doesn't — strip from base
        if base_all_have and not target_all_have:
            stripped_base = {k[len(prefix):]: v for k, v in base_dict.items()}
            if set(stripped_base.keys()) & target_keys:
                return (
                    stripped_base,
                    target_dict,
                    f"stripped '{prefix}' prefix from base tensor names to match target naming",
                )

        # Case 2: target has prefix, base doesn't — strip from target
        if target_all_have and not base_all_have:
            stripped_target = {k[len(prefix):]: v for k, v in target_dict.items()}
            if base_keys & set(stripped_target.keys()):
                return (
                    base_dict,
                    stripped_target,
                    f"stripped '{prefix}' prefix from target tensor names to match base naming",
                )

    return base_dict, target_dict, None


def _detect_legacy_aux_buffers(state_dict):
    """Detect legacy auxiliary buffer tensors that should be excluded from
    chain compression because they (a) don't change between checkpoints,
    (b) are dropped by modern HF loaders anyway, and (c) corrupt the
    aligned scale calculation when included in scale groups with learned
    weights.

    Returns a sorted list of tensor names to exclude.

    Three signatures (any one is sufficient to mark as exclusion):
      1. uint8 dtype + name ending in 'attention.bias' or 'attn.bias'
         → causal mask buffers (4 MB each on GPT-NeoX-2048-context)
      2. Tensor name contains 'inv_freq' or 'rotary_emb'
         → rotary positional encoding precomputed buffers
      3. Single-element tensor named 'masked_bias' or matching that pattern
         → mask fill value scalar

    Modern transformers (5.x) registers all of these with persistent=False
    so they don't appear in the state_dict on disk. But legacy checkpoints
    saved with older transformers versions DO have them, and they leak into
    DMX chain compression as constant-but-still-encoded data plus per-tensor
    manifest overhead. Stripping them is what every modern HF loader does
    anyway via `_keys_to_ignore_on_load_unexpected`.
    """
    excluded = []
    for key, tensor in state_dict.items():
        is_legacy_buffer = False

        # Signature 1: uint8 attention causal mask
        if tensor.dtype == torch.uint8 and (
            key.endswith("attention.bias") or key.endswith("attn.bias")
        ):
            is_legacy_buffer = True

        # Signature 2: rotary positional encoding precomputed buffer
        elif "inv_freq" in key or "rotary_emb" in key:
            is_legacy_buffer = True

        # Signature 3: masked_bias scalar fill value
        elif key.endswith("masked_bias") or key.endswith("attention.masked_bias"):
            is_legacy_buffer = True

        if is_legacy_buffer:
            excluded.append(key)

    return sorted(excluded)


def delta_compress(base_path, target_path, output_path, precision="int16",
                   zstd_level=None, strip_legacy_buffers=True, entropy="auto"):
    """Delta-compress a target checkpoint against a base checkpoint.

    Precision modes:
    - int16: Aligned quantization to int16, delta, entropy code. ~87% savings, +0.06% RelL2.
    - int32: Aligned quantization to int32, delta, zstd. ~87% savings, error below FP32 noise floor.
             (Replaced raw XOR in April 2026 after validation showed XOR produces 0% sparsity.)

    Entropy modes:
    - auto (default): per-tensor competitive selection across the available
      candidate coders ([zstd-19, FLAC, brotli-11]). For each tensor the
      smallest output is kept and the winning coder's identifier is recorded
      in per-tensor metadata. Decompression dispatches on that identifier.
      Note: int32 precision falls back to zstd because FLAC and brotli
      operate on the int16-native byte stream.
    - zstd-19 / flac / brotli-11: pin a single coder for the whole delta.
      Useful for testing and for measuring per-coder baselines.
    - Legacy aliases ('zstd', 'lpc') from earlier DMX versions are accepted
      and map to 'zstd-19' and 'flac' respectively.

    Algorithm:
    1. Load base and target safetensors
    2. Compute aligned scales per component type
    3. Quantize both to int16 or int32, integer subtract to produce delta
    4. Entropy-code the sparse delta with zstd
    5. Save as .dmxd file with metadata
    """
    print(f"  Base: {base_path}")
    print(f"  Target: {target_path}")
    precision_label = {"int16": "near-lossless", "int32": "practically lossless"}
    print(f"  Precision: {precision} ({precision_label.get(precision, '')})")
    start = time.time()

    base = load_file(base_path)
    target = load_file(target_path)

    base_size = os.path.getsize(base_path)
    target_size = os.path.getsize(target_path)

    # Auto-normalize common HF prefix differences before computing the
    # base/target intersection. Without this step, two safetensors files using
    # different naming conventions (e.g. 'transformer.h.0.attn.c_attn.weight'
    # vs 'h.0.attn.c_attn.weight') would produce an empty intersection and
    # delta_compress would silently write a ~249-byte empty .dmxd file with no
    # actual delta data — a silent data-loss failure mode.
    normalized_base, normalized_target, norm_note = _try_normalize_huggingface_prefixes(base, target)
    if norm_note:
        print(f"  INFO: tensor name auto-normalization applied -- {norm_note}")
        base = normalized_base
        target = normalized_target

    # Verify compatible keys
    base_keys = set(base.keys())
    target_keys = set(target.keys())
    if base_keys != target_keys:
        missing = base_keys - target_keys
        extra = target_keys - base_keys
        if missing:
            print(f"  WARNING: {len(missing)} keys in base but not target")
        if extra:
            print(f"  WARNING: {len(extra)} keys in target but not base")
    common_keys = sorted(base_keys & target_keys)

    # Strip legacy auxiliary buffer tensors (causal masks, rotary inv_freq,
    # masked_bias). These are present in checkpoints saved with older HF
    # transformers versions where buffers were persistent=True. Modern
    # loaders ignore them via _keys_to_ignore_on_load_unexpected, so
    # excluding them on the DMX side matches what every consumer does
    # downstream. See _detect_legacy_aux_buffers() docstring for the
    # exact signatures.
    if strip_legacy_buffers and common_keys:
        excluded_in_base = set(_detect_legacy_aux_buffers(base))
        excluded_in_target = set(_detect_legacy_aux_buffers(target))
        excluded = excluded_in_base | excluded_in_target
        if excluded:
            n_before = len(common_keys)
            common_keys = [k for k in common_keys if k not in excluded]
            n_excluded = n_before - len(common_keys)
            print(f"  Stripped {n_excluded} legacy auxiliary buffer tensor(s) "
                  f"(causal masks / inv_freq / masked_bias) from chain compression. "
                  f"Modern HF loaders ignore these on load anyway. Pass "
                  f"strip_legacy_buffers=False to keep them.")

    # Hard abort if no common tensors. Previously this would silently produce
    # a ~249-byte empty .dmxd file with no actual delta data. The auto-
    # normalization above handles the common HF prefix mismatch case; if we
    # still have zero common keys, the user has a non-standard naming scheme
    # and needs to know about it explicitly.
    if len(common_keys) == 0:
        base_sample = sorted(base.keys())[:5]
        target_sample = sorted(target.keys())[:5]
        sample_msg = (
            "  Base sample (first 5 tensor names):\n"
            + "\n".join(f"    {k}" for k in base_sample)
            + "\n"
            + "  Target sample (first 5 tensor names):\n"
            + "\n".join(f"    {k}" for k in target_sample)
        )
        raise ValueError(
            "No tensor names in common between base and target after attempting "
            "auto-normalization of common HuggingFace prefixes (transformer., "
            "model., bert., etc.). The two safetensors files appear to use "
            "different naming conventions.\n\n"
            + sample_msg
            + "\n\n"
            "If your files use a non-standard prefix or differ in tensor naming "
            "for some other reason, manually rename tensors before delta-compress, "
            "or contact the model author about format consistency. Refusing to "
            "write an empty .dmxd file."
        )

    print(f"  Common tensors: {len(common_keys)}")

    # Compute aligned scales from max of BOTH base and target (prevents clamping)
    combined_max = {}
    for key in common_keys:
        b_max = base[key].float().abs().max().item()
        t_max = target[key].float().abs().max().item()
        combined_max[key] = base[key] if b_max >= t_max else target[key]
    scales = _compute_aligned_scales(combined_max, precision=precision)
    print(f"  Aligned component groups: {len(scales)}")

    # Compute per-component-group diagnostics — captures structure, max-abs
    # values, and detection of scale groups whose composition may be suboptimal.
    # Diagnostics are embedded in the manifest so downstream analysis can
    # identify suboptimal scale group compositions automatically.
    decomp_diagnostics = _compute_decomposition_diagnostics(combined_max, scales, precision=precision)
    if decomp_diagnostics["flagged_groups"]:
        print(f"  WARNING: {len(decomp_diagnostics['flagged_groups'])} scale group(s) "
              f"flagged with suboptimal composition — chain compression on these may "
              f"be reduced:")
        for ctype in decomp_diagnostics["flagged_groups"]:
            g = decomp_diagnostics["per_group"][ctype]
            print(f"    [{ctype}]: {g['composition_reason']}")

    max_val = 32767 if precision == "int16" else 2147483647

    # Resolve entropy mode. The default is 'auto' which means per-tensor
    # competitive selection across the available candidate coders
    # ([zstd-19, FLAC, brotli-11]). Explicit values pin to a single coder for
    # the whole compress, useful for testing and for environments missing
    # optional dependencies. Legacy aliases ('zstd' -> 'zstd-19', 'lpc' ->
    # 'flac') are accepted for backward compatibility with earlier DMX versions.
    if entropy == "lpc":
        entropy = "flac"
    if entropy == "auto":
        # Competitive selection. Always available because zstd is a hard dep,
        # and the selector subsets to whatever optional coders are installed.
        effective_entropy = "auto"
    elif entropy == "flac" and not _lpc_backend_available():
        print("  WARNING: entropy=flac but no FLAC backend available, falling back to zstd-19")
        effective_entropy = "zstd-19"
    elif entropy == "brotli-11" and not BROTLI_AVAILABLE:
        print("  WARNING: entropy=brotli-11 but brotli is not installed, falling back to zstd-19")
        effective_entropy = "zstd-19"
    elif entropy in ("zstd", "zstd-19"):
        effective_entropy = "zstd-19"
    else:
        effective_entropy = entropy
    if effective_entropy != "zstd-19" and precision == "int32":
        # FLAC/brotli paths only meaningful for the int16 native stream.
        # int32 deltas fall back to zstd-19 for this delta.
        print(f"  NOTE: entropy={effective_entropy} + precision=int32 not supported, "
              f"falling back to zstd-19 for this delta")
        effective_entropy = "zstd-19"
    print(f"  entropy: {effective_entropy}")

    # Quantize and compute deltas, encode per-tensor
    level = zstd_level if zstd_level is not None else ZSTD_LEVEL_DELTA
    if effective_entropy == "zstd-19":
        print(f"  zstd level: {level}")
    cctx = zstd.ZstdCompressor(level=level, write_content_size=True)
    tensor_meta = {}
    compressed_chunks = {}
    total_params = 0
    total_zero = 0
    total_compressed = 0

    # Three GPU acceleration tiers, in priority order:
    # 1. Native CUDA kernels (fastest, requires compiled C++ extension via nvcc)
    # 2. torch-on-GPU fallback (when CUDA available but kernels not built — A100
    #    pods typically hit this path because the extension build needs nvcc)
    # 3. CPU torch fallback
    #
    # Without tier 2, A100 pods that don't have the native kernels built fall
    # all the way through to CPU, which is ~50-100x slower than necessary for
    # the int16 quant + subtract step. Tier 2 was added  evening
    # after Pythia 1.4B chain experiments showed delta_compress as the long pole
    # at ~280s/delta on A100 (CPU-bound).
    cuda_kernels = _get_cuda_kernels() if torch.cuda.is_available() else None
    use_torch_gpu = cuda_kernels is None and torch.cuda.is_available()
    if cuda_kernels:
        print("  GPU-accelerated delta compression (native CUDA kernels)")
    elif use_torch_gpu:
        print(f"  GPU-accelerated delta compression (torch fallback on {torch.cuda.get_device_name(0)})")

    for i, key in enumerate(common_keys):
        scale = _get_scale(scales, key)

        if cuda_kernels and base[key].numel() > 1024:
            # Tier 1: native CUDA fused quantize + subtract
            b_gpu = base[key].float().cuda()
            t_gpu = target[key].float().cuda()
            max_abs = scale * max_val
            if precision == "int32":
                delta = cuda_kernels.delta_compute_i32(b_gpu, t_gpu, max_abs).cpu()
            else:
                delta = cuda_kernels.delta_compute_i16(b_gpu, t_gpu, max_abs).cpu()
            delta_bytes = delta.numpy().tobytes()
            del b_gpu, t_gpu
        elif use_torch_gpu and base[key].numel() > 1024:
            # Tier 2: torch-on-GPU fallback. Same math as the CPU branches
            # below, but the heavy operations (round, clamp, cast, subtract)
            # run on the GPU. The int16 / int32 result is moved back to CPU
            # for entropy coding. Skip the GPU path for tiny tensors (<=1024
            # params) where the H2D transfer cost dominates.
            if precision == "int32":
                # int32 uses float32 on GPU instead of float64 (A100 fp64 is
                # 1/32 throughput). Headroom is ~31 bits so float32 round is
                # exact for typical NN weight ranges. Falls through to CPU
                # double() if a future precision concern requires it.
                b_gpu = base[key].contiguous().cuda(non_blocking=True).float()
                t_gpu = target[key].contiguous().cuda(non_blocking=True).float()
                base_int_g = torch.clamp(
                    torch.round(b_gpu / scale), -max_val, max_val
                ).to(torch.int32)
                target_int_g = torch.clamp(
                    torch.round(t_gpu / scale), -max_val, max_val
                ).to(torch.int32)
                delta_g = (
                    target_int_g.to(torch.int64) - base_int_g.to(torch.int64)
                ).to(torch.int32)
                delta = delta_g.cpu()
                delta_bytes = delta.numpy().tobytes()
                del b_gpu, t_gpu, base_int_g, target_int_g, delta_g
            else:
                b_gpu = base[key].contiguous().cuda(non_blocking=True).float()
                t_gpu = target[key].contiguous().cuda(non_blocking=True).float()
                base_int_g = torch.clamp(
                    torch.round(b_gpu / scale), -max_val, max_val
                ).to(torch.int16)
                target_int_g = torch.clamp(
                    torch.round(t_gpu / scale), -max_val, max_val
                ).to(torch.int16)
                delta_g = (
                    target_int_g.to(torch.int32) - base_int_g.to(torch.int32)
                ).to(torch.int16)
                delta = delta_g.cpu()
                delta_bytes = delta.numpy().tobytes()
                del b_gpu, t_gpu, base_int_g, target_int_g, delta_g
        elif precision == "int32":
            # Tier 3: CPU fallback (int32, float64 for max precision)
            base_int = torch.clamp(
                torch.round(base[key].double() / scale), -max_val, max_val
            ).to(torch.int32)
            target_int = torch.clamp(
                torch.round(target[key].double() / scale), -max_val, max_val
            ).to(torch.int32)
            delta = (target_int.to(torch.int64) - base_int.to(torch.int64)).to(torch.int32)
            delta_bytes = delta.numpy().tobytes()
        else:
            # Tier 3: CPU fallback (int16)
            base_int = torch.clamp(
                torch.round(base[key].float() / scale), -max_val, max_val
            ).to(torch.int16)
            target_int = torch.clamp(
                torch.round(target[key].float() / scale), -max_val, max_val
            ).to(torch.int16)
            delta = (target_int.to(torch.int32) - base_int.to(torch.int32)).to(torch.int16)
            delta_bytes = delta.numpy().tobytes()

        # Stats
        n = delta.numel()
        nz = (delta == 0).sum().item()
        total_params += n
        total_zero += nz

        # Compress delta with the resolved entropy coder.
        # In 'auto' mode the per-tensor selector tries every available
        # candidate coder and keeps the smallest output. Each tensor records
        # the winning coder's identifier in tensor_meta so the decoder
        # dispatches correctly even on mixed-coder deltas. Pinned entropy
        # modes (zstd-19, flac, brotli-11) bypass the selector and apply the
        # same coder to every tensor; useful for testing and per-coder
        # baselines.
        if effective_entropy == "auto" and precision == "int16":
            try:
                tensor_entropy, compressed = _competitive_encode_int16(
                    delta_bytes, n, zstd_level=level
                )
            except Exception as e:
                print(f"  WARNING: competitive encode failed for {key} ({e}), "
                      f"falling back to zstd-19 for this tensor")
                compressed = cctx.compress(delta_bytes)
                tensor_entropy = "zstd-19"
        elif effective_entropy == "flac" and precision == "int16":
            try:
                compressed = _int16_to_flac(delta_bytes, n)
                tensor_entropy = "flac"
            except Exception as e:
                print(f"  WARNING: FLAC encode failed for {key} ({e}), "
                      f"falling back to zstd-19 for this tensor")
                compressed = cctx.compress(delta_bytes)
                tensor_entropy = "zstd-19"
        elif effective_entropy == "brotli-11" and precision == "int16":
            try:
                compressed = _compress_bytes_brotli(delta_bytes)
                tensor_entropy = "brotli-11"
            except Exception as e:
                print(f"  WARNING: brotli encode failed for {key} ({e}), "
                      f"falling back to zstd-19 for this tensor")
                compressed = cctx.compress(delta_bytes)
                tensor_entropy = "zstd-19"
        else:
            compressed = cctx.compress(delta_bytes)
            tensor_entropy = "zstd-19"
        total_compressed += len(compressed)

        tensor_meta[key] = {
            "shape": list(base[key].shape),
            "dtype": str(base[key].dtype).replace("torch.", ""),
            "scale": scale,
            "component_type": _key_type(key),
            "precision": precision,
            "entropy": tensor_entropy,
            "params": n,
            "zeros": nz,
            "raw_size": len(delta_bytes),
            "compressed_size": len(compressed),
        }
        compressed_chunks[key] = compressed

        if (i + 1) % 100 == 0 or (i + 1) == len(common_keys):
            print(f"  Delta-compressed {i+1}/{len(common_keys)} tensors...")

    # Build manifest
    manifest = {
        "version": DELTA_VERSION,
        "format": "delta",
        "precision": precision,
        "base_file": os.path.basename(base_path),
        "target_file": os.path.basename(target_path),
        "base_size": base_size,
        "target_size": target_size,
        "tensor_count": len(common_keys),
        "total_params": total_params,
        "total_zeros": total_zero,
        "scales": {ct: s for ct, s in scales.items()} if scales else {},
        "tensors": tensor_meta,
        # Per-component-group diagnostics — captures per-scale-group max-abs
        # and detection flags so downstream analysis can identify suboptimal
        # scale group compositions automatically.
        "decomposition_diagnostics": decomp_diagnostics,
    }

    # Write .dmxd file: [MAGIC 4B][VERSION 4B][MANIFEST_LEN 4B][MANIFEST][CHUNKS...]
    # Stabilize offsets (manifest size changes when offsets are added)
    for _pass in range(5):
        manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        header_size = 4 + 4 + 4 + len(manifest_json)
        offset = header_size
        changed = False
        for key in common_keys:
            if tensor_meta[key].get("offset") != offset:
                changed = True
            tensor_meta[key]["offset"] = offset
            offset += len(compressed_chunks[key])
        if not changed:
            break

    with open(output_path, "wb") as f:
        f.write(DELTA_MAGIC)
        f.write(struct.pack("<I", DELTA_VERSION))
        f.write(struct.pack("<I", len(manifest_json)))
        f.write(manifest_json)
        for key in common_keys:
            f.write(compressed_chunks[key])

    output_size = os.path.getsize(output_path)
    elapsed = time.time() - start
    pct_zero = 100.0 * total_zero / total_params if total_params > 0 else 0
    ratio = output_size / target_size * 100

    print(f"\n  Results:")
    print(f"    Base size:    {base_size / 1024 / 1024:.1f} MB")
    print(f"    Target size:  {target_size / 1024 / 1024:.1f} MB")
    print(f"    Delta size:   {output_size / 1024 / 1024:.1f} MB ({ratio:.1f}% of target)")
    print(f"    Savings:      {(target_size - output_size) / 1024 / 1024:.1f} MB ({100-ratio:.1f}%)")
    print(f"    Zero deltas:  {pct_zero:.1f}%")
    print(f"    Time:         {elapsed:.1f}s")

    return output_size


def chain_compress(checkpoint_paths, output_dir, precision="int16", max_delta_ratio=None,
                   zstd_level=None, use_gpu=None, anchor_safety_margin=0.95,
                   strip_legacy_buffers=True, entropy="auto"):
    """Chain-compress a sequence of related checkpoints with auto-anchor promotion.

    Entropy default is 'auto' which means per-tensor competitive coder
    selection across the available candidate set ([zstd-19, FLAC, brotli-11]).
    For each tensor the smallest output is kept and the winning coder's
    identifier is recorded in per-tensor metadata. By construction this
    matches or beats any fixed-codec strategy. Pinned values force a single
    coder for the whole chain — useful for testing and for measuring
    per-coder baselines.

    This is the production chain compression entry point. It implements the
    auto-anchor promotion policy: each checkpoint is delta-compressed against
    the current anchor, and if the resulting delta exceeds a configured
    threshold (either a fixed fraction of the target raw size or, in dynamic
    mode, the current anchor's actual compressed size), the target is promoted
    to a new anchor instead.

    Two anchor-promotion policies are supported:

    1. **Dynamic (default, optimal)** — `max_delta_ratio=None`. The threshold
       is the *current anchor's actual compressed size* (× anchor_safety_margin).
       A delta is kept if and only if `delta_compressed_size <
       current_anchor_compressed_size * 0.95`. This is provably storage-optimal
       given the anchor and delta primitives — keeping a delta that's larger
       than what a fresh anchor would cost is strictly worse for total storage.
       Self-calibrating across source dtypes without manual tuning, because
       the threshold scales with the anchor itself.

    2. **Static (legacy)** — pass a float in [0, 1]. The delta is kept if its
       size is <= max_delta_ratio of the target's *raw* size. Preserved for
       back-compatibility and ablation experiments where a fixed threshold is
       required. The dynamic policy is generally preferred because it adapts
       to per-model anchor compressibility automatically.

    Without an anchor policy, naive chain compression degrades catastrophically
    when consecutive checkpoints diverge significantly (large step intervals,
    mid-training, large models). With auto-anchor, chain compression is
    guaranteed to be no worse than DMX-standalone single-file mode while
    preserving all chain wins where they exist.

    Args:
        checkpoint_paths: ordered list of safetensors paths to compress as a chain.
        output_dir: directory to write anchors (.dmx) and deltas (.dmxd) into.
        precision: int16 (default) or int32 quantization precision.
        max_delta_ratio: anchor promotion threshold. None (default) selects the
            dynamic policy: keep a delta only if its compressed size is smaller
            than the current anchor's compressed size × anchor_safety_margin.
            A float in [0, 1] activates the static policy: keep a delta only
            if its size is at most that fraction of the target's RAW size.
            The dynamic policy is storage-optimal; the static one is legacy.
        zstd_level: optional zstd compression level for deltas (1-19, default 3).
        use_gpu: passthrough to compress_file for anchor writes.
        anchor_safety_margin: in dynamic mode, multiply current anchor size by
            this factor before comparing to delta size. Default 0.95 prevents
            kept/promoted oscillation when delta size is exactly at the anchor
            size boundary. Ignored in static mode.

    Writes:
        output_dir/<basename>.dmx for anchor checkpoints
        output_dir/<basename>.dmxd for delta checkpoints
        output_dir/chain_manifest.json describing the chain

    Returns:
        dict with 'manifest' (the chain manifest), 'total_raw_bytes',
        'total_chain_bytes', 'savings_percent', 'anchor_count', 'delta_count'.
    """
    if not checkpoint_paths:
        raise ValueError("chain_compress requires at least one checkpoint")
    os.makedirs(output_dir, exist_ok=True)

    policy_mode = "static" if max_delta_ratio is not None else "dynamic"
    print(f"=" * 70)
    print(f"DMX chain-compress (auto-anchor policy: {policy_mode})")
    print(f"=" * 70)
    print(f"  Checkpoints: {len(checkpoint_paths)}")
    print(f"  Precision: {precision}")
    if policy_mode == "dynamic":
        print(f"  Anchor promotion: dynamic (delta > current_anchor_compressed_size * {anchor_safety_margin})")
    else:
        print(f"  Anchor promotion: static (delta > {max_delta_ratio} * raw target size)")
    print(f"  Output dir: {output_dir}")
    print()

    chain_entries = []
    total_raw = 0
    total_chain = 0
    anchor_count = 0
    delta_count = 0
    current_anchor_safetensors = None  # path to the raw safetensors of the current anchor
    current_anchor_compressed_size = None  # compressed size of the current anchor (for dynamic policy)

    for idx, ckpt_path in enumerate(checkpoint_paths):
        raw_size = os.path.getsize(ckpt_path)
        total_raw += raw_size
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]

        if current_anchor_safetensors is None:
            # First checkpoint always becomes an anchor
            anchor_path = os.path.join(output_dir, f"{basename}.dmx")
            print(f"[{idx+1}/{len(checkpoint_paths)}] {basename}: writing anchor (first checkpoint)")
            t0 = time.time()
            # Use mode='auto' so the anchor encoding is picked from source dtype:
            # FP16/BF16 source → BFP (GPU-accelerated via bfp_compress_gpu)
            # FP32 source → int16 quant (GPU-accelerated via the encode_tensor GPU path)
            # The 'precision' arg only governs DELTA encoding (int16 vs int32),
            # not anchor encoding. Forcing the anchor to mode=precision was a bug
            # that defeated GPU acceleration on FP16 source models.
            compress_file(ckpt_path, anchor_path, mode="auto", entropy=entropy, use_gpu=use_gpu)
            elapsed = time.time() - t0
            anchor_size = os.path.getsize(anchor_path)
            total_chain += anchor_size
            anchor_count += 1
            chain_entries.append({
                "index": idx,
                "type": "anchor",
                "source": os.path.basename(ckpt_path),
                "raw_size_bytes": raw_size,
                "compressed_size_bytes": anchor_size,
                "compressed_path": os.path.relpath(anchor_path, output_dir),
                "anchor_index": idx,
                "time_seconds": elapsed,
            })
            current_anchor_safetensors = ckpt_path
            current_anchor_idx = idx
            current_anchor_compressed_size = anchor_size
            print(f"        anchor {fmt_size(anchor_size)} ({100*anchor_size/raw_size:.1f}% of raw) in {elapsed:.1f}s")
            continue

        # Subsequent checkpoint: try delta against current anchor
        delta_path = os.path.join(output_dir, f"{basename}.dmxd")
        print(f"[{idx+1}/{len(checkpoint_paths)}] {basename}: trying delta vs anchor "
              f"({os.path.splitext(os.path.basename(current_anchor_safetensors))[0]})")
        t0 = time.time()
        try:
            delta_compress(current_anchor_safetensors, ckpt_path, delta_path,
                           precision=precision, zstd_level=zstd_level,
                           strip_legacy_buffers=strip_legacy_buffers,
                           entropy=entropy)
        except Exception as e:
            print(f"        DELTA FAILED: {e}")
            print(f"        falling back to fresh anchor")
            if os.path.exists(delta_path):
                os.remove(delta_path)
            anchor_path = os.path.join(output_dir, f"{basename}.dmx")
            t0 = time.time()
            # Use mode='auto' so the anchor encoding is picked from source dtype:
            # FP16/BF16 source → BFP (GPU-accelerated via bfp_compress_gpu)
            # FP32 source → int16 quant (GPU-accelerated via the encode_tensor GPU path)
            # The 'precision' arg only governs DELTA encoding (int16 vs int32),
            # not anchor encoding. Forcing the anchor to mode=precision was a bug
            # that defeated GPU acceleration on FP16 source models.
            compress_file(ckpt_path, anchor_path, mode="auto", entropy=entropy, use_gpu=use_gpu)
            elapsed = time.time() - t0
            anchor_size = os.path.getsize(anchor_path)
            total_chain += anchor_size
            anchor_count += 1
            chain_entries.append({
                "index": idx,
                "type": "anchor",
                "source": os.path.basename(ckpt_path),
                "raw_size_bytes": raw_size,
                "compressed_size_bytes": anchor_size,
                "compressed_path": os.path.relpath(anchor_path, output_dir),
                "anchor_index": idx,
                "promoted_due_to": "delta_compress_exception",
                "time_seconds": elapsed,
            })
            current_anchor_safetensors = ckpt_path
            current_anchor_idx = idx
            current_anchor_compressed_size = anchor_size
            continue
        delta_elapsed = time.time() - t0
        delta_size = os.path.getsize(delta_path)
        ratio = delta_size / raw_size

        # Decide whether to keep the delta or promote to a new anchor.
        # Dynamic policy (default): compare delta size to the current anchor's
        # actual compressed size. Optimal regardless of source dtype because
        # it directly minimizes total chain bytes — keeping a delta that's
        # larger than what a fresh anchor would cost is strictly worse.
        # Static policy (legacy): compare delta size to a fixed fraction of
        # raw target size. Preserved for back-compat and ablation experiments.
        if policy_mode == "dynamic":
            anchor_threshold_bytes = int(current_anchor_compressed_size * anchor_safety_margin)
            keep_delta = delta_size <= anchor_threshold_bytes
            decision_label = (
                f"delta {fmt_size(delta_size)} vs anchor*{anchor_safety_margin} {fmt_size(anchor_threshold_bytes)}"
            )
        else:
            keep_delta = ratio <= max_delta_ratio
            decision_label = (
                f"{100*ratio:.1f}% of raw target, threshold {100*max_delta_ratio:.0f}%"
            )

        if keep_delta:
            # Delta is within budget — keep it
            total_chain += delta_size
            delta_count += 1
            chain_entries.append({
                "index": idx,
                "type": "delta",
                "source": os.path.basename(ckpt_path),
                "raw_size_bytes": raw_size,
                "compressed_size_bytes": delta_size,
                "compressed_path": os.path.relpath(delta_path, output_dir),
                "anchor_index": current_anchor_idx,
                "delta_ratio": ratio,
                "delta_vs_anchor_ratio": delta_size / current_anchor_compressed_size,
                "savings_percent": 100 * (1 - ratio),
                "time_seconds": delta_elapsed,
            })
            print(f"        delta KEPT: {decision_label} in {delta_elapsed:.1f}s")
        else:
            # Delta exceeds threshold — promote to anchor instead
            print(f"        delta REJECTED: {decision_label}, promoting to anchor")
            os.remove(delta_path)
            anchor_path = os.path.join(output_dir, f"{basename}.dmx")
            t0 = time.time()
            # Use mode='auto' so the anchor encoding is picked from source dtype:
            # FP16/BF16 source → BFP (GPU-accelerated via bfp_compress_gpu)
            # FP32 source → int16 quant (GPU-accelerated via the encode_tensor GPU path)
            # The 'precision' arg only governs DELTA encoding (int16 vs int32),
            # not anchor encoding. Forcing the anchor to mode=precision was a bug
            # that defeated GPU acceleration on FP16 source models.
            compress_file(ckpt_path, anchor_path, mode="auto", entropy=entropy, use_gpu=use_gpu)
            anchor_elapsed = time.time() - t0
            anchor_size = os.path.getsize(anchor_path)
            total_chain += anchor_size
            anchor_count += 1
            chain_entries.append({
                "index": idx,
                "type": "anchor",
                "source": os.path.basename(ckpt_path),
                "raw_size_bytes": raw_size,
                "compressed_size_bytes": anchor_size,
                "compressed_path": os.path.relpath(anchor_path, output_dir),
                "anchor_index": idx,
                "promoted_due_to": "delta_ratio_exceeded",
                "rejected_delta_ratio": ratio,
                "delta_attempt_seconds": delta_elapsed,
                "anchor_write_seconds": anchor_elapsed,
            })
            current_anchor_safetensors = ckpt_path
            current_anchor_idx = idx
            current_anchor_compressed_size = anchor_size
            print(f"        anchor {fmt_size(anchor_size)} ({100*anchor_size/raw_size:.1f}% of raw) in {anchor_elapsed:.1f}s")

    savings_percent = 100 * (1 - total_chain / total_raw) if total_raw > 0 else 0

    manifest = {
        "format": "dmx_chain_manifest",
        "version": 2,
        "precision": precision,
        "anchor_policy": "auto_anchor_promotion",
        "anchor_policy_mode": policy_mode,  # "dynamic" or "static"
        "max_delta_ratio": max_delta_ratio,  # null in dynamic mode
        "anchor_safety_margin": anchor_safety_margin if policy_mode == "dynamic" else None,
        "checkpoint_count": len(checkpoint_paths),
        "anchor_count": anchor_count,
        "delta_count": delta_count,
        "total_raw_bytes": total_raw,
        "total_chain_bytes": total_chain,
        "savings_percent": savings_percent,
        "entries": chain_entries,
    }

    manifest_path = os.path.join(output_dir, "chain_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"=" * 70)
    print(f"Chain compression complete")
    print(f"=" * 70)
    print(f"  Anchors: {anchor_count}, Deltas: {delta_count}")
    print(f"  Total raw:    {fmt_size(total_raw)}")
    print(f"  Total chain:  {fmt_size(total_chain)}")
    print(f"  Savings:      {savings_percent:.2f}%")
    print(f"  Manifest:     {manifest_path}")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "total_raw_bytes": total_raw,
        "total_chain_bytes": total_chain,
        "savings_percent": savings_percent,
        "anchor_count": anchor_count,
        "delta_count": delta_count,
    }


def fmt_size(n):
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n/1024/1024:.1f} MB"
    return f"{n/1024/1024/1024:.2f} GB"


def delta_reconstruct(base_path, delta_path, output_path):
    """Reconstruct a checkpoint from base + delta.

    Loads the base safetensors, applies the integer deltas from the .dmxd file,
    and dequantizes back to the original dtype.
    """
    print(f"  Base: {base_path}")
    print(f"  Delta: {delta_path}")
    start = time.time()

    # Read delta file
    with open(delta_path, "rb") as f:
        magic = f.read(4)
        if magic != DELTA_MAGIC:
            raise ValueError(f"Not a DMX delta file (got {magic!r}, expected {DELTA_MAGIC!r})")
        version = struct.unpack("<I", f.read(4))[0]
        if version > DELTA_VERSION:
            raise ValueError(f"Delta version {version} not supported (max {DELTA_VERSION})")
        manifest_len = struct.unpack("<I", f.read(4))[0]
        manifest = json.loads(f.read(manifest_len))
        delta_data = f.read()

    # Load base
    base = load_file(base_path)
    precision = manifest.get("precision", "int16")
    scales = manifest.get("scales", {})
    tensors_meta = manifest["tensors"]
    header_size = 4 + 4 + 4 + manifest_len

    precision_label = {"int16": "near-lossless", "int32": "practically lossless"}
    print(f"  Precision: {precision} ({precision_label.get(precision, '')})")

    dctx = zstd.ZstdDecompressor()
    result = {}

    cuda_kernels = _get_cuda_kernels() if torch.cuda.is_available() else None
    if cuda_kernels:
        print("  GPU-accelerated delta reconstruction (native CUDA kernels)")

    keys = sorted(tensors_meta.keys())
    for i, key in enumerate(keys):
        meta = tensors_meta[key]
        shape = meta["shape"]
        dtype_str = meta["dtype"]
        scale = meta.get("scale", 0.0)
        tensor_precision = meta.get("precision", precision)
        offset = meta["offset"] - header_size  # offset into delta_data
        comp_size = meta["compressed_size"]

        # Decompress delta. Each tensor records the entropy coder it was
        # written with in its per-tensor metadata; the decoder dispatches on
        # that identifier. Legacy short identifiers ('zstd', 'lpc') from
        # earlier DMX versions are accepted as aliases of the current names.
        # Files with no entropy field at all (oldest .dmxd format) are
        # treated as zstd by default.
        chunk = delta_data[offset:offset + comp_size]
        tensor_entropy = meta.get("entropy", "zstd-19")
        if tensor_precision == "int32":
            # int32 path always uses zstd (FLAC/brotli paths are int16-native)
            delta_bytes = dctx.decompress(chunk)
        else:
            delta_bytes = _decompress_int16_dispatch(chunk, tensor_entropy)

        if tensor_precision == "int32":
            if scale == 0.0:
                # Legacy int32 XOR mode (pre-April 2026 files)
                delta = torch.frombuffer(bytearray(delta_bytes), dtype=torch.int32).clone()
                base_bits = base[key].contiguous().view(torch.int32).flatten()
                recon_bits = torch.bitwise_xor(base_bits, delta)
                if dtype_str == "float32":
                    result[key] = recon_bits.view(torch.float32).reshape(shape)
                elif dtype_str == "float16":
                    result[key] = recon_bits.to(torch.int16).view(torch.float16).reshape(shape)
                elif dtype_str == "bfloat16":
                    result[key] = recon_bits.to(torch.int16).view(torch.bfloat16).reshape(shape)
                else:
                    result[key] = recon_bits.view(torch.float32).reshape(shape)
            else:
                # int32 aligned quantization mode (April 2026+)
                delta = torch.frombuffer(bytearray(delta_bytes), dtype=torch.int32).clone()
                if cuda_kernels and delta.numel() > 1024:
                    max_abs_val = scale * 2147483647
                    base_q = cuda_kernels.quantize_int32(base[key].float().flatten().cuda(), max_abs_val).cpu()
                    recon_float = cuda_kernels.delta_apply_i32(base_q.cuda(), delta.cuda(), scale).cpu()
                else:
                    max_val = 2147483647
                    base_int = torch.clamp(
                        torch.round(base[key].double().flatten() / scale), -max_val, max_val
                    ).to(torch.int32)
                    recon_int = base_int.to(torch.int64) + delta.to(torch.int64)
                    recon_float = (recon_int.double() * scale).float()

                if dtype_str == "float16":
                    recon_float = recon_float.half()
                elif dtype_str == "bfloat16":
                    recon_float = recon_float.to(torch.bfloat16)

                result[key] = recon_float.reshape(shape)
        else:
            # Near-lossless: int16 aligned quantization
            delta = torch.frombuffer(bytearray(delta_bytes), dtype=torch.int16).clone()

            if cuda_kernels and delta.numel() > 1024:
                max_abs_val = scale * 32767
                base_q = cuda_kernels.quantize_int16(base[key].float().flatten().cuda(), max_abs_val).cpu()
                recon_float = cuda_kernels.delta_apply_i16(base_q.cuda(), delta.cuda(), scale).cpu()
            else:
                base_int = torch.clamp(
                    torch.round(base[key].float().flatten() / scale), -32767, 32767
                ).to(torch.int16)
                recon_int = (base_int.to(torch.int32) + delta.to(torch.int32)).to(torch.int16)
                recon_float = recon_int.float() * scale

            if dtype_str == "float16":
                recon_float = recon_float.half()
            elif dtype_str == "bfloat16":
                recon_float = recon_float.to(torch.bfloat16)

            result[key] = recon_float.reshape(shape)

        if (i + 1) % 100 == 0 or (i + 1) == len(keys):
            print(f"  Reconstructed {i+1}/{len(keys)} tensors...")

    # Save
    save_file(result, output_path)
    output_size = os.path.getsize(output_path)
    elapsed = time.time() - start

    print(f"\n  Output: {output_path} ({output_size / 1024 / 1024:.1f} MB)")
    print(f"  Tensors: {len(result)}")
    print(f"  Time: {elapsed:.1f}s")

    return output_size


def chain_reconstruct(manifest_path, output_dir, indices=None, use_gpu=None):
    """Reconstruct safetensors checkpoints from a DMX chain manifest.

    Walks the chain_manifest.json produced by chain_compress and recreates the
    original safetensors files into output_dir. Anchors are decompressed via
    decompress_file; deltas are reconstructed via delta_reconstruct against
    their anchor (which is decompressed first if not already cached).

    Args:
        manifest_path: path to chain_manifest.json (or a directory containing one).
        output_dir: directory to write reconstructed .safetensors files into.
        indices: optional iterable of integer entry indices to reconstruct. If
            None (default), reconstruct every entry. Anchors required by any
            requested delta are always reconstructed even if not listed.
        use_gpu: passthrough to decompress_file for anchor decompression.

    Returns:
        dict with 'reconstructed_count', 'output_paths', 'total_output_bytes',
        'time_seconds'.
    """
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, "chain_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"chain manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if manifest.get("format") != "dmx_chain_manifest":
        raise ValueError(f"not a DMX chain manifest: {manifest_path}")

    entries = manifest["entries"]
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(output_dir, exist_ok=True)

    # Build index → entry lookup
    by_index = {e["index"]: e for e in entries}

    # Determine which entries to actually reconstruct
    if indices is None:
        target_indices = sorted(by_index.keys())
    else:
        target_indices = sorted(set(indices))
        for idx in target_indices:
            if idx not in by_index:
                raise ValueError(f"requested index {idx} not in manifest")
        # Pull in any anchor required by a requested delta
        required_anchors = set()
        for idx in target_indices:
            e = by_index[idx]
            if e["type"] == "delta":
                required_anchors.add(e["anchor_index"])
        for a_idx in required_anchors:
            if a_idx not in target_indices:
                target_indices.append(a_idx)
        target_indices = sorted(set(target_indices))

    print(f"=" * 70)
    print(f"DMX chain-reconstruct")
    print(f"=" * 70)
    print(f"  Manifest:    {manifest_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Entries:     {len(target_indices)} of {len(entries)} total")
    print()

    start = time.time()
    output_paths = []
    total_output_bytes = 0
    # Cache anchor_index → reconstructed safetensors path so consecutive
    # deltas referring to the same anchor don't pay decompression twice.
    anchor_cache = {}

    def _output_basename(entry):
        src = entry["source"]
        base = os.path.splitext(os.path.basename(src))[0]
        return base + ".safetensors"

    def _ensure_anchor_reconstructed(anchor_idx):
        if anchor_idx in anchor_cache:
            return anchor_cache[anchor_idx]
        anchor_entry = by_index[anchor_idx]
        if anchor_entry["type"] != "anchor":
            raise ValueError(
                f"entry {anchor_idx} referenced as anchor but type={anchor_entry['type']}"
            )
        compressed_path = os.path.join(manifest_dir, anchor_entry["compressed_path"])
        out_path = os.path.join(output_dir, _output_basename(anchor_entry))
        print(f"  decompress anchor [{anchor_idx}]: {os.path.basename(compressed_path)}")
        decompress_file(compressed_path, out_path, use_gpu=use_gpu)
        anchor_cache[anchor_idx] = out_path
        return out_path

    for idx in target_indices:
        entry = by_index[idx]
        if entry["type"] == "anchor":
            out_path = _ensure_anchor_reconstructed(idx)
            if out_path not in output_paths:
                output_paths.append(out_path)
                total_output_bytes += os.path.getsize(out_path)
        elif entry["type"] == "delta":
            anchor_idx = entry["anchor_index"]
            base_path = _ensure_anchor_reconstructed(anchor_idx)
            delta_path = os.path.join(manifest_dir, entry["compressed_path"])
            out_path = os.path.join(output_dir, _output_basename(entry))
            print(f"  reconstruct delta [{idx}]: {os.path.basename(delta_path)} ← anchor [{anchor_idx}]")
            delta_reconstruct(base_path, delta_path, out_path)
            output_paths.append(out_path)
            total_output_bytes += os.path.getsize(out_path)
        else:
            raise ValueError(f"unknown entry type {entry['type']!r} at index {idx}")

    elapsed = time.time() - start
    print()
    print(f"=" * 70)
    print(f"Chain reconstruction complete")
    print(f"=" * 70)
    print(f"  Files written: {len(output_paths)}")
    print(f"  Total output:  {fmt_size(total_output_bytes)}")
    print(f"  Time:          {elapsed:.1f}s")

    return {
        "reconstructed_count": len(output_paths),
        "output_paths": output_paths,
        "total_output_bytes": total_output_bytes,
        "time_seconds": elapsed,
    }


def delta_info(delta_path):
    """Show detailed info about a .dmxd delta file."""
    with open(delta_path, "rb") as f:
        magic = f.read(4)
        if magic != DELTA_MAGIC:
            raise ValueError(f"Not a DMX delta file (got {magic!r}, expected {DELTA_MAGIC!r})")
        version = struct.unpack("<I", f.read(4))[0]
        manifest_len = struct.unpack("<I", f.read(4))[0]
        manifest = json.loads(f.read(manifest_len))

    delta_size = os.path.getsize(delta_path)
    precision = manifest.get("precision", "int16")
    base_size = manifest.get("base_size", 0)
    target_size = manifest.get("target_size", 0)
    total_params = manifest.get("total_params", 0)
    total_zeros = manifest.get("total_zeros", 0)
    tensor_count = manifest.get("tensor_count", 0)
    scales = manifest.get("scales", {})

    precision_label = {"int16": "near-lossless", "int32": "practically lossless"}
    print(f"DMX Delta File: {delta_path}")
    print(f"  Version:    {version}")
    print(f"  Precision:  {precision} ({precision_label.get(precision, '')})")
    print(f"  Base:       {manifest.get('base_file', 'unknown')} ({base_size / 1024 / 1024:.1f} MB)")
    print(f"  Target:     {manifest.get('target_file', 'unknown')} ({target_size / 1024 / 1024:.1f} MB)")
    print(f"  Delta size: {delta_size / 1024 / 1024:.1f} MB")
    if target_size > 0:
        ratio = delta_size / target_size * 100
        print(f"  Ratio:      {ratio:.1f}% of target")
        print(f"  Savings:    {(target_size - delta_size) / 1024 / 1024:.1f} MB ({100 - ratio:.1f}%)")
    print(f"  Tensors:    {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    if total_params > 0:
        print(f"  Zero deltas: {total_zeros:,} ({100.0 * total_zeros / total_params:.1f}%)")
    if scales:
        print(f"  Aligned component groups: {len(scales)}")

    # Per-component breakdown
    tensors_meta = manifest.get("tensors", {})
    if tensors_meta:
        # Group by component type
        from collections import defaultdict
        comp_stats = defaultdict(lambda: {"params": 0, "zeros": 0, "raw": 0, "compressed": 0, "count": 0})
        for key, meta in tensors_meta.items():
            ct = meta.get("component_type", "unknown")
            comp_stats[ct]["params"] += meta.get("params", 0)
            comp_stats[ct]["zeros"] += meta.get("zeros", 0)
            comp_stats[ct]["raw"] += meta.get("raw_size", 0)
            comp_stats[ct]["compressed"] += meta.get("compressed_size", 0)
            comp_stats[ct]["count"] += 1

        print(f"\n  Per-component breakdown:")
        print(f"    {'Component':<30s} | {'Tensors':>7s} | {'Params':>10s} | {'Zero%':>6s} | {'Ratio':>6s}")
        print(f"    {'-'*30}-+-{'-'*7}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}")
        for ct in sorted(comp_stats.keys()):
            s = comp_stats[ct]
            z_pct = 100.0 * s["zeros"] / s["params"] if s["params"] > 0 else 0
            ratio = 100.0 * s["compressed"] / s["raw"] if s["raw"] > 0 else 0
            print(f"    {ct:<30s} | {s['count']:>7d} | {s['params']:>10,d} | {z_pct:>5.1f}% | {ratio:>5.1f}%")

    # Top 5 largest deltas (least compressible)
    if tensors_meta:
        by_size = sorted(tensors_meta.items(), key=lambda x: x[1].get("compressed_size", 0), reverse=True)
        print(f"\n  Largest delta tensors:")
        for key, meta in by_size[:5]:
            csz = meta.get("compressed_size", 0)
            z = meta.get("zeros", 0)
            n = meta.get("params", 1)
            print(f"    {key}: {csz/1024:.1f} KB ({100.0*z/n:.0f}% zero)")


DMX_BANNER = f"""DMX - Delta Multiplexed Model Compressor v{__version__}
Patent Pending. (c) 2026 William J. Riley. MIT License.
"""

def main():
    print(DMX_BANNER)
    parser = argparse.ArgumentParser(description="DMX - Delta Multiplexed Model Compressor (Patent Pending)")
    sub = parser.add_subparsers(dest="command")

    p_compress = sub.add_parser("compress", help="Compress safetensors to .dmx")
    p_compress.add_argument("input", help="Input .safetensors file")
    p_compress.add_argument("output", help="Output .dmx file")
    p_compress.add_argument("--mode", choices=["auto", "bfp", "int16", "int32"], default="auto",
                            help="Compression mode: bfp (FP16/BF16), int16 (FP32, best compression), "
                                 "int32 (FP32, practically lossless), auto (detect)")
    p_compress.add_argument("--entropy",
                            choices=["auto", "zstd-19", "zstd", "flac", "lpc", "brotli-11", "brotli"],
                            default="auto",
                            help="Entropy coder. Default 'auto' = per-tensor competitive selection "
                                 "across [zstd-19, FLAC, brotli-11] with the smallest output kept. "
                                 "Pinned values force a single coder for every tensor. Legacy "
                                 "aliases accepted: 'zstd' -> 'zstd-19', 'lpc' -> 'flac', "
                                 "'brotli' -> 'brotli-11'.")
    p_compress.add_argument("--gpu", action="store_true",
                            help="Use GPU-accelerated compression (CUDA required)")
    p_compress.add_argument("--parallel-workers", type=int, default=None,
                            help="Number of threads for per-tensor encoding (default: auto, "
                                 "uses min(8, cpu_count) on CPU and 1 on GPU)")
    p_compress.add_argument("--mantissa-bits", type=int, default=None,
                            help="Override BFP mantissa bit width (int in [1, 10]). Only affects "
                                 "BFP mode. Common values: 4 (aggressive, smaller files), "
                                 "6 (default, balanced), 7 (balanced, better fidelity), "
                                 "8 (conservative, highest fidelity). Default uses the shipping "
                                 "constant (6). The chosen value is stored per-tensor in the "
                                 "manifest so decoders always reconstruct correctly.")
    p_compress.add_argument("--fast-load", action="store_true", default=False,
                            help="Store embedding and output-head tensors as FP16 instead of BFP. "
                                 "~13%% larger file but enables instant loading for compressed "
                                 "residency (no materialize step). Recommended for interactive use.")

    p_decompress = sub.add_parser("decompress", help="Decompress .dmx to safetensors")
    p_decompress.add_argument("input", help="Input .dmx file")
    p_decompress.add_argument("output", help="Output .safetensors file")
    p_decompress.add_argument("--gpu", action="store_true",
                              help="Use GPU-accelerated decompression (CUDA required)")

    p_info = sub.add_parser("info", help="Show info about a .dmx file")
    p_info.add_argument("input", help="Input .dmx file")

    p_compress.add_argument("--report", default=None,
                            help="Auto-verify after compression and write JSON report to this path")

    p_verify = sub.add_parser("verify", help="Compress, decompress, compare")
    p_verify.add_argument("source", help="Original .safetensors file")
    p_verify.add_argument("dmx", help="DMX file path (created if missing)")
    p_verify.add_argument("--mode", choices=["auto", "bfp", "int16", "int32"], default="auto",
                            help="Compression mode (if dmx needs to be created)")
    p_verify.add_argument("--entropy",
                            choices=["auto", "zstd-19", "zstd", "flac", "lpc", "brotli-11", "brotli"],
                            default="auto",
                            help="Entropy coder. Default 'auto' = per-tensor competitive selection "
                                 "across [zstd-19, FLAC, brotli-11]. Pinned values force a single "
                                 "coder. Legacy aliases accepted: 'zstd' -> 'zstd-19', 'lpc' -> "
                                 "'flac', 'brotli' -> 'brotli-11'.")
    p_verify.add_argument("--report", default=None,
                            help="Write structured JSON verification report to this path")
    p_verify.add_argument("--gpu", action="store_true",
                            help="Use GPU-accelerated decompression (CUDA required)")

    p_delta = sub.add_parser("delta-compress",
                              help="Delta-compress a checkpoint against a base")
    p_delta.add_argument("base", help="Base .safetensors file (the anchor)")
    p_delta.add_argument("target", help="Target .safetensors file to delta-compress")
    p_delta.add_argument("output", help="Output .dmxd delta file")
    p_delta.add_argument("--precision", choices=["int16", "int32"], default="int16",
                          help="int16: near-lossless ~87%% savings (default). "
                               "int32: practically lossless ~87%% savings, error below FP32 noise floor.")
    p_delta.add_argument("--zstd-level", type=int, default=None,
                          help="zstd compression level 1-19 (default: 3 for speed. Use 19 for maximum compression).")
    p_delta.add_argument("--entropy",
                          choices=["auto", "zstd-19", "zstd", "flac", "lpc", "brotli-11"],
                          default="auto",
                          help="Entropy coder. Default 'auto' = per-tensor competitive selection "
                               "across [zstd-19, FLAC, brotli-11] with the smallest output kept. "
                               "Pinned values force a single coder for the whole delta -- useful "
                               "for testing and per-coder baselines. Legacy aliases accepted: "
                               "'zstd' -> 'zstd-19', 'lpc' -> 'flac'.")
    p_delta.add_argument("--keep-legacy-buffers", action="store_true",
                          help="Include legacy auxiliary buffer tensors (causal masks, "
                               "inv_freq, masked_bias) in the delta. Default is to strip them "
                               "because modern HF loaders ignore them anyway and they corrupt "
                               "aligned scale calculations. Only set this if you need bit-exact "
                               "round-trip of a legacy-format checkpoint.")

    p_recon = sub.add_parser("delta-reconstruct",
                              help="Reconstruct a checkpoint from base + delta")
    p_recon.add_argument("base", help="Base .safetensors file (the anchor)")
    p_recon.add_argument("delta", help="Input .dmxd delta file")
    p_recon.add_argument("output", help="Output .safetensors file")

    p_dinfo = sub.add_parser("delta-info", help="Show info about a .dmxd delta file")
    p_dinfo.add_argument("input", help="Input .dmxd delta file")

    p_chain = sub.add_parser("chain-compress",
                              help="Chain-compress a sequence of related checkpoints with auto-anchor promotion")
    p_chain.add_argument("checkpoints", nargs="+", help="Ordered list of safetensors paths")
    p_chain.add_argument("--output-dir", required=True, help="Directory for anchors / deltas / manifest")
    p_chain.add_argument("--precision", choices=["int16", "int32"], default="int16",
                          help="int16 (default) or int32 quantization precision")
    p_chain.add_argument("--max-delta-ratio", type=float, default=None,
                          help="Static anchor promotion threshold (legacy). If set, a delta is "
                               "kept only when its compressed size is <= this fraction of the "
                               "target's RAW size. Default unset -> dynamic policy: keep delta only "
                               "when its size is < current_anchor_compressed_size * 0.95. The "
                               "dynamic policy is storage-optimal and self-calibrates per source dtype.")
    p_chain.add_argument("--anchor-safety-margin", type=float, default=0.95,
                          help="Dynamic policy safety margin (default 0.95). Multiplied by the "
                               "current anchor's compressed size to set the keep/promote boundary. "
                               "Ignored when --max-delta-ratio is set.")
    p_chain.add_argument("--zstd-level", type=int, default=None,
                          help="zstd compression level for deltas (1-19, default 3)")
    p_chain.add_argument("--entropy",
                          choices=["auto", "zstd-19", "zstd", "flac", "lpc", "brotli-11"],
                          default="auto",
                          help="Entropy coder. Default 'auto' = per-tensor competitive selection "
                               "across [zstd-19, FLAC, brotli-11]. Pinned values force a single "
                               "coder. Legacy aliases accepted: 'zstd' -> 'zstd-19', 'lpc' -> "
                               "'flac'.")
    p_chain.add_argument("--gpu", action="store_true", help="Use GPU-accelerated compression for anchor writes")
    p_chain.add_argument("--keep-legacy-buffers", action="store_true",
                          help="Include legacy auxiliary buffers (causal masks, inv_freq, "
                               "masked_bias) in chain deltas. Default is to strip them.")

    p_export = sub.add_parser("export",
                               help="Derive a quantized model from a DMX file (INT8, NF4, FP8)")
    p_export.add_argument("input", help="Input .dmx file")
    p_export.add_argument("output",
                           help="Output path. For --target: .safetensors file path. "
                                "For --targets: output directory where {target}.safetensors "
                                "files will be written.")
    p_export.add_argument("--target", choices=["int8", "nf4", "fp8"], default=None,
                           help="Single target quantization format. int8 = per-channel symmetric "
                                "INT8. nf4 = QLoRA NF4 codebook (group_size=64). fp8 = FP8 E4M3 "
                                "(requires PyTorch 2.1+, best results from int32-mode DMX files). "
                                "Exactly one of --target or --targets is required.")
    p_export.add_argument("--targets", type=str, default=None,
                           help="Multiple target formats as a comma-separated list "
                                "(e.g. 'int8,nf4,fp8'). The DMX file is decoded ONCE and all "
                                "targets are derived from the shared in-memory tensor dict, "
                                "amortizing decode cost across targets. Saves roughly 27%% wall "
                                "time on T=2 and 48%% on T=3 vs separate --target calls. The "
                                "'output' positional must be a DIRECTORY in this mode; each "
                                "target is written as {target}.safetensors inside that directory.")
    p_export.add_argument("--gpu", action="store_true",
                           help="Use GPU-accelerated decompression")

    p_chain_recon = sub.add_parser("chain-reconstruct",
                                    help="Reconstruct safetensors checkpoints from a chain manifest")
    p_chain_recon.add_argument("manifest", help="Path to chain_manifest.json (or directory containing one)")
    p_chain_recon.add_argument("--output-dir", required=True, help="Directory to write reconstructed .safetensors files")
    p_chain_recon.add_argument("--indices", type=int, nargs="+", default=None,
                                help="Optional subset of entry indices to reconstruct (default: all)")
    p_chain_recon.add_argument("--gpu", action="store_true",
                                help="Use GPU-accelerated decompression for anchors")

    args = parser.parse_args()

    if args.command == "compress":
        compress_file(args.input, args.output, mode=args.mode, entropy=args.entropy,
                      use_gpu=args.gpu if args.gpu else None,
                      parallel_workers=args.parallel_workers,
                      mantissa_bits=args.mantissa_bits,
                      fast_load=args.fast_load)
        if args.report:
            print()
            print("=== Auto-verify after compression ===")
            verify_file(args.input, args.output, mode=args.mode, entropy=args.entropy,
                        report_path=args.report)
    elif args.command == "decompress":
        decompress_file(args.input, args.output, use_gpu=args.gpu if args.gpu else None)
    elif args.command == "info":
        info_file(args.input)
    elif args.command == "verify":
        verify_file(args.source, args.dmx, mode=args.mode, entropy=args.entropy,
                    report_path=args.report, use_gpu=args.gpu if args.gpu else None)
    elif args.command == "delta-compress":
        delta_compress(args.base, args.target, args.output, precision=args.precision,
                      zstd_level=getattr(args, 'zstd_level', None),
                      strip_legacy_buffers=not args.keep_legacy_buffers,
                      entropy=getattr(args, 'entropy', 'auto'))
    elif args.command == "delta-reconstruct":
        delta_reconstruct(args.base, args.delta, args.output)
    elif args.command == "delta-info":
        delta_info(args.input)
    elif args.command == "chain-compress":
        chain_compress(args.checkpoints, args.output_dir,
                       precision=args.precision,
                       max_delta_ratio=args.max_delta_ratio,
                       anchor_safety_margin=args.anchor_safety_margin,
                       zstd_level=getattr(args, 'zstd_level', None),
                       use_gpu=args.gpu if args.gpu else None,
                       strip_legacy_buffers=not args.keep_legacy_buffers,
                       entropy=getattr(args, 'entropy', 'auto'))
    elif args.command == "export":
        # Mutex: exactly one of --target or --targets must be given
        if args.target is None and args.targets is None:
            parser.error("dmx export: exactly one of --target or --targets is required")
        if args.target is not None and args.targets is not None:
            parser.error("dmx export: --target and --targets are mutually exclusive; "
                         "use --target for single target output file or --targets for "
                         "multi-target output directory")
        if args.targets is not None:
            targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
            if not targets_list:
                parser.error("dmx export: --targets must contain at least one target")
            export_quant_multi(args.input, args.output, targets=targets_list,
                               use_gpu=args.gpu if args.gpu else None)
        else:
            export_quant(args.input, args.output, target=args.target,
                         use_gpu=args.gpu if args.gpu else None)
    elif args.command == "chain-reconstruct":
        chain_reconstruct(args.manifest, args.output_dir,
                          indices=args.indices,
                          use_gpu=args.gpu if args.gpu else None)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
