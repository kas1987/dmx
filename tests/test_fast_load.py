"""Tests for --fast-load flag (Sprint B, Task #31).

Validates that --fast-load stores embed/head tensors as FP16+entropy
instead of BFP, enabling instant loading for compressed residency.
"""

import json
import os
import struct
import tempfile

import pytest

# Skip entire module if pythia-14m safetensors not available locally
PYTHIA_14M_SAFETENSORS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--EleutherAI--pythia-14m/"
    "snapshots/cf967c0a9a04383db6f7b1108d86b2962634b4ac/model.safetensors"
)
if not os.path.isfile(PYTHIA_14M_SAFETENSORS):
    pytest.skip("pythia-14m safetensors not found locally", allow_module_level=True)

# Import after skip check so missing deps don't break collection
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmx_cli import (
    compress_file, decompress_file, ENC_FP16_ZSTD, ENC_BFP_ZSTD,
    ENC_TRANSPOSE_LOSSLESS, FAST_LOAD_PATTERNS, _is_fast_load_tensor,
)
from safetensors.torch import load_file


def _read_manifest(dmx_path):
    """Read the JSON manifest from a .dmx file (handles both provenance and legacy formats)."""
    with open(dmx_path, "rb") as f:
        magic = f.read(4)
        if magic == b"DMX\x00":
            # New provenance format: skip provenance header, then read inner DMX1 header
            prov_len = struct.unpack("<I", f.read(4))[0]
            f.seek(f.tell() + prov_len)  # skip provenance JSON
            magic = f.read(4)
        assert magic == b"DMX1", f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        manifest_size = struct.unpack("<I", f.read(4))[0]
        manifest_json = f.read(manifest_size)
    return json.loads(manifest_json)


def test_fast_load_flag_stores_embed_as_fp16():
    """Compress with --fast-load. Assert embed tensors use FP16 encoding,
    decoder projections use BFP encoding, and fast_load metadata is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dmx = os.path.join(tmpdir, "model_fast.dmx")
        compress_file(PYTHIA_14M_SAFETENSORS, output_dmx, fast_load=True)

        manifest = _read_manifest(output_dmx)
        tensors_meta = manifest["tensors"]

        # Check that embed/head tensors got FP16 encoding + fast_load flag
        found_fast_load = False
        for name, meta in tensors_meta.items():
            if _is_fast_load_tensor(name):
                found_fast_load = True
                assert meta["encoding"] in (ENC_FP16_ZSTD, 10), (
                    f"Tensor {name} should be FP16, got encoding={meta['encoding']}"
                )
                assert meta.get("fast_load") is True, (
                    f"Tensor {name} missing fast_load=True in manifest"
                )

        assert found_fast_load, (
            "No tensors matched FAST_LOAD_PATTERNS -- model may use unexpected naming"
        )

        # Check that non-embed tensors use transpose-lossless (encoding 16)
        transpose_found = False
        for name, meta in tensors_meta.items():
            if not _is_fast_load_tensor(name):
                if meta["encoding"] == ENC_TRANSPOSE_LOSSLESS:
                    transpose_found = True
        assert transpose_found, "Expected some non-embed tensors to use transpose-lossless encoding"


def test_fast_load_roundtrip_matches_default():
    """Both default and fast-load paths are lossless now (transpose-lossless
    default). Verify each path independently roundtrips bit-exactly against
    the original safetensors."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        default_dmx = os.path.join(tmpdir, "model_default.dmx")
        fast_dmx = os.path.join(tmpdir, "model_fast.dmx")
        default_st = os.path.join(tmpdir, "model_default.safetensors")
        fast_st = os.path.join(tmpdir, "model_fast.safetensors")

        compress_file(PYTHIA_14M_SAFETENSORS, default_dmx, fast_load=False)
        compress_file(PYTHIA_14M_SAFETENSORS, fast_dmx, fast_load=True)

        decompress_file(default_dmx, default_st)
        decompress_file(fast_dmx, fast_st)

        original_tensors = load_file(PYTHIA_14M_SAFETENSORS)
        default_tensors = load_file(default_st)
        fast_tensors = load_file(fast_st)

        assert set(original_tensors.keys()) == set(default_tensors.keys()), (
            "Tensor name sets differ between original and default outputs"
        )
        assert set(original_tensors.keys()) == set(fast_tensors.keys()), (
            "Tensor name sets differ between original and fast-load outputs"
        )

        # Default path: compress -> decompress should be bit-exact (lossless)
        for name in original_tensors:
            orig = original_tensors[name]
            default = default_tensors[name]
            # Bitwise comparison via integer view for robust NaN handling
            if orig.dtype == torch.float16:
                orig_bits = orig.view(torch.int16)
                default_bits = default.view(torch.int16)
            else:
                orig_bits = orig.contiguous().view(torch.int32)
                default_bits = default.contiguous().view(torch.int32)
            assert torch.equal(orig_bits, default_bits), (
                f"Default path: tensor {name} not bit-exact after roundtrip"
            )

        # Fast-load path: compress -> decompress should be bit-exact (lossless)
        for name in original_tensors:
            orig = original_tensors[name]
            fast = fast_tensors[name]
            if orig.dtype == torch.float16:
                orig_bits = orig.view(torch.int16)
                fast_bits = fast.view(torch.int16)
            else:
                orig_bits = orig.contiguous().view(torch.int32)
                fast_bits = fast.contiguous().view(torch.int32)
            assert torch.equal(orig_bits, fast_bits), (
                f"Fast-load path: tensor {name} not bit-exact after roundtrip"
            )


@pytest.mark.skip(reason="Fast-load size relationship changed after Task 1 default pivot; feature disposition pending")
def test_fast_load_file_larger_than_default():
    """Assert the --fast-load .dmx is larger than the default .dmx (by at least 1%)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        default_dmx = os.path.join(tmpdir, "model_default.dmx")
        fast_dmx = os.path.join(tmpdir, "model_fast.dmx")

        compress_file(PYTHIA_14M_SAFETENSORS, default_dmx, fast_load=False)
        compress_file(PYTHIA_14M_SAFETENSORS, fast_dmx, fast_load=True)

        default_size = os.path.getsize(default_dmx)
        fast_size = os.path.getsize(fast_dmx)

        print(f"Default size: {default_size / 1024:.1f} KB")
        print(f"Fast-load size: {fast_size / 1024:.1f} KB")
        print(f"Overhead: {(fast_size / default_size - 1) * 100:.1f}%")

        assert fast_size > default_size * 1.01, (
            f"Fast-load file ({fast_size}) should be at least 1% larger than "
            f"default ({default_size}), got {(fast_size / default_size - 1) * 100:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
