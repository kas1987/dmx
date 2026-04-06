"""
DMX Checkpoint Callback for Training Frameworks

Drop-in callbacks that automatically delta-compress checkpoints during training.
Supports DeepSpeed and PyTorch/FSDP.

Usage with DeepSpeed:
    from dmx_callback import DMXDeepSpeedCallback
    callback = DMXDeepSpeedCallback(
        output_dir="/checkpoints/run-42/",
        anchor_every=50,       # fresh anchor every 50 saves
        precision="int32",     # practically lossless
        zstd_level=3,          # fast compression
    )
    # In your training loop:
    callback.on_save(step, model_engine)

Usage with vanilla PyTorch:
    from dmx_callback import DMXCheckpointManager
    manager = DMXCheckpointManager(output_dir="/checkpoints/run-42/")
    # Save:
    manager.save(step=1000, state_dict=model.state_dict())
    # Load:
    state_dict = manager.load(step=1000)
"""

import json
import os
import time

import torch
import zstandard as zstd
from collections import defaultdict
from safetensors.torch import save_file, load_file


def _key_type(key):
    parts = key.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _compute_aligned_scales(state_dict, precision="int16"):
    max_val = 32767 if precision == "int16" else 2147483647
    groups = defaultdict(list)
    for key in state_dict:
        groups[_key_type(key)].append(key)
    scales = {}
    for ctype, keys in groups.items():
        mx = max(state_dict[k].float().abs().max().item() for k in keys)
        scales[ctype] = mx / max_val if mx > 0 else 1.0
    return scales


class DMXCheckpointManager:
    """Manages delta-compressed checkpoint chains.

    Stores one full anchor checkpoint + a chain of small deltas.
    Reconstruction: load anchor + apply deltas to reach any step.
    """

    def __init__(self, output_dir, precision="int32", anchor_every=50,
                 zstd_level=3):
        """
        Args:
            output_dir: Directory for checkpoint files
            precision: 'int16' (near-lossless) or 'int32' (practically lossless)
            anchor_every: Save a fresh full anchor every N checkpoint saves
            zstd_level: zstd compression level (1-19, default 3 for speed)
        """
        self.output_dir = output_dir
        self.precision = precision
        self.anchor_every = anchor_every
        self.zstd_level = zstd_level
        self.max_val = 32767 if precision == "int16" else 2147483647
        self.save_count = 0
        self.anchor_sd = None
        self.anchor_step = None
        self.scales = None
        self.manifest = {"version": 1, "precision": precision, "saves": []}

        os.makedirs(output_dir, exist_ok=True)
        self._load_manifest()

    def _manifest_path(self):
        return os.path.join(self.output_dir, "dmx_manifest.json")

    def _load_manifest(self):
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                self.manifest = json.load(f)
            # Restore state from manifest
            saves = self.manifest.get("saves", [])
            self.save_count = len(saves)
            # Find latest anchor
            for s in reversed(saves):
                if s["type"] == "anchor":
                    self.anchor_step = s["step"]
                    break

    def _save_manifest(self):
        with open(self._manifest_path(), "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _quant(self, tensor, scale):
        dtype = torch.int16 if self.max_val == 32767 else torch.int32
        return torch.clamp(
            torch.round(tensor.float() / scale), -self.max_val, self.max_val
        ).to(dtype)

    def save(self, step, state_dict):
        """Save a checkpoint, automatically choosing anchor vs delta.

        Args:
            step: Training step number
            state_dict: Model state dict (CPU tensors)
        Returns:
            dict with save info (type, path, size, time)
        """
        start = time.time()
        is_anchor = (self.save_count % self.anchor_every == 0) or self.anchor_sd is None

        if is_anchor:
            return self._save_anchor(step, state_dict, start)
        else:
            return self._save_delta(step, state_dict, start)

    def _save_anchor(self, step, state_dict, start):
        """Save a full anchor checkpoint."""
        # Convert to CPU if needed
        cpu_sd = {k: v.cpu() for k, v in state_dict.items()}

        path = os.path.join(self.output_dir, "anchor_%06d.safetensors" % step)
        save_file(cpu_sd, path)

        # Update state
        self.anchor_sd = cpu_sd
        self.anchor_step = step
        self.scales = _compute_aligned_scales(cpu_sd, self.precision)
        self.save_count += 1

        size = os.path.getsize(path)
        elapsed = time.time() - start

        info = {
            "type": "anchor", "step": step, "path": os.path.basename(path),
            "size": size, "time": elapsed,
        }
        self.manifest["saves"].append(info)
        self.manifest["scales"] = {k: v for k, v in self.scales.items()}
        self._save_manifest()

        print(f"  [DMX] Anchor saved: step {step}, {size/1e6:.1f} MB, {elapsed:.1f}s")
        return info

    def _save_delta(self, step, state_dict, start):
        """Save a delta against the current anchor."""
        cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
        keys = sorted(cpu_sd.keys())

        up_dtype = torch.int32 if self.precision == "int16" else torch.int64
        down_dtype = torch.int16 if self.precision == "int16" else torch.int32

        cctx = zstd.ZstdCompressor(level=self.zstd_level, write_content_size=True)
        chunks = []
        total_zeros = 0
        total_params = 0

        for k in keys:
            scale = self.scales[_key_type(k)]
            base_q = self._quant(self.anchor_sd[k], scale)
            curr_q = self._quant(cpu_sd[k], scale)
            delta = (curr_q.to(up_dtype) - base_q.to(up_dtype)).to(down_dtype)

            total_params += delta.numel()
            total_zeros += (delta == 0).sum().item()

            compressed = cctx.compress(delta.numpy().tobytes())
            chunks.append(compressed)

        # Write delta file
        path = os.path.join(self.output_dir, "delta_%06d.dmxd" % step)
        with open(path, "wb") as f:
            # Simple format: [num_tensors u32][chunk_sizes u32 * N][chunk_data...]
            import struct
            f.write(struct.pack("<I", len(chunks)))
            for c in chunks:
                f.write(struct.pack("<I", len(c)))
            for c in chunks:
                f.write(c)

        size = os.path.getsize(path)
        elapsed = time.time() - start
        sparsity = 100.0 * total_zeros / total_params if total_params > 0 else 0

        info = {
            "type": "delta", "step": step, "path": os.path.basename(path),
            "anchor_step": self.anchor_step, "size": size, "time": elapsed,
            "sparsity": round(sparsity, 1), "params": total_params,
        }
        self.manifest["saves"].append(info)
        self._save_manifest()

        self.save_count += 1
        print(f"  [DMX] Delta saved: step {step}, {size/1e6:.1f} MB ({sparsity:.0f}% sparse), {elapsed:.1f}s")
        return info

    def load(self, step):
        """Load and reconstruct a checkpoint at the given step.

        Returns:
            state_dict with reconstructed tensors
        """
        # Find the save entry
        saves = self.manifest["saves"]
        target = None
        for s in saves:
            if s["step"] == step:
                target = s
                break
        if target is None:
            raise ValueError(f"No checkpoint found for step {step}")

        if target["type"] == "anchor":
            path = os.path.join(self.output_dir, target["path"])
            return load_file(path)

        # Delta — need the anchor first
        anchor_step = target["anchor_step"]
        anchor_entry = None
        for s in saves:
            if s["step"] == anchor_step and s["type"] == "anchor":
                anchor_entry = s
                break
        if anchor_entry is None:
            raise ValueError(f"Anchor step {anchor_step} not found")

        # Load anchor
        anchor_path = os.path.join(self.output_dir, anchor_entry["path"])
        anchor_sd = load_file(anchor_path)
        scales = _compute_aligned_scales(anchor_sd, self.precision)

        # Load and apply delta
        delta_path = os.path.join(self.output_dir, target["path"])
        keys = sorted(anchor_sd.keys())
        up_dtype = torch.int32 if self.precision == "int16" else torch.int64
        down_dtype = torch.int16 if self.precision == "int16" else torch.int32

        import struct
        dctx = zstd.ZstdDecompressor()
        with open(delta_path, "rb") as f:
            num_tensors = struct.unpack("<I", f.read(4))[0]
            chunk_sizes = [struct.unpack("<I", f.read(4))[0] for _ in range(num_tensors)]
            chunks = [f.read(sz) for sz in chunk_sizes]

        result = {}
        for i, k in enumerate(keys):
            scale = scales[_key_type(k)]
            base_q = self._quant(anchor_sd[k], scale)
            delta_bytes = dctx.decompress(chunks[i])
            delta = torch.frombuffer(bytearray(delta_bytes), dtype=down_dtype).clone()
            recon_q = base_q.to(up_dtype) + delta.to(up_dtype)
            result[k] = (recon_q.float() * scale).to(anchor_sd[k].dtype).reshape(anchor_sd[k].shape)

        return result

    def list_saves(self):
        """List all saved checkpoints."""
        for s in self.manifest.get("saves", []):
            t = s["type"]
            step = s["step"]
            size = s["size"] / 1e6
            if t == "anchor":
                print(f"  [{step:>6d}] ANCHOR  {size:.1f} MB")
            else:
                sparsity = s.get("sparsity", "?")
                print(f"  [{step:>6d}] DELTA   {size:.1f} MB  ({sparsity}% sparse, anchor={s['anchor_step']})")


class DMXDeepSpeedCallback:
    """DeepSpeed-compatible callback for DMX delta checkpoint compression.

    Usage:
        callback = DMXDeepSpeedCallback(output_dir="/checkpoints/run-42/")

        # In your training loop:
        for step in range(total_steps):
            loss = train_step(...)
            if step % save_every == 0:
                callback.on_save(step, model_engine)
    """

    def __init__(self, output_dir, precision="int32", anchor_every=50,
                 zstd_level=3, save_optimizer=False):
        self.manager = DMXCheckpointManager(
            output_dir=output_dir,
            precision=precision,
            anchor_every=anchor_every,
            zstd_level=zstd_level,
        )
        self.save_optimizer = save_optimizer

    def on_save(self, step, model_engine):
        """Called when a checkpoint should be saved.

        Args:
            step: Current training step
            model_engine: DeepSpeed model engine (or any object with .module.state_dict())
        """
        if hasattr(model_engine, 'module'):
            state_dict = model_engine.module.state_dict()
        else:
            state_dict = model_engine.state_dict()

        return self.manager.save(step, state_dict)

    def on_load(self, step):
        """Reconstruct a checkpoint for resumption.

        Returns:
            state_dict that can be loaded with model.load_state_dict()
        """
        return self.manager.load(step)
