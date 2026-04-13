# DMX — Delta Multiplexed Model Format

**Structure-aware compression for neural networks.**

DMX transforms weight tensors into aligned integer representations, enabling efficient storage and distribution of model variants. The headline capability is **55-87% compression of safetensors / checkpoint files / checkpoint deltas with practically lossless reconstruction**.

**Safe for production training.** Resuming from DMX-reconstructed checkpoints produces 0.15% loss difference after 50 chain resumes over 10,000 training steps — verified on GPU with reproducible scripts. Delta chains use exact integer arithmetic with zero error accumulation regardless of chain length.

---

**Contents:** [What is DMX?](#what-is-dmx) · [How It Works](#how-it-works) · [Installation](#installation) · [Quick Start](#quick-start) · [Delta Compression](#delta-compression-checkpoint--model-versioning) · [Chain Compression](#chain-compression-training-run-checkpoints-every-n-step-cadences) · [Quantized Export](#quantized-export-derive-int8--nf4--fp8-on-demand) · [Benchmarks](#benchmarks) · [Why DMX Matters](#why-dmx-matters-for-training) · [Validated Results](#validated-results-checkpoint-delta-compression) · [Research](#research-directions) · [License & Patent](#license--patent)

---

```
Original:  9.1 GB  (SVD-XT, FP32 — 80% includes FP32→FP16 conversion)
DMX:       1.8 GB
```

```
Original:  7.2 GB  (Wan 2.2 14B shard, FP32 — 79.5% includes FP32→FP16)
DMX:       1.5 GB  (142/142 tensors verified)
```

```
Original:  16 GB   (Llama 3.1 8B, BF16 — 53% pure BF16 compression)
DMX:       ~7.5 GB (+0.70% perplexity on wikitext-2)
```

### Try it now

```bash
pip install dmx-compress
dmx compress your_model.safetensors compressed.dmx
```

### Download pre-compressed models

| Model | Original | DMX | Savings | Verified |
|-------|----------|-----|---------|----------|
| [Wan 1.3B](https://huggingface.co/Senat1/dmx-wan-1.3b) | 2.7 GB | 1.1 GB | 60% | 825/825 tensors |
| [Wan 2.2 shard](https://huggingface.co/Senat1/dmx-wan2.2-shard6) | 7.2 GB | 1.5 GB | 79.5% | 142/142 tensors |
| [SVD-XT](https://huggingface.co/Senat1/dmx-svd-xt) | 9.1 GB | 1.8 GB | 80% | Roundtrip verified |

---

## What is DMX?

DMX is a structure-aware compression system for neural networks. It reduces model file sizes by 55-80% while preserving quality (+0.03-0.70% perplexity), with reversible decompression back to the original format.

DMX also supports delta-based model storage with deterministic reconstruction and ROI-driven adaptive rebasing — enabling efficient versioning across model families.

- **No retraining required** — compress any pretrained safetensors model
- **Reversible** — decompress back to the original format
- **Broad compatibility** — tested on LLMs, diffusion models, video models, encoder-decoder

**DMX reduces the cost of storing, moving, and resuming large models — without breaking training.**

## Compression & Delta Storage

| Capability | Evidence |
|-----------|---------|
| Single-file compression (55-80%) | 6+ models, Llama 3 8B through SVD-XT |
| Checkpoint delta chains (87%) | GPT-2, T5, TinyLlama, Qwen 3B |
| Full checkpoint w/ optimizer (79%) | GPT-2 1000-step, weights + momentum + variance |
| Zero chain accumulation error | Exact integer arithmetic, 10K steps / 50 resumes |
| Fine-tune variant distribution (80%) | [Qwen 2.5 3B delta on HuggingFace](https://huggingface.co/Senat1/dmx-qwen2.5-3b-instruct-delta) |

## How It Works

### BFP Mode (for FP16/BF16 models — recommended)
```
Standard FP16:  16 bits per weight (5-bit exponent wasted on unused dynamic range)
DMX BFP:        ~7 bits per weight (shared exponent per group + truncated mantissa + entropy coding)
```

Trained weights cluster in a narrow magnitude range — 74% use only 3 of 31 possible exponents. DMX shares one exponent per group of 32 values, eliminating wasted dynamic range, then entropy-codes the mantissa stream.

### int16 Mode (for FP32 models — near-lossless)
```
Standard FP32:  32 bits per weight
DMX int16:      ~13 bits per weight (aligned cross-layer quantization + entropy coding)
```

Integer quantization as a preprocessing step (not a lossy final format) transforms float weights into a representation where entropy coding is effective. Aligned cross-layer quantization enforces a global coordinate system across layers, enabling structured compression.

### Adaptive per-tensor compression

DMX automatically picks the best compression for each tensor in your model — you don't choose a compressor, DMX does, per tensor, every time. Each tensor gets the strongest compression the candidate set can deliver, capturing the maximum benefit available without any manual tuning.

Actual savings depend on model architecture, source precision (FP16 / BF16 / FP32), and the quantization mode you select. Across the model families we have measured, savings typically fall in the 50–80% range vs the original safetensors file, with no manual tuning required.

### Precision: below the FP32 arithmetic noise floor

DMX INT32 mode roundtrip error falls **below the noise floor of FP32 arithmetic itself.** A single FP32 matrix multiply introduces more error than a full DMX compress → decompress cycle.

| Measurement | RelL2 |
|-------------|-------|
| FP32 matmul noise floor (1000 samples, dim=768) | ~1.5–1.8e-07 |
| DMX INT32 roundtrip (GPU path, real GPT-2 weights) | ~4.1e-08 |
| DMX INT32 roundtrip (CPU float64 path) | ~1.1e-10 |

On GPU: **~4.5x below** the FP32 noise floor. On CPU with float64 arithmetic: **~1,300x below.**

This means DMX does not degrade the model — it stores weights more faithfully than FP32 represents them during computation. See [`benchmarks/dmx_noise_floor_benchmark.py`](benchmarks/dmx_noise_floor_benchmark.py) to reproduce.

### Why DMX beats generic compression

| Method | Bits/value | Notes |
|--------|-----------|-------|
| gzip on safetensors | ~15.5 | Raw floats look like noise |
| zstd level 19 | 14.06 | Dictionary matching, no prediction |
| **DMX int16 + entropy** | **11.45** | Aligned quantization enables structured entropy coding |
| **DMX BFP + zstd** | **~4.2** | Shared exponent eliminates wasted dynamic range |

## Installation

```bash
pip install dmx-compress
```

Or from source:
```bash
git clone https://github.com/willjriley/dmx.git && cd dmx && pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+. GPU (CUDA) is optional — automatically used when available for faster compression and decompression.

## Quick Start

```bash
# Compress any safetensors model (auto-detects FP16 vs FP32)
dmx compress model.safetensors model.dmx --mode auto

# Practically lossless compression (FP32 models — error below FP32 noise floor)
dmx compress model.safetensors model.dmx --mode int32

# Compress with explicit parallel encoding (defaults to min(8, cpu_count) on CPU,
# 1 on GPU). zstd releases the GIL so threads give real parallelism.
dmx compress model.safetensors model.dmx --parallel-workers 8

# Tune BFP fidelity with the mantissa-bits slider (4-8, default 6)
# M=4: aggressive compression, lower fidelity
# M=6: default, balanced
# M=8: conservative, highest fidelity
dmx compress model.safetensors model.dmx --mode bfp --mantissa-bits 7

# Decompress back to safetensors (auto-uses GPU if available)
dmx decompress model.dmx model.safetensors

# Verify roundtrip quality (with JSON report)
dmx verify model.safetensors model.dmx --report verify.json

# View compression info
dmx info model.dmx
```

Compression output now reports savings against **two baselines** so you can see
both the source-file ratio and the dtype-independent FP32 baseline side by side:

```text
Output size: 129.95 MB
  vs source file (375 MB): 65.3% of source, +34.7% savings
  vs FP32 equivalent (750 MB): 17.3% of FP32, +82.7% savings
```

Same compressed bytes, different baselines. The FP32-equivalent column is the
canonical baseline (`total_params × 4 bytes`) and stays constant regardless of
whether the input was FP16 or FP32 source.

### Delta compression (checkpoint / model versioning)

```bash
# Delta-compress a checkpoint against a base (near-lossless, ~87% savings)
dmx delta-compress base.safetensors checkpoint.safetensors delta.dmxd

# Practically lossless delta (error below FP32 noise floor, ~87% savings)
dmx delta-compress base.safetensors checkpoint.safetensors delta.dmxd --precision int32

# Reconstruct checkpoint from base + delta
dmx delta-reconstruct base.safetensors delta.dmxd restored.safetensors

# View delta file info (sparsity, compression, per-component breakdown)
dmx delta-info delta.dmxd
```

### Chain compression (training-run checkpoints, every-N-step cadences)

DMX chain compression takes a sequence of related checkpoints (training run, fine-tune steps, branch variants) and stores them as one or more anchors plus deltas, with an automatic anchor-promotion policy that keeps the chain mathematically guaranteed to be no larger than storing each checkpoint with `dmx compress` independently.

```bash
# Chain-compress a sequence of checkpoints into one output directory
dmx chain-compress step_1000.safetensors step_2000.safetensors step_3000.safetensors \
    --output-dir ./compressed_chain

# Reconstruct every checkpoint in the chain back to safetensors
dmx chain-reconstruct ./compressed_chain --output-dir ./restored

# Reconstruct only specific entries by index
dmx chain-reconstruct ./compressed_chain --output-dir ./restored --indices 0 2
```

The auto-anchor policy promotes a checkpoint to a fresh anchor whenever its delta would be larger than re-encoding the checkpoint from scratch, so the chain is self-calibrating across source dtypes and cadences. No manual tuning required.

### Quantized export (derive INT8 / NF4 / FP8 on demand)

One DMX file can serve as the source of truth for multiple quantized formats. Instead of storing separate INT8, NF4, and FP8 files alongside the original, derive them on demand:

```bash
# Derive INT8 per-channel from a DMX file
dmx export model.dmx model_int8.safetensors --target int8

# Derive NF4 (QLoRA codebook, group_size=64)
dmx export model.dmx model_nf4.safetensors --target nf4

# Derive FP8 E4M3 (best results from int32-mode DMX files)
dmx export model.dmx model_fp8.safetensors --target fp8
```

**Multi-target export** — derive several formats in a single DMX load:

```bash
# Write int8.safetensors, nf4.safetensors, and fp8.safetensors to an output dir,
# decoding the DMX file exactly once and deriving all three from the shared
# in-memory tensors.
dmx export model.dmx ./quant_out/ --targets int8,nf4,fp8
```

When `--targets` is used, the positional `output` must be a directory (created
if missing) rather than a single file path. Each target is written as
`{target}.safetensors` inside that directory.

**Performance notes**

Speed improvements depend on your model and workload.

- On decode-dominated models (measured on Pythia 160M), exports can be
  **~27% faster with 2 targets and ~48% faster with 3 targets** thanks to
  shared decoding.
- On models where the derive step dominates (measured on Qwen 2.5 0.5B,
  where FP8 derivation is heavier), results may vary — **~22% faster with
  2 targets, but up to ~33% slower with 3 targets** because the per-target
  work outweighs what decoding-once saves.

Output is byte-equivalent either way — only wall-time varies. If wall-time
is critical for your use case, benchmark on your target model.

The derived codes are bounded to ±1 LSB of what you'd get by quantizing the original full-precision weights directly. Only RTN-family formats are derivable — calibration-dependent formats like GPTQ and AWQ are not supported because they require activation data that DMX does not store.

### Example: Compress and verify a model from HuggingFace

```bash
# Download a model
pip install huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./wan_model

# Compress it
dmx compress ./wan_model/model.safetensors wan_compressed.dmx

# Decompress and verify
dmx verify ./wan_model/model.safetensors wan_compressed.dmx --report report.json
```

### Browser decoder and 3DGS support

[dmx-web](https://github.com/willjriley/dmx-web) — Rust/WASM decoder for browser-based decompression. Includes a real-time 3D Gaussian Splatting viewer. DMX compresses 3DGS scenes at 60%+ savings with roundtrip-verified visual fidelity (PSNR 48-53 dB, SSIM 0.9997+).

---

## Benchmarks

### Storage and transfer comparison

How a 140 GB model (Llama 3 70B, FP16) compares across compression approaches:

| Method | Compressed Size | Savings | Quality Loss | Purpose |
|--------|----------------|---------|-------------|---------|
| **safetensors** | 140 GB | 0% | None | Original format |
| **gzip** | ~134 GB | ~4% | None | Generic compression (barely helps on floats) |
| **zstd-19** | ~129 GB | ~8% | None | Better generic compression (still limited) |
| **DFloat11** | ~98 GB | ~30% | None (lossless) | Lossless NN weight compression |
| **ZipNN** | ~94 GB | ~33% | None (lossless) | Lossless NN weight compression |
| **DMX M=7** | ~63 GB | ~55% | +0.03% PPL | **Near-original quality, high compression** |
| **DMX M=6** | ~56 GB | ~60% | +0.16% PPL | **Aggressive storage compression** |

For reference, quantized inference formats like GGUF Q8 (~50%) and Q4 (~75%) achieve similar or greater compression but are designed for a different purpose — running models directly at reduced precision with fused kernels. DMX and GGUF serve different needs and are not interchangeable.

If lossless is enough, use DFloat11 or ZipNN. If you need to run inference at lower precision, use GGUF. If you need high compression with near-original quality for storage and distribution, that's where DMX lives.

| Without DMX | With DMX |
|-------------|----------|
| Llama 3 70B: 140 GB download | ~36 GB download |
| 4-5 models on 1 TB | 10+ models on 1 TB |

### BFP Mode (FP16 models)

| Model | Type | Original | DMX | Savings | Quality |
|-------|------|----------|-----|---------|---------|
| Llama 3.1 8B | LLM | 16 GB | ~7.5 GB | **53%** | **+0.70% PPL** (wikitext-2) |
| Wan 2.2 shard | Video | 7.2 GB | 1.5 GB | **79.5%** | 142/142 tensors pass |
| Wan 1.3B | Diffusion | 2.7 GB | 1.1 GB | **60%** | 825/825 tensors pass |
| SVD-XT | Video | 9.1 GB | 1.8 GB | **80%** | Verified roundtrip |

*Note: SVD-XT 80% includes FP32->FP16 conversion. Wan 2.2 79.5% is on FP32 source with BFP.*

### BFP Quality-per-Bit (Llama 3 8B, wikitext-2, 289K tokens)

| Config | Bits/Weight | Perplexity | vs FP16 |
|--------|-----------|------------|---------|
| FP16 baseline | 16.0 | 5.4958 | -- |
| BFP(M=8) | 9.25 | 5.4964 | +0.01% |
| BFP(M=7) | 8.25 | 5.4973 | **+0.03%** |
| BFP(M=6) | 7.25 | 5.5045 | +0.16% |
| *GGUF Q8_0 (ref)* | *8.50* | *~5.55-5.58* | *~1.0-1.5% (different purpose — inference format)* |

### int16 Mode (FP32 models)

| Model | Type | Original | DMX | Savings | PPL Change |
|-------|------|----------|-----|---------|------------|
| SVD-XT | Video | 8.9 GB | 4.0 GB | **55.5%** | Lossless |
| GPT-2 | LLM | 475 MB | 201 MB | **57.7%** | +0.22% |
| Phi-2 | LLM | 10.6 GB | 4.2 GB | **60.1%** | +0.12% |

### Decompression Speed

| Model | Mode | CPU | GPU (--gpu) | Speedup |
|-------|------|-----|-------------|---------|
| Wan 1.3B | BFP | 185s | 13.4s | 13.8x |
| SVD-XT | BFP | 281s | 22.3s | 12.5x |
| SVD-XT | int16 | 10.5s | -- | CPU-bound |

*Benchmarked on RTX 4090 Laptop, Python 3.13. GPU path uses PyTorch CUDA ops.*

**Native CUDA kernels** are available in `kernel/dmx_kernels_v2.cu` — 12 kernels covering the full compression and decompression pipeline (quantize, delta compute, BFP compress/decompress, dequantize, delta apply). Compiled and tested on A100. int32 roundtrip error: 9.3e-10.

---

## Why DMX Matters for Training

Frontier training runs are $50M-$200M+ each. Checkpoint storage, bandwidth, and crash recovery are recurring operational costs that compound across every experiment and team. DMX addresses these directly.

### What DMX enables

- **Safe resumption from compressed checkpoints** — 0.15% loss difference after 50 chain resumes over 10K training steps
- **87% checkpoint storage reduction** — 200 checkpoints of Llama 70B: ~28 TB raw → ~3 TB (projected)
- **Full checkpoint compression including optimizer** — weights + momentum + variance: 79% savings (measured)
- **Dense checkpoint history** — save 5-10x more often without the storage penalty
- **Fine-tune distribution** — store base model once, each variant as a small delta (80% savings)
- **Weight-shift analytics** — per-layer diffs show exactly what changed between any two checkpoints

### No other tool does all of this

| Tool | Delta between versions | Chain safety demonstrated |
|------|:-:|:-:|
| **DMX** | **87% savings (structure-aware)** | **0.15% loss diff after 50 resumes / 10K steps** |
| ZipNN | XOR-based delta (~44% savings) | Not published |
| DFloat11 | ✗ (per-file only, ~30%) | N/A |
| Git LFS / DVC | ✗ (full copy each version) | N/A |
| HuggingFace Hub | ✗ (full copy each version) | N/A |
| W&B / MLflow | ✗ (full copy each version) | N/A |
| xdelta (binary diff) | ~8.5% savings | Not published |

Byte-level delta tools (xdelta, ZipNN XOR) operate on raw float bits, where IEEE 754 layout destroys numerical proximity. DMX produces dramatically sparser deltas (87% vs 44%) by encoding in a structure-aware representation where similar values map to similar integers.

### The operational impact

| Aspect | Current Status Quo | With DMX | Benefit |
|--------|-------------------|---------|---------|
| Checkpoint frequency | Sparse (forced by cost) | Dense and safe | Better science and debugging |
| Storage for 200 ckpts (70B) | ~28 TB | ~3 TB (projected) | ~9x reduction |
| Crash recovery | Reload full checkpoint | Reload small delta | Minutes instead of hours |
| Fine-tune distribution | Full copy per variant | Small delta per variant | 80% savings (measured) |
| Experimentation | Branching is expensive | Branch via small delta | 5-10x more experimental forks |

DMX transforms checkpoint management into an operational advantage. It enables safe, multi-step training resumptions, preserves per-layer diffs, and drastically reduces the storage cost of model snapshots. Engineers and researchers gain usable model history that was previously impractical, minimizing wasted GPU time, improving training continuity, and lowering cloud storage costs.

### Efficient model distribution with deltas

DMX enables a new distribution model: send the base model once, then distribute only small aligned deltas for every variant.

```
Llama 70B base:          140 GB  (stored/downloaded once)
  → chat fine-tune:       ~28 GB  (delta only)
  → code fine-tune:       ~28 GB  (delta only)
  → medical fine-tune:    ~28 GB  (delta only)

Traditional: 4 × 140 GB = 560 GB
With DMX:    140 + 3 × 28 = 224 GB  (60% savings)
```

This applies to model hubs (HuggingFace, CivitAI), enterprise model management, and any workflow where multiple variants share a common base. Reconstruction from deltas is verified safe across 10K-step training chains (0.15% loss difference after 50 resumes).

**Where this matters today (estimated scale):**

| Platform | Hosted Models | Est. Fine-Tunes | Redundant Storage | Potential Savings with Deltas |
|----------|:------------:|:--------------:|:-----------------:|:----------------------------:|
| HuggingFace | 800K+ | ~500K (est. 60%) | Petabytes of duplicated base weights | ~60-80% bandwidth reduction |
| CivitAI | 100K+ | Tens of thousands of SD variants | Each a full 2-4 GB copy of SD base | ~80% per variant |
| Enterprise (per company) | 10-100 variants | Per-customer or per-use-case fine-tunes | Full copy per deployment | ~80% storage per variant |

*Estimates based on public model counts and observed fine-tune ratios. Actual savings depend on how much each fine-tune diverges from its base.*

### Validated: Qwen 2.5 3B model family

Measured on real HuggingFace models — [reconstructable delta available](https://huggingface.co/Senat1/dmx-qwen2.5-3b-instruct-delta):

```
Qwen/Qwen2.5-3B (base)           13.6 GB — stored once
  → Qwen2.5-3B-Instruct          2.88 GB delta (78.8% savings)
  → Qwen2.5-Coder-3B             5.60 GB delta (58.8% savings — fork, heavier retrain)
```

| Variant | int16 Zeros | int16 Savings | int32 Savings | RelL2 from Base |
|---------|:-:|:-:|:-:|:-:|
| **Instruct** (SFT+RLHF) | 29.2% | **90.7%** | 67.7% | 0.014 |
| **Coder** (domain retrain) | 0.2% | **58.8%** | 14.9% | 0.828 |

The Coder variant has diverged significantly from the base (RelL2 = 0.83). When a variant drifts this far, DMX supports auto-forking — promoting it to a new base and restarting the delta chain. Coder → Coder-Instruct would delta efficiently from the Coder anchor.

**Reconstruction quality (verified roundtrip):**

| Method | Precision Loss | Industry Acceptance |
|--------|:-:|---|
| FP32 → FP16 conversion | Measurable (~1e-3) | Standard practice everywhere |
| GGUF Q8 quantization | ~1% PPL increase | Widely deployed in production |
| **DMX int16 delta** | **+0.06% RelL2** | **Less loss than FP32→FP16** |
| **DMX int32 delta** | **1.87e-7 RelL2** | **Below FP32 arithmetic noise** |

**Try the distribution workflow yourself:**
```bash
pip install dmx-compress

# Download base + delta from HuggingFace (base: 13.6 GB, delta: 2.9 GB)
huggingface-cli download Senat1/dmx-qwen2.5-3b-instruct-delta --local-dir ./qwen-delta

# Reconstruct the full Instruct model from base + delta
dmx delta-reconstruct ./qwen-delta/qwen2.5-3b-base.safetensors ./qwen-delta/instruct.dmxd qwen2.5-3b-instruct.safetensors
```
If you already have the base model locally, you only need the 2.9 GB delta — not the full 13.6 GB Instruct model.

DMX enables multi-million-dollar savings in storage and bandwidth for hubs and enterprises that maintain many fine-tuned model variants, because only small deltas need to be stored and transmitted instead of full checkpoints.

---

## Validated Results: Checkpoint Delta Compression

All results are measured on real data using an NVIDIA A100-SXM4-80GB. Full result data is in `benchmarks/`.

### Compression across architectures

| Model | Architecture | Params | Consecutive Delta Zeros | Entropy (bits) | Measured Savings |
|-------|-------------|-------:|:----------------------:|:--------------:|:----------------:|
| GPT-2 | Decoder-only | 163M | 33-67% | 1.76-3.02 | **87.3%** (measured, 498→63 MB) |
| T5-small | Encoder-decoder | 110M | **89-94%** | **0.49-0.85** | Not yet measured in bytes |
| TinyLlama | Decoder-only | 1.1B | 16-63% | 1.69-3.73 | **80%** (measured, fine-tune base→chat) |

Delta compression works across model architectures and scales. T5 encoder-decoder shows highest sparsity. TinyLlama 1.1B confirms the pattern holds at scale — sparsity increases as training progresses (16% → 63% zeros). int32 aligned entropy matches int16 at all scales tested (1.71 vs 1.69 bits at 1.1B).

### Precision tiers

Both tiers achieve comparable compression — the aligned quantization produces similar entropy regardless of bit width:

| Tier | Consecutive Entropy | Compression | Error | Use Case |
|------|-------------------:|:----------:|:-----:|----------|
| **int16 aligned** | 0.6-1.3 bits | **87%** | +0.06% RelL2 | Maximum compression |
| **int32 aligned** | 1.0-1.2 bits | **~87%** | 1.87e-7 RelL2 | Practically lossless (error below FP32 noise floor) |
| Raw bit XOR (no alignment) | 14-16 bits | 8.5% | Bit-exact | Baseline — alignment is essential |

### Full checkpoint including optimizer states

Training checkpoints include model weights + Adam optimizer states (momentum + variance), typically 3x the weight size. Validated on GPT-2 124M, 1000 training steps:

| Component | % of Checkpoint | Delta Sparsity | Entropy | Compression |
|-----------|:-:|:-:|:-:|:-:|
| Weights | 33% | 55-66% zeros | 1.8-2.6 bits | ~84% |
| Momentum (exp_avg) | 33% | 28-30% zeros | 7.5-9.0 bits | ~53% |
| Variance (exp_avg_sq) | 33% | **91-92% zeros** | **0.6 bits** | **~96%** |
| **Full checkpoint** | 100% | — | — | **~79%** |

### Safety for training resumption

Training from DMX-reconstructed checkpoints is safe for production use:

| Test | Steps | DMX Resumes | Final Loss Diff | Result |
|------|------:|:-----------:|:---------------:|--------|
| Single resume (100 steps) | 100 | 1 | 0.042% | Negligible |
| **Long-run chain (10K steps)** | **10,000** | **50** | **0.15%** | **Production-safe** |

The 10K-step test reconstructed from a DMX delta chain every 200 steps — 50 total resumes over 10,000 training steps. Final loss tracks the clean baseline within 0.15%, with no divergence trend over time.

### Zero error accumulation in delta chains (Test 8)

Chained reconstruction (base → delta1 → delta2 → ... → deltaN) produces **identical** results to direct reconstruction (base + deltaN) — verified to 10 decimal places across both int16 and int32 modes. This is not an approximation: delta application is exact integer arithmetic, so error is mathematically constant regardless of chain length. Re-anchoring is needed only for delta *size* control, never for error control.

### Fine-tune variant compression

TinyLlama 1.1B base → chat fine-tune: **80% savings** (876 MB delta vs 4.4 GB full copy). Store the base model once, distribute each fine-tune variant as a small delta.

### Projected Savings at Scale

These projections are extrapolated from observed sparsity and scaling behavior on GPT-2 (163M), T5 (110M), and TinyLlama (1.1B). The core property — very small per-step weight updates under SGD — appears scale-invariant, but we are actively validating on 8B+ models with frontier-scale schedules.

| Scenario                                      | Raw Storage     | Projected DMX     | Projected Savings |
|-----------------------------------------------|-----------------|-------------------|-------------------|
| 200 checkpoints of Llama 70B (weights only)   | 28 TB           | ~3.6 TB           | **~87%**          |
| 200 checkpoints of Llama 70B (full + optimizer) | 84 TB         | ~18 TB            | **~79%**          |
| 20 fine-tune variants of Llama 70B            | 2.8 TB          | ~700 GB           | **~75%**          |

**Key Caveats**
- Validation on 8B+ models with real frontier training schedules is in progress.
- Optimizer state compression (currently ~53%) may drop to 40–45% on highly diverse data, reducing full-checkpoint savings to ~70–73%.
- All projections assume continued zero error accumulation (exact integer arithmetic), as demonstrated in long-chain tests.

These numbers suggest DMX could reduce checkpoint storage and I/O pressure by nearly an order of magnitude while keeping training resumption safe.

---

## Research Directions

- **Multi-framework integration** — DeepSpeed, FSDP, and Megatron-LM callbacks for production training pipelines
- **Checkpoint-efficient continual learning** — delta chains for long-running training with minimal storage overhead

We welcome collaboration — reach out via GitHub Issues or Discussions.

## Format Specification

See [spec/dmx_spec_v1.md](spec/dmx_spec_v1.md) for the complete format specification.

## Paper

[DMX: Delta Multiplexed Compression for Neural Network Model Weights (PDF)](https://github.com/willjriley/dmx/raw/main/paper/DMX_Paper.pdf) — click to download

## Background

DMX is based on the principle that floating-point weights should be transformed into multiple statistically distinct, independently modeled entropy domains prior to compression. Trained neural network weights exhibit extreme exponent clustering — 74% of FP16 values use only 3 of 31 possible exponents, wasting 2.4 bits per value. DMX decomposes the floating-point representation into separate exponent and mantissa streams, each with distinct statistical properties that benefit from independent entropy coding. For FP32 models, aligned cross-layer quantization enforces a global coordinate system across layers, enabling additional integer-domain compression. The format auto-profiles each model to select the optimal compression strategy per component.

## License & Patent

**Code:** MIT License — free to use, modify, and distribute.

**Methods:** Patent Pending (U.S. Provisional Applications filed April 2026). The patented methods cover aligned cross-layer quantization for neural network weight compression and stream-separated block floating point encoding with independent entropy coding. Personal, academic, and open-source use is unrestricted. Commercial use of the patented methods may require a license from the inventor — contact bill.riley@gmail.com.

## Citation

```bibtex
@software{riley2026dmx,
  author = {Riley, William J},
  title = {DMX: Delta Multiplexed Model Format},
  year = {2026},
  url = {https://github.com/willjriley/dmx}
}
```
