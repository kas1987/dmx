# DMX

**DevOps primitives for neural networks: lossless storage, content-addressable deltas, provenance manifests, and runtime derivation — one file format for the entire model lifecycle.**

---

**Contents**

- [What DMX is](#what-dmx-is)
- [The gap DMX fills](#the-gap-dmx-fills)
- [Storage: archival-grade lossless](#storage-archival-grade-lossless)
- [Deltas: efficient updates across versions](#deltas-efficient-updates-across-versions)
- [Inference: runtime derivation](#inference-runtime-derivation)
- [Provenance: supply-chain visibility](#provenance-supply-chain-visibility)
- [Beyond neural network weights](#beyond-neural-network-weights)
- [Where DMX fits in ML DevOps](#where-dmx-fits-in-ml-devops)
- [Roadmap](#roadmap)
- [Status](#status)
- [Technical details](#technical-details)
- [Quick start](#quick-start)
- [License and patent](#license-and-patent)
- [Citation](#citation)

---

## What DMX is

DMX is a file format and set of primitives for managing neural network artifacts the way the rest of IT has managed artifacts for decades. A single DMX file serves the entire model lifecycle: lossless storage for archival and distribution, content-addressable deltas for training chains and variant families, embedded provenance for supply-chain visibility, and runtime derivation for efficient inference across heterogeneous hardware.

The core idea is that lossless storage is the foundation. Once the stored weights are bit-exact, everything else — deltas between related checkpoints, efficient distribution of model variants, fast inference-time derivations, verifiable lineage — becomes straightforward and trustworthy. Quality tradeoffs belong at inference (where they're reversible by going back to the source), not at storage (where they corrupt the foundation every downstream operation depends on).

---

## The gap DMX fills

Neural network artifact handling today looks nothing like modern software artifact handling. Multi-gigabyte files download as single blobs with no resume. Checkpoints duplicate full files instead of storing deltas. Variants of the same base model re-duplicate most of the base across every fork. There's no content-addressable storage, no standard provenance tracking, no built-in supply-chain verification. A new model variant means a new full download.

The rest of IT solved these problems decades ago. Resumable downloads, content-addressable storage, delta updates, dedup across related artifacts, signed provenance, versioned registries — these have been standard DevOps practice since the 1990s and early 2000s. ML adopted none of it.

This gap isn't because the problems don't exist in ML. It's because the file formats (safetensors, pickle) are blob-oriented with no structural awareness, and the infrastructure grew up inside research contexts where "download the weights file" was the whole workflow. With models now reaching hundreds of billions of parameters, the gap has become operationally expensive.

DMX is the set of primitives that closes this gap. Each DMX feature maps onto a standard DevOps practice:

- **Archival integrity** → lossless storage with content-addressable hashes
- **Efficient updates** → bit-exact deltas between related versions
- **Supply-chain visibility** → embedded provenance manifests
- **Flexible deployment** → runtime derivation to hardware-appropriate representations
- **Registry compatibility** → fits into existing distribution infrastructure

The sections below describe each primitive, with measured data on current behavior.

---

## Storage: archival-grade lossless

DMX stores weights losslessly. When you save a model as DMX, the bits you get back on load are identical to the bits you put in. This is bit-exact, not "practically lossless," not "within noise tolerance" — the reconstructed tensors pass strict bitwise equality against the originals, including for NaN and infinity values.

**Why this matters operationally:** archival integrity. A model stored losslessly can be reproduced exactly years later. Training can resume from any checkpoint with bit-identical weights. Delta chains can extend indefinitely without error accumulation. Downstream consumers can verify they received what the source produced. None of this works if the storage foundation is lossy.

### Measured compression

| Source format | Typical savings | Status |
|---|---|---|
| FP32 weights | ~16% | Measured on GPT-2 124M (byte transposition + zstd, bit-exact verified) |
| FP16 weights | ~24% | Measured on Qwen 1.5B (byte transposition + zstd, bit-exact verified) |
| BF16 weights | ~33% | Measured on Qwen 1.5B (byte transposition + zstd, bit-exact verified) |

The range on FP16 reflects a real phenomenon: the compressibility of weights depends on how they were trained and downcast, not just their stored format. Weights trained natively in FP16 or from a careful BF16→FP16 downcast compress differently from weights stored at low bit-widths after mixed-precision training.

### Why the numbers look modest

Neural network weights are more compressible at the top of their bit representation (where structure lives) than at the bottom (where noise lives). Lossless compression can only exploit the structured part. The remaining bits are information-theoretically incompressible without losing precision.

The savings DMX achieves are at the ceiling of what's possible without giving up bit-exactness, not at the ceiling of what aggressive lossy compression could reach. This is by design — the storage tier is the archival foundation, and it has to be exact.

For a deeper explanation of why these numbers are what they are, see the technical documentation (coming soon).

---

## Deltas: efficient updates across versions

DMX supports lossless deltas between related DMX files. Given two DMX files representing related states of the same model — two training checkpoints, a base and a fine-tune, a model and one of its forks — DMX can compute a delta that reconstructs the target from the base exactly, with no floating-point error accumulation regardless of chain length or tree depth.

**Why this matters operationally:** the same delta mechanism that powers `git` and `rsync` applied to model weights. Training runs store checkpoint chains instead of duplicated full files. Model registries store base-plus-deltas instead of full copies per variant. Users download only the changes when updating. A workflow pattern familiar from every other artifact class, finally available for ML.

Deltas apply to two distinct use cases: **training chains** (time-ordered checkpoints during a training run) and **fork/variant families** (tree-ordered derivatives of a common base). The underlying mechanism is the same; the economics and workflows are different.

### Training chains

A training run that checkpoints every N steps produces a time-ordered sequence of closely related model states. DMX stores the first checkpoint in full and each subsequent checkpoint as a delta from the previous one. Reconstructing any specific checkpoint is bit-exact — the same weights the training loop saved.

**Validated measurements:**

| Scenario | Source | Compression per delta |
|---|---|---|
| Training checkpoints (10-1000 steps apart) | GPT-2 FP32 | 35-43%, distance-dependent |
| Training checkpoints | FP16 sources | Measurement pending |
| Training checkpoints | BF16 sources | Measurement pending |

A finding worth noting: **closer checkpoints compress better than distant ones**. A 10-step delta compresses at 43%; a 1000-step delta at 35%. Frequent checkpointing — which is also safer and more debuggable — costs less per checkpoint under DMX than under full-file storage. The format rewards good checkpointing discipline rather than penalizing it.

**Chain correctness:** Delta reconstruction uses integer arithmetic with exact round-tripping. Applying a chain of deltas produces the same result as applying each delta independently. There is no error accumulation across chain length. A model reconstructed from `base + delta_1 + delta_2 + ... + delta_N` is bit-identical to the original checkpoint at position N, for any N.

**Chain self-calibration:** When a checkpoint has drifted far enough from its anchor that the delta exceeds a size threshold, DMX drops a fresh anchor and the chain continues from there. No manual tuning required.

### Fork and variant families

A base model plus N variants usually means N full downloads. Under DMX, it means one download plus N small deltas.

Models rarely exist alone. A base model typically has many derived variants — instruction-tuned forks, domain-specialized fine-tunes, LoRA merges, community variants, quantized exports. Under conventional file formats, each variant costs a full model's worth of storage and bandwidth, even though most bits are shared with the base.

DMX delta compression applies directly to this case. A distributor or power user can store the base model once and maintain each variant as a small delta from that base. A user who already has the base and wants to try a new variant downloads only the delta and reconstructs the variant locally.

This changes the economics of model family distribution:

- **For users:** Trying multiple variants of a model family becomes cheap. The first variant costs a full base download; each subsequent variant costs only its delta.
- **For distributors and model hubs:** Storage and bandwidth costs for a model family scale with the number of *distinct changes* across the family, not with the number of variants. A 50-variant family of an 8B model no longer requires 50× the base model's storage per mirror.
- **For teams:** Maintaining many internal fine-tunes, customer-specific forks, or A/B candidates becomes a per-delta cost rather than a per-full-model cost.

Reconstructed variants are bit-identical to the original variant files. There is no quality difference between a variant downloaded as a full file and the same variant reconstructed from base plus delta.

**Compression depends on how the variant was produced:**

| Variant type | Expected delta size |
|---|---|
| LoRA merge into base | Small — LoRA changes are localized and sparse |
| Light fine-tune (few hundred steps) | Small-to-moderate |
| Full fine-tune (many epochs on new domain) | Moderate — most layers touched, changes stay structured |
| Format conversion (e.g. FP16→BF16 of same weights) | Near-zero when source bits align |

Measurements across common variant types are in progress.

---

## Inference: runtime derivation

> **Note:** Compressed residency inference is available via [`dmx-runtime`](https://pypi.org/project/dmx-runtime/) (`pip install dmx-runtime`). The full runtime engine (`dmx-vram`) with additional features (KV cache compression, weight paging) is not yet publicly released.
>
> **Hardware:** Compressed residency requires an NVIDIA GPU with CUDA support (any generation — RTX, A-series, H-series). Weight materialization uses standard PyTorch CUDA ops; no custom kernels or specific compute capability required. Lossless compression and decompression (storage/distribution) runs on CPU — no GPU needed.

DMX's stored file is lossless — bit-exact with the source weights. At load time, DMX works down a cascade, starting from full source fidelity and stepping to a compressed representation only when hardware constraints require it. Users don't pick precision modes; the loader walks the cascade and picks the point that fits.

**Why this matters operationally:** one canonical artifact serves heterogeneous hardware. A 12 GB laptop, a 24 GB workstation, an 80 GB server GPU, and a 32 GB edge device can all run the same DMX file, each picking the representation that fits its constraints. No per-hardware variants to maintain, no separate pre-exports for every deployment target. The same file that went into archival distribution runs directly on whatever infrastructure needs it.

### The loading cascade

```
  Lossless DMX file on disk (FP32 or FP16 source, bit-exact)
                          │
                          ▼
    ┌────────────────────────────────────────────┐
    │  Fit at full precision in VRAM, with room  │
    │  for expected KV cache?                    │
    └────────────────────────────────────────────┘
        │ yes                          │ no
        ▼                              ▼
    ┌─────────────┐        ┌──────────────────────────────┐
    │ Load at     │        │  Derive M=7 compressed form. │
    │ full        │        │  Fit in VRAM now?            │
    │ precision   │        └──────────────────────────────┘
    └─────────────┘             │ yes                │ no
    bit-exact with source       ▼                    ▼
                          ┌─────────────┐    ┌─────────────────┐
                          │ Compressed  │    │ Compressed      │
                          │ residency   │    │ residency       │
                          │ (M=7)       │    │ + weight pager  │
                          └─────────────┘    └─────────────────┘
                          weights as BFP    some weights paged
                          in VRAM           from RAM or disk
```

The cascade starts at full source fidelity and steps to a compressed representation only when hardware can't hold the full precision form. Advanced users can override the default (force compressed residency for bandwidth-bound speed gains, pin a specific mode, or use DMX files with a custom loader that ignores the cascade entirely).

The lossless file on disk is the same regardless of which runtime representation a given machine selects. The cascade happens per-machine at load time, using the same file.

### At a glance (M=7 compressed residency)

| Metric | Value |
|---|---|
| Size reduction vs FP16 | **53%** |
| Quality cost (PPL delta) | **<0.8%** across all 9 tested architectures (v1.1.0+). Phi-2 outlier resolved by rounding improvement. |
| Decode operation | **Bit manipulation only** — integer shifts + masks, no FP arithmetic at the decode step |
| Hardware requirement | **FP16 MAC, any generation** — no Hopper, no special silicon |
| Mantissa resolution | **7 bits (128 levels) per binade** — 16× FP8 E4M3's 3-bit mantissa within the same dynamic range slice |

### How DMX decode compares

All sub-FP16 formats require a decode step to produce FP16 values for the MAC unit. The cost differs:

| Format | Decode mechanism | Cost per weight | PPL delta |
|---|---|---|---|
| Q8_0 (GGUF) | `int8 × FP16_scale → FP16` | FP multiply | Minimal (8-bit mantissa equivalent) |
| FP8 E4M3 | Hardware LUT (Hopper+ only) | Near-zero on Hopper; no native path elsewhere | Higher (3-bit mantissa) |
| **DMX M=7** | `shifts + masks → legal FP16 bit pattern` | **Integer ops only, no FP arithmetic at decode** | **+0.29% measured (Llama 3.1 8B, wikitext-2)** |

DMX's decode reconstructs a legal FP16 value from the shared exponent + 7-bit mantissa via bit manipulation. No floating-point multiply, no rounding from the decode itself, no precision loss at the reconstruction step. This runs on any GPU with integer ALUs — no generation-specific hardware required.

### Where DMX M=7 may not be the best fit

- **Maximum VRAM reduction at any quality cost:** INT4/NF4 (75% savings) beats DMX M=7 (53%) if you accept 2-15% PPL degradation.
- **Hopper-native deployments:** FP8 E4M3 is handled in hardware on H100/H200 with near-zero decode overhead — if you only deploy on Hopper+ and accept 3-bit mantissa precision, FP8 is simpler.
- **Already fits at FP16:** If the model fits in VRAM at full precision, DMX adds no inference benefit (use lossless storage for distribution savings instead).

### Measured fallback quality

When the cascade reaches M=7 compressed residency, the quality cost of the step-down has been measured across multiple architectures. Numbers below are from selective-roundtrip evaluation that matches the production `dmx-vram` loader's skip_compression pattern (embeddings, `lm_head`, and normalization layers kept at FP16; linear layers compressed to M=7 BFP).

**Methodology:** wikitext-2 full test split (288K+ tokens), sliding window PPL with max_len=1024 and stride=512, per-token NLL averaged, PPL = exp(mean NLL). All models loaded as explicit FP16. Same methodology applied to every row.

> **Note on negative deltas:** Negative values (e.g., −0.19%) are within measurement noise — they indicate DMX is statistically indistinguishable from FP16, not that compression improves quality.

| Model | Parameters | Architecture | M=7 PPL delta | BFP compression ratio |
|---|---|---|---|---|
| GPT-2 | 124M | GPT-2 | −0.19% | 53.8% |
| GPT-2 Medium | 355M | GPT-2 | −0.52% | 53.8% |
| OLMo | 1B | OLMo | +0.76% | ~53% |
| Pythia | 1.4B | NeoX | +0.77% | ~53% |
| Qwen 2.5 | 1.5B | Qwen | +0.60% | ~53% |
| Phi-2 | 2.7B | Phi | +1.37% | ~53% |
| Qwen 2.5 | 3B | Qwen | +0.58% | ~53% |
| Mistral | 7B | Mistral | +0.16% | ~53% |
| Llama 3.1 | 8B | Llama | +0.29% | 53.2% |

Across the 9 architectures tested so far, M=7 selective-roundtrip delta stays under 1.5% on wikitext-2, with 8 of 9 models under 0.8%. Performance varies by architecture more than by size: within the Qwen family, delta is similar at 1.5B and 3B (+0.60% and +0.58%). Mistral 7B and Llama 3.1 8B — both standard modern transformer architectures — anchor the larger-model range at +0.16% and +0.29%. Phi-2 previously showed +1.37% under v1.0.0 (truncation-based encoding). Layer ablation identified 82% of this delta concentrated in the last 3 layers (29-31). The rounding improvement in v1.1.0 resolved this — Phi-2 now measures indistinguishable from FP16 baseline. For architectures showing elevated delta, the `--skip-layers` flag can protect specific layers at FP16 precision. Users deploying architectures not included in this measurement set (Mamba, MoE, encoder-only, multimodal, vision transformers, etc.) should verify PPL on their specific model and task.

Negative deltas on GPT-2 and GPT-2 Medium are within measurement noise — selective roundtrip at these sizes is statistically indistinguishable from FP16 inference.

**BF16 source models:** When the source is BF16 (the native training format for most modern models), M=7 quality cost drops further — BF16 has exactly 7 mantissa bits, matching M=7's precision:

| Model | Parameters | BF16 baseline PPL | M=7 quality cost |
|---|---|---|---|
| Qwen 2.5 | 1.5B | 9.1875 | +0.00%* |
| Pythia | 1.4B | 11.2500 | +0.00%* |
| Mistral | 7B | 5.0000 | +0.63% |

*\*Identical at BF16 inference precision. The shared-exponent difference is below the output resolution of BF16 computation on these models.*

### The M-dial: quality vs compression across M levels

M is a tunable parameter: the BFP mantissa bits preserved per block of 32 values. Higher M = more precision, lower M = more compression. The cascade defaults to M=7, but lower settings unlock additional VRAM for models that don't fit at M=7.

**Measured PPL delta across M levels (BF16 source, wikitext-2):**

| Model | Parameters | M=7 PPL delta | M=6 PPL delta | M=5 PPL delta |
|---|---|---|---|---|
| Qwen 2.5 | 1.5B | +0.60% | +0.43% | +2.07% |
| Mistral | 7B | +0.16% | — | +0.14% |
| Llama 3.1 | 8B | +0.29% | +0.14% | +0.92% |

| M | File savings | PPL range (7-8B models) | Use case |
|---|---|---|---|
| M=7 | 53% | +0.16% to +0.29% | Default. Near-zero quality cost. |
| M=6 | 59-60% | +0.14% to +0.43% | Sweet spot — more compression, quality still within noise. |
| M=5 | 65-66% | +0.14% to +0.92% | Aggressive — fits 8B models on 8GB GPUs. |

Larger models are more robust to lower M settings — Mistral 7B at M=5 shows only +0.14% PPL delta. The lossless source remains the source of truth; lower M settings are derived at load time or pre-compressed for distribution.

### Measured VRAM savings (compressed residency)

Weights stay compressed as BFPLinear in VRAM during inference. Measured across 9 models, 7 architecture families. Quality is indistinguishable from FP16 on lm-eval-harness benchmarks.

| Model | Family | Size | Standard FP16 (uncompressed) | DMX M=7 (compressed) | VRAM Saved | Quality |
|---|---|---|---|---|---|---|
| Pythia | NeoX | 410M | 0.81 GB | 0.51 GB | **37%** | Identical (lm-eval verified) |
| OLMo | OLMo | 1B | 2.35 GB | 1.35 GB | **43%** | Close |
| GPT-Neo | NeoX | 1.3B | 2.73 GB | 1.60 GB | **42%** | Close |
| Pythia | NeoX | 1.4B | 2.83 GB | 1.59 GB | **44%** | Identical (lm-eval verified) |
| Qwen 2.5 | Qwen | 1.5B | 3.12 GB | 1.91 GB | **39%** | Identical (lm-eval verified) |
| Qwen 2.5 | Qwen | 3B | 6.29 GB | 3.57 GB | **43%** | Identical |
| Mistral | Mistral | 7B | 14.48 GB | 7.78 GB | **46%** | Identical |
| Llama 3.1 | Llama | 8B | 16.06 GB | 9.36 GB | **42%** | Identical |
| Qwen 2.5 | Qwen | 14B | 29.69 GB | 16.96 GB | **43%** | Identical |

VRAM savings range: **37-46%**, increasing with model size. File size savings: consistently **53-54%**.

**lm-eval-harness benchmark scores (compressed residency, 0-shot):**

| Task | Pythia 410M (FP16→Compressed) | Qwen 1.5B (FP16→Compressed) |
|---|---|---|
| HellaSwag | 0.3374 → 0.3369 (Δ -0.0005) | 0.5082 → 0.5091 (Δ +0.0009) |
| ARC-Easy | 0.5185 → 0.5240 (Δ +0.0055) | 0.7673 → 0.7677 (Δ +0.0004) |

Scores are deterministic (std=0 across runs). Max absolute delta: 0.0055. Deltas go both directions (no systematic bias). Evaluation ran through BFPLinear — weights stayed compressed in VRAM during the entire benchmark.

**Distribution preservation (KL divergence):**

KL divergence measures whether the compressed model's probability distribution over the vocabulary matches FP16 — catching "confident but wrong" drift that perplexity can miss. On BF16 source models, M=7 keeps all 7 mantissa bits; the only source of difference is the shared block exponent (one exponent per 32 values instead of per-value). Lower M values drop additional mantissa bits.

| Model | M | Mean KL | Top-1 Agreement |
|---|---|---|---|
| Llama 3.1 8B | M=7 | 0.0011 | 98.1% |
| Mistral 7B | M=6 | 0.0031 | 97.1% |
| Llama 3.1 8B | M=6 | 0.0052 | 97.0% |
| Qwen 2.5 1.5B | M=6 | 0.0105 | 94.9% |
| Mistral 7B | M=5 | 0.0255 | 95.1% |
| Llama 3.1 8B | M=5 | 0.0260 | 93.7% |
| Qwen 2.5 1.5B | M=5 | 0.0467 | 90.3% |

M=6 on 7-8B models maintains 97% top-1 agreement — nearly indistinguishable from M=7. Even at M=5, larger models preserve 93-95% agreement. Smaller models (1.5B) show more sensitivity to precision reduction. Measured on 1024 tokens of wikitext-2 across 8 sliding windows (BF16 source models).

### Try compressed residency yourself

> **VRAM requirement:** The compressed model must fit entirely in GPU memory. Compressed residency reduces VRAM by ~40%, but there is no automatic paging or CPU offload if the compressed model still exceeds your GPU's capacity. Check that your model's compressed size (roughly 53% of FP16) fits your GPU before loading. Weight paging for oversized models is on the roadmap.

```bash
# Install PyTorch first if needed (see Quick Start below)
pip install dmx-compress dmx-runtime
```

```python
from dmx_runtime import from_dmx_compressed
import torch

# Load with compressed residency (~40% less VRAM)
model = from_dmx_compressed(
    "path/to/model.dmx",
    model_id="Qwen/Qwen2.5-1.5B-Instruct"
)

# Standard HuggingFace generate — works as usual
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ids = tok("The future of AI is", return_tensors="pt").input_ids.cuda()
output = model.generate(ids, max_new_tokens=50)
print(tok.decode(output[0], skip_special_tokens=True))
```

To create the `.dmx` file:
```bash
dmx compress model.safetensors model.dmx --mode bfp --mantissa-bits 7
```

Pre-compressed models available on HuggingFace:

**M=7 (default):**
- [Senat1/dmx-pythia-160m-m7](https://huggingface.co/Senat1/dmx-pythia-160m-m7) — Pythia 160M
- [Senat1/dmx-qwen2.5-1.5b-m7](https://huggingface.co/Senat1/dmx-qwen2.5-1.5b-m7) — Qwen 2.5 1.5B
- [Senat1/dmx-mistral-7b-m7](https://huggingface.co/Senat1/dmx-mistral-7b-m7) — Mistral 7B
- [Senat1/dmx-llama-3.1-8b-m7](https://huggingface.co/Senat1/dmx-llama-3.1-8b-m7) — Llama 3.1 8B
- [Senat1/dmx-qwen2.5-14b-m7](https://huggingface.co/Senat1/dmx-qwen2.5-14b-m7) — Qwen 2.5 14B

**M=6 (60% savings):** [qwen2.5-1.5b-m6](https://huggingface.co/Senat1/dmx-qwen2.5-1.5b-m6) · [mistral-7b-m6](https://huggingface.co/Senat1/dmx-mistral-7b-m6) · [llama-3.1-8b-m6](https://huggingface.co/Senat1/dmx-llama-3.1-8b-m6)

**M=5 (65% savings):** [qwen2.5-1.5b-m5](https://huggingface.co/Senat1/dmx-qwen2.5-1.5b-m5) · [mistral-7b-m5](https://huggingface.co/Senat1/dmx-mistral-7b-m5) · [llama-3.1-8b-m5](https://huggingface.co/Senat1/dmx-llama-3.1-8b-m5)

### Beyond the default cascade

The cascade above describes DMX's default behavior. Underneath it, DMX exposes a composable set of runtime components: lossless representations (FP32/FP16, INT16 via bitcast), lossy compressed residency (M=7 default, M=6 more aggressive), pre-exported quantization formats (INT8, NF4, FP8 — see Export section), and a weight pager for VRAM that can't hold the chosen representation.

Power users can stack these differently than the default. Some examples:

- **Lossless INT16 with paging.** Weights stay bit-exact with source; when VRAM runs out, paging handles the overflow. No quality tradeoff at all.
- **M=6 compressed residency.** More aggressive compression than the default, accepting a measurable quality cost in exchange for additional VRAM headroom.
- **Pre-exported INT8 or NF4.** Generate a quantized file once, load it directly. Skips runtime derivation but locks in that format.
- **Storage-only use.** Use DMX purely for lossless archival and delta compression, loading into your own inference pipeline with no runtime derivation involved.

The tradeoff axis is **speed ←→ performance** per unit of VRAM. Full precision gives speed at nominal VRAM cost; heavier compression gives more performance per GB at some quality or latency cost. Generating a non-default representation takes one-time compute to derive; once derived, the result is cached and can be paged or switched between modes without re-derivation.

Detailed configuration is documented separately in [technical design notes](./docs/DESIGN.md).

### Export to other formats — COMING SOON

DMX can derive and export weights in other formats — INT8, NF4, FP8, and additional quantization schemes — from the lossless source. This is useful when a deployment target expects a specific format natively, or when a user wants to pre-materialize a derivation rather than computing it at load time.

Export produces a new file in the target format. The source DMX file remains unchanged. Each export is a derivation from the lossless source, so multiple exports from the same source are independent of each other — converting to INT8 and then to NF4 produces the same result as converting directly to NF4 from the DMX source.

*Export paths are implemented but not yet fully verified. Detailed documentation will follow once behavior on all source file types is confirmed.*

---

## Beyond neural network weights

The structural properties DMX exploits — byte-plane decomposition, exponent clustering, exact integer deltas — are not unique to neural network weights. They apply to any dense floating-point data with similar statistical structure.

### 3D Gaussian Splats

3DGS rendering is structurally the same pattern as model inference: a lossless source is stored at rest, and at runtime DMX derives a representation optimized for the specific downstream consumer. For model inference, the consumer is a transformer and the derivation targets hardware constraints (VRAM, bandwidth, supported precision formats). For 3DGS, the consumer is a browser renderer and the derivation targets rendering speed and streaming bandwidth, with quality held below the perceptual threshold.

The same DMX file can be the lossless source for either use. The derivation happens at load time, tuned to what the consumer needs.

**Two workflows are supported:**

- **Lossless source with runtime derivation (recommended).** Store the 3DGS scene as a lossless DMX file. At load time, the decoder derives a streaming-quality representation for rendering. The source remains available for archival, re-derivation at different quality targets, or any future downstream use. Provenance records the file as lossless.

- **Direct lossy export (available when archival fidelity is not required).** When file size is the priority and the 3DGS scene will only ever be consumed by a renderer at perceptual quality, DMX can compress directly to a lossy representation at rest. This is opt-in, not the default. The provenance manifest records that the file is lossy, so downstream consumers know they are not receiving an archival source.

**Measured rendering quality on shipped lossy path:** The lossy compression path (BFP with FLAC entropy coding) produces ~63% bandwidth reduction on typical 3DGS scenes (measured on the bonsai scene, ~265 MB → 97.2 MB), with rendering quality at PSNR 48.50 dB mean and SSIM 0.9997 across 100 viewpoints per scene. These numbers are well above the perceptual threshold for visible difference — the rendered scene is indistinguishable from the source in practice. Lossless compression numbers under the current main encoder are pending measurement.

**dmx-web** is the companion Rust/WASM browser decoder ([github.com/willjriley/dmx-web](https://github.com/willjriley/dmx-web)). It reconstructs DMX-compressed 3DGS data directly in the browser and includes a real-time Gaussian splatting viewer rendered via WebGL2. The same compressed file sitting on a server decodes and renders in a browser tab — no server-side processing, no intermediate format conversions, and no Python runtime required.

**Live demo:** [huggingface.co/spaces/Senat1/dmx-3dgs-viewer](https://huggingface.co/spaces/Senat1/dmx-3dgs-viewer)

The decoder has been validated against 2,436 tensor roundtrips across all encoding paths and entropy codecs (zstd, brotli, FLAC).

---

## Provenance: supply-chain visibility

Every DMX file carries an embedded provenance manifest that records what it is, where it came from, and what operations produced it. The manifest is part of the file itself, not a sidecar — it travels with the data and cannot be lost or replaced independently.

**Why this matters operationally:** supply-chain visibility for ML artifacts. The same pattern that SBOMs (Software Bill of Materials) provide for application builds and that signed commits provide for code, applied to neural network weights. A consumer receiving a DMX file can verify its source, trace its lineage, and detect whether lossy operations appear anywhere in its history. This matters for regulated deployment contexts, for distribution integrity, and for debugging when a model variant behaves unexpectedly in production.

The manifest provides three concrete capabilities:

**Identity.** Each file carries a hash of its source weights and a hash of its own compressed data. A user receiving a DMX file can verify that it came from the source it claims to come from and that its contents haven't been altered since creation.

**Lineage.** Every file records its parent (if derived), its delta base (if it's a delta), its lineage depth, and the hash of the original root source. A user can trace any DMX file back to the original checkpoint it descended from, through any number of intermediate operations.

**Warnings at boundaries.** When an operation produces a derivative file whose quality is bounded by an earlier lossy step — for example, a delta computed against a lossy base — the resulting file carries a warning in its manifest. Downstream tooling can detect these warnings and decide how to handle them (proceed with caveat, refuse to use as a lossless source, require explicit user confirmation). The format reports the state; consumers enforce policy.

The manifest aligns with real standards where they exist: field conventions draw from the EU AI Act Article 13 transparency requirements, the NIST AI Risk Management Framework, and the C2PA content provenance standard. This is deliberate — DMX is designed to work in regulated and distribution-integrity-sensitive contexts, not only in research settings.

### What Phase 1 delivers

The initial implementation focuses on the trust-critical subset:

- Source identity (source hash, content hash)
- Lineage (parent, delta base, lineage depth, root hash)
- Lossy-source warnings (automatic when an operation produces output bounded by an earlier lossy step)
- Integrity (cryptographic hash of the compressed data)
- Creation metadata (timestamp, model architecture, parameter count, source format)

### What's coming later

Training-pipeline integration (checkpoint step, epoch, training config hash), cryptographic signing (C2PA-compatible detached signatures), regulatory metadata fields (license, author, intended use, known limitations), and merge-tracking for weight-averaged models are defined in the manifest schema and will ship in subsequent phases.

The full manifest schema is documented in [DESIGN.md](./docs/DESIGN.md).

---

## Where DMX fits in ML DevOps

DMX maps onto the stages that existing ML infrastructure already has. It doesn't ask operators to adopt new workflows — it improves the economics of stages they're already running.

### Training and checkpointing

Training pipelines already produce checkpoints at intervals. DMX replaces bulk full-file checkpoint storage with lossless base-plus-deltas, recording ~35-43% per-delta compression on measured training chains. Frequent checkpointing becomes affordable; checkpoint rollback is bit-exact reconstruction from any point in the chain. Provenance captures checkpoint step, training configuration hash, and lineage to the root source.

### Model registry and distribution

Model registries currently store full files per variant. DMX stores the base once and maintains variants as small deltas, scaling registry storage with the number of distinct changes across a model family rather than the number of variants. A registry with 50 variants of an 8B model stops requiring 50× the base model's storage per mirror. Distribution bandwidth drops accordingly — users pull the base once, variants arrive as small delta files.

### Serving and deployment

> **Note:** Basic compressed residency is available via `dmx-runtime`. Production serving features (auto-cascade, weight paging, KV cache compression) require `dmx-vram`, which is not yet publicly released.

Production inference infrastructure has a VRAM budget per GPU and wants to maximize throughput per dollar. DMX compressed residency (measured +0.29% PPL delta on Llama 3.1 8B) reduces weight VRAM by ~40%, freeing that capacity for larger batch sizes, longer contexts, multi-model tenancy, or deployment on smaller hardware tiers. One DMX file serves all of these use cases; the runtime picks the appropriate representation per machine.

### Governance and compliance

Regulated deployment contexts (EU AI Act, NIST AI RMF, industry-specific compliance) require supply-chain documentation for model artifacts. DMX's embedded provenance manifests align with these standards — source identity, lineage tracing, lossy-operation warnings, integrity verification. The same infrastructure that provides DevOps convenience also satisfies audit requirements.

### Edge and on-device inference — ROADMAP

> **Note:** Requires `dmx-vram` runtime. Not yet publicly released.

Mobile and embedded devices have sharply constrained memory. At DMX M=7 compressed residency, models that don't fit natively (3B-8B class on 8-12 GB mobile devices) become candidates. The same file that ships to server clusters ships to on-device deployment; the cascade picks compressed residency automatically on memory-constrained hardware. *(Mobile runtime kernels are roadmap; current runtime is CUDA-based.)*

### When DMX is less interesting

DMX is less interesting if you only deploy a single model to a single hardware target and don't care about checkpoint history, distribution topology, or lineage verification. Conventional tools work fine for that case. DMX's value scales with the complexity of the artifact lifecycle you're managing.

---

## Roadmap

These are directional ideas under consideration, not committed features. They describe where the format could go if the foundations hold up.

### DMX-Server

A server component that holds DMX files (base models, variants, checkpoint chains) and serves derived artifacts on demand. Local or cloud-hosted.

A client asks for "this base model at this precision for this hardware," and the server derives the requested representation on the fly from the lossless source it holds, then streams the result. The same derivation mechanism that enables local load-time auto-selection (see Inference section) extends naturally to server-side on-demand export — the local DMX loader and DMX-Server are running the same transformation, just in different locations.

This inverts the conventional distribution model. Rather than pre-exporting every possible variant for every possible hardware target and storing them all, the server stores only the lossless canonical source plus any relevant deltas, and produces variants as needed. Popular derivations can be cached; cold derivations are cheap to recompute on demand.

### DMX-REPO (or HuggingFace extension)

Version-control-like workflow for model families. Base models as commits, variants as branches, deltas as the diffs. Whether this lives as a standalone repository format or as an extension layered onto existing model hubs (HuggingFace in particular) is under consideration — the extension path is likely the more pragmatic route because it meets users where they already are.

### Other directions under exploration

- Additional data modalities beyond neural network weights and 3D Gaussian Splats — any dense floating-point data with exploitable structure is a candidate.
- Provenance manifest expansion beyond Phase 1 — cryptographic signing (C2PA-compatible), training pipeline integration (checkpoint step, epoch, training config hash), regulatory metadata fields, and merge tracking for weight-averaged models. Schema defined; implementation staged.
- Format conversion utilities that make DMX a transit format between existing neural network file formats without requiring full re-export.

---

## Status

DMX is in active development. The format, tooling, and documentation are evolving.

**Currently shipped:**
- Lossless storage on FP32, FP16, and BF16 sources
- Lossless storage as the default CLI behavior
- Lossless delta compression on FP32 training checkpoints via the CLI
- `.dmx` files accepted directly as delta inputs (base or target, any combination)
- Provenance manifest Phase 1 embedded in every new `.dmx` and `.dmxd` file (source identity, lineage, lossy-source warnings, integrity)
- Lineage chain reference through delta operations (content_hash-based, not file-SHA-based, when inputs are `.dmx`)
- `dmx inspect` command for manifest display and content-hash verification
- Legacy lossy delta path preserved as `--lossy-quantized` opt-in
- 3DGS streaming at ~60%+ bandwidth reduction with perceptual quality (PSNR 48-53 dB, SSIM 0.9997+); dmx-web Rust/WASM browser decoder with real-time viewer; decoder validated against 2,436 tensor roundtrips

**Validated internally (not yet publicly released):**
- Auto-selected inference mode on Llama-class models (requires dmx-vram runtime)
- M=7 compressed residency with measured PPL across 9 architectures

**In validation:**
- Lossless delta compression on FP16 and BF16 training checkpoints
- Fork and variant delta scenarios (LoRA merges, fine-tunes, format conversions)
- Additional model architectures for inference quality measurement

**Planned (Phase 2 and beyond):**
- Provenance manifest expansion: cryptographic signing, training pipeline integration, regulatory fields, merge tracking
- Published benchmark corpus and reproducibility scripts
- Expanded auto-selection heuristics for emerging hardware

---

## Technical details

For the underlying mechanics — how storage compression works, how deltas are computed, how the inference auto-selection decides, and what the precision and correctness guarantees formally are — see the technical documentation (coming soon).

For measurement methodology and reproducibility — how the numbers in this document were produced, which benchmarks were used, and how to reproduce them — see the benchmark documentation (coming soon).

---

## Quick start

### Install

```bash
# Install PyTorch first (if not already installed) — pick your CUDA version from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install DMX
pip install dmx-compress
```

GPU-accelerated inference is included automatically — no compiler or extra setup needed. The package ships with a pre-compiled CUDA kernel for NVIDIA GPUs (sm_70 through sm_90: V100, T4, RTX 30xx/40xx, A100, L40, H100).

### Compress a model (lossless)

```bash
dmx compress model.safetensors model.dmx
```

```
Original:   474.7 MB  (GPT-2 124M, FP32)
Compressed: 397.6 MB  (16.2% savings)
Result:     LOSSLESS — all 148 tensors exactly match
```

### Decompress (bit-exact reconstruction)

```bash
dmx decompress model.dmx restored.safetensors
```

### Verify roundtrip

```bash
dmx verify model.safetensors model.dmx
```

```
Result: LOSSLESS - all tensors exactly match
Overall verdict: PASS
```

### Delta compression (training checkpoints)

```bash
dmx delta-compress base.safetensors checkpoint.safetensors delta.dmxd
dmx delta-reconstruct base.safetensors delta.dmxd restored.safetensors
```

### Inspect provenance

```bash
dmx inspect model.dmx
```

```json
{
  "dmx_version": "1.0",
  "source_format": "safetensors",
  "source_hash": "sha256:ae60e8b7...",
  "created": "2026-04-19T21:58:58Z",
  "param_count": 124439808,
  "lineage_depth": 0,
  "root_hash": "sha256:ae60e8b7...",
  "export_warning": null,
  "content_hash": "sha256:3107d014..."
}
```

```bash
dmx inspect model.dmx --verify
```

### Download pre-compressed models

| Model | Original | DMX | Savings | Verified |
|-------|----------|-----|---------|----------|
| [GPT-2 124M FP32](https://huggingface.co/Senat1/dmx-gpt2-lossless) | 474.7 MB | 397.6 MB | 16.2% | Bit-exact roundtrip |

---

## License and patent

**Code:** MIT License — free to use, modify, and distribute.

**Methods:** Patent Pending (U.S. Provisional Applications filed April 2026). The patented methods cover aligned cross-layer quantization for neural network weight compression and stream-separated block floating point encoding with independent entropy coding. Personal, academic, and open-source use is unrestricted. Commercial use of the patented methods may require a license from the inventor — contact **bill.riley@gmail.com**.

## Citation

```
@software{riley2026dmx,
  author = {Riley, William J},
  title = {DMX: Delta Multiplexed Model Format},
  year = {2026},
  url = {https://github.com/willjriley/dmx}
}
```

---

*End of README.*