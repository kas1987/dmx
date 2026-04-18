# M=10 / M=23 Compression Results (2026-04-17)

## Measured on local 4090 Laptop, zstd level 19

### FP16 Source Models at M=10 (effectively lossless FP16)

| Model | Params | Source | M=6 | M=6 savings | M=10 | M=10 savings |
|-------|--------|--------|-----|-------------|------|--------------|
| Qwen 2.5 1.5B | 1.5B | 3090 MB | 1239 MB | 60% | 2023 MB | **34.5%** |
| SmolLM 135M | 135M | 513 MB | 210 MB | 61% | 176 MB | **67%** * |

\* SmolLM is FP16 source but was compressed alongside FP32 models. Need to verify source dtype.

### FP32 Source Models at M=10 (downcasts to FP16, then BFP)

| Model | Params | Source | M=6 | M=6 savings | M=10 | M=10 savings |
|-------|--------|--------|-----|-------------|------|--------------|
| GPT-2 | 124M | 548 MB | 198 MB | 64% | 164 MB | 70% |
| GPT-2 Medium | 355M | 1520 MB | 551 MB | 64% | 472 MB | 69% |
| GPT-2 Large | 774M | 3247 MB | 1267 MB | 61% | 1030 MB | 68% |

**WARNING**: FP32 M=10 numbers include the free FP32→FP16 conversion (50%).
Honest DMX-only savings from FP16 baseline would be ~34-35%.

### FP32 Source at M=23 (native FP32 BFP, full FP32 fidelity)

| Model | Params | Source | M=23 | M=23 savings | Max error | Verified |
|-------|--------|--------|------|--------------|-----------|----------|
| GPT-2 | 124M | 548 MB | 392 MB | **28.5%** | 3.77e-6 | Full roundtrip |

## Key Findings

1. **M=10 on FP16 source: ~34.5% savings** — this is the honest number for Table 1
2. **M=10 on FP32 source: ~68-70%** — misleading, bundles FP32→FP16 conversion  
3. **M=23 on FP32 source: ~28.5%** — honest FP32 full-fidelity number
4. **M=10 beats M=6 on FP32 sources** — less truncation = more structure = better entropy coding
5. **M=10 loses to M=6 on FP16 sources** — 34.5% vs 60%. The extra mantissa bits cost disk space when entropy coding can't recover the difference.
6. **Compression time**: M=10 with zstd-19 is very slow (Qwen 1.5B took 117 minutes). Future: use zstd level 3-5 for testing.

## Updated Table 1: DMX as Storage

| Source format | Setting | Disk savings | What's preserved |
|---------------|---------|--------------|------------------|
| FP32 | M=23 | ~28% | Full FP32 precision |
| FP16 | M=10 | ~34% | Full FP16 precision |

## Updated Table 4: Advanced Slider (from FP16 source)

| Setting | Disk savings | Fidelity | At the cost of |
|---------|--------------|----------|----------------|
| M=10 | ~34% | Effectively lossless | — |
| M=7 | ~50% est | Within GPU variance | Not safe for training or export |
| M=6 | ~60% | 99.5%+ | Output may differ slightly |
