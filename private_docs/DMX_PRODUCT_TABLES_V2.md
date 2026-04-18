# DMX Product Tables v2 (2026-04-17)

Supersedes all previous framing. Three stories, four tables, no mixing.

---

## Table 1: DMX Storage (lossless, default)

*Your file, smaller, bit-exact. Foundation for deltas, exports, and forks.*

| Source format | DMX mode | Disk savings | Precision |
|---|---|---|---|
| FP32 | INT32 aligned | ~28% | Lossless — exact integer arithmetic |
| FP16 | INT16 aligned | ~34% | Lossless — exact integer arithmetic |

*Auto-detected from source. No settings needed.*
*Tested on: GPT-2 124M (FP32), Qwen 2.5 1.5B (FP16). Additional families pending.*

---

## Table 2: DMX Delta Compression

*Requires lossless base. Exact integer deltas — zero error, any chain length.*

| Scenario | Savings | Precision |
|---|---|---|
| Consecutive training checkpoints | ~87% per delta | Lossless (exact integers) |
| Fine-tune variant from base | ~80% per variant | Lossless (exact integers) |
| Full checkpoint + optimizer | ~79% | Lossless (exact integers) |
| Fork: try multiple training directions | Base + small delta per fork | Lossless (exact integers) |

*Tested on: GPT-2, T5, TinyLlama 1.1B, Qwen 3B. Chain resume drift: in testing (INT32 and INT16 paths).*

---

## Table 3: DMX Inference (running models)

*Derived from lossless base at load time. Smaller than INT8, higher quality.*

| Mode | VRAM | Speed | Quality |
|---|---|---|---|
| Inflate to FP16 | Same as FP16 | Full speed | +0.03% PPL (Llama 8B) |
| Compressed residency (M=7) | ~50% less | Full speed on large models | +0.03% PPL |
| Compressed + pager | Minimal | Slower | +0.03% PPL |

*Auto-detected based on hardware. Two paths to inference:*
- *Path A (auto): load lossless base → derive M=7 on the fly → cache for next time*
- *Path B (explicit): pre-export M=7/INT8/NF4/FP8 from lossless base → load directly*

*DMX M=7 (~8.25 bits) vs INT8 (~8.5 bits): smaller file, higher quality.*
*Quality: +0.03% perplexity, Llama 3.1 8B, wikitext-2, 289K tokens. Additional architectures pending.*

---

## Table 4: Inference Compression Slider (advanced)

*Default is M=7. Adjust only if you need more compression.*

| Setting | Disk savings (from FP16) | Quality | At the cost of |
|---|---|---|---|
| M=7 (default) | ~50% | +0.03% PPL — no detectable difference | — |
| M=6 | ~60% | +0.16% PPL — slight tradeoff | Output may differ slightly on complex tasks |

---

## Summary: Three settings, clean

| Setting | Purpose | When |
|---|---|---|
| Lossless (INT32/INT16 aligned) | Store, train, export, fork, delta | Default — always |
| M=7 | Run models (inference) | Auto-derived at load or pre-exported |
| M=6 | Run models, max compression | Advanced users only |

---

## What this replaces

Previous framing mixed compression, inference, and training stories. Key corrections:
- "55-80% savings" was inference-optimized lossy + bundled FP32→FP16 conversion
- "Full FP16 fidelity" at M=10 was not truly lossless (3.77e-6 error)
- M=10 and M=23 BFP are dead settings — no practical use case
- The real differentiator is the delta stack (aligned scales + exact integer deltas + competitive entropy + component grouping), not single-file compression
- Lossless storage is the foundation that enables everything else

---

## Pending work

### Phase A: Validate claims (can run in parallel — pods for GPU tests)

**Lossless storage (Table 1):**
- [ ] INT32 lossless savings range: GPT-2 (124M, 355M, 774M), Pythia 160M — FP32 source
- [ ] INT16 lossless savings range: Qwen 1.5B, Qwen 3B, Phi-3.5, SmolLM, Llama 8B — FP16 source
- [ ] Roundtrip verify: compress → decompress → bit-exact match on all above

**Delta chain (Table 2):**
- [ ] Re-run 10K step / 50 resume chain test with INT32 lossless (expect zero drift)
- [ ] Re-run 10K step / 50 resume chain test with INT16 lossless (measure drift)
- [ ] Delta savings range: verify ~87% holds across families with lossless base

**Inference (Table 3):**
- [ ] M=7 PPL on Llama 8B (have +0.03%), verify on Qwen 1.5B, Phi-3.5, GPT-2
- [ ] M=6 PPL across same families (have +0.16% on Llama, need range)
- [ ] M=7 file size vs INT8 comparison (confirm quality-per-bit claim)

### Phase B: Squeeze lossless further (research, after Phase A)
- [ ] Byte transposition ON TOP of INT32/INT16 aligned — does layering help?
- [ ] FLAC/WavPack per-stream instead of zstd (sequential correlation in aligned integers)
- [ ] Competitive per-tensor entropy on the lossless path
- [ ] Any other novel ideas for lossless savings without breaking the foundation

### Phase C: Implementation (after Phase A validates)
- [ ] Implement lossless as default in compress_file (auto-detect FP32→INT32, FP16→INT16)
- [ ] Implement auto-derive M=7 at load time with caching
- [ ] Provenance manifest embedding in .dmx files
- [ ] Update README and site to match these tables
- [ ] Red team the new framing

### Phase D: Patent review
- [ ] Review current patent filings against lossless pivot — any new claims to file?
- [ ] Lossless + exact delta chain + derive-on-demand as a combined method claim
