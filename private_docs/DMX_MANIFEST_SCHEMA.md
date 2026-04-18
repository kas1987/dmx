# DMX Provenance Manifest Schema v1.0

## Overview

Every `.dmx` file embeds a provenance manifest -- a JSON object stored in the file header that provides complete traceability from compressed weights back to their original source. The manifest serves three purposes: (1) **trust** -- any consumer can verify the full lineage of a `.dmx` file without external lookups; (2) **regulatory compliance** -- fields are aligned with EU AI Act model card requirements, NIST AI RMF, and the C2PA content provenance standard; (3) **export guardrails** -- the manifest auto-populates warnings when a derivative file is created from a low-mantissa source, preventing silent quality degradation through delta chains. The manifest is machine-readable (JSON), human-inspectable, and designed so that tooling can enforce policy checks at both write time and load time.

---

## Field Reference

### Core

| Field | Type | Required | Description |
|---|---|---|---|
| `dmx_version` | string | **required** | Manifest format version. Currently `"1.0"`. Consumers must reject files with an unrecognized major version. |
| `source_format` | string | **required** | Precision of the source weights before compression. One of `"fp16"`, `"fp32"`, `"bf16"`. |
| `source_hash` | string | **required** | SHA-256 hash of the source weight file, prefixed `"sha256:"`. Used to verify which exact checkpoint was compressed. |
| `mantissa_bits` | int | **required** | BFP mantissa width used during encoding (e.g. 7 for standard, 5 for aggressive). |
| `group_size` | int | **required** | BFP group size (number of elements sharing an exponent). Typical values: 32, 64, 128. |
| `created` | string | **required** | ISO 8601 timestamp of when this `.dmx` file was produced. Always UTC (`Z` suffix). |
| `model_architecture` | string | **required** | HuggingFace-style architecture class name, e.g. `"Qwen2ForCausalLM"`, `"GPT2LMHeadModel"`. |
| `param_count` | int | **required** | Total parameter count of the model. |

### Lineage

| Field | Type | Required | Description |
|---|---|---|---|
| `parent` | string or null | **required** | `content_hash` of the parent `.dmx` file this was derived from. `null` for first-generation compressions from a raw checkpoint. |
| `delta_base` | string or null | **required** | `content_hash` of the base `.dmx` file that a delta must be applied against. `null` if this file is self-contained (not a delta). |
| `lineage_depth` | int | **required** | Number of generations from the original source. `0` = compressed directly from raw checkpoint. `1` = derived from a depth-0 `.dmx`. Etc. |
| `root_hash` | string | **required** | `source_hash` of the original raw checkpoint at the root of this lineage. Copied forward through every generation so the origin is always one field away. |

### Training

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint_step` | int or null | optional | Global training step at which the source checkpoint was saved. `null` if unknown. |
| `checkpoint_epoch` | int or null | optional | Training epoch at which the source checkpoint was saved. `null` if unknown. |
| `training_config_hash` | string or null | optional | Hash of the training config (e.g. `sha256:` of the YAML/JSON config file). Enables exact reproduction of training conditions. `null` if unavailable. |
| `tags` | array of strings | optional | Free-form tags for filtering and search. Examples: `["finetune", "rlhf", "v2.1"]`. Defaults to `[]`. |

### Export

| Field | Type | Required | Description |
|---|---|---|---|
| `export_warning` | string or null | **required** | Auto-populated warning string when this file was derived from a source with reduced precision. See "Export Guardrail Logic" below. `null` when no warning applies. |
| `merge_parents` | array of strings | optional | For weight-averaged / merged models: list of `content_hash` values of each parent `.dmx` used in the merge. Defaults to `[]`. |
| `compatible_bases` | array of strings | optional | For delta files: list of `content_hash` values of base files this delta can legally be applied to. Defaults to `[]`. |

### Regulatory / Audit

| Field | Type | Required | Description |
|---|---|---|---|
| `license` | string or null | optional | SPDX license identifier or URL. E.g. `"Apache-2.0"`, `"https://example.com/license"`. |
| `author` | string or null | optional | Person or entity that created this `.dmx` file. |
| `organization` | string or null | optional | Organization responsible for the model or this compressed artifact. |
| `intended_use` | string or null | optional | Free-text description of intended use. Maps to EU AI Act Article 13 transparency requirements. |
| `known_limitations` | string or null | optional | Free-text description of known limitations, biases, or failure modes. Maps to EU AI Act model card fields. |
| `content_hash` | string | **required** | SHA-256 hash of the actual compressed weight data in this `.dmx` file (excluding the manifest itself). This is the file's identity for lineage references. Prefixed `"sha256:"`. |
| `signature` | string or null | optional | Reserved for C2PA-style digital signature. When populated, contains a detached signature over `content_hash` that can be verified against a public key. `null` until signing infrastructure is available. |
| `modification_history` | array of objects | optional | Ordered log of modifications. Each entry: `{"timestamp": "<ISO 8601>", "description": "<what changed>"}`. Append-only. Defaults to `[]`. |

---

## Example Manifests

### (a) Fresh compression from FP16

A Qwen2.5-3B checkpoint compressed directly from its HuggingFace FP16 weights.

```json
{
  "dmx_version": "1.0",
  "source_format": "fp16",
  "source_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "mantissa_bits": 7,
  "group_size": 64,
  "created": "2026-04-17T14:30:00Z",
  "model_architecture": "Qwen2ForCausalLM",
  "param_count": 3090000000,

  "parent": null,
  "delta_base": null,
  "lineage_depth": 0,
  "root_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",

  "checkpoint_step": 150000,
  "checkpoint_epoch": 3,
  "training_config_hash": "sha256:ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00",
  "tags": ["base", "qwen2.5"],

  "export_warning": null,
  "merge_parents": [],
  "compatible_bases": [],

  "license": "Apache-2.0",
  "author": "will@dmx.dev",
  "organization": "DMX",
  "intended_use": "General-purpose causal language model",
  "known_limitations": null,
  "content_hash": "sha256:deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
  "signature": null,
  "modification_history": []
}
```

### (b) Delta checkpoint (training step 151000, derived from step 150000)

Only the weight differences from the base are stored. The base file's `content_hash` is referenced.

```json
{
  "dmx_version": "1.0",
  "source_format": "fp16",
  "source_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "mantissa_bits": 7,
  "group_size": 64,
  "created": "2026-04-17T15:00:00Z",
  "model_architecture": "Qwen2ForCausalLM",
  "param_count": 3090000000,

  "parent": "sha256:deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
  "delta_base": "sha256:deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
  "lineage_depth": 1,
  "root_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",

  "checkpoint_step": 151000,
  "checkpoint_epoch": 3,
  "training_config_hash": "sha256:ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00",
  "tags": ["delta", "qwen2.5", "step-151000"],

  "export_warning": null,
  "merge_parents": [],
  "compatible_bases": [
    "sha256:deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
  ],

  "license": "Apache-2.0",
  "author": "will@dmx.dev",
  "organization": "DMX",
  "intended_use": "Delta checkpoint for research cadence workflow",
  "known_limitations": "Requires base file deadbeef... to reconstruct full weights.",
  "content_hash": "sha256:cafe0001cafe0001cafe0001cafe0001cafe0001cafe0001cafe0001cafe0001",
  "signature": null,
  "modification_history": []
}
```

### (c) Fork (weight-averaged merge of two fine-tuned variants)

Two fine-tuned `.dmx` files merged via weight averaging. Both parents are recorded.

```json
{
  "dmx_version": "1.0",
  "source_format": "fp16",
  "source_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "mantissa_bits": 7,
  "group_size": 64,
  "created": "2026-04-17T16:00:00Z",
  "model_architecture": "Qwen2ForCausalLM",
  "param_count": 3090000000,

  "parent": null,
  "delta_base": null,
  "lineage_depth": 1,
  "root_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",

  "checkpoint_step": null,
  "checkpoint_epoch": null,
  "training_config_hash": null,
  "tags": ["merge", "qwen2.5", "code-math-blend"],

  "export_warning": null,
  "merge_parents": [
    "sha256:aaaa0001aaaa0001aaaa0001aaaa0001aaaa0001aaaa0001aaaa0001aaaa0001",
    "sha256:bbbb0002bbbb0002bbbb0002bbbb0002bbbb0002bbbb0002bbbb0002bbbb0002"
  ],
  "compatible_bases": [],

  "license": "Apache-2.0",
  "author": "will@dmx.dev",
  "organization": "DMX",
  "intended_use": "Merged model combining code and math fine-tuning",
  "known_limitations": "Quality characteristics are a blend of both parents; evaluate independently.",
  "content_hash": "sha256:merge999merge999merge999merge999merge999merge999merge999merge999",
  "signature": null,
  "modification_history": [
    {
      "timestamp": "2026-04-17T16:00:00Z",
      "description": "Weight-averaged merge of code-ft (aaaa0001...) and math-ft (bbbb0002...) at 50/50 ratio."
    }
  ]
}
```

---

## Export Guardrail Logic

The `export_warning` field is auto-populated at write time by the DMX encoder. The rules are:

1. **Low-mantissa source.** If `mantissa_bits < 7` (the default BFP-7 precision), set:
   ```
   "WARNING: Compressed with M=<N>. Derivatives may accumulate quantization error beyond GPU variance. Re-derive from the root checkpoint (root_hash) for maximum fidelity."
   ```

2. **Deep lineage.** If `lineage_depth >= 3`, set:
   ```
   "WARNING: Lineage depth <D>. This file is <D> generations from the original checkpoint. Verify quality against root_hash before further derivation."
   ```

3. **Delta from low-M base.** If `delta_base` is non-null and the base file's `mantissa_bits < 7`, set:
   ```
   "WARNING: Delta computed against a base with M=<N>. Reconstruction fidelity is bounded by the base precision."
   ```

4. **Multiple warnings.** When more than one condition triggers, concatenate the warnings separated by `" | "`.

5. **No warning.** When no condition applies, `export_warning` is `null`.

The encoder must read the parent manifest to evaluate rules 3 (base precision). If the parent manifest is unavailable (e.g., the base file was deleted), the encoder sets:
```
"WARNING: Parent manifest unavailable. Cannot verify base precision. Treat as unverified lineage."
```

These warnings are informational. Tooling may choose to block or require confirmation on write when warnings are present, but the format itself does not enforce this -- policy is a consumer-side decision.

---

## Embedding Strategy

**Recommendation: embed the manifest in the `.dmx` file header.**

### Layout

```
[ 4 bytes: magic "DMX\x00" ]
[ 4 bytes: manifest length (uint32 LE) ]
[ N bytes: manifest JSON (UTF-8, no BOM) ]
[ ... compressed weight data ... ]
```

### Rationale

| Approach | Pros | Cons |
|---|---|---|
| **Embedded (recommended)** | Single file = single artifact. No desync risk. Works with any transport (S3, HTTP range requests, local fs). Manifest available without external lookup. | Slightly more complex parser. Manifest update requires rewrite (acceptable -- manifests are immutable after creation). |
| Sidecar `.dmx.json` | Simple to read/write independently. | Two files to keep in sync. Easy to lose the sidecar on copy/move. Breaks single-artifact trust model. |
| Embedded + optional sidecar cache | Best of both for tooling that wants fast manifest-only reads. | Added complexity. Sidecar is purely a cache, never authoritative. |

The embedded approach is recommended because the core value proposition of the manifest -- traceability -- depends on the manifest being inseparable from the weights it describes. A sidecar can be lost, replaced, or forged independently. An embedded manifest travels with the data and can be integrity-checked via `content_hash`.

### content_hash computation

The `content_hash` covers everything after the manifest block (i.e., the compressed weight data only). This means the manifest can reference its own file's data hash without a circular dependency: first write the weights, compute their hash, then write the manifest header pointing to that hash.

### Immutability

Manifests are write-once. To "update" a manifest (e.g., add a signature after the fact), create a new `.dmx` file with the updated manifest and the same weight data. The `modification_history` array records the reason. The old `content_hash` remains valid since the weight data is unchanged; only the file-level hash changes.

---

## Future Extensions

- **`signature` field**: Will carry a C2PA-compatible detached signature once signing infrastructure is in place. The signature covers `content_hash`, ensuring weight integrity is cryptographically bound to a known publisher.
- **`quantization_map`**: Per-tensor mantissa bit allocation when adaptive selection is used. Currently omitted to keep v1.0 simple; the per-tensor codec config is stored in the weight data section.
- **`entropy_codec`**: Which entropy coder was selected per tensor (zstd-19, FLAC, brotli-11). Same rationale as above -- stored in weight data for v1.0, may be promoted to manifest in v1.1 for fast inspection without decompression.
