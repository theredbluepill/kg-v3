---
type: "Reference"
title: "Kaggriculture encoder reuses Isaiah's stateless layers"
description: "The rebuilt encoder uses Isaiah's StatelessTransformerV1 classes, token roles, packing, compile hook and initialization; only stem input widths are game-specific, and a guard chunks compiled GEMMs below 2^31."
tags: ["kaggriculture-v3", "model", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/kaggriculture/types.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:python/owl/train/optimizer.py"}, {"resource": "repository:configs/model/kaggriculture.yaml"}, {"resource": "repository:tests/kaggriculture/test_model_encoder.py"}, {"resource": "repository:tests/kaggriculture/test_types.py"}, {"resource": "repository:tests/owl/model/test_model_config_files.py"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/2.1-encoder.md"}]
---

# Kaggriculture encoder reuses Isaiah's stateless layers

## What changed (rebuild Task 2.1)

- **`python/owl/kaggriculture/types.py`:** contract v4 in code. It defines the 29-field `KaggricultureObsBatch` with `check_contract()`, the pinned observation and action enums, and the size constants.
- **`python/owl/model/base.py`:** `BaseModelAPI[ObsT, ActT, ActSpecT]` and `ModelOutput` / `ModelServingOutput[ActT]` use PEP 696 defaults, so Isaiah's bare annotations keep meaning Orbit. `count_non_masked_tokens` narrows to `ObsBatch` with `isinstance` and requires other games to override it. `python/owl/train/optimizer.py` accepts any game's model.
- **`python/owl/model/kaggriculture.py`:**
  - one `ObservationInputStem` per entity group, fed float channels plus one-hot categories (widths 133/371/16/11/44/15)
  - separate per-role token parameters; players are `player_tokens + player_feature_proj(...)`, and the global token is `global_proj(...)`
  - a `TransformerBlock` trunk with `final_norm`, built from a typed `StatelessTransformerV1Config`
  - Isaiah's flash packing, compile hook and initialization helpers, reused unchanged
  - named `KaggricultureEncoded` fields, with every seat row encoded independently
  - the critic and action heads raise until Tasks 2.2/2.3
- **Compiled-GEMM guard** (from the [[compiled-gemm-template-overflows-above-2-21-rows|overflow Reference]]): `Kmax = max(D, D·mlp_ratio)`. The padded path chunks complete rows below 2^31; the packed path raises. At the preset, 5,915 padded rows are allowed per forward, against 256 per training forward.
- **`configs/model/kaggriculture.yaml`:** Isaiah's 6m ladder point with depth 8. Encoder: 5,312,768 parameters.
- **`tests/owl/model/test_model_config_files.py`:** loads the new preset through its own config class until shared registration (Task 2.3).

## Checks

- `tests/kaggriculture` has 39 passing cases:
  - contract types and boundaries
  - topology against Isaiah's 6m classes and shared fields
  - initialization
  - named offsets and pre-trunk player addition
  - masking of absent actors and shops
  - statelessness and seat isolation
  - chunk boundaries, chunked vs unchunked equality, compiled dispatch per chunk, forced-flash failure before the trunk
  - Muon groups
- `just py-prepare`: 762 passed, 3 hardware skips; Ruff, mypy (50 files) and docs freshness pass.
- Everything ran on CPU with tiny shapes. Real compilation and CUDA flash execution are not qualified here; they're GPU qualification (Phase 6).

## Limits

Task 2.2 adds Isaiah's critic: an `OutputProjectionMLP` over the two critic-value tokens, a winner softmax, value `2p(self) − 1`, output gain 1.0 excluded from Muon (5 more tests, 44 total). The action heads (2.3), shared factory registration and trainer integration are open. The full-model 6–10M budget is checked in 2.3.
