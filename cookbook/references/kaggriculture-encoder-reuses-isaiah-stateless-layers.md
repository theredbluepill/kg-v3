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

- **`python/owl/kaggriculture/types.py`:** contract v4 in code. It defines the 29-field `KaggricultureObsBatch`, the pinned observation and action enums, and the size constants. `check_contract()` requires leading dims exactly `[E, 2]` and applies per-field bounds: categories in `[0, n)`, exact counts and `globals_int` ≥ 0, ranks in `[0, 12]`, `order_limits` in `[0, MAX_ORDER_LIMIT = 10]`, `tiles_int` ≥ −1 (the sentinels); `market_int` and the float channels are signed and only need to be finite.
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

- `tests/kaggriculture` has 155 passing cases. After Codex's verification (`ops/rebuild-2026-09-29/codex/verify-stream-b-2.1.md`, findings 1–7), each claimed gap got a test that was seen failing against the old `check_contract` or a deliberate one-line mutation of the model or base: zeroed or shifted readouts, reused chunk masks, `>` for `>=` at the packed limit, no `max_seqlen`, the eager trunk in the packed path, a changed PEP 696 default, residual gain, output-layer and input-layer lists, and `hidden_state` handling. The compile-key and SiLU checks were not mutation-tested.
  - contract types: lead shapes `(2,)` and `(1,1,2)` rejected; negative counts, ranks, globals and order limits rejected; −1 tile sentinels and signed `market_int` accepted; dtype, shape and missing field (pydantic `ValidationError`) for every field
  - `conftest.make_obs` builds one game per env and writes each seat's legal view: zero-filled absent actors and shops, no rival-private actor or player channels, the applicability rules for tiles, and `can_act = frame < own + order_limits + 1`; a test asserts these
  - topology against Isaiah's 6m classes; initialization (residual, hidden √2 and input gains by singular values, token std ≈ `D^-0.5`, LayerNorm 1/0)
  - every token region and named readout at its offset, using marker stems and an identity trunk, with `still_playing = False` zeroing the player, plan and critic tokens
  - padded chunks dispatch exact row and mask slices in order, below the strict bound, with distinct per-row masks
  - CPU-mocked packed dispatch: one pack with `max_seqlen` = sequence length and one unpack, packed rows = `token_mask.sum()`, the overflow boundary (safe below, rejected at and above, before the trunk), and the compiled callable
  - `compile_transformer_trunk` keeps the state-dict keys; SiLU builds and encodes
  - every API rejects `hidden_state`; forward, serve and `evaluate_actions` raise `NotImplementedError` until Task 2.3
  - Muon and AdamW membership (stem inputs, tokens, biases, norms and `critic_head.out` in AdamW)
  - `test_base_generics_typing.py` runs mypy `--strict` with the project config and a fresh cache on in-test probes: bare `BaseModelAPI` is Orbit, Kaggriculture outputs are `ModelOutput[KaggricultureActions]`, token counting type-checks, and exactly 11 marked mixed-game lines are errors
- `just py-prepare`: 878 passed, 3 hardware skips; Ruff, mypy and docs freshness pass.
- Everything ran on CPU with tiny shapes. Real compilation and CUDA flash execution are not qualified here; they're GPU qualification (Phase 6).

## Limits

Task 2.2 adds Isaiah's critic: an `OutputProjectionMLP` over the two critic-value tokens, a winner softmax, value `2p(self) − 1`, output gain 1.0 excluded from Muon. The action heads (2.3), shared factory registration and trainer integration are open. The full-model 6–10M budget is checked in 2.3. `check_contract` checks dtypes, shapes and bounds, not the semantic zero-fill and privacy rules; those are asserted only for the synthetic fixture, and the native writer (Task 1.x) must be tested against them separately.
