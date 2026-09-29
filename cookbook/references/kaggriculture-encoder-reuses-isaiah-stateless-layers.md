---
type: "Reference"
title: "Kaggriculture encoder reuses Isaiah's stateless layers"
description: "The rebuilt encoder uses Isaiah's StatelessTransformerV1 classes, token roles, packing, compile hook and initialization; only stem input widths are game-specific, and guards chunk both the padded and packed compiled GEMMs below 2^31."
tags: ["kaggriculture-v3", "model", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/kaggriculture/types.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:python/owl/train/optimizer.py"}, {"resource": "repository:configs/model/kaggriculture.yaml"}, {"resource": "repository:tests/kaggriculture/test_model_encoder.py"}, {"resource": "repository:tests/kaggriculture/test_types.py"}, {"resource": "repository:tests/owl/model/test_model_config_files.py"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/2.1-encoder.md"}, {"resource": "repository:ops/v3-architecture-image-2026-09-29/README.md"}, {"resource": "repository:ops/v3-architecture-image-2026-09-29/prompt.txt"}, {"resource": "repository:ops/v3-architecture-image-2026-09-29/v3-agent-architecture.png"}]
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
  - the critic (Task 2.2) and the grammar action heads (Task 2.3, [[kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|heads Reference]]) share this encode
- **Compiled-GEMM guard** (from the [[compiled-gemm-template-overflows-above-2-21-rows|overflow Reference]]): `trunk_gemm_width` is the max of `max(in_features, out_features)` over every trunk `nn.Linear` (`max(D, D·mlp_ratio)` today, checked by enumeration). The padded path chunks complete rows below 2^31. Since Task 2.3, the packed path also splits rows at row boundaries (`packed_row_chunks`) and raises only when a single row cannot fit; it previously raised. At the preset, 5,915 padded rows are allowed per forward, against 256 per rollout forward.
- **`configs/model/kaggriculture.yaml`:** Isaiah's 6m ladder point with depth 8. Encoder: 5,312,768 parameters.
- **`tests/owl/model/test_model_config_files.py`:** loads the new preset; since Task 3.1 through the shared loader, like Isaiah's presets.

## Checks

- `tests/kaggriculture` has 155 passing cases. After Codex's verification (`ops/rebuild-2026-09-29/codex/verify-stream-b-2.1.md`, findings 1–7), each claimed gap got a test that was seen failing against the old `check_contract` or a deliberate one-line mutation of the model or base: zeroed or shifted readouts, reused chunk masks, `>` for `>=` at the packed limit, no `max_seqlen`, the eager trunk in the packed path, a changed PEP 696 default, residual gain, output-layer and input-layer lists, and `hidden_state` handling. The compile-key and SiLU checks were not mutation-tested.
  - contract types: lead shapes `(2,)` and `(1,1,2)` rejected; negative counts, ranks, globals and order limits rejected; −1 tile sentinels and signed `market_int` accepted; dtype, shape and missing field (pydantic `ValidationError`) for every field
  - `conftest.make_obs` builds one game per env and writes each seat's legal view: zero-filled absent actors and shops, no rival-private actor or player channels, the applicability rules for tiles, and `can_act = frame < own + order_limits + 1`; a test asserts these
  - topology against Isaiah's 6m classes; initialization (residual, hidden √2 and input gains by singular values, token std ≈ `D^-0.5`, LayerNorm 1/0)
  - every token region and named readout at its offset, using marker stems and an identity trunk, with `still_playing = False` zeroing the player, plan and critic tokens
  - padded chunks dispatch exact row and mask slices in order, below the strict bound, with distinct per-row masks
  - CPU-mocked packed dispatch: one pack with `max_seqlen` = sequence length and one unpack, packed rows = `token_mask.sum()`, and the compiled callable. Since Task 2.3 the overflow boundary splits instead of rejecting: one pack below it, per-row chunks at and above it, and distinct, ordered chunk inputs whose result equals the unchunked one
  - `compile_transformer_trunk` keeps the state-dict keys; SiLU builds and encodes
  - every API rejects `hidden_state`; since Task 2.3, `evaluate_actions` also rejects `dones`
  - Muon and AdamW membership (stem inputs, tokens, biases, norms and `critic_head.out` in AdamW)
  - `test_base_generics_typing.py` runs mypy `--strict` with the project config and a fresh cache on in-test probes: bare `BaseModelAPI` is Orbit, Kaggriculture outputs are `ModelOutput[KaggricultureActions]`, token counting type-checks, and exactly 11 marked mixed-game lines are errors
- `just py-prepare`: 878 passed, 3 hardware skips; Ruff, mypy and docs freshness pass.
- Everything ran on CPU with tiny shapes. Real compilation and CUDA flash execution are not qualified here; they're GPU qualification (Phase 6).

## Limits

Task 2.2 adds Isaiah's critic: an `OutputProjectionMLP` over the two critic-value tokens, a winner softmax (masked as Isaiah's since Task 3.1), value `2p(self) − 1`, output gain 1.0 excluded from Muon. Task 2.3's action heads and the full-model budget (6,252,223 parameters) are in the [[kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|heads Reference]]. Shared factory registration, the trunk compile dispatch and the masked critic are in the [[kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic|Task 3.1 model-side Reference]]; trainer integration is open. `check_contract` checks dtypes, shapes and bounds, not the semantic zero-fill and privacy rules; those are asserted only for the synthetic fixture, and the native writer (Task 1.x) must be tested against them separately.

## Architecture illustration

The owner-requested [architecture image](../../ops/v3-architecture-image-2026-09-29/v3-agent-architecture.png) depicts `kg/rebuild-model` at `8093d51`: the observation schema, encoder and critic are solid teal; the actor heads and target Kaggriculture training pipeline are dashed amber. The [artifact receipt](../../ops/v3-architecture-image-2026-09-29/README.md) inventories the PNG and exact ImageGen prompt, source checks, visual inspection and simplifications. This is an explanatory artifact, not new implementation or runtime evidence. The image is pinned to `8093d51`; the current tree has since merged the Task 2.3 actor heads and the Task 3.1 factory registration, which the image still draws as dashed amber. Regenerate its status before reusing it as a current-status artifact; the snapshot says nothing about work that landed after `8093d51`.
