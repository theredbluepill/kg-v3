---
type: "Reference"
title: "Kaggriculture grammar heads sit behind Isaiah's actor projection"
description: "Task 2.3 adds Isaiah's 3D→D actor input projection and batched grammar heads with exact coupled-Gumbel HIRE sampling, same-path replay with support/length/canonical flags and one host check, head GEMM-extent chunking, and a 6,252,223-parameter preset; synthetic grammar tables stand in until the Task 1.4 binding exposes the Task 1.2 native tables (which exist and match the stand-in bit-for-bit)."
tags: ["kaggriculture-v3", "model", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:python/owl/model/kaggriculture_actor.py"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/kaggriculture/gpu_grammar.py"}, {"resource": "repository:tests/kaggriculture/test_model_heads.py"}, {"resource": "repository:tests/kaggriculture/test_gpu_grammar.py"}, {"resource": "repository:tests/kaggriculture/test_model_encoder.py"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/2.3-action-heads.md"}, {"resource": "repository:ops/rebuild-2026-09-29/codex/brief-2.3-r3.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/gpu_sampling_grammar.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_sampler.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_model.py"}]
---

# Kaggriculture grammar heads sit behind Isaiah's actor projection

## What changed (rebuild Task 2.3, brief v3)

- **Topology.** `actor_input_proj = nn.Linear(3D, D)` over `[entity ‖ self player ‖ plan]`, as Isaiah's `source_actor_input_proj`. The market queue has no entity, so it uses `[plan ‖ player ‖ plan]`. `KaggricultureGrammarActor` (`python/owl/model/kaggriculture_actor.py`) holds everything game-specific:
  - `source_norm`
  - `market_position` (11 = 10 queue slots plus the forced sentinel)
  - seven prefix `slot_embeddings`
  - nine `OutputProjectionMLP` heads, one per sampled slot

  Stage input is `base + prefix / sqrt(stage + 1)`, and the prefix resets for every observation. The reference model was a design and test oracle only: it had a `prefix_norm` and a factorized `frame_input`, and neither was taken.
- **Tables.** `GrammarTables` (`python/owl/kaggriculture/gpu_grammar.py`) holds the eight Task 1.2 tables, with validation.
  - `expected_grammar_tables()` re-derives them from the brief's support rules.
  - `native_grammar_tables()` is the named hook for the Task 1.4 binding of the Task 1.2 tables. It raises `NotImplementedError` until then.
  - The model defaults to the expected tables and keeps them as non-persistent buffers, so they never enter checkpoints.
- **Sampling.** Exact Gumbel-max throughout. `couple_market_kinds` implements the HIRE correction: the raw-HIRE prefix masks HIRE where capacity is exhausted, and the argmax is re-taken over the same perturbed scores. Densities use the final-HIRE prefix. STOP is the first final NONE, which keeps its slot-7 density; the sentinel has zero density.
- **Replay.** `evaluate_actions` runs the same core, teacher-forced.
  - Python shape and dtype checks run before any kernel.
  - Every replay-derived index is clamped to a temporary value, and its validity goes into the flags.
  - Three flag groups: support, length and canonical equality.
  - `check_replay_flags` makes one compact host transfer and raises `GrammarReplayError`, naming the groups and the first failing row.
  - Sampling adds no policy-validation host synchronization: `forward` never runs the replay check. The encode is not sync-free: packed trunk dispatch sizes its packing on the host (Isaiah's `build_packed_sequence`), and an oversized packed batch transfers per-row token counts to plan chunks. (Corrected after Codex's Task 2.3 verification, `ops/rebuild-2026-09-29/codex/verify-2.3-r1.md`.)
- **Outputs.** `event [E,2,252,12]`, `per_player_entity = event.sum(-1)`, zero `launch`, per-slot entropy `components`, and critic values from the same encode.
- **Initialization and optimizer.** The embedding `.weight`s are in `get_input_layers` (std `D^-0.5`). Every `head.out` is in `get_output_layers` with gain 0.01; `reset_parameters` now follows Isaiah's output-layer loop. `actor_input_proj` and every head `.up` go to Muon.
- **Overflow guards (brief §9):**
  - `trunk_gemm_width`: the max of `max(in, out)` over every trunk Linear, enumerated by a test.
  - The packed trunk path chunks at row boundaries (`packed_row_chunks`) instead of raising.
  - `head_rows_per_chunk` keeps `rows × 252 × max(3D, D, widest head) < 2^31` (11,096 rows at D = 256).
  - The heads stay eager in production.
- **Size.** 6,252,223 parameters at the preset: encoder 5,312,768, critic 66,049, heads 873,406. That is inside the owner's 6–10M budget, so the depth stays 8.

## Checks

- `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture -q`: 256 passed, 1 skipped (the native-table comparison, which waits for the Task 1.4 binding).
- `just py-prepare`: Ruff and mypy clean, 979 passed with 4 skips. Docs freshness passes with the mapped docs updated.
- Brief §7 items 1–10 have tests:
  1. saved-sample replay (whole, split, permuted) for the dense, HIRE-capacity, STOP-at-every-position and mixed/inactive cases, to ≤ 1e-5
  2. every table row, plus the quantity cases, through both the tables and replay
  3. Gumbel chi-square over 20k draws
  4. enumeration of the actual `couple_market_kinds` against the replayed densities at budgets 0/1/2/3/10 (FP64, 1e-9, NONE and EMPTY corrections)
  5. STOP and marginalization
  6. every malformed group, including out-of-range indices with no fault
  7. an `eager`/`fullgraph=True` captured core, plus an FP64 finite difference
  8. seat isolation and statelessness
  9. topology, initialization and Muon membership
  10. the budget

  Item 11 is skipped with its reason.
- Deliberate mutations, each restored byte-for-byte (SHA-256 checked):
  - a batch-mean term in the market input fails the replay invariant (4 cases)
  - an inclusive raw-HIRE prefix fails the enumeration (3 budgets)
  - dropping the actor-ordinal slot from canonical equality fails 2 canonical cases
  - dropping the head outputs from `get_output_layers` fails the Muon and initialization tests
- Codex's independent verification of `0e989a1..5ec3af1` (`ops/rebuild-2026-09-29/codex/verify-2.3-heads/report.md`) found no functional defect and approved with one wording edit, applied above. It reran `tests/kaggriculture` (256 passed, 1 skipped) and the full non-slow suite (979 passed, 4 skipped), and three further mutations (raw replay index, partial canonical equality, inclusive raw-HIRE prefix) each failed their tests. Market-position noise independence inside `policy_core` is established by source inspection, not by a dedicated test.
- Everything ran on CPU at tiny shapes. BF16, CUDA and a real compile are not qualified here; that is Phase 6.

## Limits and reopening conditions

- The masks come from synthetic tables until the Task 1.4 binding exposes `kaggriculture_grammar_tables`. Then implement `native_grammar_tables`, switch the model default, and let `test_native_tables_match_expected_tables` run. Brief item 11 (recorded reference programs) also waits for the Task 1.4 binding and model integration of the Task 1.2 native tables.
- The actor count is the own `actor_mask` sum, which assumes own actors fill slots `0..n-1` in frame order, as the contract and fixture do. `can_act` is not read by the heads.
- The prefix embeddings (std `D^-0.5`) are added to a LayerNorm-scaled base with no second norm, as the brief specifies. At initialization the prefix signal is about `1/sqrt(D)` of the base. Whether this conditions strongly enough is untested; reopen if per-slot KL or BC accuracy shows weak item/quantity conditioning on kind.
- The unit input materializes `[rows, 241, 3D]`, which is about 0.8 GB FP32 at 1,024 rows. The reference's factorized projection is mathematically equivalent if memory binds (Phase 6).
- Trainer integration, factory registration (Task 3.1) and the startup workload assertion (Task 3.4) remain open.
