You are Codex, reviewing Claude's draft for rebuild plan Task 0.1. This is READ-ONLY: do not modify any file.

Review `docs/kaggriculture-contract.md` (draft v1) in this checkout, against:
- `ops/rebuild-2026-09-29/plan.md` (Task 0.1, components C1–C9, lessons L1–L13, principle I0b)
- the reference implementation, read with `git show kg/reference-2026-09-29:<path>`:
  - `engine_rs/src/myolie_features.rs` (the flat feature encoder)
  - `engine_rs/src/lib.rs` (Game/Farm/Private/Market/Town/Config state, and the legal-visibility rules)
  - `engine_rs/src/myolie_sampler.rs` (grammar slots and widths; what `unit_target` width 128 indexes)
  - `engine_rs/src/training.rs`
  - `python/owl/kaggriculture/{types,env,rewards,actor_codec}.py`
  - `docs/coordination-system-design.md` (the observation-schema corrections)

Check specifically:
1. Legality: is any field private to the rival, or hidden engine state (RNG, unrevealed shop results, hidden config)? Is any legal public or own-private fact missing?
2. The information audit table: is every reference feature range correctly mapped, or rightly dropped?
3. Correctness of every field's definition against the engine structs (names, units, ranges), and the proposed normalizations (flag the ones that could exceed [0, 4]).
4. The action contract: slot widths and semantics against `myolie_sampler.rs`. Resolve open point 2 (what `unit_target` indexes).
5. The env contract: seed streams, auto-reset, terminal metrics, rewards; anything that conflicts with how `training.rs` works.
6. Anything that would stop the model from following Isaiah's `StatelessTransformerV1` topology (one-hot categorical fields into `ObservationInputStem`).

FINAL REPORT: numbered findings, each with severity (blocker / should-fix / note), the exact contract section, reference evidence (file:line), and a concrete proposed edit. End with your answers to the four "Open points for review".
