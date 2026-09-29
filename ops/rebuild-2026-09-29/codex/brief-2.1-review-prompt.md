You are Codex, reviewing Claude's brief for rebuild plan Task 2.1. READ-ONLY: do not modify files.

Brief: `ops/rebuild-2026-09-29/briefs/2.1-encoder.md`. Also read `docs/kaggriculture-contract.md` (v3; your re-review of it is in progress separately), `ops/rebuild-2026-09-29/plan.md` (I0/I0b, L6, Task 2.1), and Isaiah's code on this branch: `python/owl/model/base.py`, `python/owl/model/stateless_transformer_v1.py` (`encode_observations`, `TransformerBlock`, `ObservationInputStem`, the token parameters, the flash-packing and compile hooks, `get_input_layers` / `get_output_layers`), `python/owl/model/config.py`, `python/owl/model/factory.py`, `python/owl/train/optimizer.py` (`_excluded_from_muon_param_ids`). The compiler-overflow finding is `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md`.

Answer the brief's four review questions, and additionally check:
- that each stem input width is correct for the contract v3 fields (count the channels and the one-hot sizes)
- that the proposed token order and masks match Isaiah's `encode_observations` pattern (including where the player features are added)
- whether `BaseModelAPI[ObsT, ActT]` can be introduced without breaking Isaiah's mypy-strict code and tests; name the files it touches
- whether each test is sufficient and runs cheaply on a Mac (no large allocations)
- whether anything in the brief would copy the reference model instead of rebuilding it on Isaiah's classes

FINAL REPORT: numbered findings (blocker / should-fix / note), each with a concrete brief edit. End with a verdict: APPROVE, APPROVE WITH EDITS, or REVISE.
