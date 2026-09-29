# Independent verification: Phase 4 teacher, round 1

Reviewed branch `kg/rebuild-trainer-model` at `b94b8dc2f554286e3814099318f7a51123b27f8e`, using exactly `git diff b1da613a197a334547563111c1e11453f84551b8...HEAD`. The merge base equals the requested base. Compared the diff with `ops/rebuild-2026-09-29/briefs/4-teacher.md` v2, accepted contract v4 (`docs/kaggriculture-contract.md`), governing cookbook notes, and Isaiah's source at `32b3ec900ad406eedd965f53a1a0f4490d31c589`. Two parallel read-only audits covered math/guards and contract/records; the primary verifier ran all checks and mutations below.

This review covers the implemented 4.1–4.3 changes. The brief explicitly permits those subtask reviews while T18/T19b and 4.4 remain open. This is not acceptance of completed Phase 4 or integrated Kaggriculture training.

Findings:

1. **P2 — `tests/kaggriculture/test_teacher.py:1458`: T19b is an unimplemented placeholder.** The skipped function consists of a docstring and an unconditional `AssertionError`. Removing its skip after the configs merge cannot exercise resume or fresh launch. Brief §0 and §6 require dependency-blocked tests to be written and skipped. **Fix:** implement the launch/resume assertions behind the existing named dependency skip: restoration from `checkpoint_last_best.pt`, teacher activation from initial weights, and no teacher cache in checkpoints. Until implemented, explicitly describe T19b as unwritten in the cookbook rather than merely awaiting execution. The existing refresh test covers in-place weight/table preservation, but does not replace launch/resume coverage.
2. **P3 — `docs/model-architecture.md:737`: the game-generic teacher section still says cached action KL supports only `discrete_targets`.** Kaggriculture now implements cached grammar KL. **Fix:** scope this restriction and the associated launch-mode rule to Orbit, as README already does; distinguish the Orbit actor method from Kaggriculture's policy-core KL in the surrounding explanation.

No functional defect was found in the implemented KL, value CE, cache or teacher dispatch:

- Isaiah's direction is `KL(teacher || student)` over replay-conditioned distributions. The grammar implementation retains that estimator, including the effective masks, exclusive final-HIRE prefix, STOP handling and liveness weights. It does not claim an unbiased joint-program KL.
- The KL helper preserves Isaiah's FP16/BF16/FP32 calculation behavior and retains FP64 for the finite-difference oracle. Masked cached logits use finite dtype minima.
- BaseModelAPI owns Isaiah's unchanged joint winner CE formula; Kaggriculture overrides it with the specified live-seat mean. The old free function is removed. Existing PPO dispatch helpers are amended for stateless calls; no compatibility shim or second trainer was introduced.
- Cached and combined paths encode through `encode_observations -> _run_trunk`, and their actor passes use `_policy -> _policy_chunk`. Teacher logits are sliced with each head chunk, and per-slot logits/KL are concatenated. Existing padded/packed guards remain ahead of trunk dispatch. T4 observes actual head and trunk chunk calls and compares outputs with the documented tolerance.
- Contract arithmetic is `4 * (241 * (20+16+2+32+32) + 11 * (8+16+32+32) + 2) = 102,208` bytes per FP32 seat row. Totals are 1,674,575,872 bytes for 16,384 rows and 837,287,936 bytes for 8,192 rows. Runtime `nbytes()` counts actual tensor metadata, including winner probabilities. GPU peak/reserved memory is not established by this arithmetic.
- Cache indexing follows segment dimension 0; concat validates optional presence, keys and grammar symmetrically, with a no-copy single-chunk path. The cached student rejects a grammar mismatch before encoding. The documented construction-time digest limitation remains: subsequent in-place table edits do not update that digest.
- The support-enumerating KL oracle checks probability mass, gradients are nonzero across student heads and absent on the teacher, the FP64 test compares finite differences, and cached/combined outputs are exactly compared on CPU FP32. These checks are substantive.
- Cookbook inventory, retained mutation counts, historical suite totals and CPU/synthetic-grammar limits agree with the evidence inspected. The records explicitly state that Phase 4 is incomplete. T19b's placeholder status needs the clarification above. T18 contains substantive deferred assertions; integrated rollout/trainer and launch/resume qualification have not run.

Commands and results (CPU, `OMP_NUM_THREADS=2`):

| Command | Result | Evidence |
|---|---|---|
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,372 passed, 6 skipped, 30.12 s | `pytest.log` |
| `uv run mypy python/owl scripts` | Success, 58 source files | `mypy.log` |
| `uv run pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,048 passed, 3 skipped, 13.76 s | `orbit-pytest.log` |
| `uvx --from rust-just just py-prepare` | Pass: formatting, lint, typing, tests and docs freshness; 1,372 passed, 6 skipped, tests 29.22 s | `py-prepare.log` |

The six skips are two unavailable CUDA flash-attention tests, one unavailable x86 quantized backend test, the unavailable native grammar binding, T18's trainer/value seams, and T19b's configs dependency. The separate mypy command checks `python/owl`; py-prepare checks the broader `python/` tree.

Three controlled mutations were applied sequentially to the reviewed files, tested, and restored in `finally` blocks. Each test process returned exit code 1 for the expected assertion failure, not an import or syntax error:

| Mutation | Test result |
|---|---|
| Restore FP32 casts in the categorical KL helper | FP64 finite-difference test fails: 1 failed |
| Sum value CEs across seats instead of averaging live seats | Hand reduction and copied-teacher entropy tests fail: 2 failed |
| Remove cached grammar-signature check | Replay-admissible grammar mismatch no longer raises: 1 failed |

`mutations.json` records source hashes before and after each mutation, all equal. The corresponding `mutation-*.log` files retain the failures. The full green py-prepare run followed restoration. `tracked-before.json` and `tracked-after-check.json` record whole-tree tracked-file custody. No tracked modification, staging or commit was made. Review evidence is untracked in this directory; pre-existing untracked `verify-3.1-rest-r3/` was preserved.

No GPU, real Inductor/CUDA qualification, integrated Kaggriculture training run, or Rust check was performed. These are limits, not results inferred from the CPU tests. Phase 4.4 and execution of T18/T19b remain required for full Phase 4 completion.

VERDICT: APPROVE WITH EDITS
