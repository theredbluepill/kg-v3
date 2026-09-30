# Independent verification of Tasks 3.2–3.3, round 2

Reviewed branch `kg/rebuild-semantics`, HEAD `530b8ccf8c23a3f26c96fd633e08bb2ea151d696`, using exactly `git diff e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD`. The merge base is the supplied base. The range contains `762e516` and `530b8cc`.

No outstanding correctness findings. Both findings in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-3.2-3.3-r1.md` are resolved. No further implementation edit is requested. This approves the documented trainer seams, not completed native Kaggriculture integration.

| Prior finding | Current location and fix | Independent verification |
| --- | --- | --- |
| P2: premature promotion success telemetry | `scripts/run_ppo.py:490`: the single evaluation log follows incumbent refresh, teacher update, promoted checkpoint write and barrier. | Both refresh and promoted-write fault injections leave no `eval/promoted` record. Moving the log back before promotion makes both current regression cases fail. |
| P3: overclaimed seed-stream separation | `scripts/run_ppo.py:1441` and seed-band comment, `README.md:365`, `docs/rl-api-specs.md:869`, and the cookbook now distinguish fixed-input bijections from pair collisions/range overlap, and condition training separation on a future enforced bound. | Reproduced the exact pair collision and adjacent-start counterexamples; the characterization test at `tests/scripts/test_run_ppo.py:2937` pins them. The unchanged function satisfies the literal fixed-run seed requirement. |

The review read `ops/rebuild-2026-09-29/plan.md` Tasks 3.2/3.3 and lessons L1/L2/L12, accepted contract v4 in `docs/kaggriculture-contract.md`, the governing cookbook notes, Isaiah's code/config at `32b3ec900ad406eedd965f53a1a0f4490d31c589`, and the reference implementation at `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Two read-only subagents independently inspected production semantics and prior-finding/documentation resolution; neither found an additional defect.

| Requirement | Result |
| --- | --- |
| L1: raw-bank outcomes | Terminal metrics supply float64 seat banks; their required margin and optional winner are checked. Raw banks, including ties, decide evaluation. Candidate-relative bank metrics use the actual seat assignment. The literal reference `_evaluation_outcome` matches on eight win/loss/draw/precision-sensitive seat cases. |
| L2: truncation reward and bootstrap | Kaggriculture retains the economic reward on the cut transition. The cut helper marks done/truncated and stores the pre-reset critic value. The named test checks nonzero reward plus nonuniform bootstrap through GAE. `_apply_truncation` evaluates before resetting. |
| Joint clipping | The named 40-frame test discriminates joint `exp(0.4)` clipping from per-frame clipping; the config rejects `per_entity`. |
| Value guards | Kaggriculture requires `win_loss`, gamma 1, MSE, and `per_player`, matching contract v4 and Isaiah's scaling recipe. The config tests exercise both acceptance and setting-specific failures, including the validator dispatch. |
| L12: evaluation seed | Reproducible and bijective in either input with the other fixed, over the checked domain; output is in `[2**62, 2**62 + 2**61)`. Native `Game::new` accepts i64. The seed function has no native production caller yet. |
| Evaluation count and promotion | Retains Isaiah's `cfg.env.n_envs` evaluation games and threshold 0.7. Tests distinguish 0.7 promotion from 0.69 rejection, pass evaluation steps through the loop, and verify the three requested telemetry fields. |
| Orbit compatibility | Constructor arguments, accumulated-return scoring, reward-zeroing truncation and pre-reset bootstrap retain Isaiah's behavior. The full retained Python suite and Rust fixture tests pass. New promotion telemetry and its later logging are deliberate changes. |
| No shim wrappers | Existing PPO/evaluation loops are extended directly. Extracted helpers implement the factored semantics; no legacy compatibility wrapper, second trainer, v2 model, or ctypes path is introduced. |
| Cookbook accuracy | The note, references index and prepended log agree on the repairs and current counts. They accurately describe fake-env/pure-helper evidence and missing native/config/mask/action/replay integration. Historical round-one reports remain historical evidence. |

Executed checks on this HEAD:

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`: **1,344 passed, 6 skipped**, 26.68 seconds, exit 0 (`pytest.log`). Includes Isaiah's retained Orbit tests.
- `uv run mypy python/owl scripts`: **58 source files, no issues**, exit 0 (`mypy.log`).
- `OMP_NUM_THREADS=2 UV_OFFLINE=1 uvx --offline --from rust-just just py-prepare`: **passed**; 104 files unchanged by formatting, lint and syntax checks passed, mypy over 59 files passed, **1,344 passed / 6 skipped**, 24.75 seconds (`py-prepare.log`). Docs freshness reported no updates required against the clean tracked worktree; the branch-level docs were separately inspected.
- Initial `cargo test`: 148 passed, 7 failed, 2 ignored, solely because this worktree lacked its ignored Orbit parity fixtures (`cargo-test.log`). Temporary byte-exact copies of the README's generation fixture and episodes 75926553/75930761 from `/Users/poonszesen/kg-v3/tests/fixtures` resolved this. Rerun: **155 passed, 0 failed, 2 ignored**, exit 0 (`cargo-test-with-fixtures.log`). Source paths, sizes and SHA-256 values are in `orbit-fixtures.json`. Temporary copies were removed; parity was not disabled.
- `uv run python ops/rebuild-2026-09-29/codex/verify-3.2-3.3-r2/probe.py`: eight exact reference-oracle matches, Orbit tensor identity preserved, seed counterexamples reproduced, and both injected promotion failures leave no success telemetry (`probe.json`, `probe.log`).
- Whole-range `git diff --check` flags trailing whitespace only in two retained raw pytest failure logs (`diff-check.log`). The same check excluding `ops/**` passes. These captured diagnostic-log spaces are not an implementation finding.

Mutation checks used byte-exact scratch copies of current `scripts/run_ppo.py` and `tests/scripts/test_run_ppo.py`, with unchanged test contents:

| State | Result |
| --- | --- |
| Baseline | 3 passed, 90 deselected |
| Move promotion log before promotion | 2 expected failures (refresh and checkpoint), 1 passed |
| Decide Kaggriculture winner by shaped return | 1 expected L1 failure, 2 passed |
| Restore source | 3 passed, 90 deselected |

Restoration was checked after each mutation against saved bytes and tracked source. Before/restored SHA-256: `356a05ba486e5256ef337ce0dd8766e71baf4a118d9d6c6d9b9050cd0ed8d138`. Mutation code/results are in `mutate.py`, `mutation.json`, and `mutation-*.log`. Scratch copies were removed.

Scope limits remain explicit. Native Kaggriculture environment/config registration, actual seed consumption, action/mask integration, evaluation replay plumbing and native truncation-buffer preservation await Tasks 1.4/1.5 and later integration. The six Python skips are native grammar binding, config integration, native evaluation integration, two CUDA FlashAttention tests, and the unavailable x86 quantization backend. No training or GPU qualification ran. The seed's int64 fit is source/arithmetic evidence, not a native trajectory check.

Final custody: SHA-256 comparison of **all 780 tracked files** found no change; `git diff --exit-code HEAD --` passed. Only new untracked verification artifacts remain in this directory. No cookbook or implementation adaptation was made.

VERDICT: APPROVE
