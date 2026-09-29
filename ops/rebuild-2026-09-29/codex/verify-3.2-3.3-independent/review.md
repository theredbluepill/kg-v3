# Independent verification — Tasks 3.2 and 3.3

Date: 2026-09-29. Branch: `kg/rebuild-semantics`.

Reviewed the requested three-dot diff:
`e1458d2a717d9d731a367cbb78b98616ee6649f4...762e516e690f51a7556fe2ef69d5008c56465e62`.
The merge base is exactly the requested base. The worktree started clean.
Reference oracle: `kg/reference-2026-09-29` at
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`.
Isaiah source: `32b3ec900ad406eedd965f53a1a0f4490d31c589`.

Scope: compare the changed trainer/evaluation seams with plan Tasks 3.2/3.3,
lessons L1/L2/L12, accepted contract v4, the reference semantics and Isaiah's
Orbit behavior. Run the requested CPU suites, distinguish correct behavior
with a deliberate mutation, and stop after recording findings and verifying
tracked files unchanged. No training, native Kaggriculture evaluation, GPU run,
Rust edit or Rust test was performed.

## Findings

### P2 — Report completed promotion after its lifecycle succeeds

Location: `scripts/run_ppo.py:468` (especially `eval/promoted` at line 471).

The new log emits `eval/promoted=1` before refreshing the incumbent, updating
the teacher, writing `checkpoint_last_best.pt` and completing the barrier.
README line 359 defines this as a snapshot that was replaced. An exception in
any of those operations can leave a false success record. The reference branch
logs promotion after these operations and the barrier.

Independent fault injection into the current training loop confirms both cases:
an incumbent-refresh exception and a promoted-checkpoint write exception each
leave `eval/promoted=1`, with no completed promoted checkpoint. See
`probe.json` / `probe.log`. Existing new telemetry tests cover successful
promotion and rejection at 0.7/0.69, but not failure ordering.

Fix: log the completed promotion after refresh, teacher update, checkpoint save
and barrier succeed, as the reference does. If evaluation results must be
logged earlier, keep a separate decision metric and reserve `eval/promoted`
for completion. Add a failure-ordering regression test.

### P3 — Narrow the seed-stream guarantees to those actually established

Locations: `scripts/run_ppo.py:1442`; related seed-band assertions at lines
74–76, `docs/rl-api-specs.md:869`, and
`cookbook/references/evaluation-and-truncation-follow-the-kaggriculture-objective.md:62`.

The mix is bijective in either input while the other is fixed. It is not
injective over input pairs, and distinct starting seeds do not guarantee
disjoint ranges of consecutive game seeds. The docstring's guarantee that
different evaluations or runs do not share such ranges is false. Exact-source
counterexamples, all inside the accepted input range:

| base_seed | env_steps | returned seed |
| --- | --- | --- |
| 0 | 0 | 4611686018427387904 |
| 1 | 2131737497183550101 | 4611686018427387904 |
| 0 | 787325655728545358 | 4611686018427387905 |

The first two collide across runs. The first and third yield adjacent starts
within one fixed-base stream, allowing overlap when multiple consecutive seeds
are consumed. Also, the contract currently requires only nonnegative training
seeds; it does not enforce the claimed training-stream ceiling below `2**62`.
These counterexamples concern the unconditional guarantee, not an observed
collision at normal training cadences.

Fix: retain the proved claims—reproducibility, distinct starts for distinct
evaluation steps with a fixed base, and the int64 output band/headroom. Remove
global range-nonoverlap claims and make training/evaluation separation
conditional on an enforced training-seed bound, or implement that stronger
contract in the native integration. Reconcile the docstring, RL API docs and
cookbook together. The seed function meets Task 3.3's literal fixed-run seed
requirement; this finding does not require replacing its algorithm.

## Semantic verification

| Requirement | Finding |
| --- | --- |
| L1 raw-bank outcome | Correct on the shared evaluation seam. Uses terminal float64 seat banks, independent of shaped returns; equal banks draw. Candidate-relative bank metrics respect seat assignments. |
| Reference oracle | Executed the literal reference `_evaluation_outcome` extracted from git on win, loss, draw and float64-sensitive cases, both candidate-seat assignments: 8 exact matches. Orbit returns preserve tensor identity in both versions. |
| L2 truncation | The cut helper keeps Kaggriculture economic reward, marks done/truncated and supplies the critic bootstrap. Value calculation precedes reset. Named test checks nonzero reward plus nonuniform bootstrap through GAE. |
| Orbit truncation | Isaiah's reward-zeroing and reset order remain. New rollout test exercises the Orbit trainer end to end, while existing Orbit tests pass. |
| Joint clipping | Named 40-frame test distinguishes joint ratio `exp(0.4)` clipping from per-frame ratios. The config guard rejects `per_entity`. |
| Value guards | `win_loss`, gamma 1, MSE and joint `per_player` are required for Kaggriculture. Matches contract v4 and the current fixed critic/Isaiah recipe. Orbit constraints remain separate. |
| L12 seed | Reproducible and distinct for different steps at a fixed base; output lies in `[2**62, 2**62 + 2**61)`. Native `Game::new` takes i64. Stream-nonoverlap wording needs correction above. |
| Evaluation count | Preserves Isaiah's `cfg.env.n_envs` games on the main process, without introducing `eval_n_games`. |
| Telemetry | `eval/games`, threshold 0.7 and successful/rejected promotion values are tested. Failure ordering needs the P2 fix. |
| Shared trainer / shims | Existing PPO and evaluation loops are extended directly. New helpers own the factored logic; no compatibility wrapper or second trainer found. |

## Checks and mutation evidence

- Requested `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`: **1,341 passed, 6 skipped**, 27.97 seconds, exit 0 (`pytest.log`). This includes the retained Orbit suite.
- Requested `uv run mypy python/owl scripts`: **success, 58 source files**, exit 0 (`mypy.log`). The author's 59-file `py-prepare` claim is consistent with its broader `python` target.
- `OMP_NUM_THREADS=2 uvx --offline --from rust-just just py-prepare`: **passed**, 104 files unchanged by formatting, lint/syntax checks passed, mypy succeeded on 59 files, **1,341 passed / 6 skipped** in 24.82 seconds (`py-prepare-offline.log`). The initial online uvx launcher failed to resolve PyPI DNS before running checks (`py-prepare.log`); cached offline tooling succeeded. Docs freshness reported no updates required against the clean tracked worktree, so that check is not branch-level proof; the three-dot documentation diff was inspected separately.
- Mutation on a scratch copy: replaced Kaggriculture `return banks` with `return returns` in `_evaluation_scores_and_metrics`. The named L1 test passed before the mutation, failed after it (`[0.0, 2.0] != [1.5, 0.5]`), then passed after restoration. See `mutation-*.log` and `mutation.json`.
- The restored scratch `scripts/run_ppo.py` matched both its saved original and the tracked source byte-for-byte. SHA-256: `53235fb12b9eb93354e27823df60de7ace9f3682d566ba6fd3ae8dffebea324d`. The mutation never touched a tracked file.
- Source-bound read-only probes and fault injection are reproducible with `uv run python ops/rebuild-2026-09-29/codex/verify-3.2-3.3-independent/probe.py` (`probe.json`).
- `git diff --check e1458d2...HEAD` passes.

## Scope and cookbook accuracy

The new cookbook note, index and log inventory the material changes and
correctly disclose the missing native environment, config registration,
Kaggriculture action-mask mapping and evaluation action/device/replay plumbing.
The new L1 environment is a test double; the L2 Kaggriculture check exercises
the cut helper, not a native rollout. `_evaluation_seed` has no production
caller yet, and `_create_eval_env` explicitly rejects Kaggriculture until the
native seam lands. This verification therefore approves the scoped seams,
not a completed runnable Kaggriculture training/evaluation integration.

The six skips are the native grammar binding, config integration, native
evaluation integration, two CUDA FlashAttention tests and one unavailable
x86 quantization backend. These are disclosed limits rather than successful
integration checks. Cookbook pass counts match the independent suite; its
seed-stream claims need the P3 correction above.

Final `git diff --exit-code HEAD --` passed. No tracked implementation or
cookbook changes were made. Verification evidence is kept as untracked files
in this directory; the restored scratch copy was removed after hashing.

VERDICT: APPROVE WITH EDITS
