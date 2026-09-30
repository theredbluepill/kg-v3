# Independent verification: Phase 4 teacher, round 2

Reviewed `kg/rebuild-trainer-model` at `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`, using exactly `git diff b1da613a197a334547563111c1e11453f84551b8...HEAD`. The merge base equals the requested base. Compared against the Phase 4 brief v2, accepted contract v4 (`docs/kaggriculture-contract.md`), governing cookbook records, and Isaiah's code at `32b3ec900ad406eedd965f53a1a0f4490d31c589`. Two parallel read-only audits covered implementation and prior findings/records; the primary verifier independently ran the commands and mutations below.

No remaining actionable findings. Approval covers implemented 4.1–4.3, as permitted by brief §6. Full Phase 4 is incomplete: T18 and T19b still need execution through the real configuration/trainer seams, and 4.4 is deferred. This review does not qualify integrated Kaggriculture training or GPU behavior.

## Prior findings

Every finding in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-4-teacher-r1.md` is resolved:

| Severity | Current location | Resolution |
|---|---|---|
| P2 | `tests/kaggriculture/test_teacher.py:1625`, `:1654`, `:1690` | The unconditional T19b placeholder is replaced with substantive resume, fresh-launch and real-trainer checkpoint tests. Resume uses distinct student and last-best weights, exercises real last-best construction/loading and verifies activation, identity, weights, grammar tables and targets. Fresh launch checks loaded weights, activation and zero KL. The checkpoint test checks the complete serialized key set after a teacher iteration. Dependencies are named in the skips; the cookbook accurately states what has and has not run. No further fix required. |
| P3 | `docs/model-architecture.md:742` | The cached `discrete_targets` and fixed-teacher launch-mode restrictions are explicitly scoped to Orbit. The surrounding text distinguishes Orbit actor KL from Kaggriculture policy-core KL and states the grammar-signature requirement. No further fix required. |

## Implementation and contract review

- The KL retains Isaiah's `KL(teacher || student)` direction and teacher-forced conditional estimator at the replayed prefix. Effective unit/market masks, the exclusive final-HIRE prefix, STOP, and liveness weights are reused from action density. The implementation does not claim an unbiased joint-program KL.
- Finite dtype-minimum fills are used for cached logits. The shared helper promotes FP16/BF16 to FP32 and preserves FP64. Isaiah's FP16/BF16/FP32 behavior is retained.
- Contract-v4 widths and independent seat rows are preserved. The cache uses unit `[... ,241,W]` and market `[... ,11,W]` slot tensors, plus per-seat winner probabilities. FP32 accounting is `4 * (241 * (20+16+2+32+32) + 11 * (8+16+32+32) + 2) = 102,208` bytes per seat row; rollout totals match the brief. Runtime `nbytes()` counts actual tensors.
- Cache indexing is along the leading segment dimension; concatenation validates optional fields, slot keys and grammar symmetrically. Cached admission checks the grammar before student encoding. Combined evaluation also compares table tensors. The construction-time digest's limitation for later in-place table mutation remains explicitly documented.
- Both teacher paths use guarded encoding and the shared chunked policy path. Tests inspect actual head/trunk chunk calls, compare cached/combined outputs, enumerate admissible supports, and check finite differences and nonzero student-head gradients. The tests are substantive rather than only shape checks.
- Value CE is model-owned: Isaiah's default formula is unchanged; Kaggriculture averages over live seats. The removed free function has no shim. Existing PPO dispatch helpers now omit hidden state and dones for stateless calls; no second trainer or compatibility wrapper was introduced.
- Cookbook adaptation inventory, limits and retained evidence match the inspected changes. It records CPU/synthetic-grammar qualification, the documented tolerance deviations and deferred full Phase 4 acceptance. Historical test counts are scoped to their runs.

## Executed checks

CPU only, `OMP_NUM_THREADS=2`; tiny-model tests. No GPU, training run, Rust edit or Rust check.

| Command/check | Result | Evidence |
|---|---|---|
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **1,372 passed, 8 skipped**, 28.53 s | `pytest.log` |
| `uv run mypy python/owl scripts` | **Success, 58 source files** | `mypy.log` |
| `uv run pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | **1,048 passed, 3 skipped**, 13.20 s | `orbit-pytest.log` |
| `uvx --from rust-just just py-prepare` after restoring mutations | **Pass**: formatting, lint, syntax, typing, tests and docs freshness; **1,372 passed, 8 skipped**, 29.53 s; broader mypy **59 files** | `py-prepare.log` |
| T19b resume/fresh-launch dry run | **2 passed**, before mutation and after restoration | `t19b-dryrun-baseline.log`, `t19b-dryrun-restored.log` |

The eight suite skips are two unavailable CUDA flash-attention tests, one unavailable x86 quantized-backend test, one native grammar-binding test, T18, T19b's two launch tests, and T19b's trainer-checkpoint test. The last four await the named configuration and trainer seams.

The T19b dry run copied the committed `teacher-r1-fixes/t19b_dryrun.py` into a temporary test file under `tests/kaggriculture/` and removed it afterward. It bypasses `FullConfig` validation/file loading using a constructed configuration; the existing launch tests use a fake env/trainer while leaving last-best construction/loading real. This establishes the launch test bodies' operation, not integration through the missing seams. The real trainer-checkpoint test remains unexecuted.

Whole-diff whitespace checking reports whitespace in retained operational logs only (`diff-check.log`). The scoped check over `python`, `scripts`, `tests`, `docs`, `README.md` and `cookbook` passes. No source defect is inferred from preserving raw test/transcript whitespace.

## Independent mutations and custody

Each mutation was applied sequentially, run in a fresh pytest process, and restored in a `finally` block. Both failed by the expected assertion, not collection, import or syntax errors:

| Mutation | Discriminating failure |
|---|---|
| Resume activates the student instead of the separately loaded last-best teacher in `scripts/run_ppo.py` | T19b resume dry run: **1 failed**, teacher is not `session["last_best_model"]` at `test_teacher.py:1610`. |
| Restore FP32 demotion in `categorical_kl_from_logits` | FP64 finite-difference test: **1 failed** at `test_teacher.py:555`; analytic and finite-difference gradients disagree. |

`mutations.json` contains equal before/after SHA-256 hashes for both files. The launch dry run passed again after restoration; the full green `py-prepare` followed. `tracked-before.json` and `tracked-after-check.json` establish **all 829 tracked files unchanged byte-for-byte**, unchanged HEAD, and empty staged/unstaged diffs. The temporary test file was removed. The pre-existing untracked `verify-3.1-rest-r3/` directory was preserved. Only untracked verification artifacts in this directory remain.

VERDICT: APPROVE
