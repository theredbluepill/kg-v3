# Independent verification of the Phase 4 staging merge — r2

Reviewed `kg/merge-teacher` at `2390c8e239a4d57770bdafd621c05abbcb326739`, using the three-dot diff from `b8747b6e8acece5f561d09a75bb914364a60ac05`. The merge commit is `a424d8cd20ad2734c5cd8159664d7c4be683d923`; its teacher parent is `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`. HEAD adds the r1 documentation reconciliation and archived evidence. No implementation changes were made during verification.

The diagnostic question was whether the merge preserves both parents' code, tests and contracts, including the inherited compiled-GEMM guards, and whether the prior finding is fixed. Inputs were the exact parent trees, current source, requested checks and isolated CPU mutations. Completion required reconciled parent inventories, explicit mutation outcomes and limits, check counts, and byte-for-byte restoration. Three parallel subagents audited retention and ran guard/numerical mutations; the root verifier independently read the implementation, brief and docs, ran the requested checks and checked custody.

## Findings and prior-review disposition

**No new actionable findings.** Every finding in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1.md` is accounted for below. The source report is unchanged; its SHA-256 and disposition are in `prior-finding-status.json`.

| Prior severity and location | Status | Fix verified in this HEAD |
| --- | --- | --- |
| P3 — `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md`, old lines 130/141/150; current lines 131/142/151–155 | **RESOLVED** | Reconciles the earlier prose in place: configs and Task 3.2 guards have landed; T18/checkpoint wait for trainer mapping, launch/resume for run_ppo/native env, and Task 4.4 remains undone with its prerequisite satisfied. Description, index and newest log entry agree. No further fix required. |

This approves the staging merge's implemented Phase 4.1–4.3 scope. It does not establish complete Phase 4 training integration, Task 4.4, real CUDA/BF16 behavior, distributed correctness, memory use or throughput.

## Parent retention and resolution semantics

| Source inventory | Integration parent | Teacher parent | HEAD |
| --- | ---: | ---: | ---: |
| Python test definitions | 893 | 797 | 926 |
| Rust test definitions | 327 | 226 | 327 |
| Cookbook log section titles | 100 | 75 | 106 |

These are source definitions rather than parametrized execution counts. Every integration Python test name remains. The only missing teacher name, `test_task_authored_inventory_requires_exact_replay_test`, was already renamed and expanded by integration to `test_task_authored_inventory_requires_exact_authored_set`. All 33 teacher test bodies remain AST-identical. No Rust test name or parent tracked path was lost. The complete `src/` and `engine_rs/` trees, including the trim manifest, match integration. All parent log entry bodies survive after stripping boundary whitespace; no duplicate titles exist.

Fresh scratch three-way reconstruction of all twelve paths changed by both parents yields eight byte-exact automatic merges and the four declared conflict files. The additional changes are the disclosed coverage counts, skip reasons and Reference reconciliation. See `merge-audit/report.md` and its machine-readable inventories.

The run_ppo resolution retains both required imports and game-aware teacher observation dispatch. RL docs retain the teacher-target bullet, integration's complete environment semantics and both native sections. PPO's teacher protocol, model-owned CE and stateless dispatch coexist with the integration's reward/truncation handling. Orbit's CE formula and segment-major layout remain equivalent. Kaggriculture's teacher-forced KL uses the replay-conditioned support and liveness; cache indexing/chunking, no-grad targets, live-seat mean CE and cached/combined equality are exercised. Every teacher encode routes through `_run_trunk`; four scratch probes cover precompute, cached student, combined student and combined teacher admission to the inherited GEMM-backend guard.

## Required checks

| Command | Result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed (41 + 9 + 19), 0 ignored; 0 doc tests | `engine.log` |
| `cargo test` | 254 passed, 4 ignored | `root.log` |
| `uv run python scripts/check_engine_trim.py` | Passed | `trim.log` |
| `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,673 passed, 11 skipped, 46.23 s; no sharding required | `pytest.log` |
| `uv run mypy python/owl scripts` | Passed, 63 source files | `mypy.log` |

All commands exited 0. Skips comprise one native grammar binding, two CUDA pinned-memory cases, four deferred teacher integration tests, two CUDA FlashAttention cases, one x86 quantized backend and one native Kaggriculture evaluation-env case. Check metadata is in `checks.json`.

Direct cookbook shape/source lint passes for both changed concept notes. The three-dot code/doc mappings are satisfied by the architecture doc and README (`cookbook-lint.json`, `three-dot-doc-mapping.json`). This is separate from semantic review. `git diff --check` identifies whitespace in retained raw evidence logs only, with no implementation or document whitespace finding (`diff-check.log`).

## Mutation evidence and limits

There were **73 scratch mutation attempts: 67 killed, 2 scoped survivors and 4 blocked by pre-existing integration seams**. The executable subset comprises 36 guard attempts and 33 numerical/behavioral attempts. Each mutation was restored byte-for-byte; the final restored guard union passed 32 cases, and the restored numerical union passed 53 with the four expected integration skips. Restoration receipts cover all 136 guard-scratch inputs and 199 numerical-scratch inputs (195 tracked inputs, the native module and three fixture files). All 33 new teacher test definitions are covered by attempts, with the four deferred integration oracles explicitly classified as blocked.

Coverage includes per-slot admissible-value KL, finite-difference gradients, masking/liveness, finite target fills, compile capture, head chunking, target layout/index/concat, grammar digest/hire limit, bytes, cached/combined equality, live-seat CE, teacher gradient isolation, seat isolation/statelessness, optional outputs, checkpoint state exclusion, last-best refresh, both PPO wrappers, Orbit CE, new admission guards and inherited GEMM checks across teacher paths.

The two survivors are limited and explained:

1. Demoting only teacher logits to FP32 survives the student finite-difference oracle, which tests the student derivative. A separate direct FP64 teacher-probability reference passes the original and kills that demotion. Two-sided demotion is killed by the existing finite-difference test.
2. Removing the combined-path signature comparison alone still rejects the tested constructor-level mismatch through exact table equality. Removing exact table equality is killed by an in-place edit whose stored digest remains unchanged. Cached signature removal is independently killed. The survivor reflects overlapping guards, not acceptance of the mismatched grammar.

The four deferred integration bodies were also run without skips or production bypasses. Two fail first at `python/owl/train/ppo.py:2140` on unmapped `KaggricultureActionMask`; two fail at `python/owl/train/config.py:57` with the explicit run_ppo no-environment stop. Their mutation attempts alter the cache metric, checkpoint schema, resume teacher selection and fresh-launch activation respectively, but reach those same first failures. They are **BLOCKED**, not mutation kills or verified downstream oracles. These are disclosed inherited seams, not new merge regressions.

Detailed replacements, selections, failures and restoration hashes are in `guard-mutations/` and `numerical-mutations/`. CPU fake-compiled-callable admission probes do not demonstrate real CUDA execution. No complete trainer or performance qualification is claimed.

## Custody

All 1,789 tracked file contents match their initial SHA-256 hashes, and tracked git status is empty. No tracked edit, index change or commit was made. Review evidence is confined to the untracked `ops/rebuild-2026-09-29/codex/verify-merge-teacher-r2/` directory. See `tracked-before.json`, `tracked-after-check.json` and each scratch restoration receipt.

VERDICT: APPROVE
