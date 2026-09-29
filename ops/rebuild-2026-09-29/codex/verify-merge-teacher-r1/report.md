# Independent verification of the Phase 4 staging merge

Reviewed `kg/merge-teacher` at `a424d8cd20ad2734c5cd8159664d7c4be683d923` using exactly the three-dot diff from `b8747b6e8acece5f561d09a75bb914364a60ac05`. HEAD's other parent is approved teacher commit `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`; the parents' merge base is `4cac1a18f2209c54d40bef80d44755334a1a1ed2`. Three independent sub-agent audits covered parent/doc retention, semantics, and guard mutations; the primary verifier ran the required checks and numerical mutations.

The diagnostic question was whether the staging merge preserved both parents' behavior and executable checks, including the inherited compiled-GEMM protections. Inputs were the exact parent trees, merged source, requested suites and CPU scratch probes. Completion required parent inventories reconciled, explicit failures from controlled mutants, required checks reported, and unchanged tracked-file hashes. This is review of the staging merge and implemented Phase 4.1–4.3, not qualification of complete Phase 4 or integrated Kaggriculture training.

## Finding

**P3 — `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:150` (also lines 130 and 141): superseded dependencies remain current prose.** The Reference still says T18 awaits Task 3.2 value-mode guards, T19b awaits the configs merge, and Task 4.4 waits on that same merge. Its newly appended line 151 correctly says these dependencies have landed. Line 130 also describes an old `require_orbit_env` stop and skip reason. These are contradictory current reopening conditions, contrary to the cookbook's reconcile-in-place contract.

**Fix:** rewrite the earlier current statements to say configs and value guards have landed, T18/checkpoint await the trainer mapping, launch/resume await the run_ppo/native-env seams, and 4.4 is undone with its configs prerequisite satisfied. Preserve historical checks as explicitly historical evidence. Update the note's index/log in the normal coherent record change. No tracked fix was made during this independent review.

No functional merge regression or lost parent work was found.

## Parent retention and semantics

| Source inventory | Integration parent | Teacher parent | Merge |
| --- | ---: | ---: | ---: |
| Python test definitions | 893 | 797 | 926 |
| Rust `#[test]` definitions in source | 327 | 226 | 327 |
| Cookbook log titles | 100 | 75 | 105 |

These are source definitions, not parametrized execution counts. Every integration Python test name is present. The sole absent teacher name, `test_task_authored_inventory_requires_exact_replay_test`, was already renamed by the integration parent to `test_task_authored_inventory_requires_exact_authored_set` with stronger coverage. No Rust test name was lost; integration Rust/source is byte-identical. Every parent tracked path survives. All parent log entry bodies survive after trimming boundary whitespace; there are no duplicate titles. Detailed AST/name/path comparisons are in `parent-audit/`.

The import resolution preserves `terminal_seat_banks` and `KaggricultureObsConfig`. The RL spec retains the teacher bullet, the complete Environment semantics, and native observation/grammar sections. Teacher observation dispatch preserves Orbit's adjustment, requires Kaggriculture equality, and rejects cross-game input. Auto-merged teacher code retains replay-conditioned teacher-to-student KL, liveness, head chunk slicing, no-grad targets, cached/combined equality and model-owned live-seat mean CE. Shared Orbit behavior remains covered.

All teacher encode paths pass through `encode_observations -> _run_trunk`. Four additional scratch cases independently test precompute, cached student, combined student and combined teacher against a corrupted GEMM-backend setting. They pass before mutation, all fail when the inherited backend guard is removed, and pass after restoration. This demonstrates CPU dispatch/admission, not real CUDA or cuBLAS execution.

## Required checks

| Command | Independent result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed, 0 ignored (41 + 9 + 19; doc tests 0) | `engine.log` |
| `cargo test` | 254 passed, 4 ignored | `root.log` |
| `uv run python scripts/check_engine_trim.py` | Pass | `trim.log` |
| `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,673 passed, 11 skipped, 55.62 s; no sharding needed | `pytest.log` |
| `uv run mypy python/owl scripts` | Success, 63 source files | `mypy.log` |

The 11 pytest skips are: one unavailable native grammar binding, two CUDA pinned-memory cases, four teacher trainer/run_ppo seams, two CUDA FlashAttention cases, one unavailable x86 quantized backend, and one native Kaggriculture evaluation-env test. An independent scratch run removing the four teacher skips produces exactly four failures: two `KaggricultureActionMask` mapping errors and two explicit missing-env/game-seam stops. No test-body rewrite or production bypass was used for this probe (`numerical-mutations/unskipped-integration-probe.log`).

Changed concept notes pass direct installed cookbook shape/source lint; model/training changes map to changed architecture docs and README (`cookbook-lint.json`, `three-dot-doc-mapping.json`). Lint does not establish semantic currency; the P3 above remains. `git diff --check` reports only whitespace in retained raw failure logs, not implementation/document edits.

## Mutation checks

68 isolated mutation attempts were run on scratch copies: 32 numerical/behavioral attempts and 36 admission/guard attempts, including overlapping wrapper checks. 66 failed as intended, with no setup-error kills. Every mutation was restored byte-for-byte in `finally`; source hashes and full-copy checks were then compared with the tracked originals.

Coverage includes independent admissible-value KL and finite differences; finite masks, liveness, gradients, compile capture, head chunking; target dtype/layout/optional fields, index/concat/digest/hire-limit/bytes; cached-versus-combined equality; live-seat CE, teacher-gradient isolation, statelessness, seat isolation, plain-path optional outputs; checkpoint exclusion, refresh, both PPO wrappers, Orbit CE preservation; all new cache/type/presence/shape/dtype/grammar/stateless/action-layout/replay/obs-spec admission families; helper presence/invalid-slot errors; and inherited GEMM checks across every teacher encode path.

Two individual attempts survived their original selected tests:

1. Demoting only the teacher logits to FP32 does not disrupt the student finite-difference identity. The existing finite-difference test does kill two-sided demotion. A supplemental direct FP64 teacher-probability reference passes original code and kills teacher-only demotion. This distinguishes a gradient oracle's scope from a precision reference; there is no precision defect in the reviewed code.
2. Removing the combined-path grammar-signature comparison alone still rejects mismatched tables through exact table equality before encoding. Removing exact table equality is independently killed by an in-place table mutation whose construction-time digest is unchanged. This is redundant protection, not an admitted mismatched grammar.

Final restored numerical union: 53 passed, 4 intentionally skipped (48 repository teacher cases plus four GEMM probes and one precision reference). Final restored guard union: 27 passed plus five invalid-slot cases passed. Test sources, exact replacements, assertions and hashes are in `numerical_mutations.py`, `supplemental_mutations.py`, `numerical-mutations/`, and `guard-mutations/`.

The four dependency-skipped integration oracles remain unqualified past their disclosed first failure; their downstream behavior is not claimed as verified by these mutations. No GPU, BF16, real distributed trainer or performance qualification was attempted. Phase 4.4 remains incomplete as explicitly scoped in the brief and Reference.

## Custody

All 1,668 tracked files match their initial SHA-256 hashes; tracked `git status` is empty. No index change, commit or tracked edit was made. Evidence lives only in this untracked `ops/` directory. The numerical scratch's 195 copied tracked inputs all match; guard scratch's 136 inputs all match and that generated tree was removed after recording hashes. Whole-tree receipts: `tracked-before.json`, `tracked-after-check.json`.

VERDICT: APPROVE WITH EDITS
