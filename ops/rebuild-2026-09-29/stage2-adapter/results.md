# Task 1.5 Stage 2 receipt

Base: `558ac3c2ffd475276c8221476c1a95f25229ddd8`, branch
`kg/rebuild-adapter`, this checkout only. No Git writes, dependencies, vendored
engine edits, training, release build, GPU run or performance claim.

All implementation items are done. All 2,230 collected Python cases have a
successful bounded execution: 2,224 passed and six inherited/hardware skips.
Rust passes 274 root tests with five existing ignores and 69 engine tests.
The exact monolithic `just prepare` and `just py-prepare` commands do not finish
inside the owner's Mac memory limit; therefore the requested final preparation
condition is not reported as passed. See the budget receipts and verdict below.

## 1. Changed and added files

- `README.md`: replace stale missing-native claims with the actual Task 3.1 blockers and working seeded evaluation factory.
- `docs/rl-api-specs.md`: Stage 2 binding, tests, strict tables, reward arithmetic, lifecycle and remaining pod/trainer scope.
- `docs/model-architecture.md`: native table default, one construction load, device-buffer lifetime and actual startup blocker.
- `ops/rebuild-2026-09-29/briefs/1.5.md`: append Stage 2 deviations and bounded-check limits; prior review history retained.
- `python/owl/kaggriculture/gpu_grammar.py`: constants-first strict native loader, exact eight bool NumPy arrays, existing conversion helper, no fallback.
- `python/owl/kaggriculture/rewards.py`: correct the independent oracle to literal native binary64 order at overflowing inner death sums; remove obsolete rescaling helper.
- `python/owl/model/kaggriculture.py`: native grammar table default; explicit injection retained.
- `scripts/run_ppo.py`: native seeded evaluation factory/env union; precise retained main stop and explicit unsupported policy-evaluation mapping stop.
- `src/kaggriculture/env_tests.rs`: strengthening admission-table case.
- `tests/kaggriculture/test_native_env.py`: the same strengthening native Python case.
- `tests/kaggriculture/test_env.py`: real lifecycle, terminal diagnostics, 35 stable buffers, selected-row truncation and native rollback; fake tests retained.
- `tests/kaggriculture/test_env_cuda_fence.py`: only hardware skip remains; legal native codec actions replace the fake-only invalid STOP helper; all 35 DMA checks/control remain.
- `tests/kaggriculture/test_codec.py`: unskip corpus; add actual encode rejection atomicity and batch order/rows.
- `tests/kaggriculture/test_game.py`: unskip factory seeds; verify 66 exact seeds/rank, sequential batches, disjoint ranks.
- `tests/kaggriculture/test_rewards.py`: all 11 native admission cases, validated fixture schema/custody, three live extreme cases, corrected overflow expectation.
- `tests/kaggriculture/test_gpu_grammar.py`: remove the native-table NotImplementedError skip path.
- `tests/kaggriculture/test_native_tables.py` (new): equality, corrupt constants/keys/dtype/shape/layout, ordering/device forwarding, binding failures, native default called once and injected-table bypass.
- `tests/scripts/test_run_ppo.py`: Kaggriculture factory/reproducible-world/independence/stop tests; Orbit assertions and all non-Kaggriculture definitions byte-unchanged.
- `ops/rebuild-2026-09-29/stage2-adapter/`: command logs, resource/exit receipts, scratch mutation driver, declaration/protected-source checks, this report, full command ledger, regression coverage and source custody manifests. `files.txt` lists every artifact separately.

`env.py`, `codec.py`, `game.py`, `types.py`, `config.py`, the formatted `rs.pyi`,
all protected trainer files, engine bytes, dependencies, cookbook and Isaiah
`tests/owl`/`tests/tools` are unchanged. Their merged code already called the
actual native attributes correctly; stub-only types remain annotation-only.

## 2. Checks and red/green evidence by requested item

All checks inherit `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true
UV_OFFLINE=true`; uv/uvx are offline. Python test processes were serialized.
Root bounded checks cap each command at 115 seconds and sampled process-group
RSS at 960 MiB, logging actual SIGKILL (-9) and sampled peaks rather than a
passing wrapper code. A shell receiving `SystemExit(-9)` reports 247; the ledger
preserves the raw child code. Root/check memory excludes unrelated agents.

1. Real adapter: existing runtime works upon unskip, so no artificial production
   red is claimed. New real lifecycle tests also pass on the merged runtime.
   Final `test_env.py` + DMA: 39 passed, one hardware skip. Construction/observe,
   reset, native codec step, short terminal/autoreset, metrics/snapshots/seeds,
   all 35 identity-stable outputs and late-env/seat invalid-action byte rollback
   are checked. Selected and no-op truncation preserve all six transition bytes
   and every unselected observation row, with terminal and nonterminal inputs.
2. Unskips: initial combined run is 102 passed, one failure, one CUDA skip;
   fixture KeyError exposes the provisional `transition_*` names. Correct the
   test to the actual custody-validated NPZ keys. Expanded run is 128 passed,
   one failure, one CUDA skip: at step 71 tiny W gives native [-.25,.25] versus
   rescaling-oracle approximately [-9.999888e-13,9.999888e-13]. After the oracle
   fix, rewards 42 passed, game 27 passed, codec 21 passed. Fixture: 11,504
   recorded transitions, inherited one-f32-ULP oracle allowance unchanged. Live
   extremes: three one-game/96-transition probes compare every reward exactly,
   including terminal output and positive S/D/I coverage. Corpus: 321 accepted
   programs and 43 rejected records, exact JSON/list order, lengths and rows;
   rejected decode input and encode output bytes remain unchanged.
3. Reward table: a scratch harness compiles the exact production reward module
   and extracted Rust admission test without dependencies. Combined-rule mutant
   passes the original ten cases (exit 0), fails the strengthening case (exit
   101, one failed test), then byte-exact restored code passes (exit 0). Production
   `reward.rs` was never changed. `cargo test --locked --lib
   reward_admission_predicate_cases`: exit 0, one passed, 278 filtered out.
4. Tables/default: before implementation, 22 failed/24 passed (exit 1), including
   zero loader calls in the default-model spy. After implementation: 46 passed,
   no skips (exit 0). Constants are checked before native arrays are requested;
   no missing-binding fallback. Tiny model loads once and explicit injection
   does not call the loader.
5. Evaluation factory: initial focused red has nine failures/one pass; final
   green has ten passes/97 deselected, no skips. Two intervening test-assumption
   failures are preserved: empty farmer command is invalid, and initial boards
   are seed-independent. Legal PASS/STOP and first-day RNG snapshots fix those
   assumptions. Three two-transition evaluations check fresh seed tuples, changed
   first-day worlds, exact repeat snapshots/final banks and independent instances.
6. Trainer stop: same red/green selection covers exact fresh/resume startup
   message before allocation. `_PPORolloutBuffer` accepts Orbit specs at
   `python/owl/train/ppo.py:251`, constructs Orbit `ObsBatch` at 315 and action
   storage at 402; canonical trainer uses it at 596. `_map_action_bundle` at
   2075 handles Orbit variants only, and observation mask dispatch at 2115 lacks
   Kaggriculture. `scripts/run_ppo.py:1831` and 1906 still have Orbit evaluation
   observation/action mapping. Keep the stop: no permitted Kaggriculture PPO
   rollout can execute until Task 3.1. No rollout/trainer code was added.
7. DMA: binding skip removed, hardware skip retained; actual native codec actions
   are prepared before delayed H2D copies. The test drives real `env.step`, compares
   all 35 outputs and removes `_fence` for its negative control. CPU fence spies
   pass; actual DMA/control remains unexecuted on this Mac.
8. Docs: status assertion first fails (exit 1), then passes after edits (exit 0).
   Final README/API/model docs lint and docs-fresh pass. Brief history gains the
   explicit deviations; cookbook stays untouched for Claude's review.

The subreceipts `native/results.md`, `tables/results.md`, `eval/results.md` retain
all intermediate lint/test corrections. Two independent cross-lane source reviews
found no concrete implementation gap; they ran no extra tests.

### Full preparation and resource limits

- `just prepare`: build, engine trim, formatting, Clippy, Python lint, docs lint
  and mypy (69 files) pass; default Rust test parallelism reaches sampled
  1,008,156,672 bytes and watchdog kills it (-9) after 41.147 s.
- `RUST_TEST_THREADS=1 just prepare`: root Rust 274 passed/five ignored, engine
  41+9+19=69 passed, all prior static phases pass. Python collects 2,230 cases,
  then the in-process mypy typing test hits 1,007,960,064 bytes; killed (-9)
  after 70.518 s. There is no test assertion failure.
- `just py-prepare`: format/lint/mypy pass, then the same first Python test hits
  1,008,943,104 bytes; killed (-9) after 8.884 s. This fallback command therefore
  is also not reported as a completed preparation.
- Whole Kaggriculture without the typing test also exceeds accumulated memory
  (1,081,098,240 sampled bytes); a file-sized model-head run later hits
  1,012,727,808, and a parity script run hits 1,009,238,016. All are killed by
  the watchdog; no newly added skip or relaxed assertion hides these attempts.
- Bounded completion: 57 successful test commands account for exactly every one
  of the 2,230 collected cases: **2,224 passed, six skipped, zero failed**.
  Most files run with normal project fixtures. The pure typing and pure parity
  files run with `--noconftest` because their tests use no project fixture and
  never use Torch compilation; this avoids importing the unrelated global
  Torch compile-state fixture. Their peaks are 795 MB and 699 MB. Model heads
  run as 60 non-enumeration cases, budget 10 alone (one), and budgets 0–3 (four);
  assertions/parameters are unchanged. `regression-coverage.json` gives the exact
  successful command set and denominator; it is not a full shared-process run.
- `RUST_TEST_THREADS=1 just rs-prepare`: exit 0, all Rust format/Clippy/tests,
  trim and docs freshness; root 274 passed/five ignored, engine 69 passed,
  51.227 s, sampled peak 466,354,176 bytes. `cargo fmt --check` also exits 0.
- Final declaration custody: all seven ABI declarations are AST-identical and
  unique; all eight fake signatures match. Protected-path comparison passes,
  including 105 non-Kaggriculture definitions in `test_run_ppo.py` byte-identical.
  Its mixed factory test's Orbit setup/assertions also remain unchanged.
- Required missing-binding grep exits 1 with no output (the expected no-match
  success condition); required Orbit test diff is empty; whitespace diff passes.

Full command strings, raw exits, pass/fail/skip counts and log paths appear in
`commands.md` and `command-ledger.json`; the appendix includes every verification
invocation, including intermediate reds, budget terminations and the bounded
replacement runs. Inspection/edit commands are not counted as tests.

## 3. Remaining skipped tests

In `tests/kaggriculture`:

- `test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma`: CUDA is unavailable.
- `test_observe.py::test_native_writer_uses_real_schema_and_keeps_all_pointers[pinned-1]`: pinned host memory requires CUDA.
- `test_observe.py::test_native_writer_uses_real_schema_and_keeps_all_pointers[pinned-2]`: pinned host memory requires CUDA.

In `tests/scripts`: none. No test carries the reason `needs Task 1.4 binding`.

Other inherited full-suite skips: two `tests/owl/model/test_attn.py` FlashAttention
CUDA tests and one `tests/owl/test_int8_emulation.py` x86 quantization test
(unavailable qnnpack backend). These protected tests are unchanged.

## 4. Deviations and cookbook review handoff

The appended brief section is authoritative for the complete deviation list:
existing real adapter/codec needed qualification rather than rewrites; fixture
field names/custody are corrected; Python overflow rescaling is removed to match
native order; the Rust/native strengthening tables are updated; merged stub is
checked by AST; owner brings eval-factory adoption forward from 3.1/3.3; distinct
seeds are observed through first-day RNG rather than identical initial boards;
main and policy-eval mapping stops remain; the DMA test needs valid native actions;
README is updated for its mapped runner contract; cookbook edits belong to Claude;
full checks use bounded executions because monolithic commands exceed the cap.

No ABI, config union, reward recipe, production Rust reward, native engine,
observation/model state or policy conditioning changes. No opposite-game test
body changes and no second trainer/collector.

Claude must reconcile current reward Reference lines 38–39: the Stage 1 claim
that rescaling avoids premature saturation is contradicted by the new live test.
Revise that current claim, its opening/description as needed, native-buffer status,
`cookbook/references/index.md` and `cookbook/log.md` during review. Existing owning
records: `native-game-semantics-use-v3-owned-buffers.md` and
`reward-reuse-preserves-objective-and-critic-semantics.md`. No new cookbook concept
or unsupported empirical lesson is needed. This explicit handoff follows the
owner's instruction not to edit any cookbook file in Stage 2.

## 5. Open questions and residual risks

- Blocked validation condition: literal monolithic `just prepare`/`py-prepare`
  completion under this Mac cap. All assertions and check components pass in
  bounded runs, but shared-process full preparation remains unverified here.
- Task 3.1: native rollout storage, token/length transport, observation-mask
  mapping, complete policy evaluation/replay adoption and canonical PPO smoke.
  Existing first-minibatch alarm/multirank behavior requires integration proof.
- Pod: run `test_step_does_not_overwrite_pending_dma` including fence-removal
  control, the two pinned-observation tests, and the approved early two-rank
  smoke only after Task 3.1 and dependent semantics/telemetry integrate.
- Seed custody: rank streams are exact/disjoint; mixed evaluation starts are
  reproducible. Evaluation seed bands do not promise global stream separation;
  any training-only band restriction remains downstream factory policy.
- No learning, policy-quality, GPU, BF16, compiled-path or throughput evidence
  is inferred from these CPU unit checks. No source/API question remains for
  the implemented Stage 2 boundary itself.

VERDICT: STAGE2-INCOMPLETE — the required monolithic prepare/py-prepare commands exceed the Mac memory limit; all implementation items and bounded check components pass.
