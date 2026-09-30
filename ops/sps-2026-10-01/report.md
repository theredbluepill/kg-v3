# Native SPS brief result

The native implementation and parity work are complete. The optimization is the
single default path; no configuration knob or old production path is retained.
The pristine base is retained as a small, source-bound golden and reproducible
build instructions. Original worktree commit publication is blocked by the
sandbox's read-only Git metadata. Delivery uses `/private/tmp/kg-sps-delivery`
on branch `kg/sps-rollout`; `delivery.json` records the resulting commit and
portable bundle, and verifies every delivered file against this worktree.
Nothing is pushed and no remote machine is touched.

## Changes

- `src/kaggriculture/env.rs`: run raw token admission, terminal prediction and
  grammar planning/decoding per environment on Rayon. Preserve raw-error,
  seed-overflow, grammar-error and engine-error precedence plus indexed error,
  seed and metric order. Reuse `ObsStaging`; each worker clears/writes its own
  rows. Parallel commit copies all 29 fields into disjoint caller slices and
  replaces/drops old games and snapshots on workers. Failure returns reusable
  scratch; abandoned pending batches allocate replacement scratch next time.
- `src/kaggriculture/buffers.rs`: expose the validated batch count for constant
  time publication preflight.
- `engine_rs/src/lib.rs`, `src/kaggriculture/observe.rs`: return the existing
  transactional engine candidate through `stepped_with_market_metrics`, removing
  the redundant adapter game clone. The mutating API delegates to it. The rules
  body, validation checks, RNG and reward arithmetic do not change.
- `engine_rs/TRIM_MANIFEST.json`, `VENDORED_FROM.md`,
  `scripts/check_engine_trim.py` and `tests/tools/test_check_engine_trim.py`:
  declare and independently pin only the narrow API extraction; six new
  mutation cases reject undeclared semantic/provenance drift.
- `src/kaggriculture/env_tests.rs`, `engine_rs/tests/replay_parity.rs`,
  `tests/kaggriculture/native_step_oracle.py`, `test_native_step_parity.py`:
  rollback, error-order, stale-padding, discarded-batch, engine and old/new parity.
- `tests/tools/test_observation_oracle_custody.py`: give each watchdog test a
  fresh invocation budget, independent of time spent on earlier suite tests;
  add an expired-before-launch regression. Production watchdog code is unchanged.
- `docs/{rl-api-specs,kaggriculture-contract,rules-engine,rules-parity-coverage}.md`
  and the native lifecycle cookbook Reference/index/log record architecture,
  provenance, checks and limits. The scripts, golden and receipts are here.

The existing `bindings.rs` release of the GIL during preparation and publication
is retained, as are NumPy borrow guards and Python return allocation before
commit. Caller buffers keep their addresses and the pinned-memory entry fence.
The optional typed decode optimization was not needed; grammar JSON remains.
The small transition cache and ordered metric reduction remain serial.

## Parity

Base: `b2276bc5b70073b58a27f9e5fbd52473dbd48569`, independently exported from
Git and built in release mode. Recording rejects a baseline source
that differs from this Git object. `native-golden.json` records native source,
lock/config/helper and actual extension hashes; no bulk binary is committed.

Four environments start at seeds **941003, 941010, 941017, 941024**, stride 7,
with independent action PRNGs seeded 8128–8131. Random grammar-legal unit and
market actions are admitted by the existing encoder. The current bank/margin/sign
reward is enabled. **1,440 calls × four envs = 5,760 two-seat transitions** include
**eight full games**. Default `episodeSteps=720` ends after 719 transitions,
so auto-resets occur at calls 719 and 1,438, followed by two more calls.

At **1, 4 and 8 native threads**, the optimized release exactly matches the
pristine golden's byte-stream SHA-256 for every one of the **29 observation
fields**, six transition outputs (rewards, dones, both banks and both economic
counter tensors), actions, metrics with binary float encoding, complete snapshots,
terminal records and seed state. All three verification receipts match all 39
hashes. This includes signed zeros. Three selected-row truncations (alternating,
empty, complementary), their next steps, and explicit reset are also recorded.
There is no native `truncated` output array; truncation's observation and retained
transition semantics are checked directly and by the existing adapter/trainer
suites. The random trajectory reaches ten actors.

Seven Python parity cases pass on both old and new binaries. A separate 722-call
test compares a four-env/eight-thread batch with single-env streams executed in
reverse order, including auto-reset. Malformed-peer retries preserve every output
byte and RNG result. Four new Rust tests run at 1/4/8 threads, covering error
precedence, dropped pending batches, rejected publication and dirty actor padding
after worker failure/auto-reset. Existing injected panics continue to roll back.
The engine candidate API also matches four official episodes, **2,876 transitions**,
including full state, metrics and recorded observations; early/deferred errors
leave its source unchanged.

## Mac benchmark

Apple M5, 10 cores (4 performance/6 efficiency), 24 GB, macOS 26.4 arm64;
Python 3.12.13, NumPy 2.4.4, Torch 2.9.0. Release extensions, **20 envs, four
native threads, 720 calls per replicate**, 14,400 env transitions. The timer
surrounds the actual PyO3 `rs.KaggricultureEnv.step`, including its metrics dict.
Actions and hashes are generated outside timing. Import, construction, build,
Python wrapper reward telemetry, model, GPU and PPO are excluded. One auto-reset
is included; caller buffers are reused. The workload reaches at most 13 actors.

| Five-repeat series | Baseline | Optimized |
| --- | ---: | ---: |
| Total native step seconds, samples | .789712, .801222, .811461, .938840, .911192 | .681750, .658316, .712675, .660820, .786571 |
| Median seconds / 720 calls | **.811461** | **.681750** |
| Median milliseconds / call | 1.127029 | .946875 |
| Env transitions/s from median | 17,745.77 | 21,122.11 |

This series observed **15.9849% less native time (1.1903× throughput)**. The
baseline and optimized action hash is `f76e9050…27bb19`; final observation,
transition, snapshot and seed hash is `a7d56be9…288ff`, equal in every replicate.
Full hashes and source/binary identities are in the JSON receipts.

The subsequent alternating AB/BA check was substantially noisier: baseline
median **1.272358 s**, optimized **.917302 s**, but the optimized implementation
lost **two of five pairs**. Its first sample was 2.742901 s versus baseline
1.388226 s. The complete table is in `parity-benchmark.md` and
`paired-benchmark.json`. Both series are retained; the favorable median does not
establish a stable 28% gain or exclude regressions under host scheduling noise.
The final release rebuild (after the equivalent Clippy loop cleanup) was timed
again, followed immediately by the pristine baseline, with five repeats each:

| Final release series | Baseline | Optimized |
| --- | ---: | ---: |
| Total native step seconds, samples | 1.985460, 1.762235, 1.790876, 1.828218, 2.135754 | 1.216314, 1.190350, 1.203041, 1.475720, 1.164478 |
| Median seconds / 720 calls | **1.828218** | **1.203041** |
| Median milliseconds / call | 2.539191 | 1.670890 |
| Env transitions/s from median | 7,876.52 | 11,969.67 |

This last series observed 34.20% less native time (1.5197× throughput), with the
same action and final-output hashes. `final-{baseline-,}benchmark.json` binds
those results to the final source/binary. The three series' large absolute and
relative timing variation prevents a stable speedup claim. No sample was removed
and no agent build/test overlapped the retained timings. Other machine activity,
thermal state and CPU scheduling were not controlled.

## Checks and execution environment

- `cargo check --offline`: passed after local environment setup.
- Optimized release build: `uv run --no-sync maturin develop --release
  --skip-install --offline`: passed. The plain requested command first failed
  during dependency installation because network access was unavailable.
- Requested focused Python command: **615 passed, three CUDA/pinned-memory
  skips**, 825 deselected (`python-native-final-tests.log`).
- Optimized old/new parity suite: **seven cases**, included in the source-verified
  full and focused reruns (`python-final-tests.log`, `python-native-final-tests.log`).
- Focused Rust env suite: **33 passed** (`review-native-tests.log`).
- Engine candidate replay/rollback: **two passed** (`engine-candidate-tests.log`).
- Strict engine custody suite: **85 passed**, including six new declared-mutation
  tests (`engine-trim-tests.log`); checker, Ruff and mypy passed.
- Full repository preparation command set: **all checks pass**,
  **413 Rust passed / five ignored** (root 302, engine 74, opponents 37),
  **3,016 Python passed / nine skipped** (source-verified rerun: 196.61 seconds).
  All formatter, strict Clippy (including no-default-features and both crates),
  Python 3.11 syntax, Ruff, mypy (77 files), Markdown lint, engine/opponent
  custody and docs-fresh checks pass. `prepare.log` and `prepare.sh` retain the
  exact direct-command equivalent of `just prepare`; the source-verified Python
  rerun and `mypy-final.log` supersede the copied-launcher Python results there.
  The first pass caught one
  Clippy range-loop issue, corrected by iterating over the already admitted
  two-seat length slice; the full command set then passed. The independent
  review approved that final loop substitution (`review-native.md`).
- Benchmark CLI guard checks reject zero repetitions and any attempt by verify
  or benchmark to overwrite the golden, preserving its hash; new harness Ruff
  and formatting pass (`benchmark-cli-guards.log`).
- Watchdog fixture tests: **54 passed**, both normally and with an expired
  import deadline; both timeout regressions remain enforced. Final Ruff and
  format checks pass across Python/scripts/tests and the new ops harness.
- `final-identity-check.json` verifies the current 79 native source/config/lock
  hashes and loaded release binary against final parity/timing receipts.
- Staged source/docs pass `git diff --check`. Raw `.log` receipts retain the
  command output's trailing whitespace and blank end lines; the unrestricted
  staged whitespace check flags those receipt bytes only.

`just` is not installed in this worktree. The direct recipe uses cached Ruff and
pymarkdown executables, offline Cargo/uv, a reflink copy of the already installed
local Python environment, and `maturin --skip-install`. The copy's editable path
points at this worktree. A late audit found copied console-launcher shebangs still
pointing at the original environment. Those launchers were rewritten to the local
interpreter (`venv-launcher-repair.json`). The initial Python-suite result in
`prepare.log` is not used to qualify the changed extension: the entire Python
suite and requested focused suite were rerun with `runtime_probe.py` asserting
the actual interpreter prefix and loaded extension path in the pytest process.
`python-final-tests.log` and `python-native-final-tests.log` are authoritative.
The standalone old/new recorder/verify processes used explicit local Python and
PYTHONPATH and are unaffected. Rust checks and source-checker invocations in the
original prepare remain valid. The reproduced prepare script now invokes Python
modules directly to avoid copied console-launcher paths. Missing ignored Orbit fixtures
were copied from the local `kg-v3` checkout. No dependency or lockfile changed.
Initial missing-venv, read-only uv-cache and unavailable-network failures are
retained in the logs, then addressed with writable local storage/offline tools.

The first source-verified full Python rerun found four watchdog test failures
after its import-time 110-second execution budget expired during earlier tests
(3,011 passed, nine skipped; `python-final-deadline-fail.log`). The fixture now
starts the same 115-second budget minus five-second cleanup reserve for each
test. Both the existing active-child timeout test and a new already-expired
deadline test enforce production behavior. `oracle-test-isolation.md` records
the cause, narrow repair and checks; the production deadline is unchanged.

## Limits and delivery

The finite golden and direct algorithm comparison support unchanged semantics;
they are not an exhaustive proof over all seeds/configurations. Dense 241-actor
states are covered by existing grammar/encoder tests, not this timing workload.
GPU pinned-memory tests skip on the Mac. H200/Linux, eight ranks, live-policy
trajectories, CPU contention and complete learner-update throughput remain
unmeasured. Any NVIDIA timeline attribution still requires the existing Nsight
workflow on a bounded non-live run. No recipe, policy, reward, model, training
board or remote state changed.

Git staging in this worktree failed with:
`Unable to create '/Users/poonszesen/kg-v3/.git/worktrees/kg-v3-sps/index.lock':
Operation not permitted`. That Git directory is outside the writable roots and
approval escalation is disabled. The source changes remain in this worktree.
The local delivery clone provides a commit on `kg/sps-rollout` with the required
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` final trailer and a bundle;
original-worktree branch publication cannot be completed here.
The commit/bundle identity is in the adjacent `delivery.json`; the bundle is
`delivery.bundle`. It contains this change above the prerequisite base
`b2276bc5b70073b58a27f9e5fbd52473dbd48569`. Those delivery files and the automatic
`native-transcript.log` remain outside the commit to avoid recursive or mutable
receipts. Source changes, tests, docs, the brief/code map and frozen check logs
are included. No PR is created or recommended; review and residual limits are
recorded here and in `review-native.md`.

VERDICT: BLOCKED original-worktree commit cannot write protected Git metadata; implementation, verification and local commit bundle are provided.
