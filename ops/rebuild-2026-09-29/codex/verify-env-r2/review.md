# Independent verification — Task 1.4 native environment

Reviewed `kg/rebuild-env` at `ba9b59bbf1581bda5b7e9389b74f16c8ba4c8f98`
against the requested three-dot range `e197528...HEAD`. The merge base is
`e197528820ab7cfb429e21259000957370abf1c6`. The four commits in scope are
`8d98ea8`, `9dc2d02`, `1e26760` and `ba9b59b`.

No production defect was found. One additional P3 recorder-test edit is
recommended below. Both findings in the supplied prior report are resolved.
This review changes no tracked file; its evidence is confined to this new
untracked operational directory and ignored or temporary scratch copies.

**Prior report finding statuses**

Source report: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-1.4-r1.md`.

| Prior finding | Status | Evidence and fix disposition |
| --- | --- | --- |
| P3, `tests/tools/test_record_kaggriculture_env_reference.py:128`: inventory tests fail at stale checksums before semantic guards | **RESOLVED** | Separate hash-corruption tests remain at line 129. Seventeen coherent semantic cases at line 221 refresh array metadata at line 250 and assert specific errors. All 14 inventory/hash guard-removal mutants are now caught by the existing tests. No further edit needed for this finding. |
| P3, `docs/rl-api-specs.md:1061`: constructor approval status is stale | **RESOLVED** | Lines 1060–1062 explicitly acknowledge the approved Q1 refinement and its incorporation in contract v4.2. This agrees with the brief and accepted contract. No further edit needed. |

**New finding — P3: isolate fixture budget and archive-hash guard tests.**

At `tests/tools/test_record_kaggriculture_env_reference.py:255`, the size test
changes declared bytes while retaining a small actual archive, then accepts
any `size|budget` error. Removing the sole compressed-size cap still passes:
the later actual-versus-declared size check rejects the incoherent fixture.
At line 269, the hash test appends bytes without updating declared size and
accepts `hash|size`; it therefore stops at that same size check before reaching
the archive hash. These tests do not protect their named guards. The expanded
archive digest likewise lacks a discriminating negative case.

All 52 non-frozen recorder tests still pass after separately removing the
compressed cap, compressed hash, or expanded hash check. The one frozen-fixture
test is excluded from these mutation checks because any recorder edit changes
its source hash; unrelated source drift is not a behavioral kill. Current
production checks are correct. Coherent scratch probes show that removing
these guards admits a valid over-budget fixture or inconsistent archive digests. **Fix:** use a valid fixture with a temporarily lowered
`MAX_COMPRESSED` and assert the exact budget error; separately corrupt each
archive digest while keeping size and all other custody fields coherent, and
assert the exact hash error before `np.load`. Keep the existing independent
size-mismatch case. Re-run the corresponding guard-removal mutants against the
repaired tests. The separate expanded-size enforcement in the ZIP loader
remains effective when only its manifest-level check is omitted; that
redundancy is not a production defect.

**Requested checks — fresh execution**

| Command/check | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**: 41 unit, 9 PRNG, 19 replay; no ignored tests |
| `cargo test --locked --offline -- --test-threads=1` | **274 passed, 5 ignored** |
| `uv run python scripts/check_engine_trim.py` | **PASS** |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **2,059 passed, 7 skipped**, completed in shards |
| `uv run mypy python/owl scripts` | **PASS, 64 source files** |
| Complete recorded TrainingBatch replay | **16 games, 11,504 exact transitions** |

The combined Python attempt exceeded the sampled 960 MiB guard during
collection. The sharded run covered all 49 test files in 50 successful shards;
one model-head file also exceeded the guard and completed as two case shards.
No assertion failure or uncovered file remains. The seven skips are two pinned
CUDA observation cases, two flash-attention CUDA cases, one unavailable x86
quantization backend, and two Task 1.5 integration placeholders. The native
grammar exports and native environment are exercised by their dedicated tests.
See `python-summary.json`, `python-shards.json` and the per-command receipts.

The main extension was freshly built with `uv run --offline maturin develop
--locked`; its SHA-256 matched the fresh dev-library artifact before testing.
Build/source identities and fixture hashes are in `source-and-build-custody.json`.
Checks used offline dependencies, at most two Cargo build jobs, and per-command
115-second/960 MiB sampled process-group limits. Root tests used one test thread.

**Mechanism review and scratch mutations**

- Batched construction, observe/reset/step and auto-reset retain caller-owned
  storage. Native preparation stages games, checked seed reservations,
  observations, transitions and terminal records. All workers join before
  ordered error selection; Python return objects are allocated before commit.
  Full and selected-row commits are distinct. Truncate preserves unselected
  observations and all six transition outputs, including nonzero terminal dones.
- Native reset/truncate failures are injected after a peer succeeds, with hit
  assertions, byte snapshots of all 35 outputs, game/fresh-observe snapshots,
  seeds and terminal records, followed by control-matching retry. Near-exhaustion
  paths independently prove no publication and successful final partial reset.
- World sizes 2 and 8 each consume 67 seeds per rank across construction,
  full/partial resets and terminal auto-reset. Failed steps do not advance
  streams, simultaneous reservation is ordered, and overflow is transactional.
  These are sequential CPU rank-stream checks, not a distributed launch.
- The native L6 poison oracle checks all 35 outputs and padding for observe,
  reset, ordinary step and terminal step while retaining all addresses.
  Omitting the econ-after copy fails all four cases. Task 1.5 still owns the
  pinned CUDA entry fence and DMA qualification.
- Reward admission matches the exact binary64 per-component predicate,
  including product underflow, active caps and finite/nonnegative coefficients.
  The reward uses the specified economic counters, raw terminal banks and
  two-rounding schedule. HIRE admission and dependency overflow catches are
  tested before publication; vendored engine bytes are unchanged.
- Constructor/lifecycle signatures and all 140 output extractors match the
  stub. The existing extension exports the class and four codec/table functions,
  retains header encoding, validates exact shapes/types/layout/regions, and
  keeps borrows alive across detached work. Tables match all 964 bits; codec
  tests include the frozen dense-actor/negative corpus.
- The deterministic fixture has 16 complete pinned reference `TrainingBatch`
  games, source/archive/policy custody, 719 steps per game, positive required
  coverage and a 310,365-byte NPZ. No new reference regeneration is claimed.

| Fresh mutation campaign | Outcome |
| --- | --- |
| Native core guards and timing-fixture oracle | **25/25 killed** by named semantic assertions |
| Rebuilt binding/L6/stride/codec/table/native-dones mutants | **13/14 killed**; the remaining explicit-alignment omission still rejects invalid arrays through NumPy's independent alignment check |
| Recorder inventory/hash guard removals | **14/14 killed** by existing tests after the prior fix |
| Replay field/math perturbations | **7/7 killed** |
| Duplicate grammar-path retirement guard | Baseline pass, duplicate-path mutant fails, restored pass |
| Release overflow policy | Reduced-LTO baseline pass, override-off mutant fails at caught-overflow assertion, restored pass |

No compiler error, source-hash mismatch or resource stop is counted as a
semantic mutation kill. Further recorder boundary probes and the surviving
stock custody tests are documented in `oracle/review.md`; they support the new
P3 above. The native-dones mutation fails at game 0 / seed 17000 / step 0 /
seat 0 with decoded action and field. Wrong stride fails both world sizes.
The alignment survivor is redundant defense: NumPy 0.28 independently checks
alignment when creating slices, so it does not indicate invalid-buffer admission.

After restoration, the native Rust subset passed **117 tests, 3 ignored**.
A freshly rebuilt restored scratch extension passed **395 tests**: 388 repository
native/codec/oracle cases plus seven scratch probes. These repeated cases are
not added to the requested full-suite denominator. Detailed reviews, exact
patches, failure excerpts and restoration receipts are in `core/`, `bindings/`
and `oracle/`.

Canonical fat-LTO release compilation exceeded the local memory guard before
the test body. A separately labeled release build with LTO disabled and 16
codegen units passed the overflow proof and detected removal of the engine
policy. The existing canonical pod proof was inspected: `Cargo.toml`,
`Cargo.lock`, `src/` and `engine_rs/` are unchanged from its `9dc2d02` source to
this HEAD (`core/pod-release-source-identity.json`). Historical canonical
execution is distinguished from fresh reduced-LTO evidence; neither is a new
performance measurement.

Working-tree `git diff --check` and the three-dot source/docs range excluding
`ops/` pass. The complete three-dot range reports whitespace in preserved raw
operational logs; those evidence bytes were not normalized. See `diff-checks.json`.
No training, GPU operation, end-to-end throughput claim, branch switch, commit,
or other-worktree edit occurred.

`restoration-custody.json` confirms all **2,350 tracked files** retain their
initial SHA-256, the tracked inventory and HEAD are unchanged, the installed
main extension retains its verified bytes, and staged/unstaged tracked status
is empty. All mutated scratch source files were restored byte-for-byte.

**VERDICT: APPROVE WITH EDITS**
