# Independent verification — Task 1.4 native environment, r3

Reviewed branch `kg/rebuild-env` at `1e63597957ed4995dc3eac47e996349a485670a9`
against the requested three-dot range `e197528...HEAD`; merge base
`e197528820ab7cfb429e21259000957370abf1c6`. The five commits in scope are
`8d98ea8`, `9dc2d02`, `1e26760`, `ba9b59b`, and `1e63597`.

No native-environment correctness defect was found. The prior r2 finding is
resolved. One new P3 affects the recorder's final two-file publication on an
ordinary filesystem error. No tracked modification was made; this directory
contains operational verification receipts only.

**Finding — P3, UNRESOLVED: roll back a failed fixture-pair publication.**

`scripts/record_kaggriculture_env_reference.py:369` replaces the destination
NPZ, then line 370 replaces its JSON manifest without rollback. Independently
injecting an `OSError` only at the second replacement reproduces both cases:

- A fresh output retains the NPZ with no JSON after the function raises.
- An existing loader-valid fixture loses its original NPZ while retaining its
  old manifest; subsequent loading fails with `fixture compressed size differs`.

This conflicts with the brief's no-partial-fixture publication requirement
(`ops/rebuild-2026-09-29/briefs/1.4.md:630`). The loader rejects both incomplete
pairs, so this does not permit a false passing trajectory oracle. Worker
watchdog failures also remain correctly isolated in private staging. The
severity is P3 because the defect is confined to the final recorder publication
error path; the committed fixture and native runtime remain valid.

Fix: preserve the prior pair until publication succeeds, restore it if the
second replacement raises, and remove a newly created partial pair on failure.
Add separate fresh-output and existing-output regression tests that fail the
second replacement and assert exact destination bytes afterwards. If the
requirement extends to abrupt process termination during publication, use a
single atomic generation/commit-pointer design; exception rollback alone cannot
provide that stronger guarantee. Reproduction, original/after hashes and exact
loader errors: `oracle/pair-publication-probe-details.json` and
`oracle/pair-publication-probe-runner.log`.

**Every finding in the supplied prior report**

Source: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-1.4-r2.md`.
Statuses are recorded here; the supplied historical report is unchanged.

| Prior finding | Status | Current evidence / fix disposition |
| --- | --- | --- |
| r2 P3, recorder tests formerly at lines 255/269: size/hash cases stop at earlier size mismatches | **RESOLVED** | Current tests at `tests/tools/test_record_kaggriculture_env_reference.py:274`, `:293`, `:304`, `:323` keep other fields coherent, lower the relevant cap or corrupt one size/digest, assert exact errors and forbid `np.load`. Each of six size/hash guard removals is caught by a named test. No further edit needed for this finding. |
| Inherited r1 P3: inventory tests stop at stale array hashes | **RESOLVED** | The coherent semantic cases at `tests/tools/test_record_kaggriculture_env_reference.py:244` refresh array metadata at line 250. All 14 inventory/hash guard-removal mutations fail independently. |
| Inherited r1 P3: stale constructor approval status | **RESOLVED** | `docs/rl-api-specs.md:1060` explicitly records approved Q1 and contract v4.2, consistent with the brief and accepted contract. |

**Requested checks — fresh execution**

| Command/check | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**: 41 unit, 9 PRNG, 19 replay; none ignored |
| `cargo test --locked --offline -- --test-threads=1` | **274 passed, 5 ignored** |
| `uv run python scripts/check_engine_trim.py` | **PASS** |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **2,064 passed, 7 skipped**, completed in shards |
| `uv run mypy python/owl scripts` | **PASS, 64 source files** |
| Full pinned TrainingBatch trajectory replay | **16 games, 11,504 exact transitions** |

The combined Python attempt stopped at the sampled 960 MiB guard. All 49 test
files completed in 50 successful shards; `test_model_heads.py` also required a
case split after reaching the guard. There are no assertion failures or
uncovered files. `python-summary.json` and `python-shards.json` preserve the
inventory and denominator, without counting stopped or repeated cases. Seven
skips cover two pinned-CUDA observation cases, two CUDA flash-attention cases,
one unavailable x86 quantization backend, and two Task 1.5 integration
placeholders. The native grammar/table exports have their own passing tests.

The main extension was freshly built with `uv run --offline maturin develop
--locked`; its SHA-256 equals the build artifact. Source, fixture and toolchain
identity is in `source-and-build-custody.json`. Checks used offline dependencies,
at most two Cargo build jobs and per-command 115-second/960-MiB sampled
process-group guards. Root tests used one test thread. These are CPU correctness
checks, not training or throughput measurements.

**Contract audit and mutation evidence**

Batched construction/reset/step and auto-reset use unpublished native candidates,
checked local seed reservations, staged observations/transitions and terminal
records. Every worker joins before ordered error selection; the Python return
dictionary is allocated before publication. Full and selected-row commits are
separate. Reset/truncate fault tests cover late construction/preparation errors,
panics, seed exhaustion, terminal-record preservation and control-matching retry.
Truncate preserves all six transition outputs and unselected observation bytes.

L3 tests consume 67 seeds per rank for world sizes 2 and 8 across construction,
full reset, partial reset and auto-reset, including failure and overflow paths.
Ranks run sequentially, with at most two live games. The native L6 poison oracle
checks all 35 outputs and padding for observe/reset/ordinary step/terminal step
while retaining addresses. The Python adapter and CUDA reuse fence remain
Task 1.5; this review does not qualify pinned DMA or trainer integration.

Reward validation implements the exact binary64 per-component admission rule,
including underflow and active caps; computation retains own economic counters,
raw terminal-bank outcomes and the reference's two-rounding schedule. Checked
HIRE admission and engine overflow failures occur before publication. The
vendor trim passes and no kernel bytes changed.

The constructor and four lifecycle signatures, and all 140 output extractor
names/indices/dtypes, match the stub. The existing module still exports the
header encoder and now the native class plus four cold grammar/codec functions.
Caller borrows span detached work. Tables match all 964 bits; codec tests cover
canonical padding, dense actors and the frozen accepted/rejected corpus.

Independent custody checks read all 127 pinned reference source files from Git,
the five local recorder/policy sources, archive/array hashes and required
coverage. The 310,365-byte fixture contains 16 complete 719-transition games.
Fresh native replay is bit-exact. No new reference regeneration is claimed.

| Scratch campaign | Result |
| --- | --- |
| Native core plus timing-fixture oracle | **25/25 source mutations killed**, across 19 named oracles |
| Release overflow-policy removal | Reduced-LTO baseline **1 passed**, mutant **1 failed**, restored **1 passed** |
| Rebuilt binding/L6/L3/codec/table/dones mutations | **13/14 killed**; explicit alignment removal remains rejected by rust-numpy's independent slice-alignment guard |
| Recorder semantic inventory/hash removals | **14/14 killed** |
| Recorder size/hash removals | **6/6 killed**, confirming the r2 fix |
| Source custody / ZIP inventory / export identity removals | **3/3 killed** |
| Replay field and mathematical-reward perturbations | **7/7 killed** |
| Premature validation/worker publication | **2/2 killed** |
| Duplicate grammar-path retirement guard | Baseline pass, mutant fails, restored pass |

Further coherent loader, coverage, schema and supervision probes are documented
in `oracle/review.md`; these overlap the guard inventory and are not summed as
unique mutations. The final second-replacement fault probe exposed the P3 above.
One initial scratch harness run overlapped a source mutation and tripped source
custody; it is retained as a harness error, not a semantic kill. Sequential
restored reruns pass.

All source mutants were restored byte-for-byte. No compiler failure, import
error, source-hash mismatch or resource stop is counted as a semantic kill.
Restored follow-up suites passed **117 Rust tests, 3 ignored**, and **395 Python
cases** (388 repository cases plus seven scratch probes). These repeats are not
added to the requested-suite denominator. Lane reports, patches, logs and hash
receipts are in `native/`, `bindings/` and `oracle/`.

Canonical fat-LTO release compilation hit the local memory guard before the
overflow test body ran. The fresh reduced-LTO release proof used LTO disabled
and 16 codegen units; that limitation is explicit. Existing canonical pod proof
remains source-bound: all recorded source/lock hashes match HEAD and
`Cargo.toml`, `Cargo.lock`, `src/`, `engine_rs/` are unchanged from pod source
`9dc2d02`. No pod operation or fresh performance claim occurred.

Working-tree and three-dot source/docs `git diff --check` pass. The complete
three-dot range includes whitespace in preserved raw operational logs; those
historical evidence bytes were not normalized. `restoration-custody.json`
records all **2,850 tracked files** unchanged, unchanged HEAD/inventory, empty
tracked staged/unstaged status and the unchanged freshly built main extension.

**VERDICT: APPROVE WITH EDITS**
