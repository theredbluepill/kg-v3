# Task 7.3 independent verification, round 2

Reviewed `kg/rebuild-7-3` at `822c951739620bbb5f01005c217e81f05dccee01`
against `0b8cf98ef57fc49a329dca4c8368c630586c4752...HEAD`; the merge base is
exactly that requested base. This includes the merged Task 1.4 native environment.
The source review focused on Task 7.3 and the native terminal/evaluation seams;
the required Rust suites and relevant Python suite also exercised the merged env.
No training, pod performance qualification, or submission was attempted.

## Findings

### P2 — Evaluation aborts without preserving active selected-game error custody

Location: `python/owl/kaggriculture/native_evaluation.py:141`, also the selected
decoder at line 129. Policy/decode/native-step exceptions escape the loop without
closing active recorder entries. The only explicit `fail_game` call at line 162
handles one snapshot-bank mismatch, not these failures or the other active games.
Task 7.3's brief, export contract item 6, requires native runtime errors and failed
transactions to produce explicit incomplete/error records.

Two live native probes commit one valid transition, then set a submitted length
to zero on the second policy call:

- Selected decoder rejection: two selected games remain active, both have one
  committed transition, and **zero** error sidecars are written (expected two).
- Only an unselected env gets the invalid program: selected decoding succeeds,
  the real native batch transaction rejects at `env.step`, and **zero** sidecars
  are written for the one active selected game (expected one).

Both native envs correctly retain step 1; rollback is working. The defect is
loss of the selected game's already committed tape/evidence when the evaluator
unwinds. This does not fabricate a successful replay, but violates explicit
failure custody and leaves no persisted account of the aborted selections.

**Fix:** on evaluation/decode/step failure, persist every active selected game's
committed tape as error custody, then re-raise the original exception. Include
both decoder and actual native transaction rejection tests, and preserve the
original exception if error-record publication itself fails.

Reproduction and actual outputs: [error-custody/probe.py](error-custody/probe.py),
[receipt.json](error-custody/receipt.json), [probe.log](error-custody/probe.log).

### P2 — Canonical evaluation still cannot deliver its eight replays (carried forward, partial)

Location: `scripts/run_ppo.py:1606`, with the new standalone seam at
`python/owl/kaggriculture/native_evaluation.py:48`. The branch now implements
native evaluation and verifies eight selected episodes, but the canonical
trainer still raises `NotImplementedError` in `_create_eval_env` for Kaggriculture.
Searching `python/`, `scripts/` and `tests/` finds calls to `evaluate_native_games`
only in its tests; it is not called from the trainer's evaluation path.

**Fix:** finish the Task 1.5 adapter dependency and call the recorder/evaluation
seam from canonical `_evaluate_games`, with an acceptance test through that path
showing eight completed exported episodes per evaluation. This is an explicitly
documented integration dependency, not an undisclosed exporter defect. The native
seam acceptance portion is now satisfied; end-to-end Task 7.3 is still incomplete.

## Disposition of every finding in the requested r1 report

Source report:
`/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.3-r1.md`.
The original report is left unchanged to preserve the no-tracked-modification
constraint; every finding's current disposition is recorded here.

| r1 finding | Status | Current evidence / remaining fix |
| --- | --- | --- |
| P2: Eight evaluation exports remain unimplemented | **PARTIAL** | The skipped placeholders are replaced by six passing live native tests. Eight-of-ten selection, consumed construction/reset seeds and before-reset terminal capture work. `run_ppo.py:1606` still rejects Kaggriculture evaluation; complete Task 1.5 and connect canonical evaluation. |
| P3: Export accepts `complete=false` for a DONE tape and fails its own round trip | **RESOLVED** | `replay_export.rs:335` now rejects that claim. All three completion cases pass; disabling this guard in scratch source produces one failed/two passed cases with `DID NOT RAISE`. |
| P3: Byte-check test reaches semantic rejection before the byte guard | **RESOLVED** | `test_replay_export_oracles.py:360` adds same-kind `3000.0` → `3.0e3`; the first divergence is `/steps/0/0/observation/farms/0/money`. Disabling only canonical-byte rejection in scratch Rust makes this test fail with `DID NOT RAISE`. |

## Required checks, all on the reviewed version

A fresh `uv run maturin develop --offline` build succeeded before Python checks.
No dependency synchronization workaround was necessary for the required commands.

| Command | Result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**, 0 failed (41 unit + 9 oracle + 19 parity; 0 doctests) | [engine.log](engine.log) |
| `cargo test` | **289 passed, 5 ignored**, 0 failed | [root.log](root.log) |
| `uv run python scripts/check_engine_trim.py` | **PASS**, manifest OK | [trim.log](trim.log) |
| Relevant `uv run pytest -q ...` below | **655 passed**, 0 failed, 0 skipped, 154.08 s | [pytest.log](pytest.log) |
| `uv run mypy python/owl scripts` | **PASS**, 67 source files | [mypy.log](mypy.log) |

Relevant pytest command:

```sh
uv run pytest -q \
  tests/kaggriculture/test_replay_export.py \
  tests/kaggriculture/test_replay_export_oracles.py \
  tests/kaggriculture/test_replay_export_integration.py \
  tests/kaggriculture/test_native_env.py \
  tests/kaggriculture/test_env_reference.py \
  tests/kaggriculture/test_native_grammar_bindings.py \
  tests/tools/test_check_engine_trim.py \
  tests/tools/test_replay_trim_manifest.py \
  tests/tools/test_observation_oracle_custody.py \
  tests/tools/test_record_kaggriculture_env_reference.py
```

The five root ignores are the explicit lifecycle/observe performance diagnostics,
observation-oracle generation and two retained Orbit angle audits. No Task 7.3
acceptance test is skipped. These are correctness checks, not throughput claims.

## Independent oracles and scratch mutation evidence

All **21 controlled mutations** were detected. Each mutated scratch file was
restored byte-for-byte, with before/mutant/restored hashes in its receipt. None
mutated tracked source or fixture bytes.

- **12 input mutations**, [mutations-results.json](mutations-results.json): each
  of four official fixtures (719 transitions each, **2,876 total**) passes a
  full independent state/order/reward comparison, and a coherently changed final
  reward/bank fails. The pinned live framework at seed `2**80 + 19` passes a
  nine-transition game; seed +1 and integer-to-float ACTIVE reward changes fail.
  Canonical numeric respelling, float-to-integer money, a captured bank change,
  leaked seat-private state, private key reordering and official actor-inventory
  reordering also fail at specific pointers. Eight positive baselines pass.
- **6 live evaluation mutations**, [live-mutations/receipt.json](live-mutations/receipt.json):
  wrong consumed seed, reset-state terminal capture, seven instead of eight
  selections, swapped decoded seat actions, removed schedule validation and a
  terminal API returning reset state all fail the appropriate live tests.
  Baseline and restored baseline each pass **6 tests**, no skips.
- **2 Rust guard-removal mutations**, [source-mutations.json](source-mutations.json):
  disabling the canonical byte guard fails the new same-kind respelling test;
  disabling the DONE/false-completion guard fails its regression. Baseline and
  byte-restored scratch builds each pass **4 targeted cases**. Logs identify
  the imported scratch extension, avoiding accidental use of the live binary.
- **1 manifest mutation**, [manifest-mutation.json](manifest-mutation.json): a
  changed retained-kernel digest in an independent scratch manifest is rejected;
  the original/restored manifests pass the same updater oracle.

The scratch input runner initially needed `PYTHONPATH=.` to find test helpers;
that failed harness invocation is preserved. The first scratch-only direct Cargo
extension build lacked macOS dynamic-lookup linker flags; that failed attempt is
also preserved. The corrected `cargo rustc` build adds only those link flags,
then all baseline/mutant/restored checks finish successfully. Neither was a
product-test failure or counted as a successful mutation.

The current live acceptance uses ten tiny-horizon games, exports the selected
eight, retains tokens/actions and every selected full snapshot, and independently
reverifies the published episodes in byte mode. The branch's existing default-
horizon eight-game receipt was inspected but **not independently rerun** here.
No policy quality or trainer integration is inferred from the diagnostic policy.

## Custody

[restoration-custody.json](restoration-custody.json) confirms **3,272 tracked files
are byte-identical**, `git diff HEAD` is empty, and only this new untracked evidence
directory appears in `git status`. No changes to the prior report, cookbook,
source, tests, vendored engine or lockfiles remain. Scratch sources and inputs
were restored byte-for-byte; their restoration receipts are linked above.

VERDICT: REJECT
