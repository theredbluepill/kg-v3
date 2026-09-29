Independent verification of Task 7.3, round 3

Reviewed branch `kg/rebuild-7-3` at exact requested HEAD
`f23cd4fcfab3187ad008d0bfca2408d5f8f20fdf`, using the three-dot diff against
`0b8cf98ef57fc49a329dca4c8368c630586c4752`. The merge base equals that base.
The review follows `ops/rebuild-2026-09-29/briefs/7.3-replay-export.md` and
focuses on replay export and its live native evaluation/terminal seams. The
required suites also exercise the merged Task 1.4 environment. No tracked
implementation, test, cookbook, lockfile or prior review was edited.

**Finding 1 — P2: Failed episode publication leaves successful custody.**

Location: `python/owl/kaggriculture/replay_export.py:505` (publication at
505–509; occupied-path guard at 501–502).

`ReplayRecorder._write` publishes a sidecar containing `status: complete`,
`complete: true`, successful verification and the expected episode hash before
opening/writing the episode. If that second operation fails, the evaluator's
new abort handler cannot publish error custody for the same game: the complete
sidecar already occupies the path and `fail_game` raises `FileExistsError`.

Three fault probes drive the unchanged production source over real native games
with `episodeSteps=3`:

- An episode-open `PermissionError` leaves complete custody and no episode.
- With two active selected games, that failure leaves game 0 complete without
  its episode; game 1 correctly receives error custody.
- An episode write that writes 20 bytes then raises `OSError` leaves an invalid
  JSON episode, a mismatching SHA-256, and custody still claiming complete with
  `verification.ok=true`. The second active game's custody is error. The original
  exception is correctly preserved, but its note reports that error custody for
  game 0 could not be published.

These are publication failures, distinct from the repaired decoder/native-step
failures in r2. A consumer of persisted custody sees success for a missing or
corrupt artifact. The declared non-success custody guarantee does not hold at
this boundary.

Fix: stage publication and only expose successful custody after episode
publication succeeds. On partial publication, clean up only files owned by this
attempt and preserve explicit error custody; retain existing unrelated evidence,
process the other active games, and re-raise the original failure. Add tests for
episode open/write failures and custody write/close failures. Merely swapping
these two writes without handling their failure states is insufficient.

Evidence: [single-game receipt](export-review/publication-result.json),
[two-game receipt](export-review/publication-multi-result.json),
[partial-write receipt](export-review/publication-partial-result.json), and
[partial-write reproduction](export-review/publication_partial_probe.py).
Each receipt records unchanged source SHA-256.

**Finding 2 — P2: Canonical evaluation still cannot produce eight replays.**

Location: `scripts/run_ppo.py:1606` (raises at 1607).

`_create_eval_env` still raises `NotImplementedError` for Kaggriculture.
`run_ppo` imports the retained Orbit recorder, and the new native evaluator is
mentioned in the guard text but never called by the canonical evaluation path.
The new `test_kaggriculture_canonical_evaluation_exports_eight_replays` at
`tests/scripts/test_run_ppo.py:3061` is explicitly skipped; its body is a future
acceptance placeholder. Naming the missing Task 1.5 adapter/model-token-policy
dependency accurately documents the gap but does not complete Task 7.3.

Fix: finish the Task 1.5 adapter/model policy dependency, connect the native
recorder/evaluator through canonical `_evaluate_games`, and replace/unskip the
acceptance placeholder with an executable canonical call that produces eight
complete, byte-verified episodes with seed, seat and checkpoint custody.
The standalone native eight-of-ten acceptance is passing.

**Disposition of every finding in the requested r2 report.**

Source: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.3-r2.md`.
Its two main findings and all three carried-forward r1 rows are classified here;
the original report remains unchanged.

| Finding in r2 | Status | Evidence in this version |
| --- | --- | --- |
| P2: Failed evaluations lose replay custody on decoder/native-step failure | RESOLVED | `native_evaluation.py:203` catches and re-raises after publishing active-game errors; `replay_export.py:432` attempts every active game independently. The selected-decoder and real native transaction rejection tests pass; they retain exactly one committed transition with two and one error records respectively. The publication-collision test preserves the original error while writing the other game's record. Removing the handler makes all three fail; removing publication-error isolation fails its dedicated test. An additional independent policy-failure probe writes two error sidecars and preserves exception identity. Finding 1 above concerns a different, newly probed partial-publication state. |
| P2: Canonical evaluation integration remains incomplete | UNRESOLVED | `run_ppo.py:1606–1614` still rejects it; the canonical eight-export test is skipped. See Finding 2. |
| Carried-forward r1: Eight evaluation exports | PARTIAL | The live native seam exports and reverifies eight selected games from ten at a tiny horizon. Canonical trainer evaluation still cannot do so. |
| Carried-forward r1: DONE tape with `complete=false` | RESOLVED | `replay_export.rs:335` rejects the claim at `/complete`; three completion cases pass. Disabling only this guard in a compiled scratch extension makes one case fail with `DID NOT RAISE`; restoration returns all three to passing. |
| Carried-forward r1: Byte-check regression coverage | RESOLVED | The same-kind `3000.0` to `3.0e3` mutation fails at `/steps/0/0/observation/farms/0/money` with `canonical number bytes differ`. Disabling only the canonical-byte guard makes the targeted test fail with `DID NOT RAISE`; restoring the scratch source returns it to passing. |

Supporting receipts: [policy abort](policy-abort-result.json),
[Python-source mutations](mutations/python-mutations.json),
[Rust guard mutations](mutations/source-mutations.json), and
[required pytest output](pytest.log).

**Required checks on the reviewed version.**

A fresh `uv run maturin develop --offline` succeeded before Python tests
([build receipt](build.json), [build log](build.log)).

| Command | Result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed, 0 failed, 0 ignored: 41 unit + 9 oracle + 19 parity; 0 doctests | [engine.log](engine.log) |
| `cargo test` | 289 passed, 0 failed, 5 ignored | [root.log](root.log) |
| `uv run python scripts/check_engine_trim.py` | PASS, engine trim manifest OK | [trim.log](trim.log) |
| Relevant pytest command below | 760 passed, 0 failed, 2 skipped; 152.29 seconds test time, 154.07 seconds command wall | [pytest.log](pytest.log), [command receipt](pytest.json) |
| `uv run mypy python/owl scripts` | PASS, no issues in 67 source files | [mypy.log](mypy.log) |

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
  tests/tools/test_record_kaggriculture_env_reference.py \
  tests/scripts/test_run_ppo.py
```

Both Python skips are in `test_run_ppo.py`: fresh reproducible canonical
Kaggriculture worlds and the new eight-replay canonical acceptance. None of the
three replay-export test modules is skipped. The five Rust ignores are the
explicit lifecycle/observe performance diagnostics, observation-oracle generation
and two retained Orbit angle audits. These runs establish bounded correctness,
not GPU performance or gameplay quality.

**Independent oracle and mutation evidence.**

All **25/25 controlled scratch mutations** were detected, with byte restoration
checksums. This covers each Task 7.3 oracle family, including the new r2 abort
oracles. The fault probes revealing Finding 1 are additional to that count.

- [12 input mutations](mutations/mutations-results.json): all four official
  fixtures pass full state/order/status/reward comparisons over 719 transitions
  each (2,876 total); each rejects a coherently corrupted terminal raw reward.
  The pinned framework runs a nine-transition game at seed `2**80 + 19`.
  Wrong seed, ACTIVE integer-to-float reward, equivalent exponent spelling,
  float-to-integer money, changed captured bank, leaked private state, reordered
  private keys and reordered actor-inventory keys each fail at the intended
  JSON pointer. Eight positive baselines pass.
- [10 Python/API mutations](mutations/python-mutations.json): wrong consumed
  seed, post-reset terminal capture, seven selections, swapped seat actions,
  removed schedule validation, omitted abort publication, escaping publication
  error, shallow action copy, shallow token copy and a terminal API returning
  reset state. Each targeted test fails under its mutation; restored baselines
  pass. The removed abort handler fails all three new r2 tests.
- [2 Rust source mutations](mutations/source-mutations.json): disabled byte
  guard and disabled DONE/false-completion guard. Builds use only an isolated
  scratch source/target and extension path, printed in the test logs. Live
  extension/source is untouched. Baseline and restored builds each pass the
  four targeted cases.
- [1 manifest mutation](mutations/manifest-mutation.json): changed retained
  kernel digest is rejected; original/restored manifests pass the updater
  oracle. Retained vendored kernel bytes were never mutated.

The initial scratch input-runner invocation lacked `PYTHONPATH=.` and failed
helper import. The corrected invocation completed all input cases; the retained
`mutations/input-mutations.log` records that successful rerun. The setup failure
is not counted as a product failure or detected mutation.

The live acceptance independently rerun here is eight selected games from ten,
using a diagnostic policy and a tiny horizon. It verifies every selected live
initial/terminal snapshot, per-transition banks and full snapshots, then
reverifies the published episode in byte mode. Native byte equality is the
explicitly documented canonical-JSON equality; foreign framework episodes use
semantic values, number kinds and payload key order. The existing eight-game
**default-horizon** receipt was not independently rerun. No model/PPO training,
CUDA profiling, pod qualification or submission was performed.

**Custody.**

[Final custody receipt](restoration-custody.json) compares SHA-256 for all
3,324 tracked files against the pre-review snapshot and checks tracked-path
inventory and git diffs. All are unchanged; the only untracked entry is this
new verifier evidence directory. [Scratch custody](mutations/restoration-custody.json)
records byte-for-byte restoration of mutated copies. No cookbook adaptation
record is needed because this review makes no repository adaptation.

VERDICT: REJECT
