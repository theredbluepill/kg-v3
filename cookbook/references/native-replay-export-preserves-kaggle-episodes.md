---
type: "Reference"
title: "Native replay export preserves Kaggle episodes"
description: "Task 7.3 seed replay and selected-game custody pass four independent official-fixture round trips, the real pinned-framework oracle and canonical-byte checks, with Kaggle reward representation and JSON number kinds compared strictly after Claude review. After verification r1 the recorder runs over the merged Task 1.4 native env (consumed seeds, before-reset terminal snapshots, eight default-horizon games byte-verified); after verification r2 any evaluation abort writes error custody for every active selected game and re-raises the original error; after verification r3 episode and custody are published transactionally, so a failed write leaves error custody, never successful custody. run_ppo's evaluation call is deferred to Task 3.1."
tags: ["kaggriculture-v3", "adaptation", "replays", "evaluation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:src/kaggriculture/replay_export.rs"
  - resource: "repository:src/kaggriculture/replay_export_tests.rs"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/env.rs"
  - resource: "repository:src/kaggriculture/env_tests.rs"
  - resource: "repository:src/kaggriculture/bindings.rs"
  - resource: "repository:python/owl/kaggriculture/native_evaluation.py"
  - resource: "repository:python/owl/rs.pyi"
  - resource: "repository:python/owl/kaggriculture/replay_export.py"
  - resource: "repository:tests/kaggriculture/test_replay_export.py"
  - resource: "repository:tests/kaggriculture/test_replay_export_oracles.py"
  - resource: "repository:tests/kaggriculture/test_replay_export_integration.py"
  - resource: "repository:tests/tools/test_replay_trim_manifest.py"
  - resource: "repository:tests/tools/test_observation_oracle_custody.py"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "repository:docs/kaggriculture-contract.md"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/update_trim_manifest.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/framework-source-audit.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/trim-updater-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/trim-updater-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/trim-deadline-migration-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/trim-deadline-migration-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/py-prepare-deadline-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/oracle-green-final.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-native.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-cargo.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-engine.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/final-results.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-7.3-r1.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/independent-verifier/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r1-fixes/eight_default_horizon.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r1-fixes/eight-default-horizon.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r1-fixes/live-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r1-fixes/prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/independent-verifier-r2/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/independent-verifier-r2/error-custody/probe.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/error-custody-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/error-custody-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/error-custody-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/verifier-probe-rerun.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/eval-guard-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/eval-guard-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r2-fixes/prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/independent-verifier-r3/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/receipt.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/publication-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/publication-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/run_mutations.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/mutations.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.3/r3-fixes/prepare.log"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:README.md"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.3-replay-export.md"
  - resource: "repository:ops/rebuild-2026-09-29/plan.md"
---

# Native replay export preserves Kaggle episodes

Task 7.3 adds a stateless replay adapter in the root `owl` crate and an opt-in
Python recorder. The target is a Kaggle episode reconstructed from the consumed
seed, supported configuration and submitted actions, with independent evidence
separate from self-consistent serialization. Four Python-recorded official
fixtures and a small live framework game pass independent replay comparisons.
Since verification r1 the recorder also runs over the live native env (see
[Verification r1 corrections](#verification-r1-corrections)); the trainer's
evaluation loop does not call it yet. The [[../decisions/the-policy-is-stateless-and-observation-only|stateless
policy Decision]] keeps seed/checkpoint/seat custody entirely in host metadata.

Existing-concept searches for replay, export, round-trip and seed header found
the [[rebuild-data-preparation-preserves-replay-identity|preparation Reference]],
the historical [[bc-bootstrap-uses-native-replay-features-and-current-heads|BC
Reference]] and the [[native-game-semantics-use-v3-owned-buffers|native boundary
Reference]]. Their prep or reference-branch evidence did not qualify a current
Kaggle writer. The preparation note now links this separate runtime claim and
retains its historical checks.

## Boundary and adaptation inventory

- `src/kaggriculture/replay_export.rs`: typed seed header and action tape,
  v4 configuration admission through the existing validator, exact arbitrary-
  width integer seed, native replay only through `Game::from_seed_header`,
  Kaggle export/import, canonical-byte and semantic round-trip checks, and a
  separate captured-snapshot/bank comparator. Existing native grammar handles
  optional token evidence against each pre-step state; no second decoder exists.
- `src/kaggriculture/replay_export_tests.rs`: synthetic contract, rejection,
  placeholder-poisoning and divergence controls in the root crate.
- `src/kaggriculture/mod.rs`, `python/owl/rs.pyi`: two stateless JSON-text PyO3
  functions in the existing extension; failures become contextual `ValueError`.
  Each public call verifies installed hashes and the supplied full configuration,
  specification and envelope against the framework. The pure Rust adapter takes
  a specification already verified by its caller.
- `python/owl/kaggriculture/replay_export.py`: installed-source hash validation,
  framework-derived specification, identity-seeded selection, selected-game
  recording with copied evidence, complete/partial/error output and source-bound
  sidecars. The count defaults to `cfg.rl.eval_replay_games`; actual seat
  assignments are supplied for stratified selection.
- `tests/kaggriculture/test_replay_export.py`: recorder selection, seed custody,
  simultaneous completions, simulated buffer reuse and non-success handling.
- `tests/kaggriculture/test_replay_export_oracles.py`: independent fixture-to-
  episode conversion, real pinned-framework envelope checks, byte round trips,
  wide-seed and non-vacuity cases.
- `tests/kaggriculture/test_replay_export_integration.py`: live native-env
  seed custody, before-reset terminal capture and eight-of-ten export (the
  original five skipped placeholders, replaced in verification r1).
- `python/owl/kaggriculture/native_evaluation.py`, `src/kaggriculture/env.rs`,
  `src/kaggriculture/bindings.rs`, `src/kaggriculture/env_tests.rs`: the native
  evaluation seam and the terminal-record snapshot (verification r1).
- `tests/tools/test_observation_oracle_custody.py`: refresh the imported
  generator's deadline per independent unit test. The new default full replays
  consume the collection-time deadline before later custody tests run; this
  test-isolation repair leaves production limits and explicit timeout controls
  unchanged.
- `ops/rebuild-2026-09-29/7.3/update_trim_manifest.py` and
  `tests/tools/test_replay_trim_manifest.py`: test-first, idempotent registration
  from integration `0b8cf98ef57fc49a329dca4c8368c630586c4752`, preserving
  `retained`, `excluded` and `authored` unchanged. Only declared non-engine
  paths and task receipt files are admitted; unexpected input is rejected.
- `engine_rs/TRIM_MANIFEST.json`: generated non-engine change registration
  only. All vendored kernel bytes and its Cargo manifests/lockfile stay frozen.
- `docs/rl-api-specs.md`, `docs/rules-parity-coverage.md`, this Reference,
  `cookbook/references/index.md`, `cookbook/log.md`, and the scoped preparation
  Reference repair: public API, evidence scope and retrievable adaptation record.
- `ops/rebuild-2026-09-29/7.3/`: planned expectations separate from actual
  red/green logs, source audit, mutation receipts, timing and final check results.

No model, PPO, reward, grammar-semantic, opponent, panel or packaging change is
part of this adaptation. Orbit's recorder and tests remain unchanged.

## Envelope and comparison contract

`steps[0]` holds the initial state; submitted action `t` is stored beside its
successor observation in `steps[t+1][seat].action`. Each seat gets only its own
private state. The pinned framework omits shared `step` from seat 1; the game
interpreter populates both copies of the other five shared fields. Import
restores only specification-marked shared fields. Nested inventories, sheds,
market maps and tile objects keep insertion order, including zero-valued keys.
Money is float64 and integers exact. Rewards keep Kaggle's representation: the
schema default integer `0` while a seat is ACTIVE and `float(money)` once DONE
(pinned `kaggriculture.py:963`), not the native snapshot's f64 `0.0`.

`info.seed` is the consumed seed; `configuration.seed` is null after resolution.
Native provenance under `info.v3_native_replay` never claims Python-framework
execution. Required native trace initial/schedule/bank fields are explicitly
documented placeholders and are poisoned in a regression to distinguish seed
construction from loading oracle answers.

Exporter-produced episodes require canonical serialization byte equality.
Foreign episodes compare values before recursive key order. Value comparison
normalizes decimal spelling (`1e-05` equals `1e-5`) but never equates a JSON
integer with a float. Captured native evidence parses its typed f64 `rewards`
as floats; everything else in it compares strictly. Semantic comparison
normalizes specification-shared omissions and observation-wrapper order only;
payload key order remains observable. Its explicit allowlist is:

- `/info` except `seed`: `core.py` copies arbitrary host metadata; native
  provenance is newly supplied and is not an oracle claim.
- `/steps/*/*/info`: framework-owned agent/runtime metadata.
- `/steps/*/*/observation/remainingOverageTime`: the framework subtracts measured
  agent duration, which seed-only native replay does not execute.
- `/configuration/actTimeout` and `/configuration/runTimeout`: framework
  execution budgets, outside the native game configuration.

No action, seed, gameplay state, status or reward is ignored. Divergence reports
name a JSON pointer and transition. Separate captured evidence can supply initial
and terminal snapshots, one bank pair per transition and any indexed full
successor snapshots. A coverage report names which of those were provided;
export/import self-agreement alone is circular.

An attempted negative control removed an explicit zero from an ineffective
SELL order. That alternate tape produced the same states and survived replay;
the failed expectation is retained in `oracle-green-attempt1.log`. A state oracle
cannot identify which of two ineffective submitted actions was originally sent.
The writer preserves the supplied action verbatim, while the sidecar's episode
hash provides original-byte custody. The replacement byte-round-trip control
changes a numeric spelling in native-produced state, independently of action
effect, and requires a canonical-byte divergence with a pointer.

## Independent sources and actual checks

The brief's Mac/framework gap is superseded: this worktree's locked environment
has `kaggle-environments==1.32.7`. The source audit recomputed and matched all
four archived pins before runtime oracle use:

| Installed file | SHA-256 |
| --- | --- |
| `core.py` | `0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e` |
| `utils.py` | `537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b` |
| `envs/kaggriculture/kaggriculture.py` | `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e` |
| `envs/kaggriculture/kaggriculture.json` | `a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867` |

The independent oracle suite passes **10 cases in 98.89 seconds**. Each fixture
is independently converted from Python-recorded data to a Kaggle episode, then
imported, replayed and re-exported. Every public/private value and payload key
order, status, raw reward and terminal bank matches; initial/terminal snapshots,
719 transition-bank pairs and 719 captured successor snapshots are also checked.

| Official fixture | Transitions | Full positive comparison seconds |
| --- | ---: | ---: |
| 95324500 | 719 | 11.368691 |
| 95901360 | 719 | 14.083801 |
| 95921764 | 719 | 15.644422 |
| 95990191 | 719 | 16.688783 |

All four run in the default Python suite, with no ignored/slow fixture case.
The isolated live framework oracle takes 0.014333 seconds and 197,558,272 bytes
peak RSS. Its seed is `1208925819614629174706195` (`2**80 + 19`), with
`episodeSteps=10`, `turnsPerDay=3`, `maxMarketOrdersPerTurn=4`,
`townShopUnlockInterval=1` and `weedSpawnChance=0.2` (other pinned defaults).
Nine scripted transitions cross three day rolls, unlock three shops, spawn weeds
and reach DONE. Byte round-trip equality and wide-seed survival pass separately.

The mutation receipts establish non-vacuity:

| Mutation | First JSON pointer | Transition |
| --- | --- | --- |
| Each fixture raw reward changed to shaped reward | `/steps/719/0/reward` | 718 |
| Framework resolved seed incremented | `/steps/3/0/observation/farms/0/tiles/0/1` | 2 |
| Native float money spelled as integer | `/steps/0/0/observation/farms/0/money` | initial |
| Captured bank incremented | `/captured/banks/1/0` | 1 |
| Seat 0 shed leaked into seat 1 | `/steps/1/1/observation/private/shed/WHEAT` | 0 |
| Initial private seed-map keys reversed | `/steps/0/0/observation/private/seeds` | initial |
| Fixture actor inventory keys reversed | `/steps/22/1/observation/private/inventories/3` | 21 |
| Framework ACTIVE reward `0` written as `0.0` | `/steps/1/0/reward` | 0 |

The updater suite has an actual missing-module collection red and then 11 green
cases. Two later cases give **13 passed in 0.01 seconds** after a recorded
registration red: only the precisely declared pre-deadline-isolation manifest
form may migrate, while another missing declaration is rejected. Coverage also
includes immutable inventory, idempotence, receipt extension, reason drift and
unexpected changed paths. Final focused native checks pass 12
tests; root Cargo passes 266 with four existing ignored; the separate engine
passes 69. The trim checker passes. The complete command inventory and final
preparation outcomes are retained in the source-bound task results receipt.

After the test-isolation repair, both preparation commands pass.
`just py-prepare` has **1,737 Python passes / 16 skips** in 163.88 seconds (172.885
seconds command wall); full `just prepare` has **1,737 Python passes / 16 skips**
in 154.85 seconds (195.435 seconds command wall), **266 root Rust passes / four
ignored**, and **69 engine passes**. These totals include the default full
fixture replay cases and the explicit binding-dependent skips; they establish
local preparation success, not live evaluation wiring or pod qualification.

The first full Python preparation run recorded four existing observation-custody
test failures, 1,731 passes and 16 skips in 163.47 seconds. Their generator module
set `RUN_DEADLINE` during collection; the added full replay checks consumed that
120-second budget before these independent unit tests launched subprocesses.
An autouse test fixture now resets that deadline for each custody test. Tests
deliberately exercising expiration still override it. This is a necessary
test-isolation deviation from the initial file inventory, with no production
observation or timeout-policy change; the red is preserved rather than hidden.

## Consequences and limits

`native_evaluation.evaluate_native_games` is the live seam, but
`run_ppo._evaluate_games` still stops at `_create_eval_env` for Kaggriculture:
calling the seam from the trainer needs the Task 1.5 torch adapter, the
model's token policy and the Task 3.1 trainer game seam (deferred there after
verification r3). Reopen the canonical acceptance when `_create_eval_env`
builds `KaggricultureEnv`; the skipped
`test_kaggriculture_canonical_evaluation_exports_eight_replays` states it. Pod-scale evaluation and recorder overhead were not
measured. The eight-game receipt used a deterministic diagnostic program, not
a model, so it qualifies custody and round trip, not play.

The framework oracle is bounded to small supported configurations; it cannot
establish every admitted configuration or framework timeout/error behavior.
Four official fixtures qualify their recorded worlds, not exhaustive rules.
Selected live games capture a full snapshot at every transition, which is what
exact live-state certification needs; it costs memory (below). Rerun source/hash
checks if the pinned framework changes. No model-input path accepts replay host
identity metadata.

## Claude review corrections

Claude's review of Codex's run (commit `517edc4`) found that native export wrote
ACTIVE rewards as `0.0` where the pinned framework writes integer `0`. The
semantic comparator and the Python oracle's structural equality both treated
`0` and `0.0` as equal, so the framework oracle passed vacuously on that field.
Export now takes the ACTIVE reward from the specification default. Both
comparisons now distinguish number kinds. Recorded reds and greens are in
`ops/rebuild-2026-09-29/7.3/claude-review/`: `rust-red.log`, `python-red.log`
and `mutation-number-kind.log` (the new framework mutation passes once the kind
check is removed).

Replay derives the grammar hire cap as `turnsPerDay * maxMarketOrdersPerTurn + 1`
instead of recording the policy's `hire_limit`. Engine `end_of_day` clears hands
(`engine_rs/src/lib.rs:4505`), so the cap equals the day's maximum
`actors + hires` and never rejects a reachable program. A test hires on every
order of every turn and reaches the cap exactly. With the cap lowered by one,
that test fails (`mutation-derived-cap.log`). Claude's first red for an explicit
`hire_limit` field rested on a wrong premise (hands persisting across days); it
is kept as `rust-red-attempt1.log`, and that change was reverted.

Malformed captured evidence passed to the recorder now raises `ValueError` and
writes error custody, instead of a bare `KeyError` that left the game active.

## Verification r1 corrections

Independent verification r1 (`ops/rebuild-2026-09-29/codex/verify-7.3-r1.md`,
REJECT) raised one P2 and two P3 findings; all three are fixed test-first.

- **P3, completion claims.** `replay_from_seed` now also rejects a tape that
  reaches DONE with `complete=false` (at `/complete`), so export can no longer
  emit bytes that fail their own import. A Rust case and a parametrized Python
  test (every exporting completion variant verifies in byte mode; the false
  claim on a DONE tape fails export) were red against the old extension.
- **P3, byte-guard isolation.** The oracle test keeps the integer-spelling
  number-kind control and adds a same-kind respelling (`3000.0` to `3.0e3`) that
  must fail with `canonical number bytes differ` at
  `/steps/0/0/observation/farms/0/money`. This closes a regression-test gap; the
  guard itself already rejected the mutation.
- **P2, live evaluation export.** `kg/rebuild-env` (Task 1.4, Codex-approved)
  is merged into this branch. The native terminal record now keeps the
  completed game's snapshot JSON, serialized inside the terminal step before
  auto-reset, exposed as `KaggricultureEnv.terminal_snapshot(i)` with the
  record's lifetime. `evaluate_native_games` assigns ordinals as
  `_evaluate_games` does, reads each consumed seed from `seed_state()`, decodes
  executed tokens against the pre-step observation, records every transition's
  full snapshot for selected games and fails a game whose snapshot banks differ
  from the published transition banks. The five skips are now executable tests;
  four source mutations (seed off by one, terminal snapshot read after
  auto-reset, swapped seat actions, reversed ordinal order) each fail them
  (`r1-fixes/live-mutations.log`).

Receipt `r1-fixes/eight-default-horizon.json` (source `1e26f11`): two envs, eight
default-horizon games, **8/8 complete**, each with 719 transitions, captured
initial and terminal snapshots, 719 bank rows and 719 full snapshots, and
byte-mode reverification of the published episode, in 37.88 seconds. Peak RSS was
1.33 GB, above the Mac's 1 GB tiny-check guideline, because each selected game
holds its full per-transition snapshots until export. Final `just prepare`:
2,196 Python passes / 11 skips, 289 root Rust passes / five ignored, 69 engine
passes (`r1-fixes/prepare.log`).

## Verification r2 corrections

Independent verification r2 (`independent-verifier-r2/review.md`, REJECT)
raised two P2 findings.

- **P2, failed evaluations lost replay custody (fixed test-first).** Policy,
  decoder and native-step exceptions escaped `evaluate_native_games` and left
  selected games active, with no record on disk. The loop now catches any
  exception and calls the new `ReplayRecorder.fail_active_games`, which writes
  one `status: error` sidecar per active game. Each sidecar holds the
  transitions recorded before the abort, its reason is
  `evaluation aborted: <type>: <message>`, and no episode is published. The
  original exception is then re-raised. A custody write that fails, such as an
  occupied path, leaves that game active, does not stop the other games, and
  is attached to the original exception as a note. The bank-mismatch check now
  just raises and uses the same path. Three live tests cover a selected-decoder
  rejection (two records), a real native batch-transaction rejection coming
  from an unselected env (one record), and a publication failure. All three
  were red first (`r2-fixes/error-custody-red.log`). Both source mutations
  (handler removed; publication failure escapes) fail them
  (`r2-fixes/error-custody-mutations.log`). The verifier's own probe, rerun on
  the fixed source, gives the expected 2 and 1 records, with no active games
  left (`r2-fixes/verifier-probe-rerun.json`). Limit: if the abort happens
  while one committed native step's transitions are still being recorded,
  tapes not yet updated leave that step out, and their error names the cause.
- **P2, canonical evaluation integration (still open, dependency).** Completing
  it needs the Task 1.5 adapter (`env.py`, `codec.py`, `rewards.py`, still in
  progress on `kg/rebuild-adapter`) and a model token policy. Neither is on this
  branch, so no trainer integration is claimed. The `_create_eval_env` error now
  names `native_evaluation.evaluate_native_games` as the seam to drive (the test
  was updated first, `r2-fixes/eval-guard-red.log`/`-green.log`). An explicit
  skipped acceptance test records the reopening condition: eight complete,
  byte-verified exports from one canonical `_evaluate_games` call. The README
  evaluation paragraph now points to the seam as well.

Final `just prepare` (`r2-fixes/prepare.log`): 2,199 Python passes / 12 skips
(three new custody tests plus the new explicit skip), 289 root Rust passes /
five ignored, 69 engine passes, trim manifest OK.

## Verification r3 corrections

Independent verification r3 (`independent-verifier-r3/review.md`, REJECT)
raised two P2 findings.

- **P2, false successful custody (fixed test-first).** `_write` published a
  `status: complete` sidecar before the episode. An injected episode open or
  partial-write failure left success custody for a missing or 20-byte invalid
  episode with a mismatching hash, and the abort handler could not replace it.
  Publication is now transactional: each file is staged in a dot-prefixed
  temporary file, fsynced and hard-linked into place (linking, unlike rename,
  never replaces an existing path); the episode goes before its custody, then
  the directory is fsynced. On any failure the files this attempt published are
  removed and an error sidecar (`replay publication failed: ...`, no episode
  hash or verification) is published before the original exception is
  re-raised. If that also fails, the game stays active for the abort handler
  and the failure is a note. Nine tests (episode or custody write/close
  failure, unpublishable error custody, fsync/link order, no-replace, and the
  verifier's partial write over two live native games) were red first
  (`r3-fixes/publication-red.log`); eight source mutations each fail them
  (`r3-fixes/mutations.json`). The verifier's probes patch the final episode
  path, which is no longer opened, so the live test carries their scenario.
- **P2, canonical evaluation (deferred to Task 3.1).** Not implemented on this
  branch by instruction; the dependency and reopening condition are recorded
  in `r3-fixes/receipt.md` and the task receipt. Task 7.3 is not landable as
  complete canonical evaluation export until then.

Final `just prepare` results are in `r3-fixes/prepare.log`.
