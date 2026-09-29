# Task 7.3 replay export — implementation receipt

Worktree `/Users/poonszesen/kg-v3-t73`, branch `kg/rebuild-7-3`, base
`0b8cf98ef57fc49a329dca4c8368c630586c4752`. Changes are uncommitted. No Git
write, dependency addition, network request, training, GPU, panel or pod run.
The vendored kernel is unchanged; its manifest is updated only by the declared
updater. Source and installed-extension fingerprints: `source-custody.json`.

## Files and purpose

- `src/kaggriculture/replay_export.rs`: typed exact seed/header/tape, native seed replay, Kaggle envelope export/import, first-pointer value/order/byte divergence, independent captured evidence comparison.
- `src/kaggriculture/replay_export_tests.rs`: synthetic rejection, timing/privacy/quantity, poisoned placeholders, token/native error, exact numeric and captured-evidence controls.
- `src/kaggriculture/mod.rs`: declare adapter and register two stateless PyO3 functions; enforce the installed pinned framework at each Python entry.
- `python/owl/rs.pyi`: JSON-text API signatures.
- `python/owl/kaggriculture/replay_export.py`: verified framework schema, reproducible selected-game recorder, copied seed/terminal custody, partial/error records and hashed episode sidecars.
- `tests/kaggriculture/test_replay_export.py`: recorder tests plus real native writer and direct-API schema/source/configuration admission.
- `tests/kaggriculture/test_replay_export_oracles.py`: independent four-fixture and bounded real-framework round trips, byte equality and mutation failures.
- `tests/kaggriculture/test_replay_export_integration.py`: five explicit Task 1.4 binding-dependent skips with intended API.
- `tests/tools/test_replay_trim_manifest.py`: updater immutability, inventory, unsafe-path, drift and idempotence controls.
- `tests/tools/test_observation_oracle_custody.py`: give each independent unit test its own command-deadline window so earlier full fixture tests do not exhaust an import-time deadline.
- `ops/rebuild-2026-09-29/7.3/update_trim_manifest.py`: idempotent non-engine registration, rejecting unexpected input.
- `ops/rebuild-2026-09-29/7.3/run_checks.py`: offline command/status/timing receipts.
- `engine_rs/TRIM_MANIFEST.json`: updater-generated non-engine inventory only; retained/excluded/authored unchanged.
- `docs/rl-api-specs.md`: public replay JSON API and comparison/custody contract.
- `docs/rules-parity-coverage.md`: current replay coverage, independent sources and gaps.
- `cookbook/references/native-replay-export-preserves-kaggle-episodes.md`: adaptation inventory, source correction, evidence and limits.
- `cookbook/references/rebuild-data-preparation-preserves-replay-identity.md`: retain historical preparation evidence and link the new runtime qualification.
- `cookbook/references/index.md`: retrieve the new runtime claim and scoped preparation history.
- `cookbook/log.md`: prepended adaptation entry.
- Other files under `ops/rebuild-2026-09-29/7.3/`: individual red/green logs, hash audit, expectations, oracle results and final status receipts, inventoried by the updater.

## Actual final commands

The exact command/status/wall-time records are in `final-results.json`, with
stdout/stderr in its listed logs. **All seven required commands exit 0.**

| Command | Actual result | Wall seconds |
| --- | --- | ---: |
| `cargo test --offline --lib kaggriculture::replay_export` | 12 passed, 0 ignored (258 filtered) | 2.429 |
| `cargo test --offline` | 266 passed, 4 existing ignored | 15.290 |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 69 passed, 0 ignored | 22.372 |
| `uv run --offline python scripts/check_engine_trim.py` | exit 0; frozen manifest inventories preserved | 1.165 |
| `uv run --offline maturin develop` | exit 0; debug extension rebuilt | 1.725 |
| `uv run --offline pytest tests/kaggriculture/test_replay_export.py tests/kaggriculture/test_replay_export_integration.py tests/owl/test_replay.py tests/tools/test_check_engine_trim.py -q` | 123 passed, 5 Task 1.4 skips | 6.738 |
| `uvx --offline --from rust-just just prepare` | 1,737 Python passed, 16 skipped; root 266 passed / 4 ignored; engine 69 passed | 195.435 |

Required language-specific checks also pass: `just rs-prepare` (30.718 seconds)
and `just py-prepare` (172.885 seconds; 1,737 passed, 16 skipped). Full preparation
also passes formatting, lint, static typing and docs freshness. The new manifest
test suite has 13 passes. Every current file is listed separately in
[changed-files.md](changed-files.md).

## Test-first evidence

| Piece | Recorded red | Recorded green |
| --- | --- | --- |
| Seed/header/tape, completion and native token decode | `native-red.log`: missing adapter/API | `native-green.log`: first 3 cases; `final-native.log`: final suite |
| Envelope/import, byte and captured comparator | `envelope-red.log`: missing functions | `envelope-green.log`: first 5 cases |
| Real framework specification shape | `framework-shape-red.log`: `/specification/observation/properties` rejected | direct field-map correction; `oracle-green-final.log` |
| Exact decimal semantic comparison | `numeric-red.log`: `0.00001` versus `1e-5` mismatch | `native-mutational-green.log`; final native suite |
| Typed token-seat count and byte-divergence pointer | `guards-red.log`: 2 failures | `guards-green.log`: 10 cases |
| Strict terminal-state import | `terminal-shape-red.log`: empty private accepted | `final-native.log`: malformed private/market/town rejected |
| Empty captured replay | `captured-empty-red.log`: index panic | `final-native.log`: explicit error |
| Recorder base and lifecycle | `recorder-red.log`: 19 absent-implementation failures | `recorder-green.log`, `recorder-final-green.log` |
| Recorder admission and failure custody | `recorder-custody-red.log`, `recorder-admission-red.log` | matching green logs |
| Direct PyO3 pinned schema/source/budgets | `recorder-direct-schema-red.log`: 8 failures / 2 existing passes; `recorder-budget-red.log`: 4 failures | `recorder-direct-schema-green.log`: 14 passed |
| Four fixture/framework/byte/evidence oracles | `oracle-red.log`, `oracle-inventory-red.log`, `oracle-byte-pointer-red.log` | `oracle-green-final.log`: 10 passed |
| Custody unit-test deadline isolation | `py-prepare-deadline-red.log`, `prepare-deadline-red.log`: 4 deadline failures | `observation-custody-isolation-green.log`: 53 passed |
| Manifest updater | `trim-updater-red.log`: missing module | `trim-updater-green.log`: 11 passed; `trim-deadline-migration-green.log`: final 13 passed |

Two incorrect negative-control assumptions remain visible. A huge integral
market quantity is accepted/capacity-limited by the unchanged kernel; the actual
native-error case uses `PICKUP WHEAT null`, a recorded Python/native rejection
class. Removing the zero from an ineffective SELL made a valid alternate tape,
so it did not fail state replay (`oracle-green-attempt1.log`). The replacement
byte mutation changes numeric spelling while leaving its value equal. The writer
preserves submitted actions exactly; original-byte custody comes from its hash.
Supplementary already-passing coverage is not represented as a new implementation
red. Lint failures and fixes are retained separately (`clippy.log`,
`py-prepare-lint-red.log`, `prepare-lint-red.log`).

## Round trips and mutations

`oracle-results.md` records full configuration, timings and mutation transcripts.
All four brief fixture hashes match, and all four Python-recorded fixtures pass:
719 transitions each, 2,876 total. Positive comparison times are respectively
11.368691, 14.083801, 15.644422 and 16.688783 seconds. Every public/private state,
map insertion order, status, raw reward, terminal bank, initial/terminal capture,
719 bank pairs and 719 full successor snapshots is compared independently.
No fixture is ignored or marked slow.

The installed framework is now 1.32.7. All four full archived SHA-256 values
match (`framework-source-audit.json`), correcting the brief's obsolete Mac gap.
The isolated oracle runs one game with `episodeSteps=10`, `turnsPerDay=3`,
`maxMarketOrdersPerTurn=4`, `townShopUnlockInterval=1`, `weedSpawnChance=0.2`,
seed `1208925819614629174706195` (`2**80 + 19`), nine transitions, three day rolls,
weeds, shop unlocks and terminal DONE. Actual game time 0.014333 seconds, child
peak RSS 197,558,272 bytes; the subprocess timeout is 120 seconds. Framework
`toJSON` versus export and import/replay/re-export both pass. A separate writer
smoke uses `2**100 + 19`, `episodeSteps=4`, `turnsPerDay=1`, three transitions.
The native Rust wide-seed case uses `2**100 + 1`. All survive JSON without floats.

Canonical export/import/re-export bytes pass. Foreign comparison checks values
then recursive object order, restoring only schema-shared observation fields.
Seat 1 omits `step` but the game interpreter writes the other five shared fields
onto both seats. Only the observation wrapper is reordered for shared omission;
payload order remains observable. The foreign-only allowlist is exactly:

- `/info` except `seed`: arbitrary host metadata copied by pinned `core.py::toJSON`; native provenance differs explicitly.
- `/steps/*/*/info`: framework agent/runtime metadata copied by the same serializer.
- `/steps/*/*/observation/remainingOverageTime`: pinned `__loop_through_interpreter` charges measured agent duration; native replay has no agent execution clock.
- `/configuration/actTimeout` and `/configuration/runTimeout`: framework execution budgets, validated against the pinned schema before they may differ.

No actions, gameplay state, rewards, statuses or seed are ignored. Native-produced
inputs permit none of those foreign differences: canonical bytes must match.

| Successful negative control | First pointer | Transition |
| --- | --- | --- |
| Four fixture terminal raw reward/bank -> shaped 0.8 | `/steps/719/0/reward` | 718 |
| Live framework seed + 1 | `/steps/3/0/observation/farms/0/tiles/0/1` | 2 |
| Native money float -> equal integer spelling | `/steps/0/0/observation/farms/0/money` | initial |
| Captured bank + 1 | `/captured/banks/1/0` | 1 |
| Seat-private leakage | `/steps/1/1/observation/private/shed/WHEAT` | 0 |
| Initial private key order reversed | `/steps/0/0/observation/private/seeds` | initial |
| Official inventory key order reversed | `/steps/22/1/observation/private/inventories/3` | 21 |
| Action moved to earlier successor | `/steps/1/0/observation/farms/0/money` | 0 |
| Transition dropped | `/steps/3/0/observation/step` | 2 |
| Synthetic shaped terminal reward | `/steps/6/0/reward` | 5 |
| Synthetic captured bank changed | `/captured/banks/2/1` | 2 |
| Synthetic private map reordered | `/steps/1/0/observation/private/seeds` | 0 |

Additional source/schema and adapter negative tests produce contextual pointers
and explicit errors; they are validation coverage, not independent game oracles.

## Deviations and unresolved work

- The prompt supersedes the brief's placement/commit/live-wiring steps: all new Rust is in root `owl`, engine bytes stay frozen, no Git writes, and no `_evaluate_games` edit or substitute Task 1.4 binding.
- Oracle tests have a separate file, run explicitly and by normal preparation. All four full fixtures run by default rather than being ignored.
- `UV_NO_SYNC=1` prevents `uv run` from unexpectedly building an editable release wheel before the requested debug maturin command. An initial parent attempt was interrupted; another agent's automatic editable build completed before this setting was adopted (`trim-updater-red.log`); its resource use was not separately measured. All final builds and check receipts use the locked installed environment and debug maturin. No lockfiles/dependencies changed.
- The longer default fixture tests exposed four existing custody-unit-test failures from an import-time deadline expiring before their execution. A test-only fixture gives each unit test its own 120-second window; the production shared limit and deliberate deadline-enforcement test are unchanged. `py-prepare-deadline-red.log` records four failures, and all 53 custody tests pass after isolation (`observation-custody-isolation-green.log`).
- The Python-facing native boundary invokes a metadata/schema validator before detached Rust replay; pure Rust APIs accept an already verified caller-supplied specification. No game state is rebuilt by Python.
- Task 1.4 live consumed-seed/reset/terminal custody and eight-game evaluation wiring remain five explicit skips, with intended API comments.
- Pod-scale eight-game evaluation and recorder overhead are unmeasured. Framework timeout/error episodes and every supported configuration are not qualified.
- Exact future live-state certification requires every captured full snapshot or an independent oracle; coverage fields report evidence actually supplied.

## Open dependency after verification r3

Canonical trainer evaluation export is deferred to plan Task 3.1 (game seam in
the trainer, lane A). `run_ppo._create_eval_env` still rejects Kaggriculture
and the canonical eight-export acceptance test stays skipped until Task 3.1
lands. The r3 publication fix and this dependency are recorded in
[r3-fixes/receipt.md](r3-fixes/receipt.md).
