# Task 7.1 result: stopped at the required API boundary

Task 7.1 is **not implemented**. The requested standalone placement hits its
explicit STOP condition. On integration
`b8747b6e8acece5f561d09a75bb914364a60ac05`, the frozen engine does not expose
items required by the five byte-exact reference imports:

| Required item | Compiler evidence | Affected controllers |
| --- | --- | --- |
| `fib` | Private function, `engine_rs/src/lib.rs:3785`, E0603 | Starter |
| `Game.config` | Private field, `engine_rs/src/lib.rs:1190`, E0616 | Starter's `hire_step` |
| `Game::farms`, `Game::privates`, `Game::step_index` | Absent methods, E0599 | All four |
| `Game::market`, `Game::town` | Absent methods, E0599 | R04, EcoBot, E776 |

The missing accessors lived in excluded reference `policy_rows.rs:1106–1118`.
No engine visibility change, restored module, wrapper Game, copied accessor or
root-crate import was attempted. The exact compiler output and per-bot call
sites are in `native-api-probe-red.log`; support source, tool versions and
dependency lock bytes are in `native-api-probe.json`. `native-api-probe.md`
contains reproduction commands and all five source/data SHA-256s. No production
`opponents_rs/` exists. Scratch imports remain ignored in
`.codex-tmp/7.1-native-api-probe/`.

## Actual checks

All commands used the task's offline/thread/TMPDIR exports. Tests additionally
used `RUST_TEST_THREADS=1`. No Git metadata was written; no network, training,
GPU, new model diagnostic or opponent panel run occurred. The requested
repository preparation includes its existing model unit tests. `ps` was denied
by the sandbox, so aggregate live RSS was not measured; no <1 GB measurement is
claimed. The full preparation took 154.020 seconds in aggregate; individual
reported Rust/Python test suites took less than two minutes. No new opponent
game ran.

| Requested command | Actual result | Receipt |
| --- | --- | --- |
| `cargo test --locked --offline --manifest-path opponents_rs/Cargo.toml` | Exit 101: manifest absent after mandated stop; no tests executed | `opponents-test.log` |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | Exit 0: 69 passed (41 library, 9 RNG, 19 replay); 0 failed/ignored; doc tests 0 | `engine-test.log` |
| `uv run --offline python scripts/check_engine_trim.py` | Exit 0 before and after bookkeeping update | `engine-trim.log` |
| `uv run --offline python scripts/check_opponent_import.py` | Exit 2: checker absent after mandated stop | `opponent-import.log` |
| `uv run --offline pytest tests/tools/test_check_opponent_import.py tests/tools/test_check_engine_trim.py tests/scripts/test_kaggriculture_parity.py tests/owl/kaggriculture/test_opponents.py -q` | Exit 4: missing opponent test path; no tests ran | `targeted-tests.log` |
| `uvx --offline --from rust-just just prepare` | Exit 0: root Rust 254 passed/4 ignored; engine 69 passed; Python 1,625 passed/7 skipped; build/fmt/lint/typing/docs pass | `prepare.log` |

The seven Python skips are native grammar binding (1), CUDA pinned memory (2),
FlashAttention CUDA (2), quantization backend (1), and native Kaggriculture
evaluation environment (1). `commands.json` records the full argument arrays,
statuses and elapsed times. Existing engine and repository successes do not
establish opponent action parity.

### Test-first and mutation evidence

- Native admission red: offline `cargo check --locked --offline --manifest-path
  .codex-tmp/7.1-native-api-probe/Cargo.toml --all-targets` exits 101 with 42
  library errors and 51 test-compilation errors. There is no green opponent
  implementation. E776's additional E0308/E0277 errors remain unattributed
  while accessors are unresolved.
- Updater red: the new test file initially fails to import missing
  `update_trim_manifest`, before any test body runs (`updater-red.log`, exit 1).
- Updater green: three tests pass (`updater-green.log`, exit 0), checking
  retained/authored/excluded preservation, byte-idempotence and rejection of
  four unexpected-input mutations (one in each manifest inventory section).
- Actual manifest idempotency, unchanged engine inventories, direct cookbook
  lint, script Ruff checks and final whitespace checks are recorded in
  `closure.log`. Only `TRIM_MANIFEST.json` changes inside the engine tree.
- No oracle tampering, swapped-seat or changed-seed mutation was run, because
  no opponent oracle comparison could be built. Custody-checker attacks,
  lifecycle, hidden-state perturbation and determinism checks are unimplemented.

## Opponent parity and coverage

| Bot | Seats | Traces / seeds | Actions compared seat 0 / seat 1 | Mismatch result |
| --- | --- | --- | --- | --- |
| starter | Neither executed | 0 / none | 0 / 0 | Not evaluated; compile blocker |
| r04 | Neither executed | 0 / none | 0 / 0 | Not evaluated; compile blocker |
| ecobot | Neither executed | 0 / none | 0 / 0 | Not evaluated; compile blocker |
| e776 | Neither executed | 0 / none | 0 / 0 | Not evaluated; compile blocker |

For every bot and seat, observed counts are zero for openings, day resets,
weed presence, market shortages/rejected orders, hires, final-day sales and
mid-episode reset/replay. Every category remains uncovered. No native action
versus Python action comparison occurred, so there is no first action mismatch
to report and no claim of a zero-mismatch pass.

New oracle fixture size: **0 / 4,000,000 B**. There are no oracle fixture hashes.
The scratch E776 policy tape (117,954 B) is executable policy data, not a new
oracle; its SHA-256 is
`da0d5d1bd326cb5bf068c2065ba1fe8f7e644107db806d7f9a1eae4dafd89692`.
The four controller hashes match the brief; all five scratch inputs total
335,571 B. `native-api-probe.json` pins them.

Read-only original-submission custody verifies 18 files against both commit
`e8884aae82eddeb7a1aeae99ecceeca7c830d67e` and sibling working bytes. This
includes the three specified `main.py` hashes, EcoBot provenance, E776's
manifest and its 14 listed files. No Python source was copied. Exact paths,
sizes, SHA-256s and original notice text are in
`python-oracle-source-audit.json`. R04 has no agent `PROVENANCE.md`;
EcoBot/E776 declare no software license, and redistribution remains unresolved.

## Scope and remaining work

The implementation stops at the user's explicit inaccessible-engine-item rule.
Consequently the production crate/manifest/registry/runner, Python policy-loader
extension, oracle generation/comparison, custody checker, justfile wiring and
binding-dependent skipped tests are not created. Adding unavailable components
or fake passing tests would obscure that stop. No panel, replay-export,
packaging, model, PPO, reward or grammar code changes.

The trim updater deliberately preserves the five excluded reasons rather than
claiming completed imports. It changes only `non_engine_changes`. It accepts
only the exact pinned integration manifest or its own exact output and refuses
unexpected input. This is the only deviation from the planned successful-import
manifest transformation. The probe is disposable scratch instead of a broken
production crate. No source dependency was unavailable offline.

Reopening items:

- A revised placement/API contract must resolve private `fib`, private
  `Game.config` and the five missing methods while specifying byte custody.
- All controller behaviour, lifecycle, visibility, determinism and original
  Python parity checks remain required, with default-config-only qualification.
- Original license/notice custody remains unresolved, especially EcoBot/E776.
- Task 1.4 is still needed for learned-seat binding tests.
- Aggregate RSS of the requested baseline checks was not established.

## Changed file inventory

The following list is generated from the updater's explicit inventory. Each
path records a diagnostic or documentation purpose, not opponent functionality.

- `engine_rs/TRIM_MANIFEST.json` — updater-generated non-engine change inventory only.
- `docs/rules-parity-coverage.md` — document the compile blocker and zero opponent coverage.
- `cookbook/references/frozen-engine-api-blocks-standalone-opponent-import.md` — record the verified public-API blocker and reopening condition.
- `cookbook/references/index.md` — index the stopped-import Reference.
- `cookbook/log.md` — prepend the stopped-import finding.
- `ops/rebuild-2026-09-29/7.1/plan.md` — separate diagnostic expectations from actual results.
- `ops/rebuild-2026-09-29/7.1/results.md` — report actual commands, scope, coverage and remaining work.
- `ops/rebuild-2026-09-29/7.1/native-api-probe.md` — describe the compile failure and exact reproducer.
- `ops/rebuild-2026-09-29/7.1/native-api-probe.json` — pin compile inputs, support source and compiler-log hash.
- `ops/rebuild-2026-09-29/7.1/native-api-probe-red.log` — retain the compiler's inaccessible-item diagnostics.
- `ops/rebuild-2026-09-29/7.1/native-api-probe-red.exit` — retain compile exit status 101.
- `ops/rebuild-2026-09-29/7.1/commands.json` — record requested final commands and actual exit statuses.
- `ops/rebuild-2026-09-29/7.1/opponents-test.log` — record missing crate after the mandated stop.
- `ops/rebuild-2026-09-29/7.1/engine-test.log` — record unchanged engine regression results.
- `ops/rebuild-2026-09-29/7.1/engine-trim.log` — record the engine custody check.
- `ops/rebuild-2026-09-29/7.1/opponent-import.log` — record missing opponent checker after the stop.
- `ops/rebuild-2026-09-29/7.1/targeted-tests.log` — record requested pytest collection failure after the stop.
- `ops/rebuild-2026-09-29/7.1/prepare.log` — record the requested repository preparation result.
- `ops/rebuild-2026-09-29/7.1/closure.log` — record bookkeeping, documentation and final byte checks.
- `ops/rebuild-2026-09-29/7.1/python-oracle-source-audit.json` — record read-only original-submission hashes and notice gaps.
- `ops/rebuild-2026-09-29/7.1/update_trim_manifest.py` — idempotently register only non-engine diagnostic changes.
- `ops/rebuild-2026-09-29/7.1/test_update_trim_manifest.py` — test preservation, idempotency and unexpected-input refusal.
- `ops/rebuild-2026-09-29/7.1/updater-red.log` — retain the test-first missing-updater failure.
- `ops/rebuild-2026-09-29/7.1/updater-green.log` — record bookkeeping test results after implementation.
