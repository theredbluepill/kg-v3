# Task 1.1 — trimmed rules kernel result

Task 1.1 implementation, independent Codex review, Claude's implementation
review and commit `0dc9bdd`, then the verification-round-1 edits (see
[Verification round 1 edits](#verification-round-1-edits) at the end).
Current state: engine **59 passed / 0 ignored / 0 failed**, root Rust
**155 passed / 2 ignored / 0 failed**; the checker and final `just prepare` pass.
Sections before "Claude implementation review" record Codex's uncommitted
pre-review pass (57 engine tests) and are kept as history, not current counts.

Scope: Task 1.1 brief plus Claude's required Python/pytest and separate-package
workflow edits. Branch `kg/rebuild-codex`, starting HEAD
`05a3f6f9a48cb8610e28caae542ed10eaae43296`. Reference pin
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. No commit, Git write, network or training.

Pinned `lib.rs`: 185,626 bytes, SHA-256
`c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`.

## Deviations and reasons

- Claude's requested Python checker/pytest and separate engine preparation replace
  Node and root-only preparation. `non_engine_changes` inventories paths/reasons.
- The mandatory source hashes conflict with blanket formatter/Clippy cleanliness.
  `rs-prepare.log` retains the raw formatting failure in only `lib.rs` and
  `econ_attrib.rs`; `engine-clippy-audit.log` retains six style findings:
  `too_many_arguments` (1), `collapsible_if` (4), `needless_range_loop` (1).
  Root `rustfmt.toml` ignores exactly those two files. Only engine Clippy allows
  those three lint names; authored replay code re-denies them. Preparation also
  runs the byte/inventory checker. Rules bytes and parity comparisons are unchanged.
  Independent review supports these limited exceptions, which Claude must see.
- Added a deliberately corrupted initial-snapshot failure in addition to the
  brief's broken-comparator failure, to demonstrate the replay catches bad state.
- Final authored inventory is exact after independent review caught the original
  checker accepting an empty authored list. 47 pytest cases include 12 translated
  brief cases and schema/edit/inventory regressions.
- The checker and reference-file materialization overlapped after missing-module
  red; missing-manifest red still preceded manifest creation. An initial receipt
  generator extraction was too broad and failed syntax parsing; it was corrected
  by locating Appendix C explicitly. The failed initial manifest invocation and
  a transient parallel manifest/replay mismatch remain recorded, then pass after
  proper generation. No rules change was needed.
- Owner's explicit no-commit instruction overrides the brief's commit step and
  skill integration menus; everything remains in this worktree for Claude.

## Test-first evidence

- `checker-tests-red.log`: missing checker, 1 pytest collection error.
- `checker-repository-red.log`: missing trim manifest, exit 1.
- `pretrim-red.log`: exactly 7 missing-module compile errors, exit 101.
- `private-order-red.log`: 0 passed / 1 failed, because plain JSON equality does
  not reject reordered inventory keys; `private-order-red.diff` records mutation.
- `comparator-green.log`: 2 passed after full comparator restoration.
- `replay-state-red.log`: 0 passed / 1 failed with initial step deliberately
  incremented; `replay-state-red.diff` records mutation. Pinned fixtures unchanged.
- `replay-one-green.log`: 1 passed after restoring actual initial state.

## Custody and boundaries

All commands run offline with `CARGO_BUILD_JOBS=3`, `CARGO_NET_OFFLINE=true`,
`UV_OFFLINE=true`; Explicit Cargo verification uses `--locked`; inherited root preparation commands
run offline and root lockfile identity is verified unchanged. Cargo alone
regenerates the engine lockfile, pruning only Rayon, rayon-core, crossbeam-deque,
crossbeam-epoch, crossbeam-utils and either. No remaining dependency changes
version/checksum. `cargo-lock.diff` records the full generated difference.

Standalone package; no root path dependency/workspace or L4 repair yet. Reopen L4
at the first compiled root consumer. These tests cover four pinned official worlds
and unit/RNG cases, not exhaustive malformed-input equivalence, fresh Python
differential testing, native adapter/model/PPO integration, GPU performance or
playing strength. Existing root fixture coverage is preserved with no skip flags.

`before/` preserves three pre-existing dirty cookbook files byte-for-byte;
`baseline-identities.json` holds their SHA-256 values and root source identity.
`independent-review.md` records the read-only review and its limits. `replay-counts.json`
records episode seeds, terminal banks, exact compressed hashes and denominators.

## Files added and changed

Added 14 engine files (all covered by the manifest except its own inventory):

- `engine_rs/src/lib.rs`, `engine_rs/src/py_random.rs`, `engine_rs/src/econ_attrib.rs`
- `engine_rs/tests/py_random.rs`, `engine_rs/tests/replay_parity.rs`
- `engine_rs/fixtures/episode-95324500.jsonl.gz`
- `engine_rs/fixtures/episode-95901360.jsonl.gz`
- `engine_rs/fixtures/episode-95921764.jsonl.gz`
- `engine_rs/fixtures/episode-95990191.jsonl.gz`
- `engine_rs/Cargo.toml`, `engine_rs/Cargo.lock`, `engine_rs/LICENSE`
- `engine_rs/VENDORED_FROM.md`, `engine_rs/TRIM_MANIFEST.json`

Added `scripts/check_engine_trim.py`, `tests/tools/test_check_engine_trim.py`.
Changed `justfile`, `rustfmt.toml`, `docs/rules-parity-coverage.md`,
`cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md`,
`cookbook/decisions/index.md`, `cookbook/log.md`. Added this receipt directory with
command logs/status JSON, scripts, snapshots and diffs. Baseline dirty cookbook
changes are included in those three paths and preserved separately, not erased.
No root Cargo manifest/lockfile, `uv.lock`, or root Rust source changed.

## Replay result

| Episode | Seed | Transitions | Snapshots | Terminal banks (seat 0 / 1) |
| --- | --- | --- | --- | --- |
| 95324500 | 181681617 | 719 | 720 | 97,126 / 32,640 |
| 95901360 | 804786120 | 719 | 720 | 143,344 / 151,788 |
| 95921764 | 2089097928 | 719 | 720 | 7,843 / 94,230 |
| 95990191 | 1447832391 | 719 | 720 | 87,792 / 99,703 |
| Total | 4 episodes | 2,876 | 2,880 | Exact match |

## Command outcomes

Every numbered failure below is an observed failure, retained rather than replaced
by a later successful receipt. `run_check.py` records real subprocess exit codes
and compares expected red codes without pipeline masking. JSON receipts carry
exact arguments and elapsed wall time. Standalone generator/identity checks are
also retained as scripts/output; they do not introduce dependencies.

| Command/check | Observed outcome | Receipt |
| --- | --- | --- |
| `cargo test --locked --offline` baseline | PASS: 155 passed, 2 ignored, 0 failed | `baseline-root-test.log` |
| Branch, HEAD, pin, versions, `cargo tree --locked --offline -e features -i serde_json` | PASS: expected branch/pin; root default/std only | `baseline-environment.log` |
| `uv run --offline pytest tests/tools/test_check_engine_trim.py -q` before checker | Expected FAIL: 1 collection error | `checker-tests-red.log` |
| `.venv/bin/python scripts/check_engine_trim.py` before manifest | Expected FAIL: missing manifest, exit 1 | `checker-repository-red.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --lib --locked --offline` before trim | Expected FAIL: 7 missing-module compile errors, exit 101 | `pretrim-red.log` |
| `cargo remove --manifest-path engine_rs/Cargo.toml --offline rayon` | PASS: 6 unused packages pruned | `cargo-remove-rayon.log`, `cargo-lock.diff` |
| Initial extracted manifest generator / `python scripts/check_engine_trim.py` | FAIL: extraction SyntaxError; checker missing manifest, exit 1 | `manifest-initial.log`; corrected `generate_manifest.py` |
| Checker during replay materialization | FAIL: authored inventory mismatch, exit 1 | `checker-repository-pending-replay.log` |
| `python scripts/check_engine_trim.py` after trim generation | PASS | `manifest-trim.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --lib --locked --offline` | PASS: 41 passed, 0 failed | `retained-lib.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --test py_random --locked --offline` | PASS: 9 passed, 0 failed | `retained-rng.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --test replay_parity private_order_mismatch_is_rejected --locked --offline` | Expected FAIL: 0 passed, 1 failed, 6 filtered | `private-order-red.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --test replay_parity mismatch_is_rejected --locked --offline` | PASS: 2 passed, 5 filtered | `comparator-green.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --test replay_parity kernel_public_api --locked --offline` | PASS: 1 passed, 6 filtered | `public-api.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --test replay_parity episode_95324500 --locked --offline` with corrupted snapshot | Expected FAIL: 0 passed, 1 failed, 6 filtered | `replay-state-red.log` |
| Same command with restored snapshot | PASS: 1 passed, 6 filtered | `replay-one-green.log` |
| `rustfmt --edition 2024 engine_rs/tests/replay_parity.rs` | PASS: only authored file formatted | `replay-rustfmt.log` |
| `python scripts/check_engine_trim.py` after replay/hash refresh | PASS | `manifest-replay.log` |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | PASS: 41 lib + 9 RNG + 7 replay/API/comparator = 57; 0 ignored; 0 doctests | `engine-complete.log` |
| `cargo metadata --manifest-path engine_rs/Cargo.toml --locked --offline --format-version 1` | PASS: standalone engine root, 19 registry packages | `engine-metadata.log` |
| `uvx --offline --from rust-just just rs-prepare` before exceptions | FAIL: pinned source formatting differences | `rs-prepare.log` |
| `cargo clippy --manifest-path engine_rs/Cargo.toml --all-targets --locked --offline -- -D warnings` | FAIL: 6 inherited style findings, exit 101 | `engine-clippy-audit.log` |
| `uvx --offline --from rust-just just rs-prepare` with scoped exceptions | PASS: root 155/2 ignored, engine 57/0 ignored | `rs-prepare-pinned.log` |
| `uvx --offline --from rust-just just py-prepare` before review fix | PASS: 766 passed, 3 skipped | `py-prepare.log` |
| Focused checker pytest/Ruff/mypy before review | PASS: 44 tests; Ruff clean; mypy 1 source clean | `checker-focused-validation.log` |
| Focused checker pytest/Ruff/mypy after exact-authored review fix | PASS: 47 tests; Ruff clean; mypy 1 source clean | `checker-review-fix.log` |
| **Final `CARGO_BUILD_JOBS=3 cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`** | **PASS: 57 passed, 0 failed, 0 ignored; 0 doctests** | `final-engine-test.log` |
| **Final `python scripts/check_engine_trim.py`** | **PASS: 12 retained + 113 excluded + 1 authored** | `final-manifest.log` |
| **Final `uvx --offline --from rust-just just prepare`** | **PASS: root 155/2 ignored; engine 57; Python 769/3 skipped; all 0 failed** | `final-prepare.log` |
| `git diff --exit-code -- Cargo.toml Cargo.lock uv.lock src` and SHA identity assertions | PASS: root source/dependency identity unchanged, 3 dirty cookbook snapshots preserved | `identity-verification.json` |
| `uvx --offline --from rust-just just docs-lint` after receipt reconciliation | PASS | `final-docs-lint.log` |
| `uvx --offline --from rust-just just docs-fresh` after receipt reconciliation | PASS | `final-docs-fresh.log` |
| Cookbook lint payload on updated Decision | PASS: empty diagnostic object | `cookbook-lint.log` |
| `git diff --check` | PASS | `final-diff-check.log` |

Final prepare includes the manifest CLI, root fmt/Clippy/tests, separate engine
fmt/Clippy/tests, maturin build, Ruff formatting/lint, Python 3.11 syntax check,
mypy (49 source files), markdown lint and documentation freshness. Exact commands
are in `final-prepare.log`. No warning is treated as an unreported pass: engine
style allowances are above; all remaining Clippy warnings are denied.
Python skips are two unavailable FlashAttention CUDA cases and one unavailable
x86 quantized backend case (`qnnpack`). Root ignores remain the two pre-existing
slow/advisory cases, not fixture skips.

Toolchain: pinned nightly-2026-04-18; rustc 1.97.0-nightly
(e9e32aca5 2026-04-17), Cargo 1.97.0-nightly (eb94155a9 2026-04-09).
`uv` project checks use Python 3.12.13; literal `python` CLI uses system Python
3.14.5. The checker passes on both, with 3.11 syntax compatibility checked by
preparation. Apple gzip 479 decompresses pinned fixtures. See identities for the
actual version read-back. No performance/learning claim is made from test timing.

## Claude implementation review

Claude reviewed this uncommitted implementation before committing it.

- **Verified:** every retained file matches its brief reference SHA-256. Eight are
  byte-identical to `kg/reference-2026-09-29`. `lib.rs` equals the reference
  with exactly lines 19, 21–25 and 27 removed. `Cargo.toml` differs only by the
  cdylib, binary and Rayon lines. `Cargo.lock` only drops the Rayon closure.
  `VENDORED_FROM.md` is append-only. A word-boundary search finds no `ffi`,
  `native_agents`, `myolie_*`, `policy_rows`, `joint_matching`, `training` or
  `rayon` reference in retained sources, tests or Cargo files.
- **Defect fixed (test-first):** `assert_public` used plain `Value` equality,
  which ignores key order. Public `market.inventory`/`market.prices` are
  `IndexMap` state, so their insertion order was unchecked. `assert_private`
  covered only `shed`, `seeds` and inventory objects. The new
  `public_order_mismatch_is_rejected` and
  `nested_private_order_mismatch_is_rejected` tests failed against the old
  comparator (`claude-review-order-red.log`: 2 passed, 2 failed). A recursive
  key-order check over every object in both states makes them pass. All four
  replays still pass (`claude-review-order-green.log`). A throwaway probe had
  already found 0 order mismatches across all 2,876 transitions before the
  change. The manifest's authored hash was regenerated with
  `generate_manifest.py`; only that hash changed.
- **Defect fixed:** the `justfile` engine clippy/test lines hard-coded
  `--offline`, which would fail `just prepare` on a machine without a crate
  cache, such as a fresh pod. They now use `--locked` only, as the brief review
  specified. Offline hosts set `CARGO_NET_OFFLINE=true`.
- **Accepted deviations:** exactly two `rustfmt.toml` ignores (`lib.rs` and
  `econ_attrib.rs`) and three command-line Clippy allowances for the engine
  only. The authored replay test re-denies those lints, and source attributes
  override command-line levels. Editing vendored bytes would be the only other
  way to comply.
- **Limit noted:** the trace headers' `rng_schedule` and `shop_schedule` are
  not compared directly. Their effects are checked only through the resulting
  state.

Final checks (`CARGO_BUILD_JOBS=3`, `CARGO_NET_OFFLINE=true`, `UV_OFFLINE=true`):

| Command | Outcome | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | PASS: 41 lib + 9 RNG + 9 replay/API/comparator = 59; 0 ignored | `claude-review-engine-test.log` |
| `uv run python scripts/check_engine_trim.py` | PASS | `claude-review-manifest.log` |
| `uvx --offline --from rust-just just prepare` | PASS: root 155/2 ignored; engine 59; Python 769/3 skipped | `claude-review-prepare.log` |
| `git diff --exit-code -- Cargo.toml Cargo.lock uv.lock src`; `git diff --check` | PASS | — |

## Verification round 1 edits

Codex verification round 1 (`verify-r1/verify-1.1-r1.md`, verified HEAD
`0dc9bdd`) approved with edits and no blocking findings. Both non-blocking edits
are addressed on `kg/rebuild-codex`:

- **Checker hardening (defect, test-first).** Updated manifest declarations
  could authorize LICENSE changes or remove the trim provenance appendix, and
  `Cargo.lock` was checked only against manifest-declared edits. The checker now
  permits declared edits only on `lib.rs`, `Cargo.toml`, `Cargo.lock` and
  `VENDORED_FROM.md`; derives the expected lockfile as the reference minus the
  six-package Rayon closure and the single Rayon dependency edge; and requires
  the Task 1.1 appendix (heading through the next `## ` section, SHA-256
  `1d089760…d76c93`) to follow the historical bytes. Later appended sections
  remain allowed. `check()` now delegates to `reference_files()` and
  `verify_task()`. Eight new tests drive `check()`/`main()` on a copy of the
  committed package, with the pinned reference read from Git. Before the repair:
  3 failed / 52 passed, covering the LICENSE, provenance and lockfile tests
  (`verify-r1/checker-hardening-red.log`). The eighth-`lib.rs`-removal
  regression already passed. After the repair: 55 passed.
- **Receipt reconciliation.** The opening now states the committed 59-test
  state and marks the 57-test sections as pre-review history. `replay-counts.json`,
  `replay-one-green.json`, `replay-rustfmt.json` and `replay-state-red.json` were
  matched by the root `.gitignore` pattern `replay-*.json`. They were therefore
  absent from the commit. Their SHA-256 values match `evidence-sha256.json`, and
  they are now force-added.

Checks (`CARGO_BUILD_JOBS=3`, `CARGO_NET_OFFLINE=true`, `UV_OFFLINE=true`):

| Command | Outcome | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | PASS: 41 + 9 + 9 = 59; 0 ignored | `verify-r1/engine-test.log` |
| `uv run --offline python scripts/check_engine_trim.py` | PASS | `verify-r1/manifest.log` |
| `uv run --offline pytest tests/tools/test_check_engine_trim.py -q` | PASS: 55 | `verify-r1/checker-hardening-green.log` |
| `uvx --offline --from rust-just just prepare` (first attempt) | FAIL: 6 Ruff findings in new code (B905, ARG005, RUF043) | `verify-r1/prepare-1.log` |
| `uvx --offline --from rust-just just prepare` | PASS: root 155/2 ignored; engine 59; Python 777/3 skipped | `verify-r1/prepare.log` |
