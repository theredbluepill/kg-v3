# Task 1.1b results — live differential parity against Kaggle's engine

Owner request (2026-09-29): “can you add a parity check after your rust engine,
with kaggle envcironments? thanks a lot”.

- **Question:** Do Kaggle's own Python engine and the vendored Rust kernel
  produce the same public state, private state, statuses and rewards? Do they
  agree on invalid input, including inputs Python rejects?
- **Code path:** `scripts/kaggriculture_parity/generate_traces.py` runs
  kaggle-environments 1.32.7 in an isolated uv environment, and
  `engine_rs/tests/replay_parity.rs` replays the traces with `replay_text`.
- **Discriminating observation:** the first differing line, step and field.
- **Stopping condition:** every trace agrees, or each divergence is minimized,
  kept and documented without editing the pinned kernel.

## Setup

- Worktree `/Users/poonszesen/kg-v3-parity`, branch `kg/rebuild-parity` from
  `0dc9bdd`, `CARGO_BUILD_JOBS=3 uv sync`. Generated Orbit fixtures were copied
  from the main worktree because Git ignores them.
- Engine pin: the Cargo metadata specifies `kaggle-environments-version = 1.32.7` and
  `python-engine-sha256 = bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.
  The isolated install at
  `~/.cache/uv/archive-v0/iUJXGFAmYOZiEWSq0JtEg/.../envs/kaggriculture/kaggriculture.py`
  has that hash. The project lock (kaggle-environments 1.29.0, no Kaggriculture
  environment) is unchanged.

## Commits

- `33e1428` adds the generator, sweep runner, extended Rust harness, 15
  committed traces, checker validation, tests and first failing evidence.
- `8ca378b` makes the sweep pass absolute paths. `cargo test` runs from
  `engine_rs/`, so relative `KAGG_PARITY_*` paths failed. The first
  known-divergence attempt failed that way, and its log was overwritten by the
  rerun described below.
- The documentation and cookbook commit follows these two.

## Checks (this branch)

| Check | Result |
| --- | --- |
| `CARGO_BUILD_JOBS=3 cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 66 passed: 41 lib, 9 RNG and 16 replay; 0 ignored |
| `uv run python scripts/check_engine_trim.py` | `engine trim manifest: OK` |
| `uv run pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py` | 12 + 61 passed, including live byte-identical regeneration |
| `CARGO_BUILD_JOBS=3 uvx --offline --from rust-just just prepare` (`prepare.log`) | exit 0; root Rust 155 passed/2 ignored; engine 66; Python 795 passed/3 platform skips; docs fresh |
| Engine Clippy with the three inherited allowances | clean |

Non-vacuity tests show that the replay fails, at the stated step and field, when
any of these is perturbed:

- `money` at step 100 fails as `public.farms[0].money`;
- the private `seeds` key order at step 50 fails as `private[1].seeds`;
- the seat-0 action at step 0 fails at step 0;
- an accepted step relabelled `rejected` fails as `rust_accepted`;
- a Python-rejected step relabelled `transition` fails as `rust_error`.

## Recorded sweep

`uv run python scripts/kaggriculture_parity/sweep.py --games 40 --traces
engine_rs/target/kaggriculture-parity/sweep-final` ran at `8ca378b` (46 s, single
process). The log is `sweep.log`, and the summary is `sweep-summary.json`.

- **Games:** 40 games (base seed 20,260,929, step 7,919), with 21,824 transitions
  and 155 Python-rejected steps. Every game agrees.
- **Probes:** 303 probes with 1,515 transitions; 266 agree and 37 diverge. Of
  those divergences, 12 are D1 and 25 are D2, and none are unclassified.
- **Totals:** 343 traces and 23,339 transitions. The Rust test exit code is 101
  because the known probe divergences fail the directory test.

## Divergences (engine kernel not edited)

- **D1:** Python `int()` accepts non-ASCII Unicode decimal digit strings
  (`"٣"`, `"３"`). Rust errors on them as PICKUP/PLACE counts
  (probes 057, 058, 090, 091). Rust drops them as BUY_SEED, SELL, BUY_PRODUCT
  and BUY_ANIMAL quantities, while Python executes three units (probes 167, 168,
  204, 205, 241, 242, 278, 279). It was first found in the free-hire edge game
  at step 13: `BUY_PRODUCT FERTILIZER "٣"`, expected money 2,159, Rust 2,460.
- **D2:** Python raises `TypeError` on an unhashable array or object in these
  positions: a unit verb, a PLANT crop (including in a missing hand), a
  PICKUP/PLACE item, or a BUY_SEED/BUY_ANIMAL item. Rust accepts the step as a
  no-op. See probes 030–035, 102–113, 118, 144, 145, 179, 180, 290 and 291.

Seven repros in `engine_rs/fixtures/generated/divergence-*.jsonl.gz` (minimal
since the verification fixes below) are expected failures. Rust asserts their exact line, step, kind and field.
Evidence is kept in `evidence/first-failing-games/`: the first five failing
full-game traces, with `rust-report.json` giving each first divergence (steps
13, 14, 15, 30 and 45).

`with-known-divergences/` contains 12 games generated with
`--include-known-divergences` (at `33e1428` plus the uncommitted path fix, seed
777). Eight diverge; D1 accounts for 2 and D2 for 6, and 0 are unclassified.

## Verification fixes (verify-1.1b-r1)

Codex verified `9ee7fb8` (report
`/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-1.1b-r1.md`):
APPROVE WITH EDITS, no blocking findings. Commit `6217868` fixes each edit.
Tests were written before the fixes; the Rust rollback tests (unresolved names)
and the repro manifest test were observed failing first:

- **Sweep classification.** `classify` matched inputs only, so a trace corrupted
  at `public.day` on a D1 line was reported as D1 with `new_divergences: 0`.
  `confirmed_class` now requires the observed signature: D2 needs a `rejected`
  record, `TypeError: unhashable type` and Rust `rust_accepted`; D1 needs a
  Python-accepted transition and a Rust recheck, with only that line's Unicode
  digits spelled in ASCII, that passes the line. Receipt:
  `verify-r1/negative-classification.log` (the reviewer's corruption,
  `{'unclassified': 1}`, new 1).
- **Null probe.** `None` was the no-script sentinel, so probe 296 (`null` whole
  action) submitted PASS. `scripted_choices` submits scripted actions exactly;
  probe 296 now records `null`, Python accepts it and Rust agrees.
- **Rollback check.** Rejected steps compared order-insensitive `Value`s and
  skipped statuses/rewards/done. `rollback_difference` compares public and
  private values and key order, statuses, rewards and completion; three Rust
  tests cover key-order-only, private-order and scalar mutations.
- **Minimal repros.** The seven divergence fixtures are one-step games
  (`episodeSteps` 2) whose first seat-0 action carries only the divergent
  field; they diverge at line 1, step 0 with the recorded kinds and fields.
  They were regenerated live; the eight game traces are byte-identical. The 15
  committed files now total 665,021 bytes.
- **Wording.** Rich orders are 10^12 units; quantity probes cover the four
  quantity-taking market verbs; the null probe and minimal repros are described
  as above.

Checks at `6217868` (Mac, offline):

| Check | Result |
| --- | --- |
| `uvx --offline --from rust-just just prepare` (before commit, same tree) | exit 0; root Rust 155 passed/2 ignored; engine 41 + 9 + 19 replay; Python 800 passed, 3 platform skips; docs fresh (`verify-r1/prepare.log`) |
| `sweep.py --games 40` | 343 traces, 23,339 transitions, 306 agree, 37 diverge (D1 12, D2 25), 0 new; 12/12 D1 rechecks pass; 46.8 s (`verify-r1/sweep-summary.json`, `sweep.log`) |
| `sweep.py --games 12 --base-seed 777 --no-probes --include-known-divergences` | 8/12 diverge, D1 2, D2 6, 0 new (`verify-r1/with-known-divergences/`) |
| Corrupted D1 line | unclassified, new 1 (`verify-r1/negative-classification.log`) |

Residual: D1 confirmation depends on the ASCII recheck replay, so a defect that
coincides with a D1 input and disappears under ASCII spelling would still be
classified D1.

## Limits

- The generated policies are weak, and the official episodes remain the only
  competitive worlds.
- Kaggle framework behavior outside the interpreter, such as timeouts, agent
  errors and `INVALID` statuses, is not modeled.
- The sweep is a bounded sample.
- A pod-scale sweep is still open.
- No training, GPU work or network access occurred. The isolated environment
  used the offline uv cache.
