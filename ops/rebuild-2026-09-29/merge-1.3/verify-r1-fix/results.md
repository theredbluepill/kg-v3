# Merge 1.3 verification r1 fix: engine inputs in oracle source custody

Finding (Codex verification of `e197528`, REJECT, one P2): the observation-oracle
source snapshot in `scripts/kaggriculture_observation_oracle/regenerate.py`
hashed engine `TRIM_MANIFEST.json`, `src/lib.rs` and `Cargo.lock` but not the live
build inputs `engine_rs/Cargo.toml` and `engine_rs/src/py_random.rs`. Edits to
either survived every recheck and the regenerated output was installed.
Verifier report: `ops/rebuild-2026-09-29/verify-e197528-independent/review.md`
(untracked on this branch at the time of the fix) and
`kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-1-3-r1.md`.

## Repair

- `ENGINE_SOURCE_PATHS` adds `engine_rs/Cargo.toml`, `engine_rs/src/econ_attrib.rs`
  (also a compiled module, `mod econ_attrib;`) and `engine_rs/src/py_random.rs` to
  the captured and rechecked set. They are recorded in `source_identity.dirty_files`,
  so the strict `engine` key schema and the committed corpus stay valid.
- `require_declared_engine_sources` runs at capture and at every recheck: the set of
  `engine_rs/src/**/*.rs` (plus `engine_rs/build.rs` if present) must equal the
  declared engine sources, so a new engine module fails regeneration.

## Test-first evidence

`tests/tools/test_observation_oracle_custody.py` gains seven cases:
three drift regressions (py_random.rs at producer, engine Cargo.toml at admission,
econ_attrib.rs at export), identity records every engine build input, undeclared
engine module rejected before and during execution, and the declared set covers the
live repository engine crate.

- `red.log`: before the repair, 7 failed, 46 passed.
- `green.log`: after the repair, 53 passed.

## Limits

- The committed corpus manifest was generated before this repair, so its identity
  does not record the engine `Cargo.toml`, `py_random.rs` and `econ_attrib.rs`
  hashes. `git diff 469e8ec HEAD` over those files and engine `src/lib.rs` is empty,
  so their committed bytes equal those at the corpus root commit; any uncommitted
  working-tree state at generation time is unrecorded. No regeneration was run.
- Root-crate modules outside `src/lib.rs` and `src/kaggriculture/` custody paths that
  compile into the producer's test binary (e.g. `src/kaggriculture/grammar*.rs`,
  other root modules) are not enumerated; the verifier did not raise them.

## Full check

`CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` exit 0
(`prepare.log`): root Rust 254 passed with four ignored; engine 41 + 9 + 19 = 69;
Python 1,625 passed with seven skips (1,618 before plus the seven new cases). The
first attempt failed ruff E501 on the new message line; the wrapped line passed.
