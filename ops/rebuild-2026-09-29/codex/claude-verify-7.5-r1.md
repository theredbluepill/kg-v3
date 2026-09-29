Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify Task 7.5 r1: Kaggriculture parity coverage summary

Date: 2026-09-30. Target: branch `kg/rebuild-7-5` at `e9aafba0f0cc8a18d7ecd15b4c075f64f9dfdbd8`
(worktree `/Users/poonszesen/kg-v3-t75`, not modified). Scratch: detached worktree
`/tmp/cv-7.5-r1` at the same HEAD. Fixtures `generation/` and `orbit_wars_replays/` copied in from
the main checkout; `owl.rs` built there with `CARGO_BUILD_JOBS=2 uv run --offline maturin develop --release`
(exit 0, own target dir).

## Scope

- `git diff faed717...HEAD` covers the Tasks 1.4/1.5 merge (up to `bde3374`, already Codex-approved in
  `codex/verify-merge-env-adapter-r2.md`, VERDICT: APPROVE) plus the two Task 7.5 commits.
  I reviewed the Task 7.5 commits in full: `a8afb33` (docs section, eight in-place edits, `7.5/` receipts, cookbook
  note/index/log) and `e9aafba` (the record that Codex's verification hit its usage limit). I did not re-review
  the approved merge. My test runs below exercise it.
- Spec: plan Task 7.5 ("`docs/rules-parity-coverage.md` gains a Kaggriculture section stating what is tested
  and what isn't"), invariant I9, and the Codex brief `codex/task-7.5-prompt.md` (which explicitly permits the
  labelled BC branch item). Governing rule: every claim traces to a test or receipt at the integration tip,
  and gaps are stated explicitly.

## Claim audit (against source at HEAD)

- Paths: every backticked file, test and receipt path in the new section exists (script check, including the
  short names under `tests/kaggriculture/` and `src/kaggriculture/admission.rs`). These named entry points exist:
  `episode_*` (4), `generated_fixtures_replay`, `env_directory_traces`, `compare_observation_oracle`,
  `test_every_frozen_oracle_record_passes_the_actual_schema`, `test_native_matches_training_batch_16_complete_games`,
  `test_project_environment_satisfies_the_engine_pin` and `load_pinned_kaggle`. Commit `933d661` exists and is not an
  ancestor of HEAD, as the page says.
- Anchors: all 7 distinct in-page links resolve to unique GitHub slugs, and there are no duplicate headings.
- Pins: `pyproject.toml:11` and `uv.lock:1774-1775` give 1.32.7. `engine_rs/Cargo.toml:10-11` gives the version and
  the SHA-256 printed. The reference commit `65f0eac5…` is in `check_engine_trim.py:21` and `TRIM_MANIFEST.json:3`.
- Numbers recomputed from the fixtures and receipts:
  - The generated manifest has 8 non-divergence traces and 3,960 transitions, plus 7 repros.
  - The sweep summary has 40 games, 21,824 game transitions, 303 probes and 23,339 transitions in total
    (1,515 probe transitions). There are 306 agreeing traces (266 probes, since all 37 divergences are probes),
    split D2 25 / D1 12.
  - The grammar manifest has 320 accepted programs (256 sampled + 64 replay) and 44 mutations.
  - The observation manifest `reference_shape` is [512, 2, 8176], and its sources are 384+96+32 = 512.
  - The env reference fixture has 16 games, seeds 17000-17015 and 719 steps per game (11,504).
  - The codec corpus totals 321/43 (`test_native_grammar_bindings.py:266,270`).
  - The extreme reward test has 3 ids × 96 steps, with `episodeSteps` 97.
  - The admission predicate table has 11 rows (`env_tests.rs:64-76`).
  - The 10 py-prepare skips break down as 1 CUDA + 2 pinned + 2 flash-attn + 1 x86 + 4 Task 3.1 seam.
  - `run_ppo.py:181-186` raises for Kaggriculture at the Task 3.1 seam.
  - `phase-status.md:118-121` supports the 7.1/7.3/7.4 status wording.
- The root Rust carry-forward is supported. `git diff --stat 7f797a3 HEAD -- '*.rs' '*Cargo.toml' '*Cargo.lock'` is
  empty, and `merge-env-adapter/prepare.log:477` shows 274 passed, 0 failed, 5 ignored.
- The BC day-end hedge is conservative and correct. All six mismatch turns are ≡ 23 mod 24. The official traces record
  `turnsPerDay: 24`, but the sampled BC episode configurations are not tracked, so declining to confirm is right.
- The 14 "not tested" bullets are specific. None of them is contradicted by a test at this tip.

## Checks (scratch, HEAD e9aafba)

| Command | Result |
| --- | --- |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 41 + 9 + 19 + 0 doc = 69 passed, 0 failed, 0 ignored; max RSS 270 MiB |
| `uv run --offline --no-sync python scripts/check_engine_trim.py` | exit 0, `engine trim manifest: OK` |
| `uv run --offline --no-sync python scripts/check_doc_freshness.py` | exit 0 (clean tree) |
| `pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py` | 97 passed, 0 skipped (live regeneration ran); max RSS 500 MiB |
| `pytest test_env_reference.py test_native_env.py test_env.py` | 400 passed; max RSS 282 MiB |
| `pytest test_codec.py test_native_tables.py test_game.py test_rewards.py test_native_grammar_bindings.py` | 163 passed; max RSS 282 MiB |
| `pytest tests/kaggriculture/test_observe.py` | 55 passed, 2 skipped (pinned memory needs CUDA) |
| `pytest tests/kaggriculture` (whole directory) | 1,088 passed, 7 skipped; **max RSS 1.94 GiB** (over the 1 GB Mac limit; see P3-1) |
| cookbook lint (PostToolUse mode) on the new note | no problems (`{}`) |
| cookbook lint `--staged-sources --require-log` over bde3374..e9aafba | exit 0 |

I did not run `just py-prepare` or the root `cargo test` myself. The whole-directory pytest run already exceeded
1 GB, and Codex's root run tripped its 960 MiB guard. The change is docs, receipts and cookbook only, so the cited
shards cover every test entry point the page names.

## Mutations (scratch only; each restored with `git checkout`)

Task 7.5 adds no new code guard or oracle. So each mutation targets an oracle or guard the page claims is tested,
to check that the claim is not vacuous.

| # | Mutation | Claim probed | Caught by |
| --- | --- | --- | --- |
| M1 | `replay_parity.rs` `order_difference`: key-order check disabled (`if false && …`) | "recursive object key order" | 5 failures: `private_order_mismatch_is_rejected`, `public_order_mismatch_is_rejected`, `nested_private_order_mismatch_is_rejected`, `rollback_check_rejects_a_key_order_only_mutation`, `generated_trace_perturbed_private_order_is_rejected` |
| M2 | `rollback_difference`: statuses and rewards comparisons disabled | "Rejected actions must leave the checked state unchanged" | `rollback_check_rejects_status_reward_and_done_mutations` |
| M3 | `engine_rs/Cargo.toml` `python-engine-sha256` last hex digit changed | installed pin/hash test; trim checker's Cargo-pin consistency | `test_project_environment_satisfies_the_engine_pin`, `test_live_kaggle_engine_regenerates_committed_traces`; `check_engine_trim.py` exit 1 (`engine_rs/Cargo.toml: current hash`) |
| M4 | Cargo `kaggle-environments-version` changed to 1.32.6 | version rejection | `test_project_environment_satisfies_the_engine_pin`, `test_live_kaggle_engine_regenerates_committed_traces` |
| M5 | `python/owl/kaggriculture/rewards.py`: drought term ×0.5 | Task 1.5 reward arithmetic vs independent oracle | 4 failures in `test_rewards.py`, including `test_native_fixture_rewards_match_independent_oracle` and `test_extreme_value_native_rewards_match_independent_oracle[tiny-W-overflowing-inner-sum]` |
| M6 | `generate_traces.verify_engine`: hash comparison disabled | "generator refuses a different … hash" | `test_hash_guard_accepts_only_the_pinned_engine` |

All six mutations were caught, and none survived. Under M5, `test_env_reference.py` still passed. That is consistent
with the page's claim that its reward formula is independent of `rewards.py`.

The end-to-end "refuses before writing" test (`test_generator_refuses_before_writing_when_the_pin_differs`) covers
only the version path. The hash path is covered by the `verify_engine` unit test (M6) plus the call order in
`main` (`generate_traces.py:1118-1123` runs before any write). The page's wording ("checks version/hash rejection")
is accurate at that granularity.

## Findings

No P1 or P2 findings. Every number, path, test name and anchor I checked in the new section matches the source.
The gaps are explicit.

- **P3-1 (process and claim precision)**
  - Where: `docs/rules-parity-coverage.md:249-251` ("Checks run … use the owner's offline CPU/thread limits … Claude
    then ran `just py-prepare` unguarded").
  - Problem: the py-prepare row has no receipt for the offline/thread environment. `7.5/py-prepare.log:1-8`
    shows uv rebuilding and reinstalling `owl`, and no env export is recorded. The whole `tests/kaggriculture`
    directory alone peaked at 1.94 GiB RSS in my scratch run, so the unguarded full-suite run very likely
    exceeded the owner's 1 GB Mac check limit. That is the limit Codex's guard was enforcing.
  - Fix: scope the environment sentence to the Codex-run rows. State that the py-prepare row's environment and
    peak memory were not recorded. In future, use targeted shards (as here) instead of an unguarded full run.
- **P3-2 (stale receipts)**
  - Where: `ops/rebuild-2026-09-29/7.5/structure-check.txt:5` and `ops/rebuild-2026-09-29/7.5/results.md`.
  - Problem: `structure-check.txt:5` reports "Seven coverage rows and five current-command rows: PASS", but the
    committed table has four command rows after Claude's replacement. `results.md:11-13` says no cookbook edit was
    made, and its "Unverified and omitted" list says full Python totals are absent. The appended Claude review
    supersedes both, but the earlier lines are not marked as superseded.
  - Fix: add a one-line note that `structure-check.txt` describes Codex's pre-review draft. Mark the two
    `results.md` statements as superseded by the Claude review section, keeping the original text.
- **P3-3 (cookbook wording)**
  - Where: `cookbook/log.md:5` ("Plan Task 7.5 … is done") and the note's Changed paths, which says "eight minimal
    corrections of statements that are stale".
  - Problem: the task has no independent verdict (the same entry says so), and the plan checkbox is unticked.
    Also, C01 in `claims.md` is an added pointer, not a stale-statement correction, so it is really seven
    corrections plus one pointer.
  - Fix: say "implemented on `kg/rebuild-7-5`, pending independent verification", and "seven corrections plus
    an opening-paragraph pointer".
- **P3-4 (BC custody)**
  - Where: `docs/rules-parity-coverage.md:212-229`.
  - Problem: the BC paragraph traces only to a commit on unmerged `kg/rebuild-bc-now`. The brief permits this, and
    `7.5/bc-audit.txt` at the tip records the aggregates and the `pairing.json` SHA-256 (`a2c2db57…`). But the page
    itself cites no tip-resident receipt. If that branch is rebased or deleted, the citation dangles.
  - Fix: cite `ops/rebuild-2026-09-29/7.5/bc-audit.txt` in the BC paragraph as the tip-resident receipt (it holds
    the SHA and sums).
- **P3-5 (merge-time follow-up, not a branch defect)**
  - Where: `ops/rebuild-2026-09-29/phase-status.md:122` still lists 7.5 as "not started".
  - Fix: update that row at landing, as the tracker requires. The page and the note already state that a refresh
    is needed when 7.1, 7.3, 7.4 or 3.1 merges.

## Owner-rule check

This change is docs only. It adds no trainer, model or v2 code. The page states that `scripts/run_ppo.py` stays the
canonical trainer and stops at the Task 3.1 seam. Nothing touches the policy's inputs or state. No deadline or
submission timing appears.

## Cleanup

`git -C /Users/poonszesen/kg-v3-t75 worktree remove --force /tmp/cv-7.5-r1` exited 0, and the path is gone.
`/Users/poonszesen/kg-v3-t75` is still at `e9aafba` with 0 tracked modifications
(`git status --short --untracked-files=no` is empty). No push, commit or branch change was made.

VERDICT: APPROVE WITH EDITS
