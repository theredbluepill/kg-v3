Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: opponent-mix review r1 follow-ups (`0f70773..71daf36`)

- Branch `kg/rebuild-opponent-mix`, HEAD `71daf36` (one commit after `0f70773`,
  "Apply the opponent-mix review r1 follow-ups (tests, docs, receipts only)").
- Checks ran in a scratch detached worktree,
  `/private/tmp/claude-501/verify-oppmix-71daf36` (fresh `uv run maturin develop`,
  dev profile, torch 2.9.0, Python 3.12, Apple M5, `OMP_NUM_THREADS=2`).
  `/Users/poonszesen/kg-v3-oppmix` was not edited, apart from this report
  file, which is uncommitted. Its `git status` was clean before this file was
  written.
- Context: the owner asked, 2026-09-30, "Sure you can just launch the same
  reward on CHA22 run now?". The vs-cha22 run launches from `0f70773` before
  this lands, so the question is whether `71daf36` changes anything that run
  computes.
- Note: `ops/opponent-mix/review-r1.md` was uncommitted when this task was
  set, but `71daf36` commits it.

## 1. Nothing the live run computes changed

The runtime path trees are byte-identical between `0f70773` and `71daf36`.
`git rev-parse <rev>:<path>` gives the same tree/blob hash on both commits for
`scripts`, `python`, `src`, `opponents_rs/src`, `engine_rs`, `configs`,
`Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock` and `build.rs`. So
training math, rewards, evaluation logic, logged values, configs and the built
extension are all unchanged.

Every changed file (`git diff --name-status 0f70773..HEAD`, 14 files):

| Path | Kind | Runtime effect |
| --- | --- | --- |
| `opponents_rs/OPPONENT_MANIFEST.json` | non-test/non-doc: custody data | None. Only `authored[README.md].sha256` changes, to match the edited README. Only `scripts/check_opponent_import.py` (a custody check) reads it. `opponents_rs/src/native_agents.rs` mentions it only in a comment. |
| `opponents_rs/README.md` | doc (custody) | None. It adds the `agents/cha22/main.py` → original SHA-256 mapping and the Apache-2.0 redistribution gap (P3-3). |
| `ops/opponent-mix/r1_mutations.py` | non-test/non-doc: ops script | None. A mutation harness that is never imported by the package. |
| `ops/opponent-mix/{r1-mutations,digest-environment,prepare}.log`, `ops/opponent-mix/review-r1.md` | receipts | None. |
| `docs/rl-api-specs.md` | doc | None (P3-2 telemetry readings, P2-1 test pointer, P3-5 timeout note). |
| `cookbook/decisions/index.md`, `cookbook/log.md`, `cookbook/decisions/train-ppo-...-learner-mask.md`, `cookbook/references/cha22-opponent-imports-...md` | cookbook | None. |
| `tests/kaggriculture/test_opponent_mix.py`, `tests/scripts/test_run_ppo.py` | tests | None. |

Empirical cross-check (it covers only the self-play default path; the mix
path rests on the identical trees above): `ops/opponent-mix-2026-09-30/baseline_digest.py`
at HEAD reproduces the committed receipts:

- `OMP_NUM_THREADS=2` gives native `257eae38…`, trainer `3ffd53a0…`. That
  equals `post-digest.json`.
- `OMP_NUM_THREADS=4` gives trainer `efe76677…`, as
  `digest-environment.log` records.

This also confirms the P3-4 receipt.

## 2. Mutations M4 and M5 are now caught

For each mutation I applied it in the scratch worktree, ran the named test and
restored the file with `git checkout`. The worktree was clean afterwards.

| Mutation | Test | Result |
| --- | --- | --- |
| M4: `own`/`opponent` swapped (`scripts/run_ppo.py:2102-2103`) | `test_fixed_bot_evaluation_attributes_banks_to_the_learned_seat` | CAUGHT (`assert 0.0 == 1.0`) |
| M5: `values[rows.flip(0)] = row_values` (`python/owl/train/ppo.py:3397`) | `test_forward_learner_rows_matches_the_full_batch_rows` | CAUGHT (`Tensor-likes are not close`) |

The branch's own `ops/opponent-mix/r1_mutations.py`, run independently here,
gives the same two CAUGHT lines as `r1-mutations.log`.

I also ran my own variants (not in the branch):

| Variant | Result |
| --- | --- |
| M4b: seat label flipped only (`seats.append(1 - seat)`) | CAUGHT (per-seat win rate `0.0 == 1.0`) |
| M4c: seat read from the post-step `env.learner_mask` | CAUGHT (the existing guard raises "env 0 must have exactly one learned seat") |
| M4d: opponent bank only swapped | CAUGHT (`0.5 == 1.0`) |
| M5b: `values[rows.roll(1)]` | CAUGHT |

Test quality:

- The M4 test runs the real `_evaluate_against_bot` on the real hosted-Cha22
  env, with 3-step games and 4 envs. It replaces only the terminal banks, with
  seat-distinct values.
- It pins the reset learner pattern (`[[F,T],[T,F]]*2`), per-seat counts, win
  rates, own and opponent means, the margin, and that `model.training` is
  restored.
- The M5 test now plays four sampled steps over mixed hosted and self-play
  envs. It asserts that all 6 learned values are distinct, so a permutation
  cannot match by coincidence.
- It also asserts that scripted rows get zero actions, log-probs and values.

## 3. Fast tests, lint, docs-fresh

Scratch worktree, HEAD `71daf36`:

- **Tests.** `uv run pytest -q tests/kaggriculture/test_opponent_mix.py
  tests/scripts/test_run_ppo.py tests/tools/test_check_opponent_import.py
  tests/owl/kaggriculture/test_opponents.py` gave 286 passed and 1 skipped. The
  skip is the known sibling-repo original-sources check.
- **Custody check.** `uv run --offline python scripts/check_opponent_import.py`
  passed, so the rehashed manifest matches the README.
- **Lint.** `uvx ruff check` and `uvx ruff format --check` on the two test
  files and `r1_mutations.py` both passed.
- **Markdown lint.** `uvx pymarkdownlnt scan docs/rl-api-specs.md
  opponents_rs/README.md` gave exit 0.
- **docs-fresh.** `just` is not on PATH here, so I ran the recipe's command,
  `uv run python scripts/check_doc_freshness.py`. The script diffs the working
  tree against HEAD, so I ran it after `git reset --soft 0f70773`, which
  exposes all 14 changed paths. It printed "No doc updates required" (exit 0).
  I then restored HEAD to `71daf36`.
- **Cookbook lint.** With the same soft reset, `node
  .claude/hooks/cookbook-lint.mjs --staged-sources --require-log` gave exit 0.
- **Branch prepare log.** The branch's `ops/opponent-mix/prepare.log` records a
  full `just prepare` with 2934 passed, 9 skipped, `EXIT=0`. I did not re-run
  the full suite.

## Documentation accuracy spot-checks

- **P3-2.** `train/return_common_mean` and `train/return_zero_sum_abs_mean` use
  `_masked_mean` over `both_seats` (`ppo.py:855-863`). With an empty mask the
  distributed path returns `0 / clamp_min(1e-8)` = 0, which is finite, as
  documented.
- **P3-2, reward keys.** `reward_bank_mean` is exactly `0.0` when
  `econ_bank_weight == 0` (`kaggriculture/env.py:499-510`), as documented.
- **P3-5.** `distributed.py:109` calls `init_process_group(backend="nccl",
  device_id=...)` with no timeout, so the torch default applies. The 10-minute
  figure is the torch NCCL default; I did not re-derive it from the torch 2.9
  source. The 33 s last-best pause is marked as the orchestrator's report, not
  re-measured.

## Findings

No P1 or P2. Residual items, informational only and not blocking:

- **P3-5 stays open.** The vs-bot evaluation wall time is still unmeasured,
  and the docs correctly make it a first-checkpoint check.
- **P3-2 telemetry.** It is documented, not fixed. That is correct while the
  run is live.
- **P3-4 base re-run.** The `0f70773` base was not re-run at two threads (the
  cookbook says so). The HEAD re-run here matches `post-digest.json`, and the
  runtime trees are identical, so this has no consequence.

VERDICT: APPROVE
