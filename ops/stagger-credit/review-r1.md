Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Review r1: kg/rebuild-stagger-credit (initial stagger + credit-horizon presets)

- **Target.** Branch `kg/rebuild-stagger-credit`, HEAD `ba2c089`, base `3e89425`. The diff is 22 files, +2232/-13.
- **Method.**
  - All work ran in scratch detached worktrees: `/private/tmp/kg-stagger-review` at `ba2c089` and `/private/tmp/kg-stagger-base` at `3e89425`. Each used the stagger worktree's venv and a copy of its `rs.abi3.so`.
  - `/Users/poonszesen/kg-v3-stagger` was not edited. The only exception is this file, which is not committed.
  - I read `ops/earn-money-2026-09-30/plan.md` (M2, M3, X2 "Stagger seam", F2, F3) in the integration worktree.
  - I also ran a trial merge against `kg/rebuild-critic-offset` (`732438c`) and then aborted it.
- **Scope.** Fast CPU checks only. Nothing was trained, and no GPU run was made.

## What I checked, and the results

| Check | Result |
| --- | --- |
| New tests: `test_initial_stagger.py` and `test_run_ppo.py` | 187 passed in 6.9 s |
| Default off, native and trainer digests (`OMP_NUM_THREADS=2`, the branch's `baseline_digest.py` run against both trees) | Base `3e89425` and HEAD are both native `257eae38…a590` and trainer `3ffd53a0…fa2`. They are equal and match `digests.md` |
| Default off, **the existing `truncation_prob` path**, which this change rewrote to use the per-env `_truncation_at`. This path had no receipt, so I wrote my own digest: 4 updates, `truncation_step` 1, `truncation_prob` 0.6, horizon 4, λ = 1, with 6 cut rows exercised; it hashes the metrics, `truncated`, `bootstrap_values` and the weights | Base and HEAD are both `1df44212…c058`. Byte-identical |
| Default-off `config_sha256` of every preset | Equal (`config-digest-*.json`). The key is omitted by the wrap serializer (`ppo.py:214-221`), the same pattern as `kaggriculture/config.py:95` |
| Determinism and distinctness of the offsets | Seeded by `(seed, tag, rank*n_envs+i)` (`ppo.py:301-307`). Deterministic, independent of the world size, and different across ranks and seeds (tested) |
| Only the first game is cut | The flag is cleared on a natural end or a cut (`ppo.py:1339-1343`). `u = 719` coincides with the natural end and completes normally. Tested with offsets [1, 3, 4, 5], and mutation M1 is caught |
| Bootstrap at the cut | The critic is evaluated on `next_obs` rows before `truncate_envs` overwrites them (`ppo.py:1395-1413`). `dones` and `truncated` are set, and the transition reward is kept. With γ = λ = 1 the GAE delta is `r + V_boot - V` (`advantages.py:204-211`). A brute-force Monte Carlo check passes. M3, M7 and M12 are caught |
| Cut games stay out of bank and eval metrics | `train/bank_games` = `game_ends` = [1, 4, 4, 4], and `total_games_played` is 13. The native truncate path publishes no `TransitionCache`. Evaluation is a separate `n_games` path (`run_ppo.py:1690`) with no trainer truncation |
| No history or identity in observations | The policy input is byte-equal to the env's published observation, and `hidden_state` is None at every step (tested). The offsets and counters are trainer-local and are not checkpointed |
| Preset arithmetic: divisibility, samples per step, teacher rows | Correct. 16×256×4 = 32×256×2 = 16,384 env steps. There are 16 optimizer steps, and 256 env steps (512 seat rows) per rank per step. The teacher chunk is min(128, 16)×256×2 = 8,192 rows, 837,287,936 B, the same as the base (tested through `ppo_forward_workloads`) |
| Interaction with `opponent_mix` | Works. The learner mask is refreshed after `write_step`, so a cut row keeps the old game's seat (tested with Starter). A hosted-controller reset on a new episode is covered on the Rust side (`opponent_env_tests.rs:261`) |
| Post-merge interaction with critic-offset | On that branch `compute_value` includes the seat offset (`kg-v3-critic python/owl/model/kaggriculture.py:1201-1210`), so the cut bootstrap stays on the same value scale after the merge. `ppo.py` and `run_ppo.py` auto-merge. Four files have textual conflicts (P3-4) |
| Docs and cookbook | `rl-api-specs.md`, `kaggriculture-contract.md`, `README.md`, and a Decision with index and log entries. Owner quotes are verbatim, and interpretation is kept separate from adoption. `prepare.log` ends `EXIT 0` (2,960 passed) |

### Mutations (16 applied, one at a time, each reverted)

| # | Mutation | Result | Caught by |
| --- | --- | --- | --- |
| M1 | Stagger resample keeps the flag, so later games are cut again | caught | `test_stagger_cuts_only_first_games…` |
| M2 | Offsets ignore the rank (`rank*n_envs+env` becomes `env`) | caught | `test_offsets_are_deterministic…` |
| M3 | Bootstrap read after `truncate_envs`, so from the post-reset observation | caught | `test_stagger_cuts_only_first_games…` |
| M4 | Phase bucket read after the step increment | caught | `test_stagger_cuts_only_first_games…` |
| M5 | A cut is counted as a game end | caught | `test_stagger_cuts_only_first_games…` |
| M6 | The default-off dump includes `initial_stagger` | caught | `test_default_off_is_omitted…` |
| M7 | The cut drops the transition reward | caught | `test_training_smoke::test_truncation_bootstraps…` |
| M8 | Cut one step late (`u+1`) | caught | `test_stagger_cuts_only_first_games…` |
| M10 | Offsets drawn with `endpoint=False`, which loses the natural-end residue | caught | `test_offsets_are_deterministic…` |
| M12 | GAE drops the truncation bootstrap | caught | `test_gae_lambda_one…` |
| M13 | Orbit accepts `initial_stagger` | caught | `test_stagger_is_kaggriculture_only…` |
| M14 | 4-rank preset λ set back to 0.9 | caught | `test_credit_presets…[4rank]` |
| M15 | Stagger allowed together with `truncation_prob` | caught | `test_stagger_cannot_combine…` |
| M16 | The default-off trainer enables truncation | caught | `test_default_off_trainer…` |
| M9 | `run_ppo._initial_stagger` passes `rank=0` on every rank | **survived** (187 passed) | none (P3-2) |
| M11 | `_stagger_metrics` skips `all_reduce_sum` | **survived** (187 passed) | none (P3-2) |

14 of the 16 mutations were caught.

## Findings

### P1

None.

### P2

- **P2-1. The "bank_critic" presets ship the own-bank reward on the zero-sum critic.**
  - **Where.** `configs/kaggriculture_4rank_bank_critic_credit.yaml:79` and `configs/kaggriculture_2rank_bank_critic_credit.yaml:37` (`model: kaggriculture`).
  - **The problem.** At λ = 1 the critic enters the advantage **only** through bootstraps: every 256-step segment end, plus every stagger cut. Those are exactly the places where the zero-sum critic (V0 + V1 = 0) cannot carry the common-mode remainder of the own-bank term. Both seats' remaining own-bank payment is ≥ 0, but the pair's bootstrap sums to 0. So a launch of these files as committed has a biased bootstrap in both seats, and the file names suggest a bank critic that is not there.
  - **Mitigation already in place.** The header (`:44-46`) and the cookbook say to switch `model:` when `kg/rebuild-critic-offset` lands, and `kaggriculture_4rank_bc_finetune_bank.yaml` is an earlier precedent for a bank reward on the zero-sum critic.
  - **Required at merge.** Switch both presets to `kaggriculture_critic_offset`. Update the undo-diff in `test_credit_presets_keep_the_global_work_and_per_rank_rows` (`tests/kaggriculture/test_initial_stagger.py`), which currently compares against a margin preset with the same `model:`. Do not launch these presets before that. A merge-time test asserting that a `*_bank_critic*` preset uses the offset model would make this enforced rather than a note.
- **P2-2. At 4 ranks the last-best promotion gate drops from 64 to 16 games.**
  - **Where.** `scripts/run_ppo.py:1690` (`n_games=cfg.env.n_envs`), with the header at `configs/kaggriculture_4rank_bank_critic_credit.yaml:40`.
  - **The effect.** At the 0.7 threshold, a student that is truly 50/50 is promoted by chance with probability 3.8% at 16 games, 1.0% at 32, and 0.08% at 64. At a true win rate of 0.6 those become 16.7%, 11.6% and 5.8%.
  - **Why it matters.** The teacher (`last_best`) anchors the teacher-KL term. So a credit arm and the horizon-64 control would train under different teacher-refresh noise, which confounds the F2/F3 attribution the plan asks for.
  - **Status.** Disclosed, and `rl.eval_games` is listed as a follow-up. Before the arms are compared, either add `rl.eval_games` (keep 64) or record the confound in the run's question and loss conditions.

### P3

- **P3-1. Fewer independent games per optimizer step.**
  - **The change.** Samples per optimizer step keep their row count, but independent games per global step fall from 16 to 4. That is one 256-step segment per rank (`segments_per_minibatch: 1`). Global per-minibatch advantage normalization (`ppo.py:3802-3829`) then centres and scales over only 4 games, and λ = 1 raises the variance.
  - **The risk.** If a credit arm degrades, the result cannot separate "longer credit" from "4× fewer trajectories per step". The limits section and header disclose this. The plan's F3 pairs it with F1's larger batch, and the preset does not.
  - **Suggestion.** Add a twin or a logged diagnostic that separates the two.
- **P3-2. Two wiring points are untested (M9 and M11 survived).**
  - Add a unit test that calls `run_ppo._initial_stagger(cfg, DistributedContext(rank=1, …))` and asserts that it equals `initial_stagger_steps(rank=1)` and differs from rank 0.
  - Add a test that monkeypatches `all_reduce_sum` in `_stagger_metrics` (`ppo.py:1326`).
  - A rank-0-only offset bug would re-lockstep ranks with each other. A missing reduction would report rank 0's phases as if they were global.
- **P3-3. Factual slips about the environment count.**
  - `configs/kaggriculture_4rank_bank_critic_credit.yaml:16` says "at 256 global envs". The preset has 16 × 4 = 64 global envs. The 22.8 game ends are right (64 × 256 / 719).
  - `cookbook/decisions/stagger-game-phases-and-lengthen-the-credit-window.md:116` says "At 256 envs every sixth of the game is covered, as tested". The presets draw 64 offsets, and the test covers 256. With 64 draws the chance of an uncovered sixth is about 5e-5, so the practical claim holds, but it is not what was tested.
  - The comment at `tests/kaggriculture/test_initial_stagger.py:193` has the same "256 envs" slip; that test also covers (2, 32) and (4, 16) layouts.
- **P3-4. The merge with `kg/rebuild-critic-offset` (`732438c`) has textual conflicts.**
  - The conflicting files are `docs/rl-api-specs.md`, `tests/scripts/test_run_ppo.py`, `cookbook/log.md` and `cookbook/decisions/index.md`.
  - After resolving them, run both branches' new tests on the merged tree, including one staggered update with the offset model, where the cut bootstrap includes the offset.
- **P3-5. Throughput and compile time at horizon 256 are unmeasured, as disclosed.**
  - Each iteration runs 256 sequential rollout forwards of 32 rows, with 16 envs over 4 native threads.
  - `compile_compute_gae` (`advantages.py:57`) compiles a Python loop that unrolls to 256 steps.
  - Per CLAUDE.md, measure the equivalent complete work (valid learner turns per full update, and the time per phase) in a bounded pod capture before the attribution arms. This is not a gate.

## Verdict rationale

- **No correctness defect was found in the seam.**
  - Default off is byte-identical, for both the plain path and the existing `truncation_prob` path.
  - The cut, the bootstrap, the metric hygiene, statelessness, determinism, `opponent_mix` and the preset arithmetic all hold under the tests and 14 caught mutations.
- **The P2 items are merge and launch conditions, not code defects on this branch.**
  - P2-1 cannot be fixed before the critic-offset branch exists on this base.
  - P2-2 is disclosed.
- **Conditions for approval.**
  - P2-1 is resolved in the merge commit.
  - No arm launches before that.
  - P2-2 is either fixed or recorded as a confound in the S1/F3 run question.

VERDICT: APPROVE
