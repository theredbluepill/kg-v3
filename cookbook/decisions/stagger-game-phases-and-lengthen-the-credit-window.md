---
type: "Decision"
title: "Stagger game phases and lengthen the credit window"
description: "Owner decision 2026-09-30 (\"- Lockstep game phases and the short credit window for long-payback investments. for sure.\"): fix both beside the per-seat critic. rl.initial_stagger (default false, omitted from the config dump) cuts each env's first game at a step drawn from env.seed and the global env index (uniform over 1..719) through the stateless truncation path; the critic bootstraps the cut, a cut publishes no bank telemetry, and later games run 719 transitions, so every rollout mixes phases and carries game ends (train/game_phase_frac_0..5, train/game_ends, train/stagger_cuts). The credit window is configuration: configs/kaggriculture_{4,2}rank_bank_critic_credit.yaml set horizon 256, gae_lambda 1.0 and the stagger with the owner-approved bank + margin + sign reward, scaling n_envs to 16/32 per rank and segments_per_minibatch to 1/2 so 16,384 env steps, 16 optimizer steps and the minibatch and teacher rows per iteration are unchanged. The values are the plan's F2/F3 targets on the v2 precedent, not owner-given; model.critic_offset is a follow-up. Default-off digests match the base. CPU checks only; nothing trained."
tags: ["kaggriculture-v3", "training", "decisions", "adaptation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30: \"- Lockstep game phases and the short credit window for long-payback investments. for sure.\""
sources: [{"resource": "user-directive:2026-09-30:lockstep-phases-and-short-credit-window-for-sure"}, {"resource": "user-directive:2026-09-30:switch-back-to-self-play-and-earn-money-for-real"}, {"resource": "repository:python/owl/train/ppo.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:configs/kaggriculture_4rank_bank_critic_credit.yaml"}, {"resource": "repository:configs/kaggriculture_2rank_bank_critic_credit.yaml"}, {"resource": "repository:tests/kaggriculture/test_initial_stagger.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:README.md"}, {"resource": "repository:src/kaggriculture/env.rs"}, {"resource": "repository:ops/stagger-credit-2026-09-30/baseline_digest.py"}, {"resource": "repository:ops/stagger-credit-2026-09-30/config_digest.py"}, {"resource": "repository:ops/stagger-credit-2026-09-30/digests.md"}, {"resource": "repository:ops/stagger-credit-2026-09-30/prepare.log"}]
---

# Stagger game phases and lengthen the credit window

## Decision

The owner, verbatim, on 2026-09-30, quoting the main agent's list of what a per-player critic does not fix:

1. **"- Lockstep game phases and the short credit window for long-payback investments. for sure."**
2. Earlier: **"let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?"**

The owner wants both problems fixed, alongside the per-seat critic offset that is being built on `kg/rebuild-critic-offset`.

**Interpretation, not owner adoption.** The mechanism and the numbers come from the orchestrating workflow's task, which follows the earn-money plan (`ops/earn-money-2026-09-30/plan.md` in the integration worktree: M2, M3, X2 "Stagger seam", F2, F3) and v2's precedent. They are not owner-given:
- the first-game stagger, uniform over 1..719;
- horizon 256 with `gae_lambda` 1.0 and gamma 1.0;
- the per-rank split of 16 × 256 at 4 ranks and 32 × 256 at 2 ranks.

v2 reported that staggered starts cut training-KL volatility from 1.98 to 0.45, and that 256-turn segments with λ = 1 gave the first clear own-bank and margin gain. This note has not re-checked those v2 numbers.

## Why

Every game lasts `episodeSteps - 1` = 719 transitions, and all envs reset together, with no `truncation_prob`. So each 64-step rollout trains a single game phase, and only about one rollout in eleven contains a game end (plan M2). With γ 1.0, λ 0.9 and horizon 64, credit reaches about 10 turns, while plant, animal and land investments pay back over 100-300 or more turns (plan M3). The per-seat critic changes what the value can represent. It changes neither of these.

## Design (as implemented)

**Stagger (`rl.initial_stagger: bool = false`).**
- **Offsets.** `initial_stagger_steps(seed, rank, n_envs, episode_steps)` draws one uniform `u` from `1..episode_steps - 1` per env. Each draw is seeded by `(env.seed, a stream tag, rank × n_envs + i)`. It is independent of the world size, and ranks differ.
- **Wiring.** `run_ppo._initial_stagger` passes the offsets to `PPOTrainer(initial_stagger=...)`. The trainer refuses a mismatch between the config and the offsets.
- **Seam.** The existing stateless truncation path (`_apply_truncation`) now compares each env's step counter with a per-env cut step, `_truncation_at`. That is `rl.truncation_step` for `truncation_prob` runs and `u` under the stagger. Only first games are flagged, and a new game is never flagged. At the cut:
  - the critic values the cut state;
  - `_cut_truncated_envs_` marks the row done and truncated, keeps the economic reward and stores the bootstrap;
  - `truncate_envs` resets the env.
- **End of the first game.** `u = 719` coincides with the natural end, so that first game completes normally, and the phase is uniform over all 719 residues.
- **Telemetry stays clean.** The native truncate path commits no `TransitionCache` and returns no metrics (`src/kaggriculture/env.rs`, `prepare_reset` with `truncate`). So a cut game adds nothing to `train/bank_games`, `train/own_bank_*` or `train/total_games_played`.
- **New telemetry.** Only under the stagger, gathered over ranks: `train/game_phase_frac_{0..5}` (acted observations per sixth of a game), `train/game_ends` and `train/stagger_cuts`.
- **Rules.** Kaggriculture only (`FullConfig`), `episodeSteps >= 2`, stateless models, and no combination with `truncation_prob` or `truncation_step`.
- **Resume.** A resumed launch restarts every env at step 0, and the same offsets desynchronize it again.
- **Opponent mix.** The stagger works with `env.opponent_mix`, because `truncate_envs` refreshes the learner mask. A test covers one hosted-bot env and one self-play env.

**Credit window (configuration only).** At γ = λ = 1, `compute_gae` already returns the Monte Carlo return:
- to the first `done` in the segment, plus the bootstrap at a cut;
- otherwise to the segment end, plus the critic value there.

**Presets.** `configs/kaggriculture_4rank_bank_critic_credit.yaml` is `kaggriculture_4rank_margin.yaml` (J/2's recipe) with these changes:
- **Reward.** The owner-approved reward from the critic branch's `kaggriculture_4rank_bank_critic.yaml` (agent-proposed values, owner "OK go ahead."):
  - `econ_bank_weight` 0.25, `econ_bank_scale` 150000, `econ_bank_cap` 0.25;
  - `econ_margin_weight` 0.25, `econ_margin_scale` 100000, `econ_margin_cap` 0.25;
  - `econ_shaping` 0, so `terminal_scale` 0.5.
- **Stagger and credit.** `initial_stagger` true, `horizon` 256, `gae_lambda` 1.0.
- **Per-rank split.** `env.n_envs` 16 and `segments_per_minibatch` 1.

`configs/kaggriculture_2rank_bank_critic_credit.yaml` is its twin with 32 envs and 2 segments. **`model.critic_offset` does not exist on this base.** When `kg/rebuild-critic-offset` lands, the presets' `model:` should switch to `kaggriculture_critic_offset`.

## Work and memory accounting

| Per iteration | Before (4 ranks: 64 × 64, spm 4) | Credit preset (16 × 256, spm 1) |
| --- | --- | --- |
| Global env steps | 16,384 | 16,384 |
| Optimizer steps | 16 | 16 |
| Env steps per optimizer step, per rank | 256 (4 segments × 64) | 256 (1 segment × 256) |
| Segments (envs) per global optimizer step | 16 | 4 |
| Minibatch forward rows per rank | 512 | 512 |
| Teacher chunk rows / cache per rank | 8,192 / 837,287,936 B | 8,192 / 837,287,936 B |
| Rollout forward | 64 calls × 128 rows | 256 calls × 32 rows |
| Last-best evaluation games (rank 0) | 64 | 16 |
| Expected game ends per iteration | ~23, clustered into 1 rollout in ~11 | ~22.8 in every rollout after the first cycle |

The 2-rank twin has the same totals: 1,024 minibatch rows, 16,384 teacher rows (1,674,575,872 B), and 32 evaluation games instead of 128.

**Measured on CPU:** the rows and bytes, through `ppo_forward_workloads` and `check_workload_headroom`. **Not measured:**
- GPU peak memory;
- rollout wall time, since the rollout runs 4× more sequential forward and native steps with ¼ of the rows each, and each native step parallelizes over 16 envs;
- the compile time of the 256-step unrolled GAE loop;
- the effect of 4 segments per global optimizer step on gradient variance.

The quarter-size evaluation makes the ≥ 0.7 promotion noisier. `rl.eval_games` is a possible follow-up.

## Verification (this version, CPU)

- **`tests/kaggriculture/test_initial_stagger.py`, 15 tests.**
  - The default-off dump omits the key.
  - Validation refuses the invalid combinations.
  - The offsets are deterministic, in range, independent of the rank split, and differ across ranks and seeds (61 of 64 distinct: birthday collisions).
  - A schedule check at 4 × 16 × 256, 2 × 32 × 256 and 4 × 64 × 64 finds game ends in every rollout after the first cycle, with a mean of about 22.8. Without the stagger, most rollouts have none.
  - **Native trainer.** Offsets [1, 3, 4, 5] on 5-transition games give:
    - cuts only in rollout 1, of envs 0-2, at `u - 1`, done and bootstrapped, and never later;
    - `train/game_ends` = `train/bank_games` = [1, 4, 4, 4];
    - uniform phase fractions afterwards;
    - policy inputs byte-equal to the env's published observation, with no hidden state;
    - the default-off trainer carries no stagger state or keys.
  - **With a fixed-opponent mix** (Starter in one of two envs): cuts [2, 0], game ends [0, 2], one game vs the bot and one self-play, and the learner seat switches at the cut.
  - GAE with λ = 1 over 256 steps equals the brute-force Monte Carlo return plus bootstrap, and a reward at step 255 moves the step-0 advantage one for one.
  - The presets keep the global work and the rows.
- **`tests/scripts/test_run_ppo.py::test_stagger_credit_two_update_run_through_main`.** The presets' reward, stagger and λ = 1 on the tiny CPU model through `run_ppo.main`: cuts [2, 0], game ends and bank games [0, 1], and the stagger recorded in `config.yaml`.
- **Default off is byte-identical.** Measured at `OMP_NUM_THREADS=2`:
  - the native digest `257eae38…` and the trainer digest `3ffd53a0…` equal the base `3e89425`;
  - every preset's `config_sha256` is unchanged.

  See `ops/stagger-credit-2026-09-30/digests.md`.
- **Full `just prepare`** exits 0: root Rust 298 passed with 5 ignored plus the engine and opponent crates, Python 2,960 passed with 9 skipped, plus ruff, mypy, markdown lint and docs-fresh (`ops/stagger-credit-2026-09-30/prepare.log`; the gitignored Orbit replay fixtures were copied from the integration worktree).

## Consequences and limits

- Nothing has been trained. Whether staggering or λ = 1 over 256 turns restores the investment chains (strawberry, sheep and land days) is the plan's F2/F3 question. It needs runs with a null twin and two seeds.
- λ = 1 raises advantage variance, with 4 segments per global step. The plan pairs F3 with a larger batch per step (F1), which is not done here.
- Offsets are independent draws, not stratified, so phases cluster slightly. At 256 envs every sixth of the game is covered, as tested.
- Reopen the stagger design if `train/game_ends` stays zero after the first cycle, or if the phase fractions stay skewed beyond sampling.
