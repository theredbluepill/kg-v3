---
type: "Decision"
title: "The Kaggriculture v3 board"
description: "Living ranked board of v3 training options, first version 2026-09-30. Lead: continue mirror self-play at Muon 1e-4 with the 720-step whole-game window (lambda 1, 12 envs/rank) from the latest promotion (f61006d9; pcy5knet running). Then the 256-step window, better BC or imitation fidelity (open, X1a), and the landed but unrun per-seat critic offset and stagger. Scoped tombstones: Muon 2e-3 (collapse in pz3xhg9e), the 64-step lambda-0.9 window (every run slid), from-scratch starts (bank 0), and fixed-opponent cha22 training (owner: back to self-play). No v2 ranking is imported."
tags: ["kaggriculture-v3", "board", "training", "decisions"]
status: "current"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30: \"let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?\"; \"OK go ahead.\" (reward); \"ok go with 720 and relaunch the run\"; \"let eval complete, I pretty much sure it will promote, but let's run 2e-3 /1e-4 on the new run\"; \"Sure please do\" (1e-4 relaunch)"
sources:
  - resource: "user-directive:2026-09-30:switch-back-to-self-play-and-earn-money-for-real"
  - resource: "user-directive:2026-09-30:ok-go-ahead-bank-margin-sign-reward"
  - resource: "user-directive:2026-09-30:ok-go-with-720-and-relaunch-the-run"
  - resource: "user-directive:2026-09-30:let-eval-complete-and-run-2e-3-on-the-new-run"
  - resource: "user-directive:2026-09-30:sure-please-do-relaunch-at-1e-4"
  - resource: "repository:cookbook/references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving.md"
  - resource: "repository:cookbook/references/earn-money-pod-runs-use-run-local-launch-copy-off-and-switch-scripts.md"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/results.md"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/tables-best-f610.md"
  - resource: "repository:ops/earn-money-2026-09-30/x1/summary.md"
  - resource: "repository:ops/earn-money-2026-09-30/plan.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnlr/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnB/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/vs-cha22-4rank/stop.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-lr2e3-4rank/launch.md"
  - resource: "repository:configs/kaggriculture_4rank_bank_critic_credit.yaml"
  - resource: "wandb-run:spoon/kg-v3/pcy5knet"
  - resource: "wandb-run:spoon/kg-v3/pz3xhg9e"
---

# The Kaggriculture v3 board

## Position, 2026-09-30

The owner's words (verbatim):
- "let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?"
- "OK go ahead." (the agent-proposed .25 own bank /150k + .25 cash difference /100k + .5 terminal sign reward)
- "ok go with 720 and relaunch the run"
- "let eval complete, I pretty much sure it will promote, but let's run 2e-3 /1e-4 on the new run"
- "Sure please do" (relaunch at 1e-4 after the 2e-3 collapse)

The owner chose those actions. The ranking and the loss conditions below are the agent's reading of the evidence. They are not owner adoption.

Currently pulled: **option 1**, because it is the only line with two successive promotions and a paired anchor gain at each. `pcy5knet` is running.

**Common yardstick.** Local Kaggle-harness games on seeds 93001-93008, both seats, 16 games per anchor (`ops/earn-money-2026-09-30/anchor-games/`). The margin is own bank minus anchor bank:

| Policy | vs smaller_market_shock | vs cha22 |
| --- | --- | --- |
| BC `fd854587` | -67.2k | -60.6k |
| fc6b (256 window) | -41.2k | -36.0k |
| f610 (720 window) | -21.1k | -28.8k |

No policy has won a game against cha22, smaller_market_shock or v43. f610 did not play v43.

## Ranked options

**1. Continue Muon 1e-4 with the 720 window from the latest promotion (lead).**
- **Recipe.** Self-play, the owner-approved reward, `rl.horizon=720` (one whole game per segment), λ 1.0, 12 envs/rank, Muon 1e-4 / AdamW 5e-6, start f610 (`f61006d9…`), W&B `pcy5knet`.
- **Evidence.** `cmwjclbe` (720, from fc6b) promoted f610 at 20M. Paired on the same games, f610 beat fc6b's margin in 23/32 (+13.7k mean) and BC's in 32/32. `pcy5knet`'s self-play own bank holds 90-96k at 26.3M steps, with no evaluation yet.
- **Loss conditions (agent-proposed).** Either of these moves it down:
  - no promotion at two consecutive 10M evaluations;
  - a promoted checkpoint whose paired anchor margin on the yardstick seeds is not above f610's.
- **Stop condition.** The collapse signature (approx_kl, clip fraction, entropy and teacher KL rising together while the bank falls, as in `pz3xhg9e`) stops the run.

**2. The 256-step window (λ 1, 16 envs/rank).**
- **Evidence.** `h3lpxy6q` is the only BC-start run that improved. Its 10M evaluation beat BC 16/16 (own 117.6k vs 90.4k), and it promoted fc6b. fc6b beat BC's margin in 58/64 anchor games.
- **Why second.** The 720 run from fc6b then gained further. The two windows were never compared from the same start, so 720 over 256 is not attributed.
- **Pull if** option 1 meets a loss condition.
- **Loss conditions.** It slides from BC on a second seed, or it does not beat f610 on the yardstick.

**3. Better BC / imitation fidelity (open, no run).**
- **Evidence.** X1a: BC's self-play bank (75.6k) is about 74% of its teacher's 102.8k per ladder game. The teacher wastes about 9 times fewer commands and produces about twice BC's harvest, water and feed. X1b: cha22 against itself makes about 95.4k per seat.
- **Why third.** The comparison is not like for like: the teacher played ladder games against varied opponents, and BC played mirror games on evaluation seeds. The information, decision and architecture attribution is open. f610 already banks 97-113k against these anchors.
- **Loss condition.** A refit BC that does not raise the paired self-play bank over the current BC on at least 32 seeds (the plan's paired evaluation).

**4. The per-seat critic offset and the stagger (built, untested in a run).**
- **What exists.** `model.critic_offset` (merge `9cb115f`) and `rl.initial_stagger` (merge `a807752`), with the preset `configs/kaggriculture_4rank_bank_critic_credit.yaml` (256, λ 1, the offset critic, the stagger). CPU checks only.
- **Interpretation.** At 720 each segment is one whole game, so the critic acts only as a baseline and never bootstraps a return. A stagger would break the one-game-per-segment alignment. So these matter mainly for the 256 window.
- **Loss condition.** When run, it does not beat option 2's fc6b on the yardstick.

## Tombstones (scoped)

- **Muon 2e-3 / AdamW 1e-4 (Isaiah's LRs), from a promoted policy with the 720 window.** In `pz3xhg9e` (from f610), iterations 51-69 against 1-25 showed approx_kl 0.144 vs 0.017, clip fraction 0.55 vs 0.19, teacher KL 5.56 vs 0.36, and own bank 83.8k vs 94.5k. The last logged window was 45.9k. The owner approved the stop and the 1e-4 relaunch. This matches the collapse of the earlier full-LR arms from BC.
  - Scope: one seed, this reward, this start.
  - Reopen with a trust region or a KL stop.
- **The 64-step λ 0.9 window.** J/2 (`nw3klj2s`), A2 (`04cy2m6s`), `hz4bpjnq` and M (`r350xr3w`) all slid, to 30-61k from peaks of 64-82k. M lost both of its evaluations.
  - Scope: BC or J/2 starts, Muon 1e-4, rewards J, A, J .2/.8 and M. It was never run with the current reward.
  - Reopen if that pairing does not slide.
- **From-scratch starts.** `wk142q4b` (term M), `d6sh0akf` (the current reward, 1e-4) and `uujuarkx` (the current reward, 2e-3, 784 iterations, about 12.8M steps) banked 0 with draw rate about 1.0 at every game end: the random policy drains its 3,000 starting cash.
  - Scope: random init, these rewards, up to 784 iterations.
  - Reopen only with a new exploration or curriculum mechanism.
- **Fixed-opponent cha22 training.** `kifqbyx5` was stopped at iteration 286, before any evaluation. The owner said "let's switch back to self play no matter what, …". By iteration 102 its own bank had fallen from 62.8k to 39.6k and its margin from -80.6k to -112.1k, under the near-zero-signal term M.
  - Scope: the owner's direction, and not a verdict on fixed-opponent training.
  - Reopen only by the owner.

## History

This is the first version of the board. The evidence is in [[../references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving|the credit-window Reference]] and [[../references/earn-money-pod-runs-use-run-local-launch-copy-off-and-switch-scripts|the run inventory]]. The earlier Decisions [[replace-the-reward-with-half-cash-difference-and-half-terminal-sign|term M]], [[train-ppo-against-a-fixed-opponent-with-a-learner-mask|fixed opponent]], [[add-a-per-seat-critic-offset-for-the-own-bank-reward|critic offset]] and [[stagger-game-phases-and-lengthen-the-credit-window|stagger]] record how these options were built.
