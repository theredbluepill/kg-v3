---
type: "Decision"
title: "The Kaggriculture v3 board"
description: "Living ranked board of v3 training options; position 2026-10-01 after the competition deadline. Lead: 720-turn lambda-1 mirror self-play at Muon 1e-4 with reward own bank 0.30 + cash difference 0.30 + econ shaping 0.01 + win/loss 0.30, from the BC lineage, with promotions checked on the 48-game fixed-shop anchor panel (c50 6-42 -> 210M 48-0, +7.5k). Then PFSP league training (owner declined for the sprint; top future candidate), the 256-step window, better BC, and the unrun critic offset and stagger. Scoped tombstones: lambda scheduling (owner declined), rule 2 (no gain), LR change (never triggered), Muon 2e-3, the 64-step lambda-0.9 window, from-scratch starts, fixed-opponent cha22 training. No v2 ranking is imported."
tags: ["kaggriculture-v3", "board", "training", "decisions"]
status: "current"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
decider: "Owner: \"let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?\"; \"ok go with 720 and relaunch the run\"; \"actually give back 0.075 from [own bank 0.375 / cash difference 0.375] to win/lose loss to make win/lose to be 0.3, and relaunch the current run again. Let's do that.\"; \"let anchors speak will be good\"; \"i meant let it be\" (lambda); \"maybe it's ok let's just focus in pure self-play, no need league\"; \"let's not apply rule2.\""
sources:
  - resource: "user-directive:2026-09-30:switch-back-to-self-play-and-earn-money-for-real"
  - resource: "user-directive:2026-09-30:ok-go-with-720-and-relaunch-the-run"
  - resource: "user-directive:2026-09-30:give-back-0.075-to-win-lose-to-make-it-0.3"
  - resource: "user-directive:2026-09-30:let-anchors-speak-will-be-good"
  - resource: "user-directive:2026-09-30:let-it-be-no-lambda-schedule"
  - resource: "user-directive:2026-09-30:pure-self-play-no-need-league"
  - resource: "user-directive:2026-10-01:60m-is-close-use-rule1-rule2-locally-then-lets-not-apply-rule2"
  - resource: "repository:cookbook/references/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel.md"
  - resource: "repository:cookbook/references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving.md"
  - resource: "repository:cookbook/references/late-investment-filter-drops-only-purchases-that-cannot-sell-in-time.md"
  - resource: "repository:cookbook/references/final-turn-liquidation-sells-the-shed-on-the-last-resolved-turn.md"
  - resource: "repository:cookbook/references/earn-money-pod-runs-use-run-local-launch-copy-off-and-switch-scripts.md"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/README.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnF/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/launch.sh"
  - resource: "repository:ops/late-invest-2026-10-01/ab-rule2/ab-rule2.md"
  - resource: "repository:ops/earn-money-2026-09-30/x1/summary.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-lr2e3-4rank/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/vs-cha22-4rank/stop.md"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "repository:cookbook/decisions/promote-and-relaunch-on-anchor-panel-evidence.md"
  - resource: "repository:cookbook/decisions/keep-the-sprint-in-pure-self-play-without-pfsp-or-lambda-scheduling.md"
  - resource: "repository:cookbook/decisions/spend-kaggle-submission-slots-only-on-owner-agreed-checkpoints.md"
  - resource: "repository:configs/kaggriculture_4rank_bank_critic_credit.yaml"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
  - resource: "wandb-run:spoon/kg-v3/pz3xhg9e"
---

# The Kaggriculture v3 board

## Position, 2026-10-01

The competition deadline (2026-09-30 23:59 UTC) has passed and every pod is terminated. No run is live.

The owner's choices (verbatim) are in `decider`. The owner chose the reward, the window, the LR policy, pure self-play and the submissions ([[promote-and-relaunch-on-anchor-panel-evidence|panel promotion]], [[keep-the-sprint-in-pure-self-play-without-pfsp-or-lambda-scheduling|pure self-play]], [[spend-kaggle-submission-slots-only-on-owner-agreed-checkpoints|submission slots]]). The ranking and loss conditions below are the agent's reading of the evidence, not owner adoption.

Currently pulled: **option 1**. Its lineage went from c50 to 210M with five trainer promotions and a 48-0 panel at 170M, 180M, 200M and 210M ([[../references/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|sprint episode]]).

**Common yardstick.** The fixed-shop anchor panel is 48 games per checkpoint:
- seeds 93001–93008 x both seats x {smaller_market_shock, cha22, v56};
- rule 1 on;
- games paired by (anchor, seed, seat), with the SE over the 8 seed means;
- run by one command on a Linux CPU pod (`ops/sprint-2026-09-30/anchor-games/cpu-pod/`).

It is a selection panel, not held-out. Margin is own bank minus anchor bank.

| Checkpoint | W-L | Mean margin |
| --- | --- | --- |
| c50 (start) | 6-42 | −7.2k |
| 90M (submitted) | 38-10 | +1.94k |
| 120M | 30-18 | +0.49k |
| 130M | 46-2 | +2.85k |
| 170M (submitted) | 48-0 | +6.90k |
| 200M | 48-0 | +7.89k |
| 210M (submitted) | 48-0 | +7.51k (−383 ± 537 vs 200M) |

The trainer's last_best is 180M. 200M and 210M cannot be separated on the panel.

## Ranked options

**1. Continue 720-turn λ 1 self-play with the 0.30/0.30/0.01/0.30 reward (lead).**
- **Recipe.**
  - `configs/kaggriculture_4rank_margin.yaml` with `rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0`.
  - Reward: econ_bank 0.3/150000/0.3, econ_margin 0.3/100000/0.3, econ_shaping 0.01 (cap 0.1), win/loss 0.30.
  - Muon 1e-4 / AdamW 5e-6; teacher last_best; promotion at ≥ 0.70 every 10M.
  - Start from the BC lineage (BC → fc6b → f610 → c50 → 210M).
  - Launcher: `ops/rebuild-2026-09-29/sprint-8gpu/launch.sh`. Check each promotion on the panel.
- **Evidence.**
  - Trainer promotions at 70M, 80M, 140M, 170M and 180M. No paneled promotion fell below its predecessor; 80M and 180M were within about one SE.
  - Panel 6-42 → 48-0, −7.2k → +7.5k.
  - Self-play and the panel disagreed at 60M (self-play tie, panel better) and at 120M (self-play +2.8k, panel −1.73k ± 1.13k vs 100M). So the panel, not self-play alone, chose the owner's manual promotions.
- **Loss conditions (agent-proposed).**
  - Two consecutive paneled checkpoints that do not beat the last best on the paired panel.
  - A trainer-promoted checkpoint that loses to its predecessor beyond one SE.
  - The collapse signature (approx_kl, clip fraction, entropy and teacher KL rising while the bank falls).
- **Note.** The panel is saturated at 48-0, so it now separates candidates only by margin. A held-out panel or harder opponents are needed to keep ranking.

**2. PFSP league training (not run; owner declined it for the sprint).**
- **Owner.** "maybe it's ok let's just focus in pure self-play, no need league". An implementation was started on branch `kg/pfsp` and stopped with no commits.
- **Why second.** It is the top future candidate. Mirror self-play mis-ranked 60M and 120M against the anchors, and the anchor panel is saturated. Sampling past checkpoints and anchors as opponents targets both problems. This is untested.
- **Pull if** the owner reopens league training or option 1 meets a loss condition.
- **Loss condition.** On the paired panel and a held-out panel, it does not beat an option-1 run of equal steps from the same start.

**3. The 256-step window (λ 1, 16 envs/rank).**
- **Evidence.** `h3lpxy6q` from BC beat BC 16/16 at 10M and promoted fc6b ([[../references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving|credit-window Reference]]).
- **Why third.** 720 then carried the lineage to 210M. The two windows were never compared from the same start.
- **Loss condition.** From the same start and reward, it does not match option 1 on the panel.

**4. Better BC / imitation fidelity (open, no run).**
- **Evidence.** X1a: BC's self-play bank (75.6k) is about 74% of its teacher's 102.8k per ladder game (`ops/earn-money-2026-09-30/x1/summary.md`).
- **Why fourth.** PPO now wins every panel game from that BC, so a better start is a smaller lever.
- **Loss condition.** A refit BC that does not raise the paired self-play bank over the current BC on at least 32 seeds.

**5. The per-seat critic offset and the stagger (built, untested in a run).**
- **What exists.** `model.critic_offset` and `rl.initial_stagger`, with the preset `configs/kaggriculture_4rank_bank_critic_credit.yaml`. CPU checks only.
- **Interpretation.** At 720 each segment is one whole game, so the critic never bootstraps, and a stagger would break that alignment. These matter mainly for option 3.
- **Loss condition.** When run, it does not beat option 3 on the panel.

## Tombstones (scoped)

- **λ scheduling.**
  - The owner declined it: "i meant let it be".
  - Never run. Scope: the sprint. Reopen only by the owner.
- **Rule 2 (late-investment filter).**
  - On c50, −1 ± 21 per game against rule 1 alone over 48 paired games. Only wheat seeds were blocked.
  - The owner said "let's not apply rule2." No shipped package carries it ([[../references/late-investment-filter-drops-only-purchases-that-cannot-sell-in-time|rule 2]]).
  - Scope: c50, which made few late purchases.
  - Reopen on a policy that makes large late purchases.
  - Rule 1 is adopted in all three sprint packages ([[../references/final-turn-liquidation-sells-the-shed-on-the-last-resolved-turn|rule 1]]).
- **LR change.**
  - "let anchors speak will be good". The trigger was to change only if the anchors stalled and KL and clip fraction shrank. It never fired, so Muon stayed at 1e-4 through 210M.
  - Scope: this lineage to 210M. Reopen if the trigger fires.
- **Muon 2e-3 / AdamW 1e-4 from a promoted policy.**
  - `pz3xhg9e` (from f610) collapsed: approx_kl 0.144 vs 0.017, teacher KL 5.56 vs 0.36, bank 83.8k vs 94.5k.
  - Scope: one seed, the earlier reward. Reopen with a trust region or a KL stop.
- **The 64-step λ 0.9 window.**
  - Every run slid (J/2, A2, `hz4bpjnq`, M).
  - Scope: BC or J/2 starts, the earlier rewards. Never run with the current reward.
- **From-scratch starts.**
  - Bank 0 with a draw rate of about 1.0 up to 784 iterations.
  - Reopen only with a new exploration or curriculum mechanism.
- **Fixed-opponent cha22 training.**
  - `kifqbyx5` was stopped at iteration 286 when the owner said "let's switch back to self play no matter what, …".
  - Scope: the owner's direction, not a verdict on the method. Reopen only by the owner.

## History

This version replaces the 2026-09-30 board. That board led with pcy5knet and used the earlier Kaggle-harness yardstick (BC −67.2k → f610 −21.1k vs smaller_market_shock). It is preserved byte-exact in git at commit `07c8fc99` (sha256 `712f64fa6bc89e1d43546dfb22680fd748bd6be172d9814b9ab29d0319eb1e9b`). Its evidence lives in [[../references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving|the credit-window Reference]] and [[../references/earn-money-pod-runs-use-run-local-launch-copy-off-and-switch-scripts|the run inventory]]. The sprint runs, the panel, the submissions and custody are in the [[../references/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|sprint episode]]. Earlier Decisions on [[replace-the-reward-with-half-cash-difference-and-half-terminal-sign|term M]], [[train-ppo-against-a-fixed-opponent-with-a-learner-mask|the fixed opponent]], [[add-a-per-seat-critic-offset-for-the-own-bank-reward|the critic offset]] and [[stagger-game-phases-and-lengthen-the-credit-window|the stagger]] record how the options were built.
