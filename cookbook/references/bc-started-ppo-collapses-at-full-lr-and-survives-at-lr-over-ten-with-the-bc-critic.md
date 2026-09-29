---
type: "Reference"
title: "BC-started PPO collapses at full LR and survives at LR / 10 with the BC critic"
description: "Pre-landing, single-seed-per-arm diagnostic on kg/pod-ppo-prelanding e74d67e (2 ranks, 46 iterations, 753,664 env steps and 1,024 self-play games per arm, not Codex-verified). From the BC best at scaling_6m's LRs the self-play economy collapses by game 4 (control 73k to 0.1k; also with the BC critic head, critic stop-gradient or a critic warm-up). LR / 10 separates survival from collapse in all three matched pairs. At LR / 10 the BC critic head (load mode model_only) gives explained variance 0.84 at two seeds against at most 0.29 with a fresh head, and it lifted banks through game 3 at both seeds (+7.3k, +7.2k), but its game-4 rise reproduced at one seed of two (+11.6k, -4.8k). Recipe J = model_only + both LRs / 10 is a survival recipe, not shown to improve on BC. Every arm ran inside the 1,000-step LR warm-up, so peak LR is untested. No held-out evaluation."
tags: ["kaggriculture-v3", "training", "bc", "ppo"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources: [{"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/final-report.md"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/attribution.md"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/comparison.md"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/ablate-J-combined/result.md"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/ablate-D-lr10-bccritic/result.md"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/SHA256SUMS"}, {"resource": "prelanding-branch:kg/pod-ppo-prelanding/ops/rebuild-2026-09-29/pod-6000-2026-09-29/6.2/result.md"}, {"resource": "external-url:https://wandb.ai/spoon/kg-v3/runs/7k07gp7c"}, {"resource": "external-url:https://wandb.ai/spoon/kg-v3/runs/32pahqok"}, {"resource": "external-url:https://wandb.ai/spoon/kg-v3/runs/hftcr4xa"}, {"resource": "external-url:https://wandb.ai/spoon/kg-v3/runs/pkz85wlw"}, {"resource": "user-directive:2026-09-29:attribute-the-learning-with-subagents"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_8rank.yaml"}]
---

# BC-started PPO collapses at full LR and survives at LR / 10 with the BC critic

The first PPO run from the [[bc-best-starts-ppo-with-a-fresh-critic-head|BC best]] (plan 6.2) collapsed the self-play economy. The owner approved an attribution study ("can you just use your subagents and massively attrivite the learening? you get my go for all"). This note records what the study supports and where it stops.

**Scope.** Every arm ran on the unverified pre-landing merge `kg/pod-ppo-prelanding` `e74d67e`, not on this integration tree. The setup was 2 ranks, `configs/kaggriculture_2rank.yaml`, BC best `fd854587…6f51`, 46 iterations, 736 optimizer steps, 753,664 env steps and 1,024 completed self-play games per arm. There was one seed per arm: env seed 0 everywhere except J. Nothing is Codex-verified. No arm ran a held-out evaluation, so there is no win rate or bank margin against any opponent. The receipts are frozen on that branch at `ddcbf84`: `ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/`, with `SHA256SUMS` over 282 files and the full table in `final-report.md` (`830378e`). The check behind this note: every arm's iteration table reproduces byte for byte from its `run.log` (`attribution.md`; J rechecked in `final-report.md`), and the per-arm figures below were recomputed from those tables.

## Measurements

Banks are game-end self-play banks at games 1 → 4, the mean of the two seats. "Full" LR is Muon 0.002 / AdamW 1e-4 (`scaling_6m`). "/ 10" is 0.0002 / 1e-5.

- **Full LR collapses.** Control (fresh head) 73.2k → 0.1k. H (BC head) 76.7k → 0.1k. I (critic stop-gradient) 70.9k → 0.1k. C (value-only critic warm-up) was at 0 by game 2. A (`teacher_kl_coef` 0.1) delayed it: 71.4k → 41.4k.
- **LR / 10 survives 46 iterations.** B (fresh head) 73.3k → 62.2k. E (`vf_coef` 0.5) −8.2k, F (anchor 0.1) −6.7k, G (stop-gradient) −6.7k. Parameter movement scales with LR (0.09–0.11x at 0.1x) and does not depend on head, value weight or value path.
- **BC critic head at LR / 10** (`--load-model-weights-mode model_only`):
  - D, seed 0 (W&B `hftcr4xa`): 72.2k, 76.4k, 79.6k, 83.8k.
  - J, env seed 1,000,000 (W&B `pkz85wlw`): 72.0k, 74.6k, 79.2k, 67.2k.
  - Explained variance averaged 0.84 in both, against 0.29 or less in every fresh-head arm. Trunk movement matched B's within 5 %.
  - Game-end teacher KL rose monotonically in both, to 1.41 and 1.45 at game 4.
- **Seed gap.** D and J agree within 1.8k at games 1–3 and differ by 16.6k at game 4. That is 3.3x the 5.1k game-1 spread across arms.
- **Warm-up.** The LR warms up over 1,000 optimizer steps, which is 62.5 iterations. Every arm stopped at iteration 46, at 0.736 of peak LR. The collapse is shown at Muon LRs of about 7e-4 to 1.5e-3; survival is shown only up to 1.47e-4.

## Reading and consequence

- **Step size is the strongest factor:** 3 of 3 matched pairs. The fresh critic head is not what drags the BC actor. At equal LR the trunk and actor moved the same with either head. On real rollouts the BC head's critic gradient was larger than the fresh head's, the reverse of the synthetic handoff probe.
- **The BC head adds a much better critic** and, at two seeds, a better bank trend through game 3. Its game-4 benefit is unreplicated. It changes three things at once, which are not separated: the initial critic function, the teacher's value target and the initial gradient size.
- **The anchor helps only at full LR.** At LR / 10, stop-gradient, `vf_coef` 0.5 and the anchor show no bank effect beyond the spread.
- **Recipe J** is `model_only` (already `run_ppo`'s default) plus `-o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`, with everything else at config default. It is the survival recipe for BC-started PPO. It departs from `scaling_6m`'s LRs for a case Isaiah never had (PPO from imitation), with the measured collapse as its reason.
- **At 8 ranks.** `configs/kaggriculture_8rank.yaml` keeps the global batch, 16 steps per iteration and schedule, so the same overrides apply unchanged. No 8-rank run exists: 6.3b has not started.
- **Proposed loss conditions** for a longer run (anchored to observations, not validated):
  - banks below the run's game-1 bank for two consecutive games;
  - game-end teacher KL above 2.3;
  - a fall of more than 16.6k in the first two games after the warm-up peak;
  - any nonfinite metric;
  - no held-out gain over the BC best.

## Limits and reopening

This supports no ranking or selection, so no board is opened. The arms are diagnostics, not candidates. Open questions:

- Peak-LR behaviour.
- Whether J's game-4 drop is noise or the start of a slower drift.
- Seed variance, with two draws of one recipe.
- The head's coupled changes, and the head × anchor interaction.
- Per-action-family attribution.
- Strength against held-out opponents.

Reopen recipe J if a run from it meets a loss condition. Also reopen it if a replicate at LR / 10 with a fresh head matches the BC head's banks through game 3, or if the recipe is rerun on the integration tree and diverges from these receipts.
