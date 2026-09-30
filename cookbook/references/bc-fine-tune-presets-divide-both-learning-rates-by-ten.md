---
type: "Reference"
title: "BC fine-tune presets divide both learning rates by ten"
description: "Recipe J as first-class presets: configs/kaggriculture_2rank_bc_finetune.yaml and configs/kaggriculture_8rank_bc_finetune.yaml equal their ranked configs except muon_lr 0.0002 and adamw_lr 1e-5 (both / 10, same schedule), and are launched from the BC best with --load-model-weights-mode model_only. Evidence is a pre-landing 2-rank ablation from the BC best: LR / 10 prevented the self-play collapse in every matched pair, and the kept BC critic head gave mean explained variance 0.84 at two env seeds against <= 0.29 for a fresh head. Bank gains over BC are not established (game 4 − game 1: D +11.6k, J −4.8k), and no arm ran past the LR warm-up. At 8 ranks the global workload is unchanged, so the LR transfers by construction; that is reasoning, not an 8-rank measurement. A diff test pins the presets to their bases; CPU config checks only."
tags: ["kaggriculture-v3", "training", "bc", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources: [{"resource": "repository:configs/kaggriculture_2rank_bc_finetune.yaml"}, {"resource": "repository:configs/kaggriculture_8rank_bc_finetune.yaml"}, {"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_8rank.yaml"}, {"resource": "repository:configs/scaling_6m.yaml"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:README.md"}, {"resource": "repository:docs/containerization.md"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:docs/write-up.md"}, {"resource": "repository:ops/rebuild-2026-09-29/8rank-run/README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/8rank-run/prepare.log"}, {"resource": "repository:ops/rebuild-2026-09-29/8rank-run/mutations.log"}, {"resource": "pod-prelanding-branch:ddcbf84:ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/comparison.md"}, {"resource": "pod-prelanding-branch:ddcbf84:ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/attribution.md"}, {"resource": "pod-prelanding-branch:ddcbf84:ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/final-report.md"}, {"resource": "pod-prelanding-branch:ddcbf84:ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/ablate-D-lr10-bccritic/result.md"}, {"resource": "pod-prelanding-branch:ddcbf84:ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/ablate-J-combined/result.md"}]
---

# BC fine-tune presets divide both learning rates by ten

**What.** `configs/kaggriculture_2rank_bc_finetune.yaml` and `configs/kaggriculture_8rank_bc_finetune.yaml` ("recipe J") are byte copies of `configs/kaggriculture_2rank.yaml` and `configs/kaggriculture_8rank.yaml` with two values changed and a header prepended:

- `optimizer.muon_lr` 0.002 → **0.0002**;
- `optimizer.adamw_lr` 1e-4 → **1e-5**.

The schedule (1,000 warm-up steps, cosine decay over 400,000, `lr_min_ratio` 0.01), PPO coefficients, teacher (`teacher_kl_coef` and `teacher_value_coef` 0.005), shapes and the owner's [[../decisions/halve-the-kaggriculture-checkpoint-interval-to-10m-steps|10M checkpoint cadence]] are the base configs'. The launch loads the BC best with `--load-model-weights <checkpoint_bc_best.pt> --load-model-weights-mode model_only`, which keeps the BC critic head (see the [[bc-best-starts-ppo-with-a-fresh-critic-head|handoff Reference]]).

## Evidence

A 2-rank PPO-collapse ablation from the BC best (`fd854587…6f51`) on the unmerged branch `kg/pod-ppo-prelanding`. It ran on the pod at `e74d67e`, and its receipts are at `ddcbf84` in `ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/`. Every arm ran 46 iterations (753,664 env steps, 1,024 self-play games, 16 optimizer steps per iteration) with no nonfinite metric. Banks are completed-game final banks, the mean of the two seats.

- **LR / 10 prevented the collapse.** In every matched pair, dividing both LRs by 10 turned a game-4 bank of about 0.1k into 62–84k: control → B (92 → 62,212), H → D (74 → 83,790) and I → G (103 → 63,414). Every full-LR arm except the anchored A fell below 1k by game 4. Parameter movement scaled with the LR (0.09–0.11× at 0.1× LR).
- **The BC critic head gave the better critic.** At LR / 10 with `model_only` (D at env seed 0, J at 1,000,000) the mean explained variance over iterations 1–46 was 0.84 at both seeds, against 0.29 or less for every fresh-head arm (B, E, F, G).
- **Not established: banks rising above BC.** Through game 3 both BC-head seeds rose (+7.3k and +7.2k over game 1). At game 4, D rose to +11.6k and J fell to −4.8k. The 16.6k gap between the seeds at game 4 is three times the 5.1k game-1 spread. Self-play banks are not strength, and no held-out evaluation ran.

## Limits

- **Inside the warm-up.** The LR at iteration 46 was 0.736 of peak, so no arm ran at the preset's 2e-4 peak, which the schedule reaches at iteration 63 (1,000 optimizer steps at 16 per iteration).
- **Two seeds, one recipe.** Neither the anchor (`teacher_kl_coef` 0.1) nor stop-gradient was combined with the BC head, and the head's three coupled changes (initial critic function, teacher value target, initial value-gradient size) are not separated.
- **Pre-landing and unreviewed.** The ablation code equals the integration trainer only by inspection of the `ppo.py` diff (W&B gating, bank telemetry, checkpoint-key validation); it is not Codex-verified.
- **8 ranks by construction.** `configs/kaggriculture_8rank_bc_finetune.yaml` keeps the 2-rank global workload (256 envs, 16 optimizer steps and 16 global segments per step, 16,384 env steps per iteration), so the per-step LR carries over unchanged. That is reasoning, not an 8-rank measurement: no 8-rank PPO run has happened (plan Task 6.3b). Rank seeds differ (`base_seed + rank + k × 8`), so an 8-rank run plays different games from any 2-rank arm.
- **Relation to Isaiah.** These presets depart from `scaling_6m`'s LRs for a case Isaiah never had: "I stuck with pure self-play reinforcement learning and avoided any sort of imitation learning initialization" (`docs/write-up.md`). The measured full-LR collapse from the BC start is the reason, as the [[../decisions/recipe-choices-align-to-isaiah-without-owner-escalation|recipe-choices Decision]] requires for a residual difference. The base ranked configs keep Isaiah's LRs for a scratch start.

## Adaptation inventory and checks

- The two preset files; `README.md` (config list, the Kaggriculture launch example and the BC warm start paragraph now use `model_only` with a preset); `docs/containerization.md` and `scripts/run_ppo.py`'s `--load-model-weights-mode` help, which now name `model_fresh_critic_head` as the diagnostic comparison.
- `tests/kaggriculture/test_configs.py`:
  - the global-workload and per-rank-shape tests are parametrized over both presets;
  - `test_finetune_preset_divides_both_learning_rates_by_ten` checks the literals and exactly 1/10 of the base (which equals `scaling_6m`'s optimizer);
  - `test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs` compares the whole loaded `FullConfig` with the two LRs restored, so it is a diff test;
  - the teacher, env, cadence and model-build tests include the presets.
- Mutations (`ops/rebuild-2026-09-29/8rank-run/mutations.log`): the checks kill a changed non-LR key, a dropped LR change and a reverted cadence.
- The 8-rank run package that launches the 8-rank preset is `ops/rebuild-2026-09-29/8rank-run/`.
- Reopen if a run at the preset's peak LR shows the collapse returning (game-end banks below the run's own game-1 bank for two consecutive games), or if a replicate or held-out evaluation shows the BC head no better than a fresh head.
