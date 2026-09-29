---
type: "Decision"
title: "Preserve the PPO recipe across BC bootstrap"
description: "The BC-period two-rank 4096-env fixed-minibatch recipe is preserved as a historical snapshot with separate checkpoint custody; later runs follow Isaiah alignment and teacher-adapter completion."
tags: ["kaggriculture-v3", "training", "bc"]
status: "adopted"
generated: {"by": "openai/codex", "at": "2026-09-29"}
decider: "Owner: 記住我們訓練配置謝謝"
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-from-bc-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/handoff-cancellation.json"}, {"resource": "user-directive:2026-09-29:diagnose-bc-and-preserve-training-config"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/ppo-config-preserved.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/continuous30-2026-09-29/custody.json"}]
---

# Preserve the PPO recipe across BC bootstrap

Owner: “ok 診斷一下吧， 然後v2 我們有BC過，不如我們進行個多小時BC再回來搞？記住我們訓練配置謝謝”. Owner subsequently clarified: “接回ppo是另外一件事，有做好bc嗎？” BC and PPO are separate tasks; automatic PPO continuation is disabled. The owner then explicitly requested: “我們現在有最佳checkpoint開始半小時PPO試試”. This authorizes a separate bounded half-hour PPO experiment from the saved BC best, with fresh optimizer and unchanged saved recipe; it does not re-enable automatic future handoffs. Stop the remaining BC for this switch, retain its best/latest and the pod. The new run starts its incumbent from BC weights; old PPO incumbent artifacts remain preserved. For that BC-period experiment, do not adopt v2's model or change PPO batches; batch changes for later runs follow the alignment Decision below. Interpret “一個多小時” as a bounded75-minute BC launch; that exact duration is an implementation choice.

Owner subsequently directs: “你可以對齊全部參數吧 基本建設都在了isaiah, 我們盡量對齊”, and rejects the remaining teacher gaps. This supersedes freezing the old recipe for future runs, while preserving historical snapshots and checkpoint custody. The selected source is upstream scaling_6m; new GPU YAMLs are partial until adapters and qualification are complete. The owner later states that the remaining batch-cadence and critic choices are not owner decisions; [[recipe-choices-align-to-isaiah-without-owner-escalation|the alignment Decision]] resolves them toward Isaiah. The 4,096-environment fixed-minibatch figures below describe the preserved historical recipe only.

The historical saved complete YAML is `ops/bc-bootstrap-2026-09-29/ppo-config-preserved.yaml`. PPO uses two PRO6000 GPUs,4,096 environments/rank, horizon64, one PPO epoch, one segment/minibatch, accumulation2:128 learner-seat observations/rank/minibatch and512-seat global effective optimizer batch. AdamW learning rate1e-4, weight decay.01; BF16, compiled current transformer trunk and batched heads,8,353,727 parameters. That historical run used joint per-player clipping, target KL.03, gamma1, GAE lambda.95, MSE critic and normalized advantages. Full optimizer/model details live in the snapshot rather than this summary.

That historical recipe also used economic coefficient.2 (starvation/drought4/1, cap.25, ineffective0, terminal W/L/D scale.75; still the current shaping), 128 complete raw-bank selection games and a checkpoint interval of100,000 global game steps (now 20M, as upstream). Separate current/best checkpoints and the original 70% promotion logic remain current. The retained GPU pod stays running after bounded work unless the owner changes that instruction.

BC uses a separate run and optimizer state, current v3 observation/action representations and the owner's first source volume4llk4uaf20. Demonstration NLL is not a promotion result. Keep the original PPO final/current-best artifacts hash-bound; a subsequent weights-only PPO initialization must explicitly name the new experiment and reset optimizer state rather than pretending BC preserved PPO moments. Do not replace the preserved recipe with temporary diagnostic or BC batch settings.
