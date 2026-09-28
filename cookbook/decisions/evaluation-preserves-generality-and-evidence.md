---
type: "Decision"
title: "Evaluation preserves generality and evidence"
description: "Measure win rate and bank margin across opponents, retain source-bound evidence and telemetry, and keep submission authority with the owner."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:.gitignore"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/candidates-are-measured-by-win-rate-and-margin.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/no-candidate-agent-overfits-to-a-single-opponent.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/myolie-uses-wandb-for-telemetry-and-monitoring.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/repository-tracking-preserves-evidence-and-excludes-local-state.md"}]
---

# Evaluation preserves generality and evidence

The owner's current reuse instruction carries forward v2's documented evaluation and custody discipline. Candidates are judged on win rate plus final-bank margin across differently playing opponents. Record versions, seeds, both seats, denominators, legality, completion and runtime. Keep diagnostic margin distinct from official win/loss/tie outcome. A gain against one opponent is scoped evidence; use fresh seeds and held-out opponents for generalization, and distinguish selection panels from final qualification. Local gate success does not establish public-ladder robustness.

Keep experiment identity stable through retries, using separate attempt/checkpoint metadata. Select a v3 name when an experiment exists; do not import the v2 prefix or conceal continuation by renaming. The owner decides which artifact is submitted; development authorization alone is not a submission decision.

Original v2 telemetry directive:

> 你幫忙下一個cookbook record 去用wandb做telemetry做監控

Use W&B for training/evaluation telemetry alongside durable local evidence, with explicit run identity, phase, source/checkpoint hashes and completed-work denominators. Configure v3 identifiers; do not publish into v2's project/group. Telemetry implementation, authentication and live readback need their own verification. Record outages and pending synchronization honestly; a dashboard is not an independent causal check.

Original bulk-artifact directive:

> why are we pushing the training artifacts as well? They should be gitignored.

Track source, lockfiles, compact results, receipts and hash manifests. Exclude credentials, environments, caches, bulk weights/optimizer state and generated corpora; retain path, bytes and SHA-256 custody manifests for local artifacts. Ignoring is not deletion. No competitive result, cloud connection or submission is established here.
