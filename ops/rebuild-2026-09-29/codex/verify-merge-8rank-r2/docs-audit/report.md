# Independent docs and cookbook audit

Reviewed HEAD `ca37089` against the three-dot base `0b8cf98ef57fc49a329dca4c8368c630586c4752`. This agent made no tracked modification. Runtime tests, parent/test inventory and scratch mutation coverage are separate verifier tasks.

## Prior finding status

The requested prior verdict was read from `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-8rank-r1.md` (SHA-256 `699d7375f9407ee0873d46dd915ab0354c7887c439831bca95fc1df95cca9610`). It contains one finding.

| Prior finding | Status | Current evidence |
|---|---|---|
| P3, configs Reference:59: inherited claim that GPU chunking is unmeasured contradicts the merged timing/chunk-count evidence | RESOLVED | `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:59` now credits the `e1458d2` teacher-proxy run's 1/2/3 sparse/mid/dense trunk chunk counts, and explicitly retains numerical-correctness, real teacher path and config-level CPU-only gaps. The description also scopes the timing, and `cookbook/log.md:3` records the correction. This matches the model-only SPS and compiled-GEMM References and `ops/rebuild-2026-09-29/results.md` under “Flash and chunks”. |

## Semantic checks

- Read the changed YAML configs, README, model-architecture change, coverage receipt sentence, cookbook Decisions/References/index/log changes, and plan changes. Read the governing adaptation, recipe, throughput and resource-fit contracts, and cited component-evidence notes.
- Independently recomputed the 8-rank shape: 8 × 32 = 256 global envs; 32 / 2 = 16 optimizer steps; 8 × 2 = 16 global segments per step; 8 × 32 × 64 = 16,384 transitions per iteration. Seat-row workloads are rollout 64, minibatch 256, teacher 4,096 and evaluation 64. These match the README, model docs, configs Reference, Decision, plan and test assertions.
- The teacher chunk field remains 128 and clamps to 32 per rank; it is not divided into 16. One whole 4,096-seat-row teacher chunk fits both the stated 5,915 trunk-row and 11,096 head-row conservative limits.
- Verified `configs/scaling_6m.yaml`, `winner_ce_6m.yaml`, `winner_ce_6m_4x5090.yaml` and `scaling_50m.yaml` are byte-identical to upstream commit `32b3ec900ad406eedd965f53a1a0f4490d31c589`. The documented 6M division precedent, versus the 8-GPU 50M recipe's unchanged 256 envs/rank, agrees with those files.
- The corrected c1/c2 options remain distinct and unadopted. The full winner-CE recipe really changes reward/value mode, loss, schedule and weight decay. The batch-only option gives 65,536 transitions per iteration / 16 steps = 4,096 transitions per optimizer step, exactly four times scaling_6m; the documented comparison at equal env steps is therefore necessary to support adopting that deviation.
- Task 6.3b preserves the owner's “likely” wording, identifies the new billable pod decision, retains two ranks for smoke/diagnostics, and covers dense full-iteration memory, complete-work/learner-turn denominators, actual all-reduce/interconnect cost, CPU resources, per-rank alarm, W&B and seed streams. Task 1.4 now explicitly tests seed disjointness at world size 8.
- The integration's resource-fit distinction is retained: max_memory_allocated target, reserved-memory observation, and the differing 94.97 GiB torch versus 97,887 MiB reporting denominators. Complete-trainer qualification remains open.
- Read `kg/rebuild-gpu-checks:ops/rebuild-2026-09-29/results.md` directly. The 1.389/2.372-second component timings, 0.90/0.95 efficiencies, GPU-0-only timing attribution and +0.295/+0.300-second mid versus train-dominated dense attribution match the merged Decision. The source still lives on the pending branch, as expressly labeled; the current tree does not incorrectly claim that this is eight-GPU or end-to-end execution.
- Multi-GPU Decision frontmatter has 24 unique sources: the integration parent has 11 and the branch parent 23, consistent with their union. No duplicate source key was found.
- The coverage sentence clearly names historical merge-check counts and receipt scope. It does not claim new Rust changes. The source/config docs continue to state the explicit no-env startup stop and the uncompleted trainer integration.

## Findings and limits

No new actionable finding. The prior P3 is fully resolved.

`git diff --check base...HEAD` flags whitespace inside the newly committed raw r1 transcript and command-output logs only. This is retained evidence formatting, not a semantic defect, and it was not rewritten.

This read-only review did not run GPU/DDP or end-to-end training. The existing production-integration and complete-work qualification gaps remain accurately disclosed. Full requested check results and parent-preservation results belong to the root verifier's report.
