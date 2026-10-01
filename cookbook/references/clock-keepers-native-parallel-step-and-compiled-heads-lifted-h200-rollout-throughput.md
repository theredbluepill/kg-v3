---
type: "Reference"
title: "Clock keepers, the native parallel step and compiled heads lifted H200 rollout throughput"
description: "Final-sprint throughput finding, 2026-09-30, 8x H200 pod (driver 570.211.01, 96 vCPU, cgroup quota 81.6 CPU), 720-turn recipe at 20 envs/rank. Baseline ~3,650 env steps/s. The host ran intel_cpufreq passive + schedutil, and rayon env workers sampled at a median 800 MHz against 3.2 GHz on the main thread; one SCHED_IDLE busy loop per CPU (clock_keeper.sh) gave ~5,250 env steps/s (iteration 31.8 s -> 21.9 s) with no code change, while spinning only 56 of the CPUs was worse. On a 1x H200 diagnostic (single rank, keepers on), median env steps/s over iterations 2-7 was 840 baseline, 929 with all four Python switches, 1,034 with the native parallel step, 1,263 with native + compile_actor_heads and 1,268 with native + all switches. Live with 17b3068d + rl.compile_actor_heads=true: ~8,290-8,310 env steps/s (iteration 13.9 s), KL/clipfrac/teacher-KL/EV curves matching the old code. Compiled heads match eager within the eager-vs-fp64 floor. A pre-existing finding: bf16 trunk logp depends on batch shape (40-row rollout vs 1,440-row replay), giving start-of-update ratios 0.51-1.36 and 0.5-1.1% clip fraction before any optimizer step. One pod, one recipe; no Nsight capture."
tags: ["kaggriculture-v3", "throughput", "finding", "rollout", "compile"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "user-directive:2026-10-01:please-accelerate-sps-boost-with-codex"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/README.md"
  - resource: "repository:ops/sprint-2026-09-30/clock_keeper.sh"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/keepers_start.sh"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/summ.py"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps_train.sh"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps-diag-off-20261001.log"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps-diag-on-20261001.log"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps-diag-native-off-20261001.log"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps-diag-native-on-20261001.log"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/sps-diag-native-actor-20261001.log"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/bench-keeper-native.json"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/bench-nokeeper-native.json"
  - resource: "repository:ops/sprint-2026-09-30/throughput/parity-gpu/gpu_head_parity2.py"
  - resource: "repository:ops/sprint-2026-09-30/throughput/parity-gpu/results2.json"
  - resource: "repository:ops/sps-2026-10-01/report.md"
  - resource: "repository:ops/sps-2026-10-01/report-python.md"
  - resource: "repository:ops/h200-driver-gate-2026-10-01/run-statement.md"
  - resource: "repository:python/owl/model/kaggriculture.py"
  - resource: "wandb-run:spoon/kg-v3/xuft2e2i"
  - resource: "wandb-run:spoon/kg-v3/dxhey4da"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
---

# Clock keepers, the native parallel step and compiled heads lifted H200 rollout throughput

## Claim

On the sprint's 8x H200 pod, the 720-turn recipe went from about 3,650 to about 8,300 env steps/s in three steps. The first was a host fix with no code change. The other two were the default-off switches that Codex built that day. Rollout was the bottleneck throughout: env.step took about 68% of rollout time. This is one pod, one recipe and one day. It is a finding, not a recipe change. The training recipe and its reward did not change.

## Measurements

| Stage (8x H200, 20 envs/rank) | Env steps/s | Iteration |
| --- | --- | --- |
| Start (code before 17b3068d) | ~3,620–3,650 | 31.8 s |
| + one SCHED_IDLE clock keeper per CPU | ~5,250 | 21.9 s (rollout 27 s → 18 s) |
| + native parallel step and `rl.compile_actor_heads=true` (17b3068d) | ~8,290–8,310 | 13.9 s (rollout 10.5 s, update 2.7 s, teacher 0.7 s) |

For comparison, the 4x RTX PRO 6000 pod ran the same recipe at 12 envs/rank and about 2,210 env steps/s.

**1x H200 diagnostic** (single rank, keepers on, 10 CPUs, 7 iterations per arm). The values are the median of `perf/steps_per_second` over iterations 2–7, re-derived for this note with `summ.py` from the archived logs.

| Arm | Env steps/s | vs baseline |
| --- | --- | --- |
| Baseline | 840 | |
| All four Python switches | 929 | +10.5% |
| Native parallel step only | 1,034 | +23% |
| Native + `compile_actor_heads` only | 1,263 | +50% |
| Native + all four Python switches | 1,268 | +51% |

Rollout packing, the pinned action D2H and skipping the telemetry validation added nothing measurable. They were left off.

## The clock keeper

- **Mechanism.** The host used `intel_cpufreq` in passive mode with the `schedutil` governor (800 MHz–4.0 GHz). Sampling showed the rayon env workers at a median 800 MHz, while the main thread ran at 3.2 GHz. A burst test ran 2.7–2.9x slower at about 37% duty, and 1.02x with a SCHED_IDLE spinner.
- **Fix.** One SCHED_IDLE busy loop per CPU (`ops/sprint-2026-09-30/clock_keeper.sh`). SCHED_IDLE gives the loop CPU time only when nothing else wants it, so it holds the clock up without taking work time. It also keeps cores out of deep C-states.
- **Negative result.** Spinning only 14 CPUs per node (56 in total) was worse, at 3,800–4,450 env steps/s, because workers landed on unspun slow cores. Spin every CPU.
- **Cost.** The container's cgroup quota was 81.6 CPUs, and the spinners count against it: about 15% throttling and 11.6% run-queue wait on the main thread.
- **Operation.** Stop the keepers before a restart or compile, and start them again after the first iteration. Once a shell bug skipped the restart, and the run spent about 3 minutes at 3,700 env steps/s.

## The code switches

These are the [[native-game-semantics-use-v3-owned-buffers|native parallel step]] (fa0cffc2, branch `kg/sps-rollout`) and the [[rollout-optimizations-preserve-sampling-behind-default-off-switches|four Python switches]] (d564cd2, `kg/sps-python`), merged as `17b3068d` on `kg/sps-combined`. Every switch is off by default.

- **Profile before the change.** py-spy was blocked by `ptrace_scope` and the missing CAP_SYS_PTRACE, so the profile used the run's own `[nt-probe]` timers and `/proc` sampling. It showed serial Rust work on the main thread: the ObsStaging publish memcpy, grammar decode and serde_json, Game clones and a 5.6 MB staging allocation per step. It also showed about 600 eager ops per forward in the actor heads, and 6 host-device syncs per step, with nothing overlapping. No Nsight capture was taken.
- **Review.** Four Claude reviewers and Codex reviewed the change. The verdict was GO WITH CONDITIONS, with no confirmed blocker. The conditions are listed in the rollout-optimization Reference.
- **Live behaviour.** On 8x H200, the approx-KL, clip fraction, teacher KL and explained-variance curves matched the old-code run iteration by iteration. Each rank compiled 5 shapes. The first-update approx-KL and clip fraction of the four diagnostic arms were 0.00186/0.00181/0.00184/0.00182 and 0.0087/0.0082/0.0083/0.0080.

## Compiled-head GPU parity

The test ran on 1x H200 (driver 570.211.01) with the 110M weights in bf16, 20 envs x 720 steps (`throughput/parity-gpu/`).

- **Log-probabilities.** Compiled vs eager heads, per player: median 0.0057, p99.9 0.070, max 0.100 nats. The eager vs fp64 floor is 0.0057/0.053/0.068. The mean signed bias is +2e-5, and the implied ratio is 0.91–1.105 (mean 1.0001).
- **Values** are bitwise identical.
- **Gradients.** Relative error 1–9%, cosine ≥ 0.996. The eager vs fp64 error is 6–9.5%.
- **Sampling.** With identical noise, 0.6% of tokens flip.
- **Path.** Rollout and update both run `_policy_chunk` → `_compiled_actor_core` (`python/owl/model/kaggriculture.py`).

## Pre-existing finding: bf16 trunk numerics depend on batch shape

This does not come from the speed-ups. The rollout runs the trunk on 40 rows, and the update replays the same data on 1,440 rows. In bf16, the stored rollout logp and the replayed logp then differ: median 0.021, p99.9 0.23, max 0.37–0.67 nats. At the start of an update, before any optimizer step, the ratio spans 0.51–1.36 (mean 0.9999), and the clip fraction is already 0.5–1.1%.

So part of PPO's measured clip fraction and KL is numerical, not policy change. The [[ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|replay-drift alarm]] reads a signed mean log-ratio. Here the mean ratio was 0.9999, and how this spread interacts with that alarm was not measured. No repair was tried.

## Limits

- **Scope.** One 8x H200 pod and one recipe. The live 8,300 figure comes from the sprint fact sheet and W&B (`dxhey4da`, `r4zqqs49`). The `r4zqqs49` training log was lost with the pod.
- **Not separated in the live run.** The live run turned on the keepers and the code at different times, so the code gain on 8 ranks is not separated from run-to-run drift.
- **Hardware specific.** The clock-keeper effect depends on this host's governor and cgroup quota. Check both on any new pod before using it.
- **Measures.** Throughput is env steps/s at an unchanged per-update batch, not learner turns per full update at a different batch. Nsight Systems was not used; no timeline claim is made.
- **Batch-shape effect.** Its effect on learning is not measured.

## Promotion basis

The independent check is the re-derivation of the diagnostic medians from the archived logs. The live figures are the fact sheet's. Existing concepts searched: the rollout-optimization and native References (both left H200 unmeasured), the [[model-only-sps-ceiling-bounds-per-rank-throughput|model-only SPS ceiling]], the [[../decisions/throughput-means-correct-complete-work|throughput Decision]] and the replay-drift alarm. The consequence: any new pod should check its CPU governor first, and the speed-up switches have H200 evidence. Results are in the [[the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|sprint episode]].
