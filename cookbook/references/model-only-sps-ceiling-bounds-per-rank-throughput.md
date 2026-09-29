---
type: "Reference"
title: "Model-only SPS ceiling bounds per-rank throughput"
description: "A conditional component estimate, not end-to-end SPS: at e1458d2 the preset 6,252,223-parameter model gives model-only ceilings of 2,085 / 1,757 / 957 env steps/s per rank (sparse / mid / dense) on a synthetic update schedule where training takes about 60% of the wall. The engine gets at most 53 / 63 / 116 us per env step to lose no more than 10%. Allocated peak 40.3 GiB; reserved peak 85.0 GiB is unexplained."
tags: ["kaggriculture-v3", "throughput", "cuda"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/results.md"}, {"resource": "repository:ops/rebuild-2026-09-29/run-statements/model-sps-ceiling.md"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/bench_model_sps.py"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/pod/results_sparse.json"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/pod/results_mid.json"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/pod/results_dense.json"}, {"resource": "repository:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/MANIFEST.sha256"}, {"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3-rebuild/runs/model-sps-ceiling-2026-09-29/"}]
---

# Model-only SPS ceiling bounds per-rank throughput

**This is a conditional component estimate, not a throughput result.** It answers one question: how many env steps per second per rank could the model alone sustain if the engine, host copies, GAE, logging and the DDP all-reduce cost nothing? The [[../decisions/throughput-means-correct-complete-work|throughput Decision]] forbids reading component speed as end-to-end speed, so this note is an upper-bound yardstick. It must not be ranked on a board.

## Setup

- Source `e1458d2` (full model with heads), preset `configs/model/kaggriculture.yaml`, 6,252,223 parameters.
- fp32 params under bf16 autocast, TF32, flash forced ([[pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120|pod environment]]), trunk compiled with `max-autotune-no-cudagraphs` (dynamic), heads eager.
- One RTX PRO 6000 Blackwell (GPU 0 of pod `w7ia3zvxqsvs3g`), synthetic `make_obs` observations at uniform densities, synthetic grammar tables.
- Synthetic schedule per update (128 envs × horizon 64 = 8,192 env steps):
  - 64 rollout forwards of 256 rows (A);
  - 16 PPO-shaped train steps of 1,024 rows (B);
  - one teacher-proxy pass over 16,384 rows (C);
  - one value pass of 256 rows (D).
  - update wall = 64·A + C + 16·B + D, from CUDA-event medians over 20 iterations.

## Result

| density (tokens/row) | update wall | ceiling SPS/rank | engine budget for ≤ 10 % loss |
|---|---|---|---|
| sparse (222) | 3.93 s | 2,085 | 53 µs per env step |
| mid (303) | 4.66 s | 1,757 | 63 µs per env step |
| dense (709) | 8.56 s | 957 | 116 µs per env step |

- **Share of the synthetic update:** the 16 train steps take about 60 %, the teacher proxy 20–22 % and rollout sampling 18–19 % at every density. This does not attribute B's cost to heads backward, Muon or any other sub-phase.
- **Engine budget:** to lose at most 10 % of the ceiling, engine time E per update must satisfy E ≤ M/9, where M is the model's update wall. The 64 rollout steps run serially with the engine, so each batched 128-env engine step competes with an 11–25 ms sampling forward.
- **Memory:** peak `max_memory_allocated` is 40.275 GiB (dense B), 42.4 % of the 94.97 GiB torch reports. The caching allocator's **reserved** peak reached 84.994 GiB (89.5 %) during dense C. The explanation that C reused B's cached blocks is an untested hypothesis.
- **Chunks:** the teacher proxy's trunk ran in 1, 2 and 3 chunks for sparse, mid and dense at 16,384 rows, as predicted by the guard. This counts chunks on the GPU; it was timing only, with no correctness comparison.

## Limits

- **Not end-to-end.** Engine stepping, host↔device copies, GAE, logging, W&B, the DDP all-reduce and the rollout's Python loop are excluded. "≈ 2× on two ranks" assumes a small all-reduce that was not measured.
- **Surrogates.** B is PPO-shaped (joint-ratio clipping, Muon, no teacher KL), not the `run_ppo` loss. C is `evaluate_actions` under `no_grad`, not the real teacher path. Their costs are not proven bounds on the eventual trainer.
- **Timing only.** No finite-loss or gradient check and no sampling-versus-replay equality check were retained.
- **Compile path.** At `e1458d2`, `configure_model_compile` still rejected the Kaggriculture model; the probe called the trunk compile directly.
- **No timeline.** `nsys` absence on the pod is operator-reported. In-step phase attribution is unresolved.
- **Workload shape.** Densities are uniform within a batch; real rollouts mix them.

## Consequences

- The Task 1.4 and Phase 6 native engine has an explicit per-step budget: about 53–116 µs per env step to cost no more than 10 % of the model-only ceiling.
- Complete-work SPS in 6.3 can be set beside this ceiling, but the gap does not by itself attribute cost to the engine, host work or the all-reduce: B and C are surrogates and the densities are uniform. Attribution needs matched phase measurements inside the real trainer. A trainer that exceeds the ceiling likewise needs those measurements to explain it.
- The 6.1 memory smoke must resolve the reserved-memory observation (allocator snapshot or phase-order control). The multi-GPU Decision's 85 % target is defined on allocated memory, so reserved memory near 90 % is a separate risk.

## Verification

- Codex reviewed the receipts twice: `verify-sps-ceiling-r1` (APPROVE WITH EDITS, edits applied in `ddf1fb2`) and `verify-sps-ceiling-r2` (APPROVE; 21 of 21 checksums verified, and the 20 non-README artifacts unchanged across `cb4af49`, `ddf1fb2` and HEAD; the README and its checksum changed with the r1 edits). The reports are local working transcripts in `ops/rebuild-2026-09-29/codex/`, not committed.
- On this checkout, `shasum -a 256 -c MANIFEST.sha256` in `model-sps-ceiling-2026-09-29/` reports 21 of 21 files OK (2026-09-29).
- A second measurement episode reproduced the schedule: the default-backend arm of the ATEN-only GEMM A/B gave update walls of 4.667 s (mid) and 8.561 s (dense), against 4.662 s and 8.564 s here (`results.md`, "ATEN-only GEMM A/B").

Concept search before writing: the [[../decisions/throughput-means-correct-complete-work|throughput Decision]] (measured scopes), the historical [[shared-ppo-adapts-game-batches-without-a-second-loop|shared-PPO Reference]] (reference-branch SPS) and the [[../decisions/start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]] (85 % memory target). None bounds the rebuilt model's throughput. Full evidence: `ops/rebuild-2026-09-29/results.md`, "Model-only SPS ceiling (component)".
