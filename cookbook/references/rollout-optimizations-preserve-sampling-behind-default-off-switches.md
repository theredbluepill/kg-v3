---
type: "Reference"
title: "Rollout optimizations preserve sampling behind default-off switches"
description: "Four opt-in Python rollout switches compile grammar heads with eager RNG, build exact packing metadata from current CPU observation masks, reuse fenced pinned action buffers, and skip duplicate telemetry finite scans. Default CPU trainer and config digests match b2276bc5; CPU Inductor and transport tests pass. Final-sprint H200 evidence (2026-09-30): on a 1x H200 diagnostic all four switches gave +10.5% (929 vs 840 env steps/s) and native step + compile_actor_heads +50%; packing, pinned D2H and telemetry skip added nothing measurable and stayed off. compile_actor_heads with the native step ran live on 8x H200 (~8,300 env steps/s) with KL/clip curves matching old code; compiled vs eager heads match within the eager-vs-fp64 floor. Review: GO WITH CONDITIONS (8-variant recompile limit, config flag rejected by older code on resume, rank-0 first-eval compile under the NCCL timeout, test gaps)."
tags: ["kaggriculture-v3", "adaptation", "throughput", "compile", "rollout"]
status: "verified-scoped"
generated: {"by": "openai/codex; sprint H200 section by anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "repository:ops/sps-2026-10-01/brief-python.md"
  - resource: "repository:ops/sps-2026-10-01/code-map.md"
  - resource: "repository:ops/sps-2026-10-01/verification-plan.md"
  - resource: "repository:ops/sps-2026-10-01/report-python.md"
  - resource: "repository:python/owl/model/kaggriculture.py"
  - resource: "repository:python/owl/model/kaggriculture_actor.py"
  - resource: "repository:python/owl/model/stateless_transformer_v1.py"
  - resource: "repository:python/owl/train/ppo.py"
  - resource: "repository:python/owl/train/utils.py"
  - resource: "repository:python/owl/train/config.py"
  - resource: "repository:python/owl/kaggriculture/config.py"
  - resource: "repository:python/owl/kaggriculture/env.py"
  - resource: "repository:python/owl/kaggriculture/rewards.py"
  - resource: "repository:python/owl/game.py"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:scripts/bench_rollout_step.py"
  - resource: "repository:tests/kaggriculture/test_actor_compile.py"
  - resource: "repository:tests/kaggriculture/test_rollout_packing.py"
  - resource: "repository:tests/kaggriculture/test_rollout_switches.py"
  - resource: "repository:tests/kaggriculture/test_reward_telemetry_fast_path.py"
  - resource: "repository:tests/kaggriculture/test_model_compile.py"
  - resource: "repository:tests/scripts/test_bench_rollout_step.py"
  - resource: "repository:README.md"
  - resource: "repository:docs/model-architecture.md"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/throughput/sps-diag-evidence/summ.py"
  - resource: "repository:ops/sprint-2026-09-30/throughput/parity-gpu/results2.json"
  - resource: "repository:cookbook/references/clock-keepers-native-parallel-step-and-compiled-heads-lifted-h200-rollout-throughput.md"
  - resource: "external:https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/block_info.h"
  - resource: "external:https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_kernel.h"
  - resource: "external:https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_launch_template.h"
---

# Rollout optimizations preserve sampling behind default-off switches

The owner asked to carry out the Python SPS brief completely. The brief quotes
"please accelerate SPS boost with CODEX" and "please accelerate on SPS
diagnosis", and requires separate default-off switches without remote access or
Rust edits. The switches are implementation choices, not adopted training
recipe changes. No preset, reward coefficient, observation schema, checkpoint
state or standing-board fact changes.

Existing-concept search covered packing/synchronization, pinned actions,
telemetry and compiled actor heads. The existing cuBLAS-only Decision, native
buffer Reference, grammar-head Reference and shared-PPO Reference supplied the
constraints; none implemented these four switches. The code map's production
timing is supplied context, not a measurement performed by this change.

## Contract and adaptation inventory

- `rl.compile_actor_heads=false`: `kaggriculture.py`,
  `kaggriculture_actor.py`, `train/utils.py` and `scripts/run_ppo.py` compile
  `actor.policy_core` independently of trunk compilation. The entry point
  claims the checked ATEN-only backend, every actor call rechecks it, and
  actor-only launches check the stack before creating a run directory. Heads
  retain their existing extent guard. Nine eager exponential draws preserve
  dense slot shapes, dtype, chunk order and RNG consumption; replay/greedy draw
  none. Fullgraph static compilation accepts only `default` or
  `max-autotune-no-cudagraphs`; CUDA graphs are explicitly refused. Replay and
  teacher modes and new shapes can specialize separately.
- `rl.rollout_packing=false`: `train/ppo.py` scopes each current host
  observation to the model's sampling forward and final value bootstrap.
  The model derives the same token mask, optionally selecting learner rows in
  flattened seat order. `pack_sequence_from_cpu_mask` validates/builds exact
  indices and lengths on CPU and copies metadata asynchronously from fresh
  pinned storage on CUDA. No padded tokens are sent through the packed trunk.
  The context clears even on exceptions; it is current-observation data, not
  between-turn memory. Update, teacher and truncation-bootstrap calls keep
  original packing. An optional device-mask equality check remains for debug.
- `rl.pinned_action_d2h=false`: `train/ppo.py` preallocates CPU int64
  tokens/lengths on CUDA launches. It issues both nonblocking copies on the
  producer's current stream, records one event, and synchronizes that event
  before returning storage to the synchronous native step. The next step may
  reuse the storage only after consumption. CPU keeps the contiguous copy
  path; the observation-buffer fence remains.
- `env.skip_reward_telemetry_validation=false`: env config, factory, adapter
  and reward helpers skip only duplicate finite scans of native-validated
  banks. Native admission and shape/dtype checks remain. Four scans disappear
  when both bank and margin telemetry terms are active. Public reward helpers
  validate by default; metric arithmetic and FP64 operation order are unchanged.
- `PPOConfig`, `FullConfig` and the env config serialize none of the false
  flags, preserving old config hashes. Orbit rejects the RL switches.
- `scripts/bench_rollout_step.py` compares baseline, each switch and all
  switches with completed CPU action transfers, optional native step/H2D,
  separate cold timing, action/observation/RNG hashes and source/config/binary
  identity. It does not need W&B or perform an update. Tests, mapped docs and
  the cuBLAS-only Decision/index describe the added paths.

## Evidence and consequence

The independent flash-attn 2.8.3 source check confirms the existing padded
`max_seqlen` is a legal upper bound: `block_info.h` derives actual lengths from
adjacent cumulative entries; `flash_fwd_kernel.h` exits excess query blocks;
`flash_fwd_launch_template.h` uses the bound for the grid. CPU tests compare
exact packed metadata/results and full model outputs with a varlen test double;
the real CUDA test covers extra launch blocks but is skipped on this Mac.

With all flags off, native and two-update trainer digests exactly equal base
`b2276bc5` at `OMP_NUM_THREADS=2` (respectively `257eae38…`, `3ffd53a0…`), and all
existing preset config hashes match. Two canonical PPO updates with packing,
pinned transfer and telemetry enabled match actions, logp, entropy, values,
rewards, weights, metrics and RNG exactly on CPU. Transport doubles verify
nonblocking copies, event-before-native ordering and buffer reuse.

Real CPU Inductor tests check three sampling seeds with exact tokens, lengths,
values and RNG; logp/entropy tolerance is absolute `2e-6`. Replay gradients
match eager at `rtol=2e-4`, `atol=2e-5`. External noise is bit-exact with an eager
core in FP32/FP64/BF16 and across head chunks. Compile guards, explicit mode
rejection, malformed transport/packing and public reward validation also pass.
The report records broad checks and the bounded CPU benchmark numbers.

Future use should enable each flag separately on a diagnostic GPU, retain
source-bound parity receipts, then compare equivalent complete PPO updates.
CPU parity and component timing establish neither GPU correctness nor a speed
gain for the live recipe. Fresh pinned metadata allocation may offset saved
sync time; static actor variants may add compile latency/memory. Sampling near
an argmax tie can differ after compiler rounding despite identical noise.
CUDA/BF16 numerics, nondefault-stream DMA, FlashAttention, multi-rank behavior,
representative trained states and complete-update throughput remain unmeasured.
No remote or learner was touched; no result is promoted to the training board.

## H200 evidence from the final sprint (2026-09-30)

The diagnostic H200 run did happen, during the final sprint. The measurements
and their limits are in the
[[clock-keepers-native-parallel-step-and-compiled-heads-lifted-h200-rollout-throughput|H200 throughput Reference]].
For these switches, the results are:

- **1x H200 diagnostic** (single rank, keepers on; median env steps/s over
  iterations 2–7):
  - baseline 840;
  - all four switches 929 (+10.5%);
  - native step + `rl.compile_actor_heads` 1,263;
  - native step + all four switches 1,268.

  `rl.rollout_packing`, `rl.pinned_action_d2h` and
  `env.skip_reward_telemetry_validation` added nothing measurable. They stayed
  off in production.
- **Live.** `17b3068d` with only `rl.compile_actor_heads=true` ran on 8x H200
  at about 8,300 env steps/s (iteration 13.9 s), with 5 compiled shapes per
  rank. The approx-KL, clip fraction, teacher KL and explained-variance curves
  matched the old-code run iteration by iteration.
- **GPU parity, compiled vs eager heads** (bf16, 110M weights):
  - per-player logp: median 0.0057, p99.9 0.070, max 0.100;
  - the eager-vs-fp64 floor for comparison: 0.0057/0.053/0.068;
  - values: bitwise equal;
  - gradients: cosine ≥ 0.996;
  - sampling with identical noise: 0.6% of tokens flip.
- **Review conditions (GO WITH CONDITIONS, no confirmed blocker).**
  - The compiled heads use `fullgraph` with `dynamic=False` under an
    8-variant recompile limit that crashes when exceeded; production used 5.
  - The flag is written to `config.yaml`, and older code rejects it on resume.
  - Rank 0's first-evaluation compile runs under the NCCL timeout.
  - Test gaps: `replay_parity.rs:372` compares the new API with itself, and
    the golden lacks the fixed-opponent path.

Still unmeasured: CUDA packing or DMA gains at other batch shapes, and Nsight
timelines.
