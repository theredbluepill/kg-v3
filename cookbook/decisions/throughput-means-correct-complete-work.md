---
type: "Decision"
title: "Throughput means correct complete work"
description: "High throughput is required; preserve semantic parity and measure equivalent work rather than inferring gains from language or kernels."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/high-throughput-is-mandatory.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/references/myolie-native-sampling-grammar.md"}]
---

# Throughput means correct complete work

Original v2 owner directive:

> cookbook decision 下一個-高吞吐量是必須不是選擇

Under today's instruction to reuse useful discipline, high throughput is a requirement for the v3 collection, model and training design. Preserve correctness while reporting collection rate, valid learner turns through a full update, phase timings and the total workflow cost. Separate startup/compilation from steady state. State hardware, precision, batch/rollout work, recording settings, transfer and evaluation boundaries. No absolute SPS floor or allowed regression percentage has been supplied.

Rust preparation must actually execute in the path being measured. Preserve game ordering, no-op/STOP distinctions, conditional legality, masks, sampled behavior density and feature semantics; pin source/lockfile/binary and inputs/outputs. Compare equivalent work with an independent oracle where applicable. A faster parser or sampler does not establish a full-training speedup.

The v2 native-sampler Reference is useful contrary evidence: its saved shared-host run lowered sampler CPU time while complete collection elapsed time increased. That is a scoped counterexample to extrapolating component gains, not a ban on native sampling and not a v3 benchmark. Reducing environments, histories or data volume is a workload change unless comparable end-to-end measurements show an optimization.

Use [[../workflows/profile-cuda-bottlenecks-with-nsight-systems|Nsight Systems]] for NVIDIA timeline attribution. Temporary downscaling can diagnose a fault but does not close the throughput requirement. Measured scopes so far (the rejected old decoder and the historical fixed-minibatch sweep) live in `ops/gpu-sps-2026-09-29/results.md`. None of them establishes playing strength, dense-game capacity or an owner-specified performance floor, and each recipe change, including the Isaiah-aligned cadence, needs its own complete-work qualification.
