---
type: "Workflow"
title: "Profile CUDA bottlenecks with Nsight Systems"
description: "Use a representative NVIDIA timeline for GPU performance attribution without adding mandatory runs or disrupting a learner."
tags: ["kaggriculture-v3", "workflows"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
sources: [{"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "skill:/Users/poonszesen/.codex/skills/cookbook-setup/references/gpu-profiling.md"}]
---

# Profile CUDA bottlenecks with Nsight Systems

For this NVIDIA CUDA training/inference project, **NVIDIA Nsight Systems (`nsys`) is the canonical timeline profiler**. This conditional setup extension follows the Kaggriculture owner's 2026-09-10 directive preserved by the cookbook-setup skill: “please add nsys as canonical tool for us in this project cookbook to make sure we use nsys for efficient profiling? Add this as conditional requirement in $cookbook-setup as well, thanks”.

Before attributing throughput, utilization, memory or CPU/GPU bottlenecks, reuse a representative source/configuration-matched capture or obtain a short bounded capture of the actual relevant phase. Record warmup, process/batch identity, command, `nsys`/driver/runtime versions, overhead, raw `.nsys-rep` and exported report paths, relevant kernels/APIs/memcpy/synchronization, and unmeasured scope. Keep large traces outside cookbook prose.

On an authorized NVIDIA host, first inspect `nsys --version`, `nsys profile --help` and `nsys status -e`. If absent, follow the matching [official installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) for a compatible CLI outside the training environment; preserve package/version/install commands in the run receipt. Do not upgrade drivers/CUDA or weaken container security to make profiling work. Consult the matching [user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) before selecting flags. No installation is needed on this non-NVIDIA setup host merely to create the contract.

Choose non-terminating capture behavior (`--kill=none`, and `--wait=all` where supported) while preserving runner checkpoint/watchdog handling. Never kill or restart a learner merely to satisfy profiling; plan launch-time instrumentation at an authorized safe boundary. CPU sampling permission failures do not by themselves establish CUDA tracing failure. Preserve concrete errors and label fallback evidence; utilization snapshots, VRAM percentages and unsynchronized Python timers cannot establish a kernel bottleneck.

This adds no mandatory smoke, training/evaluation, per-run admission, new watcher or approval step. Existing sufficient captures qualify; obvious correctness repairs need not wait. At setup `nsys` is not installed on this macOS host, and no CUDA capture or GPU attribution has been performed.
