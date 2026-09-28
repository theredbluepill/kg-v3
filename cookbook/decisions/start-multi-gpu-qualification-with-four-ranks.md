---
type: "Decision"
title: "Start multi-GPU qualification with two ranks"
description: "Start SPS qualification with two RTX 5090/RTX PRO 6000 GPUs; the earlier four-rank preference is superseded and GPU execution remains unverified."
tags: ["kaggriculture-v3", "compute"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner, later two-GPU-first directive on 2026-09-28 supersedes the initial four-rank preference."
sources: [{"resource": "user-directive:2026-09-28:four-rank-first"}, {"resource": "user-directive:2026-09-28:two-gpu-sps-first"}, {"resource": "repository:ops/cookbook-setup-checks.md"}]
---

# Start multi-GPU qualification with two ranks

The current owner preference is **two RTX 5090 or two RTX PRO 6000 GPUs first** for SPS qualification. This supersedes the initial four-rank-first preference below; the original four/eight-rank authorization remains historical context. The filename is retained so existing links remain valid.

The original directive was:

> you can use Nx5090 / RTx pro 6000 for 4rank/8rank, recommended 4 rank first

Use two ranks as the initial multi-GPU qualification target, with RTX 5090 or RTX PRO 6000 as the permitted GPU families; four/eight ranks are later scales. This specifies a resource direction, not a demonstrated scaling benefit, exact batch size or optimal recipe. Preserve the starter's shared PPO/DDP path and qualify correctness, memory and complete-update throughput with actual workload evidence before extrapolating.

Check live capacity and price before a billable deployment. The initial read-only RunPod audit returned no four-GPU stock for the requested full-GPU families across its returned CUDA versions; catalog base prices are not an available deployment quote. Exact reads and limits are retained in the setup receipt. Subsequent two-GPU allocation attempts in US-NE1 and EUR-IS1 failed despite LOW stock responses; no new pod was created. The existing two-PRO-6000 pod remains EXITED and untouched. No GPU execution is claimed by this Decision.

W&B authentication is configured locally in `~/.netrc` with mode `0600` (file permissions independently inspected; key never printed or recorded). That setup is not proof of remote telemetry upload or GPU execution. Use project `kg-v3`, keep local evidence, and verify live synchronization during the authorized execution phase.
