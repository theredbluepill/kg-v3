---
type: "Decision"
title: "Start multi-GPU qualification with two ranks"
description: "Qualify on two RTX 5090/RTX PRO 6000 GPUs before four; keep the pod running after bounded runs at the owner’s request; host history lives in ops."
tags: ["kaggriculture-v3", "compute"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner, later two-GPU-first directive on 2026-09-28 supersedes the initial four-rank preference."
sources: [{"resource": "user-directive:2026-09-29:extra-bc-rtx6000-codex"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/cookbook-cleanup-2026-09-29/host-history.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/host-replacement.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "user-directive:2026-09-28:four-rank-first"}, {"resource": "user-directive:2026-09-28:two-gpu-sps-first"}, {"resource": "repository:ops/cookbook-setup-checks.md"}]
---

# Start multi-GPU qualification with two ranks

The original owner directive was:

> you can use Nx5090 / RTx pro 6000 for 4rank/8rank, recommended 4 rank first

The owner later preferred **two RTX 5090 or two RTX PRO 6000 GPUs first** for qualification; that supersedes the four-rank-first preference. Four and eight ranks are later scales. This sets a resource direction, not a demonstrated scaling benefit or optimal recipe. Under Isaiah's multi-GPU rule the global configuration stays identical across rank counts (see the [[recipe-choices-align-to-isaiah-without-owner-escalation|alignment Decision]]); qualify correctness, memory and complete-update throughput on two ranks before scaling to four.

On 2026-09-29 the owner authorizes verification runs: “Yo ucan use 2/4-rank RTX 6000 to verify your work.” Two- and four-rank RTX PRO 6000 runs may verify the rebuilt port; read live price and state before creating or reusing a pod, and write a run statement first.

Pod retention: the owner directive “不用關pod” means bounded runs stop training and save checkpoints while the pod keeps running. The owner also authorizes “換主機，訓練後保留新 pod” to replace an unusably slow host. Read live state and price before any billable creation or replacement, and do not silently leave several preparation hosts running.

W&B authentication is configured locally in `~/.netrc` with mode `0600` (key never printed or recorded). That setup does not prove remote upload; verify live synchronization during each authorized run and use project `kg-v3`.

Host, stock and pod history (IDs, regions, prices, download probes) is operational evidence, kept in `ops/cookbook-cleanup-2026-09-29/host-history.md` and `ops/gpu-sps-2026-09-29/results.md`.
