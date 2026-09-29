---
type: "Reference"
title: "Kaggle packaging reuses the starter submission path"
description: "Task 7.4 brief: the Kaggriculture agent ships through Isaiah's in-image extension build and tarball. It encodes one seat with the training write_seat through a new seat binding, strips the local loader's __raw_path__ key, and bounds a 1 s turn plus a 60 s bank. Design only; nothing built or timed."
tags: ["kaggriculture-v3", "adaptation", "packaging", "kaggle-runtime"]
status: "draft"
generated: {"by": "anthropic/claude-opus-5.5", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.4-packaging.md"
  - resource: "repository:Dockerfile.kaggle"
  - resource: "repository:scripts/build_kaggle_submission.sh"
  - resource: "repository:python/main.py"
  - resource: "repository:python/owl/agent/agent.py"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/config.rs"
  - resource: "repository:Cargo.toml"
  - resource: "repository:docs/kaggriculture-contract.md"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/agent.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/core.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/envs/kaggriculture/kaggriculture.json"
  - resource: "kaggle-mcp:competitions/kaggriculture/pages"
  - resource: "kaggle-mcp:competitions/kaggriculture/discussion/739874"
---

# Kaggle packaging reuses the starter submission path

## Claim

The Task 7.4 brief (`ops/rebuild-2026-09-29/briefs/7.4-packaging.md`) plans
Kaggriculture packaging on the starter's existing path. It adds no new build
system:

- `Dockerfile.kaggle` compiles the `abi3-py311` PyO3 extension inside Kaggle's
  own amd64 image;
- `scripts/build_kaggle_submission.sh` tars `owl`, `main.py` and the slim
  checkpoint.

The existing-concept search covered packaging, submission, Kaggle and runtime;
no prior note covered this. The only earlier mention is the Evaluation
Decision's rule that the owner decides submission, and this note follows it.

## Reusable findings

These come from sources read on 2026-09-29.

**Timing.**

- Kaggriculture gives 1 s per turn (`actTimeout`) plus a 60 s per-seat overage
  bank. It inherits `runTimeout` 1200 s.
- Kaggle staff say production runs the two agents in parallel. The local
  file-agent runner is sequential.
- A file agent is `exec`'d lazily inside the first timed call. Imports, weight
  load and warm-up all bill to turn 0. `sys.path` includes the agent directory
  only during that `exec`.

**Loader behaviour.**

- The loader picks the **last callable** in `main.py`.
- It injects `configuration["__raw_path__"]`. Passing the Kaggle configuration
  straight to the Rust envelope would be rejected (`config.rs`: only
  `actTimeout`, `runTimeout` and `seed` are allowed as extra keys). The agent
  must build the game configuration from an allowlist.

**Encoding.**

- The only merged Kaggriculture binding needs both seats' private state. A
  live agent has one seat's view.
- The brief adds a seat binding that calls the same `write_seat` as training.
  It rejects a pure-Python encoder unless the extension is proven unloadable
  on Kaggle.

**Failure modes.**

- An exception, a non-dict action or a timeout each lose the episode. The
  brief keeps Isaiah's catch-all default action at the process boundary. A
  strict mode re-raises for every test, benchmark and qualifying episode.

**Competition rules.**

- Only the latest 2 submissions count for the final leaderboard.
- The deadline is 2026-09-30T23:59Z.
- The FAQ's size, RAM and vCPU values are unrendered placeholders, so they are
  unknown.

## Consequence

- Packaging work must reuse the starter's scripts, parameterized by
  entrypoint.
- Latency must be measured on amd64 inside the Kaggle image, not inferred
  from the Mac or from GPU throughput.
- Any submission stays the owner's decision, made with the displacement rule
  in view.

## Limits

Nothing here is verified by execution. The following were **not** done:

- no build, binding, test, episode or timing ran;
- no Docker image was pulled;
- the production image (Kaggle's master Dockerfile says `python:v163`, while
  the starter pins a May 2026 `python-simulations` digest) and its torch
  version are unconfirmed;
- the arithmetic latency estimate in the brief (about 10 GFLOP per padded
  709-token row) is not a measurement.

The brief lists these under "Open questions". Reopen this note when Task 7.4
lands with receipts.
