---
type: "Reference"
title: "Kaggle packaging ships the BC agent as a native, validated tarball"
description: "Task 7.4 ship build on kg/rebuild-7-4-ship: a 24.7 MB submission.tar.gz with main.py, owl, a pod-built CPython 3.11 abi3 x86-64 rs.abi3.so (GLIBC_2.35 max) and the slim BC best (fd854587...6f51). It encodes one seat with the training write_seat, runs a greedy CPU fp32 forward at 1 thread, decodes natively, and returns PASS on a caught fault. Strict local episodes qualified with 719 calls per seat and zero faults, once in a fresh Kaggle-image container under emulation and twice on the pod. Three more episodes in Kaggle mode (non-strict, fallback live) in a fresh pod venv also passed with zero faults and zero fallbacks: BC against starter in each seat, and self-play through the kaggle-environments CLI. Not submitted. Deviates from the brief: no in-image Docker build, no replay-parity or latency-benchmark receipts, no W&B."
tags: ["kaggriculture-v3", "adaptation", "packaging", "kaggle-runtime"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.4-packaging.md"
  - resource: "repository:python/kaggriculture_main.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_agent.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_view.py"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:scripts/build_kaggriculture_submission.py"
  - resource: "repository:scripts/kaggle_local_episode.py"
  - resource: "repository:tests/kaggriculture/test_kaggle_agent.py"
  - resource: "repository:tests/kaggriculture/test_kaggle_view.py"
  - resource: "repository:tests/scripts/test_build_kaggriculture_submission.py"
  - resource: "repository:tests/scripts/test_kaggle_local_episode.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/kaggle-runtime-receipt.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/native-module-receipt.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-native-build.sh"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/custody.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/kaggle-image-episode-self-seed7.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-episode-self-seed20260930.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-episode-starter-seat1-seed20260931.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/validation.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/kaggle_mode_episode.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/summarize_cli_run.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/run_validation.sh"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/episode-starter-seat0-seed20261000.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/episode-starter-seat1-seed20261001.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/cli-run-self-seed20261002-summary.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/venv-kaggle-freeze.txt"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/agent.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/core.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/envs/kaggriculture/kaggriculture.json"
  - resource: "kaggle-mcp:competitions/kaggriculture/pages"
---

# Kaggle packaging ships the BC agent as a native, validated tarball

## Claim

Branch `kg/rebuild-7-4-ship` builds a Kaggle submission for the BC best checkpoint
and validates it by local execution. It carries the approved Task 7.4 brief
(`ops/rebuild-2026-09-29/briefs/7.4-packaging.md`, merged at `eb40556`). Nothing
was uploaded or submitted; submission is the owner's decision.

The existing-concept search covered packaging, submission, Kaggle, runtime and
BC. This note was the only one, as the design Reference for the brief. It is
revised in place (file name kept for incoming links) because its claim moved
from design to a built and executed artifact.

## What ships

`artifacts/7.4/submission.tar.gz` (gitignored): 24,704,468 bytes, SHA-256
`00e67809f9d4d3e40d477dc309be1df241dfc7c71a92f36241401ba873be3839`, built from
clean commit `6c49863`. The limit is 100 MiB. Its `manifest.json` (SHA-256
`4872a2d3…4a94`) binds every file hash, the source commit and tree, and the
two receipts below. `ops/rebuild-2026-09-29/7.4/custody.json` records the
archive hash and size.

- `main.py` (from `python/kaggriculture_main.py`):
  - sets 1 torch thread and 1 interop thread;
  - imports every bundled `owl` module at top level, because the agent
    directory is on `sys.path` only during Kaggle's lazy `exec`;
  - asserts a release extension, loads the model once and runs one warm-up
    turn;
  - defines `agent` last, because the loader takes the last callable.
- Weights: the BC best, original SHA-256
  `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`, slimmed
  to `{"model"}` only (25 MB fp32, 6,252,223 parameters). It is loaded strictly
  as `model_only`, with `force_flash_attn` overridden to false at runtime.
  The loader rejects hidden, recurrent, memory and opponent state names.
- Encoding: `encode_kaggriculture_seat_into` writes one `[1,1]` row through the
  same Rust `write_seat` as training. An empty rival private stands in for
  state the agent cannot see. The configuration comes from an allowlist that
  drops Kaggle's `__raw_path__`.
- Action: greedy (`deterministic=True`) decode through the native grammar, then
  a native encode/decode round trip that must reproduce the action exactly.
- Fallback: in non-strict mode, any exception or failed validation returns
  `{"farmer":["PASS"],"hands":[],"market":[]}` and increments a counter. So does
  a remaining overage bank below 2 s. Strict mode (`KAGGRICULTURE_AGENT_STRICT=1`)
  re-raises instead. The three qualifying episodes used it; the three
  Kaggle-mode episodes below did not.
- Native module: `owl/rs.abi3.so`, 3,477,136 bytes, SHA-256 `3558c26f…5375530`.
  It was built on the pod from clean `f66acf8` with CPython 3.11.13 and
  `maturin build --release --compatibility linux`. It is an ELF64 x86-64
  module whose highest glibc symbol version is `GLIBC_2.35`. A
  `manylinux_2_28` attempt was refused for newer symbols
  (`native-module-receipt.json`, both build logs). The builder refuses a
  non-x86-64 module, a glibc above the limit, a receipt whose hash differs from
  the module, a dirty tree, a checkpoint hash mismatch and an archive over
  100 MiB.

## Execution evidence

All three episodes ran the extracted tarball's `main.py` through
`kaggle-environments` 1.32.7's real file-agent loader with `debug=False`, in
strict mode. Each confirmed kaggriculture.py hash `bc8a5487…653e`.
Qualification needs zero `ERROR`/`TIMEOUT`/`INVALID` statuses at every step,
719 agent calls per seat from the harness's own counter, and zero exceptions.
It also needs zero raw returned actions failing the validator before Kaggle
normalizes them.

| Receipt | Host | Seats | Result |
|---|---|---|---|
| `kaggle-image-episode-self-seed7.json` | fresh container of the local Kaggle image (Python 3.11.13, glibc 2.35, torch 2.6.0+cu124), amd64 emulated on the Mac, `--network none --cpus=1.6 --memory=6.5g` | self-play, seed 7 | qualified; `owl` and `owl.rs` loaded from `/kaggle_simulations/agent`; turn 0 0.97 s / 0.79 s; steady p99 0.42 s, max 0.58 s (emulated) |
| `pod-episode-self-seed20260930.json` | pod EPYC 9535, CPython 3.11.13 + torch 2.6.0+cpu, `nice -n 19`, no GPU visible | self-play, seed 20260930 | qualified; turn 0 0.47 s; steady p99 0.124 s, max 0.129 s; banks 40,844 / 48,504 |
| `pod-episode-starter-seat1-seed20260931.json` | same | BC in seat 1 vs built-in `starter` | qualified; steady p99 0.125 s, max 0.150 s; banks 3,756 (starter) / 80,110 (BC) |

The minimum remaining overage bank stayed 60.0 s in every episode. No call
returned the fallback PASS.

`just py-prepare` passed on the ship branch with 2,777 passed and 18 skipped
(`py-prepare.log`).

### Kaggle-mode validation

`ops/rebuild-2026-09-29/7.4-ship/validation/validation.json` summarizes three
more episodes. The archive (`00e67809…3839`) was unpacked fresh on the pod, and
all 65 manifest files were re-hashed with no mismatch. The venv was new: uv,
CPython 3.11.13, and only torch 2.6.0+cpu, numpy 2.4.6, pydantic 2.12.4,
pyyaml 6.0.3 and kaggle-environments 1.32.7 with their dependencies. Runs used
`nice -n 19` with no GPU visible. This is Kaggle mode: `debug=False`,
`KAGGRICULTURE_AGENT_STRICT` unset (the fallback is live), the agent directory
not on `sys.path`, and a working directory outside it. The env configuration
gives `actTimeout` 1 s, a 60 s `remainingOverageTime` bank, 720 steps and a
1,200 s `runTimeout`.

| Episode | Steps, bad statuses | Calls, exceptions, invalid raw | Turn 0 / steady mean / steady max (s) | Min overage (s) | Peak RSS (MiB) | Banks | Fallbacks |
|---|---|---|---|---|---|---|---|
| BC seat 0 vs `starter`, seed 20261000 | 720, 0 | 719, 0, 0 | 1.316 / 0.122 / 0.260 | 59.68 | 459 | BC 39,981, starter 3,636 | 0 |
| BC seat 1 vs `starter`, seed 20261001 | 720, 0 | 719, 0, 0 | 1.264 / 0.121 / 0.242 | 59.74 | 458 | BC 33,352, starter 3,596 | 0 |
| CLI `kaggle-environments run`, self-play, seed 20261002 | 720, 0 | 719 per seat | seat 0: 1.219 / 0.122 / 0.162; seat 1: 0.457 / 0.122 / 0.136 | 59.78 / 60.0 | 625 | 62,278 / 61,665 | 0 |

"Fallbacks" means the agent's own counters for caught errors and overage-budget
PASSes, its printed fallback lines, PASS returns and non-empty stderr; all were
zero. The kaggriculture interpreter has no invalid-action penalty or status:
illegal unit actions are silent no-ops. Legality is therefore the framework
statuses plus the agent's validator applied to every raw returned action.
kaggle-environments 1.32.7 ships no dedicated agent-validation action, so its
CLI `run` of the submission against itself, the shape of Kaggle's validation
episode, stands in. Peak RSS is for the whole process: the env, the opponent
and the agent (or both agent copies). Replays stay on the pod in
`/root/ship/validate/out/`, and their hashes are in the receipts.

## Reusable findings

- **Timing.** The budget is 1 s per turn plus a 60 s per-seat overage bank.
  Imports, weight load and warm-up bill to turn 0 because the file agent is
  `exec`'d lazily inside the first timed call. In a fresh process the first
  copy's turn 0 took 1.22 to 1.32 s on the pod. That exceeds `actTimeout` and
  draws about 0.3 s from the 60 s bank, which is legal. A second copy in the
  same process took 0.46 s, because torch was already imported.
- **Loader.**
  - It picks the last callable in `main.py`.
  - It injects `configuration["__raw_path__"]`, which the Rust config envelope
    rejects, so the agent builds the configuration from an allowlist.
- **Failure modes.**
  - A non-dict return is silently normalized to PASS.
  - At termination the interpreter overwrites every status and reward, so a
    final-call fault ends as `DONE`.
  - Final statuses therefore prove nothing. The harness captures every
    `Agent.act` result itself, and a unit test injects a final-call fault.
- **Native build without Docker.** A `--compatibility linux` build on a newer
  glibc host is acceptable only when the highest referenced glibc symbol
  version is at or below the runtime's. The fresh runtime container must then
  load and play the module. Both held here.
- **Harness defect found and fixed.** `kaggle_local_episode.py` parsed
  `--episode-steps` but never passed it to `run_episode`. An intended 4-step
  load check therefore ran a full episode. The flag is now forwarded, and
  `test_cli_forwards_the_episode_step_bound` fails on the old code.

## Deviations from the approved brief

- **Build route.** There was no build inside Kaggle's image: the pod has no
  Docker, and the brief forbids building on the Mac. The brief's fallback
  route was used instead, and the in-image load held. `Dockerfile.kaggle` was
  neither repaired nor used.
- **Builder script.** The starter's `build_kaggle_submission.sh` was not
  reused. A Python builder (`scripts/build_kaggriculture_submission.py`)
  stages the prebuilt module instead. The build context is the clean working
  tree, not a `git archive` export.
- **Load-check host.** The fresh-container check ran on the Mac under amd64
  emulation, not on the pod. Because of the harness defect above, it was a
  full episode, which the brief reserves for owner approval on the Mac. It
  took about 6 minutes at 1.6 CPUs.
- **Not done:**
  - T6 latency benchmark: 1 versus 2 threads under an equal 1.6-vCPU quota on
    amd64 hardware, with an external watchdog;
  - T7 replay-parity test against the native env and the training path;
  - W&B telemetry for the episode and benchmark;
  - the dated Kaggle-limits receipt;
  - README and `docs/containerization.md` updates.

## Limits

- It is not established that Kaggle production runs this image. The image is a
  local build of Kaggle's Dockerfile on `python:v163`.
- Production JSON key order is unverified (brief F9); the inventory and shed
  rank channels depend on it.
- No timing here is Kaggle hardware. The pod numbers had no vCPU quota, and the
  container numbers are emulated.
- The episodes validate packaging and legality on six seeds, not strength.
  The three wins over `starter` are one seed per seat assignment; the
  evaluation panel is Task 7.2.
- The Kaggle-mode venv uses the torch 2.6.0 CPU wheel, not the image's
  `+cu124` build, on a glibc 2.39 host.
- Only the latest 2 submissions count for the final leaderboard, so a new
  submission displaces an older one. The owner decides.

Reopen when:

- the validation episode's agent logs from an owner-approved submission exist
  (production image, key order, hardware timing);
- a replay-parity or latency receipt lands;
- the checkpoint or the native module changes.
