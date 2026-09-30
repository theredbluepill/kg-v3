# Independent verification: kg/rebuild-8rank, r2

Reviewed `b51b0c0382a13a72854860ae17d7798f0875e4ee...7ad45fc6e209e4ad46712644e62fbf3e228ff119` on 2026-09-29. The supplied base is the merge base. No new actionable findings. No tracked files changed; the existing untracked r1 receipt directory is preserved.

## Prior review findings

Source: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-8rank-r1.md` (copied to `prior-review.txt` for custody). It contains exactly one finding. Its disposition is recorded here without altering that historical report.

| Status | Severity and location | Finding, applied fix and verification |
|---|---|---|
| **RESOLVED** | **P3 — cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md:45** | Component timing GPU attribution. The Decision now says the timings ran alone on one GPU (GPU 0) of the two-GPU pod, and separately states that the overall bundle used GPUs 0 and 1. This matches the cited Check 4, line 412. It also preserves the correct mid/dense cost attribution. No further fix required. |

There are no PARTIAL or UNRESOLVED prior findings and no new findings.

## Verified scope

- `configs/kaggriculture_8rank.yaml` loads through the real `FullConfig.from_file`. It divides scaling_6m's 256 envs and 16 segments/minibatch by 8 into 32 and 2, with accumulation 1 and horizon 64. Each division is whole, the rank has 16 complete minibatches, and the retained teacher chunk setting of 128 clamps to 32, one complete chunk per rank.
- Global work is unchanged: 256 envs, 16 optimizer steps per iteration, 16 global segments per optimizer step and 16,384 transitions per iteration. The actual `_minibatch_indices` generator supplies the step count. Optimizer and PPO settings match scaling_6m except the divided segments/minibatch. Across the 2/4/8-rank configurations, only the stated per-rank shapes differ.
- Startup rows are 64 rollout, 256 minibatch, 4,096 teacher and 64 evaluation. Each needs one trunk and one head call. The new startup case drives `run_ppo.main` from the shipped YAML and reaches its headroom output, followed by the documented not-wired error before run-directory, environment or model allocation. It uses a stubbed single-process distributed context; this verifies eight-rank **per-rank shapes**, not actual distributed execution. `shape-check.json` preserves a separate calculation using the real loader and workload code.
- Plan Task 1.4 at line 209 adds world sizes 1/2/4/8 to the prospective seed-stream test, covering ranks and successive resets. Its `seed + rank`, stride `world_size` wording agrees with L3 and the native-env brief. This is a planned test, not a claim that the native world-8 integration already exists.
- Plan Task 6.3b at lines 295–303 keeps dense-state memory, complete-work env/learner-turn throughput, phase costs, actual interconnect/all-reduce, CPU provisioning, every-rank log-ratio alarm, telemetry and world-8 seed checks open. A new pod remains subject to live-price approval. The plan assumes no measured eight-rank or end-to-end result.
- The Decision at line 38 quotes the owner exactly: “we'd likely use 8Xrtx 6000 in main run, 2 rank for diagonose/building.” Line 40 explicitly retains “likely” as an expectation, without adopting hardware or a run. The plan reproduces the same wording.
- Decision line 47 correctly distinguishes Isaiah's scaling_50m precedent: 8× B200, 4B environment steps, 256 envs/rank, spm 8 × accumulation 2, horizon 64. The global batch grows to 2,048 envs. Pinned upstream `32b3ec9` has no eight-GPU 6M config; its separate winner_ce_6m_4x5090 configuration supplies the 6M division precedent.
- Decision lines 48–50 and plan line 303 correctly state winner_ce_6m's different reward/value objective, winner CE, win-only reward and model value mode, warmup 200, decay 150,000, minimum LR ratio 0.02 and Muon weight decay 0.05. The Decision also names its entity limit, player mix and teacher chunk. The full recipe and batch-only variant are explicitly unadopted. The full recipe changes reward semantics; adopting only its 512×128/spm32 batch would be a recipe deviation, with 4× transitions per optimizer step under scaling_6m's schedule, requiring measurement.
- Decision line 45 accurately quotes 1.389 s/2.372 s model-only walls, 11,796/6,906 hypothetical global env steps/s and 0.90/0.95 scaling efficiencies. It excludes engine, copies, GAE, logging and all-reduce, identifies synthetic observations/fresh weights and GPU 0, and disclaims end-to-end prediction. Evidence is pinned in `component-evidence.json` to `24380f3eb0e9c7090d874f8f1a45c0d4e3477f40:ops/rebuild-2026-09-29/results.md`; this review did not rerun that GPU experiment. The source additionally limits it to one run per shape and no Nsight timeline.

## Fresh checks

Host: macOS 26.4 arm64, Python 3.12.13, torch 2.9.0, no CUDA; OMP_NUM_THREADS=2 was already in the environment. Cargo and rustc versions are in `host.json`.

| Command | Result | Receipt |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 87 passed (41 + 18 + 9 + 19), 0 failed; 0 doc tests | `engine.log` |
| `cargo test` | 164 passed, 2 ignored, 0 failed | `root.log` |
| `uv run python scripts/check_engine_trim.py` | Passed | `trim.log` |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,534 passed, 5 skipped, 0 failed in 38.17 s; no sharding/resource retry | `python.log` |
| `uv run mypy python/owl scripts` | Passed, 61 source files | `mypy.log` |
| `git diff --check b51b0c0...HEAD` | Passed | `diff-check.log` |

The five skips are unavailable native grammar binding (1), FlashAttention/CUDA (2), x86 quantized backend on arm64 (1), and native Kaggriculture env/seed integration (1). No config or startup case was skipped. An additional ad hoc shape-check invocation initially passed strings where `FullConfig.from_file` requires `Path`; correcting the verification invocation produced `shape-check.json`. This was not a repository failure.

## Scratch mutations and custody

Fresh scratch copy: `.codex-tmp/verify-8rank-r2-mutations`. All **15/15** deliberate mutations were detected by intended test failures (no collection failures). The baseline and restored targeted tests each passed **17/17**, with 137 deselected.

| Mutated condition | Oracle that detects it |
|---|---|
| spm 2 → 4 | Global work, divided shape, cross-rank shape and eight-rank headroom (4 failures) |
| envs 32 → 64 | Those four checks plus startup (5 failures) |
| teacher setting 128 → 64, effective chunk still 32 | Explicit retained teacher-setting assertion |
| Remove env world-size divisibility check | Required error identifying the nondivisible env quantity |
| Remove spm world-size divisibility check | Required ValueError replaced by ZeroDivisionError |
| Remove partial-minibatch check | Expected exception no longer raised |
| Remove partial-teacher-chunk check | Expected exception no longer raised |
| Change optimizer LR | New cross-rank optimizer equality |
| Change native threads | Cross-rank env equality including eight ranks |
| Change model preset | Cross-rank model equality including eight ranks |
| Change entropy coefficient | Cross-rank PPO equality including eight ranks |
| Remove startup workload-check call | Eight-rank startup headroom output missing |
| Replace teacher min with max | Independently isolated eight-rank row oracle |
| Add one trunk call | Independently isolated eight-rank trunk-call oracle |
| Add one head call | Independently isolated eight-rank head-call oracle |

`mutations/results.json` preserves exact commands, changes, exit codes and per-case hashes; individual logs preserve failures. All **187 copied scratch files**, including the native extension, were restored byte-for-byte, with matching before/after manifests and a restored passing baseline. No tracked source was mutated.

All **1,049 tracked worktree files** retain their original SHA-256 hashes in `tracked-before.json` and `tracked-after.json`. Staged and unstaged tracked diffs remain empty and HEAD is unchanged. Review additions are untracked receipts under this directory and ignored scratch files; previous receipts were not overwritten.

No real eight-rank launch, native Kaggriculture training, CUDA compilation/memory, live telemetry or complete-work GPU throughput was exercised. Those remain the documented downstream qualification scope, and are not defects in this config/plan change.

VERDICT: APPROVE
