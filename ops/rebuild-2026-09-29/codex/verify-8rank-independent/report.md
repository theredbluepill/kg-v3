# Independent verification of kg/rebuild-8rank

Review date: 2026-09-29. Reviewed HEAD `f9cbc6cc6bc380fde6d17140f92fba757cf990b2` against `b51b0c0382a13a72854860ae17d7798f0875e4ee...HEAD`; merge base is the supplied base. The worktree began clean. This review makes no tracked edits and starts no training or GPU run.

## Finding

- **P3 — cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md:45 — Correct the component timing hardware attribution.** The new paragraph says the component estimate “ran on 2 GPUs of a 2-GPU pod.” Its cited evidence, Check 4 in `kg/rebuild-gpu-checks:ops/rebuild-2026-09-29/results.md:412`, explicitly says “The Phase 2 timings ran alone on GPU 0.” The overall diagnostic bundle used GPUs 0 and 1; those are distinct scopes. **Fix:** state that the timings ran on one GPU (GPU 0) of the two-GPU pod. The same paragraph can also retain the evidence's precise attribution: the smaller train step is the common cost; at mid density the rollout and train-step contributions are nearly equal (+0.295 s / +0.300 s), while dense loss is train-step dominated. The evidence still supports the numeric estimates and their explicitly component-only scope.

Evidence source is pinned at `63921603e95ce8065f082c4a4ee031df4003dbb4`; `component-evidence.json` records its full file SHA-256 and numbered excerpt. There are no code/config correctness findings.

## Verified behavior and planning

- The 8-rank YAML loads through the real `FullConfig.from_file`. It has 32 envs/rank, 2 segments/minibatch, accumulation 1, horizon 64. Applying Isaiah's 6M division to `scaling_6m` yields 256 global envs, 16 optimizer steps/iteration, 16 global segments/optimizer step and 16,384 transitions/iteration. Both divided quantities and per-rank minibatches divide exactly. The teacher setting remains 128 and clamps to 32 segments, one complete chunk per rank.
- Optimizer and PPO settings match `scaling_6m` except for per-rank segments/minibatch; the 2-, 4- and 8-rank settings otherwise match each other. Workload oracles use the real minibatch generator.
- Startup workloads are 64 rollout rows, 256 minibatch rows, 4,096 teacher rows and 64 evaluation rows. Each fits one trunk and one head call. The integration test uses the shipped YAML and real loader, reaches the workload output, and then reaches the documented not-wired stop before allocating a run directory, env or model. This is a CPU startup-shape check with a stubbed distributed context, not an executed eight-rank launch.
- Task 1.4 now includes world sizes 1/2/4/8 for the planned disjoint seed-stream check. It agrees with L3 and the native-env brief: `base_seed + rank + k * world_size`, including successive resets, with checked overflow. This branch adds a plan item, not a completed native seed test.
- Phase 6.3b leaves dense-state memory, complete-work throughput, all-reduce, actual topology, CPU provisioning, every-rank log-ratio alarm, W&B and world-8 seed evidence open. It does not adopt an unmeasured result or authorize a new pod.
- The owner quote matches the supplied canonical wording exactly, including “likely” and “diagonose/building.” The Decision explicitly distinguishes expectation from final adoption.
- The retained `scaling_6m`, `winner_ce_6m`, `winner_ce_6m_4x5090`, `scaling_50m`, `scaling_12m`, `scaling_25m` and win-only model files are unchanged from pinned Isaiah `32b3ec900ad406eedd965f53a1a0f4490d31c589`. The pinned tree's only 8-GPU config is `scaling_50m`: 8 B200s, 256 envs/rank, spm 8, accumulation 2, horizon 64. The 6M division precedent comes from the separate `winner_ce_6m_4x5090` recipe.
- `winner_ce_6m` is correctly described as a different reward/value recipe, with the different schedule and weight decay. Neither its full recipe nor its batch shape alone is adopted or presented as an aligned fallback. Component timing numbers and efficiencies match their cited evidence; engine/copies/GAE/logging/all-reduce and end-to-end qualification are explicitly excluded.

## Requested checks

Host: macOS 26.4 arm64, Python 3.12.13, torch 2.9.0, CUDA unavailable; rustc 1.97.0-nightly (2026-04-17), cargo 1.97.0-nightly.

| Check | Result | Receipt |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 87 passed (41 + 18 + 9 + 19); 0 failed; 0 doc tests | `engine.log` |
| `cargo test` | 164 passed, 2 ignored, 0 failed | `root.log` |
| `uv run python scripts/check_engine_trim.py` | engine trim manifest: OK | `trim.log` |
| `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,534 passed, 5 skipped, 0 failed, 56.40 s; no sharding or resource-guard retry needed | `python.log` |
| `uv run mypy python/owl scripts` | Success; 61 source files | `mypy.log` |
| `git diff --check b51b0c0...HEAD` | passed | command result |

The five Python skips are the unavailable native Kaggriculture grammar binding (1), unavailable FlashAttention/CUDA backend (2), unavailable x86 quantized backend on this arm64 host (1), and the not-yet-wired native Kaggriculture env/seed integration (1). None is a config or startup workload case.

## Scratch mutation checks

All **12/12 scratch mutations failed for their intended reasons**, covering each new oracle/guard. The original and restored targeted selections each passed **17 tests**, with 137 deselected. Full mutation commands, failure output, per-case hashes and coverage mapping are in `mutations/report.md`, `mutations/results.json` and the individual logs.

| Mutated condition | Observed test failures |
|---|---:|
| 8-rank segments/minibatch 2 → 4 | 4: global workload, per-rank shape, cross-rank shape, 8-rank headroom |
| 8-rank envs 32 → 64 | 5: the above shape/headroom checks plus 8-rank startup |
| Teacher setting 128 → 64 (effective chunk remains 32) | 1: explicit teacher constant |
| Remove n_envs world-size divisibility guard | 1: expected explicit error identifies the wrong quantity |
| Remove spm world-size divisibility guard | 1: ZeroDivisionError instead of required ValueError |
| Remove partial-minibatch guard | 1: required exception not raised |
| Remove partial-teacher-chunk guard | 1: required exception not raised |
| Change optimizer LR | 1: cross-rank optimizer equality |
| Change native threads | 1: cross-rank env equality |
| Change model preset | 1: cross-rank model equality |
| Change entropy coefficient | 1: cross-rank PPO equality |
| Remove startup workload-check call | 1: new 8-rank startup case loses required headroom output |

All 174 scratch source/config files match their pre-mutation hashes and the original worktree bytes. The before/after scratch manifests are byte-identical (SHA-256 `1080a71e2f1b9d23010e3c9ace8e63276605e8034f776d3ed1e3626716a204a5`). All 1,049 tracked worktree files likewise retain their original SHA-256 hashes; staged and unstaged tracked diffs are empty. Only this untracked verification receipt directory was added.

## Scope limits

No real 8-rank distributed execution, native Kaggriculture training, CUDA memory/compilation, live telemetry or end-to-end throughput was exercised. Those remain the explicitly planned downstream qualification. This review verifies the branch's config, CPU startup workload, regression oracles and documentation against the stated scope.

VERDICT: APPROVE WITH EDITS
