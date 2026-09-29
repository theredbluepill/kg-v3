# Independent 8-rank mutation verification

Reviewed branch `kg/rebuild-8rank` against `b51b0c0...HEAD`. All mutations used `.codex-tmp/verify-8rank-independent-mutations`, with `PYTHONPATH` pointing to the scratch `python/` tree and the preexisting `.venv/bin/python`. No tracked source was edited. Source `.pyc` caches were removed between cases to prevent timestamp/length caching from hiding a mutation.

## Baseline and restoration

The relevant original/restored selection (`isaiah_per_rank or ranked_config or eight_rank_workload or main_loads_kaggriculture`) passed **17 tests**, with **137 deselected**, before mutation (0.27s) and after restoration (0.24s). The full requested suites are reported separately by the parent verifier.

All **12 mutations were killed** (pytest exit 1 and the intended assertion/exception failure). The per-case before/mutated/restored SHA-256 hashes are in `results.json`; all 174 scratch source/config files have equal before/after hashes. The hash manifests are byte-identical, SHA-256 `1080a71e2f1b9d23010e3c9ace8e63276605e8034f776d3ed1e3626716a204a5`.

## Mutation mapping

| Mutation | Result | Restoration |
|---|---|---|
| `wrong_spm_4` | 4 failed, 46 deselected in 0.06s | byte-exact |
| `wrong_envs_64` | 5 failed, 149 deselected in 0.19s | byte-exact |
| `teacher_constant_64` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `remove_env_divisibility_guard` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `remove_spm_divisibility_guard` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `remove_minibatch_divisibility_guard` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `remove_teacher_divisibility_guard` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `wrong_optimizer_lr` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `wrong_env_threads` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `wrong_model` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `wrong_ppo_entropy` | 1 failed, 49 deselected in 0.05s | byte-exact |
| `remove_startup_workload_call` | 1 failed, 103 deselected in 0.15s | byte-exact |

The wrong-spm case fails the 8-rank global-workload, per-rank-shape, cross-rank-shape and headroom oracles. The wrong-envs case also fails the new 8-rank startup parameter. Teacher 128→64 preserves the effective 32-env chunk, so the explicit constant assertion alone detects this change. Removing the n_env divisibility guard causes the world-3 check to report the wrong quantity (`segments_per_minibatch`), which fails the expected explicit error message. Removing the spm guard causes a `ZeroDivisionError` at world 32, which fails the required `ValueError`. Removing either partial-division guard fails with “DID NOT RAISE”. Cross-rank environment, model, optimizer and PPO mutations fail their corresponding equality assertions. Removing `_check_model_workload` fails the new 8-rank startup case because its expected headroom lines disappear.

## Shape review

`kaggriculture_8rank.yaml` is 32 envs/rank, horizon 64, spm 2, accumulation 1 and teacher precompute slice 128 clamped to 32. The resulting global workload is 256 envs, 16 optimizer steps per iteration, 16 global segments per optimizer step and 16,384 transitions per iteration, equal to `scaling_6m`. Per-rank rollout/minibatch/teacher/evaluation seat-row counts are 64 / 256 / 4,096 / 64, each one trunk/head call under the documented padded limits. Optimizer and RL config equality to `scaling_6m` (except divided spm) passes.

No config/test correctness defect found. These checks are CPU shape/startup checks; they do not establish eight-rank execution, GPU throughput, seed independence or training behavior.
