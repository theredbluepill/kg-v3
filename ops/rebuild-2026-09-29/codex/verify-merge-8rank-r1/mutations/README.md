# Independent 8-rank merge mutation verification

Scope: newly added or extended config, shape, headroom and startup oracles in the three-dot diff from `0b8cf98ef57fc49a329dca4c8368c630586c4752` to this merge. Production runtime guards are unchanged; helper guard mutations are confined to the newly added test helper.

All mutations ran in `.codex-tmp/verify-merge-8rank-r1-mutations/`, a physical copy of configs, Python package, relevant tests/conftests and `scripts/run_ppo.py`. No tracked source was edited, even temporarily. Commands use the worktree's absolute `.venv/bin/python`, scratch `PYTHONPATH`, `OMP_NUM_THREADS=2`, `-B`, and pytest without cache. Every targeted scratch file was restored byte-for-byte in `finally`, with SHA-256 verification against its original and the untouched live source. `results.json` and `extended-results.json` preserve commands, mutation replacements, hashes, outputs, timing and restoration results; the two driver scripts reproduce the checks (the first requires a fresh scratch path).

Baseline/restored checks: 17 passed and 36 deselected in each core pass; 4 passed in each supplemental pass (two overlap core cases). All **20 mutations were killed**, with exit 1 and the expected assertion or exception text. No mutation survived or produced a collection/import error.

## Coverage matrix

Config test names below are in `tests/kaggriculture/test_configs.py`; startup is in `tests/scripts/test_run_ppo.py`. Every corresponding log is next to this report.

| Added/extended oracle | Scratch mutation | Failing test / intended observation | Receipt |
|---|---|---|---|
| Per-rank spm equals global spm/world size | 8-rank YAML spm 2 → 4 | `test_ranked_config_per_rank_shapes_are_scaling_6m_divided`: differing spm | `shape-spm-4.log` |
| Per-rank envs and teacher slice equal global division | 8-rank YAML n_envs 32 → 64 | same test: differing envs / teacher slice | `shape-envs-64.log` |
| Teacher setting remains fixed 128 despite slice clamp | teacher setting 128 → 64 (clamped slice stays 32) | same test: `assert 64 == 128` | `teacher-fixed-64.log` |
| Non-whole global env division must raise its named error | bypass only n_envs world-size guard | `test_isaiah_per_rank_shape_fails_loudly_when_not_divisible`: world size 3 fails expected-error regex, sees spm error instead | `remove-envs-divisibility-guard.log` |
| Non-whole global spm division must raise its named error | bypass only spm world-size guard | same test: world size 32 reaches ZeroDivisionError instead of the required ValueError | `remove-spm-divisibility-guard.log` |
| Partial per-rank minibatch must raise | disable n_envs modulo spm×accum guard | `test_isaiah_per_rank_shape_rejects_partial_minibatches_and_teacher_chunks`: DID NOT RAISE for accumulation 3 | `remove-partial-minibatch-guard.log` |
| Partial teacher chunk must raise | disable teacher divisibility guard | same test: DID NOT RAISE for teacher chunk 24 at world size 8 | `remove-partial-teacher-guard.log` |
| Expanded per-rank optimizer equality | 8-rank muon_lr .002 → .003 | `test_ranked_configs_differ_only_in_per_rank_shapes`: optimizer equality fails | `cross-rank-optimizer-drift.log` |
| Expanded per-rank env equality | 8-rank native_threads 2 → 3 | same test: env equality fails | `cross-rank-env-drift.log` |
| Expanded per-rank PPO equality | 8-rank ent_coef 1e-6 → 2e-6 | same test: PPO equality fails | `cross-rank-ppo-drift.log` |
| Expanded per-rank model equality | 8-rank model expands to the same config with depth 7 | same test: model equality fails | `cross-rank-model-drift.log` |
| Exact 8-rank rollout rows | production workload omits seat multiplier for rollout | `test_eight_rank_workloads_fit_the_model_chunking`: rollout 32 vs 64 | `workload-rollout-seats.log` |
| Exact 8-rank minibatch rows | production workload omits seat multiplier for minibatch | same test: minibatch 128 vs 256 | `workload-minibatch-seats.log` |
| Exact 8-rank teacher rows and trunk/head counts | production workload removes clamp to n_envs | same test: teacher (16384,3,2) vs (4096,1,1) | `workload-teacher-unclamped.log` |
| Exact 8-rank evaluation rows | production workload omits seat multiplier for evaluation | same test: evaluation 32 vs 64 | `workload-evaluation-seats.log` |
| Expanded real startup caller / headroom output | remove `_check_model_workload` call in copied `run_ppo.main` | `test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation`: all 3 rank cases fail because output lines are empty | `remove-startup-workload-caller.log` |
| Expanded global-workload equality | 8-rank n_envs 32 → 64 | `test_ranked_config_global_workload_equals_scaling_6m`: differing workload | `global-workload-envs64.log` |
| Expanded optimizer equality to scaling_6m | 8-rank muon_lr .002 → .003 | `test_ranked_config_optimizer_and_ppo_equal_scaling_6m`: optimizer equality fails | `scaling-optimizer-drift.log` |
| Expanded env/reward recipe oracle | 8-rank econ_shaping .2 → .1 | `test_config_env_and_cross_section_rules`: reward recipe equality fails | `env-reward-recipe-drift.log` |
| Expanded loader/model-construction smoke | remove Kaggriculture factory dispatch branch | `test_configs_load_through_full_config_and_build_the_model`: `assert_never` rejects unreachable config | `remove-model-factory-dispatch.log` |

Limits: CPU oracle sensitivity only. No GPU launch, DDP correctness, throughput or model quality qualification is implied. The world-size 32 guard mutation is killed by the missing explicit ValueError, with a downstream ZeroDivisionError; this is recorded rather than presented as a numerical assertion. The `-k 8rank` filters in the core driver also match the scratch directory, so those runs include all three rank cases: shape mutations fail the changed 8-rank case while 2/4 ranks pass; the startup-caller deletion fails all three. Supplemental cases use exact node IDs. Repeated recipe assertions and existing runtime guards were not exhaustively mutation-tested.
