Independent Task 1.5 verification
Scope: kg/rebuild-adapter HEAD 832b8363c1de958d45f6412ba58fdf0b01da83e5 against e197528.
Read material branch diff and first-parent history (43354c8, 6bc5290, 558ac3c, d0d65f7, 832b836), brief 1.5 including overriding review/deviation history, brief 1.4 ABI/rewards, contract/API/model docs, owner/cookbook rules and read-only Stage 2 prompt. Task 1.4 internals were excluded except requested reward integration/merge checks.
Merge: 7/7 brief stub declarations unique and AST-identical; all 101 first-parent and 104 second-parent log sections retained verbatim; index links/native-reference sources and historical content retained.
Production inspection: no concrete defect found in 35-buffer ownership, first-entry CUDA+pinned fence, strict zero-copy actions/masks, rank streams, independent reward oracle/native arithmetic, table validation/default, eval seeds or deliberate trainer stops. No introduced recurrent/history/opponent identity inputs, v2 model, alternate trainer, production dynamic dispatch or native fallback.

Checks (offline, OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true):
- uv run --offline pytest tests/kaggriculture tests/scripts/test_run_ppo.py -q -rs: exit 0, 1109 passed, 0 failed, 3 skipped (16.34 s).
- cargo test --locked --lib reward_admission: exit 0, 1 passed, 0 failed, 0 skipped, 278 filtered out.
- grep -rn "needs Task 1.4 binding" tests python scripts: exit 1, empty (expected).
- Restored focused suite with explicit dev rebuild: exit 0, 1109 passed, 0 failed, 3 skipped (29.11 s).
- Restored reward/native-env suite after supplemental mutations: exit 0, 387 passed, 0 failed, 0 skipped.
- Boundary probes of current code: exit 0, 10 passed, 0 failed; confirm missing test discriminators are rejected in production.
- MATURIN_PEP517_ARGS=--profile=dev uvx --offline --from rust-just just prepare: exit 0. Python 2224 passed, 0 failed, 6 skipped; root Rust 274 passed, 0 failed, 5 ignored; engine 41+9+19 passed, 0 failed/ignored; mypy 69 source files, formatting, lint, doc checks pass.
- Static merge audit: exit 0. engine_rs, tests/owl and python/owl/train unchanged. Protected-path stat has only merged Task 1.4 tests/tools changes: test_check_engine_trim.py +7/-2 and new test_record_kaggriculture_env_reference.py +531.
- git diff --exit-code, git diff --check, git status --short: exit 0, all empty. No commit, stage, stash, branch, worktree operation or cookbook edit.

Verification execution error: the first two automatic uv Rust mutation rebuilds used the implicit release profile (Cargo.toml enables LTO). The third was interrupted (exit 130), the source was restored, and all six Rust mutants plus restoration were rerun with explicit --profile=dev. This violated the no-release/LTO host constraint; final Rust mutation evidence is from dev builds. Preliminary logs are retained with implicit-release-interrupted-or-superseded suffix. No training, GPU or throughput run occurred.

Mutation accounting: 131 distinct mutations, 108 killed and 23 SURVIVED after appropriate test expansion. The table has 133 scored dev/Python attempts because two Rust mutations initially survived a narrow test selector and were rerun against the wider reward tests (rows 121/122 -> 131/132). Each mutation was restored byte-exactly; git diff --exit-code was checked after every restoration. Every scored failing run exited 1 with named pytest failure; every survivor exited 0. Rows 117/118 also have superseded preliminary release-profile failures; attempted release row 119 was interrupted, not scored.

Mutation table (guard-Ln means replace that if condition with False; each Field constraint was independently disabled; required-field mutations add explicit defaults):
ID | Mutation | File | Exit | Failed test(s) or SURVIVED | pytest count summary
000 | fence-no-pin | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_fence_condition[cuda-False-0] - ... | 1 failed, 38 passed in 0.08s
001 | fence-no-device | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_fence_condition[cpu-True-0] - As... | 1 failed, 38 passed in 0.05s
002 | fence-after-reset | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_fence_precedes_every_native_write | 1 failed, 38 passed in 0.06s
003 | fence-after-step | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_fence_precedes_every_native_write | 1 failed, 38 passed in 0.05s
004 | fence-after-truncate_envs | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_fence_precedes_every_native_write | 1 failed, 38 passed in 0.05s
005 | python-raw-weights | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients4-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients4-False] | 2 failed, 40 passed in 0.20s
006 | python-combined-rule | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients10-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients10-False] | 2 failed, 40 passed in 0.17s
007 | factory-drop-rank | python/owl/game.py | 1 | FAILED tests/kaggriculture/test_game.py::test_factory_rank_streams - assert (... ; FAILED tests/kaggriculture/test_game.py::test_real_binding_seed_streams - ass... | 2 failed, 25 passed in 0.05s
008 | factory-stride-one | python/owl/game.py | 1 | FAILED tests/kaggriculture/test_game.py::test_factory_rank_streams - assert (... ; FAILED tests/kaggriculture/test_game.py::test_real_binding_seed_streams - ass... | 2 failed, 25 passed in 0.03s
009 | codec-publish-early | python/owl/kaggriculture/codec.py | 1 | FAILED tests/kaggriculture/test_codec.py::test_codec_batch_failure_never_publishes_partial_output | 1 failed, 20 passed in 0.08s
010 | model-expected-default | python/owl/model/kaggriculture.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_model_default_loads_native_tables_once | 1 failed, 21 passed in 0.03s
011 | eval-unmixed-seed | scripts/run_ppo.py | 1 | FAILED tests/scripts/test_run_ppo.py::test_kaggriculture_eval_factory_arguments[cpu] ; FAILED tests/scripts/test_run_ppo.py::test_kaggriculture_eval_factory_arguments[cuda] ; FAILED tests/scripts/test_run_ppo.py::test_kaggriculture_native_evaluations_draw_fresh_reproducible_worlds | 3 failed, 26 passed, 78 deselected in 0.13s
012 | run_ppo.py-guard-L183 | scripts/run_ppo.py | 1 | FAILED tests/scripts/test_run_ppo.py::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation[kaggriculture_2rank.yaml-256-16384-3] ; FAILED tests/scripts/test_run_ppo.py::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation[kaggriculture_4rank.yaml-128-8192-2] ; FAILED tests/scripts/test_run_ppo.py::test_resume_startup_checks_the_runtime_adapted_workload | 3 failed, 8 passed, 96 deselected in 0.16s
013 | run_ppo.py-guard-L1425 | scripts/run_ppo.py | 1 | FAILED tests/scripts/test_run_ppo.py::test_kaggriculture_policy_evaluation_names_remaining_mapping_blocker | 1 failed, 106 deselected in 0.07s
014 | env.py-guard-L36 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_requested_pinning_without_cuda_rejected_before_allocation | 1 failed, 38 passed in 0.06s
015 | env.py-guard-L41 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_requested_pinning_silent_unpinned_allocation_rejected | 1 failed, 38 passed in 0.06s
016 | env.py-guard-L50 | python/owl/kaggriculture/env.py | 0 | SURVIVED | 39 passed in 0.05s
017 | env.py-guard-L143 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_invalid_masks_rejected[mask0] - ... | 1 failed, 38 passed in 0.06s
018 | env.py-guard-L145 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[dtype] ; FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[length_dtype] ; FAILED tests/kaggriculture/test_env.py::test_invalid_masks_rejected[mask1] - ... | 3 failed, 36 passed in 0.07s
019 | env.py-guard-L147 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[shape] ; FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[stride] ; FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[length_shape] ; FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[length_stride] ; FAILED tests/kaggriculture/test_env.py::test_invalid_masks_rejected[mask2] - ... ; FAILED tests/kaggriculture/test_env.py::test_invalid_masks_rejected[mask3] - ... | 6 failed, 33 passed in 0.08s
020 | env.py-guard-L177 | python/owl/kaggriculture/env.py | 0 | SURVIVED | 39 passed in 0.05s
021 | env.py-guard-L386 | python/owl/kaggriculture/env.py | 1 | FAILED tests/kaggriculture/test_env.py::test_invalid_actions_rejected[type] | 1 failed, 38 passed in 0.06s
022 | env.py-guard-L484 | python/owl/kaggriculture/env.py | 0 | SURVIVED | 39 passed in 0.05s
023 | env.py-guard-L175 | python/owl/kaggriculture/env.py | 0 | SURVIVED | 39 passed in 0.04s
024 | rewards.py-guard-L90 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.18s
025 | rewards.py-guard-L92 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.17s
026 | rewards.py-guard-L94 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.16s
027 | rewards.py-guard-L118 | python/owl/kaggriculture/rewards.py | 0 | SURVIVED | 42 passed in 0.16s
028 | rewards.py-guard-L129 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.16s
029 | rewards.py-guard-L131 | python/owl/kaggriculture/rewards.py | 0 | SURVIVED | 42 passed in 0.18s
030 | rewards.py-guard-L133 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.16s
031 | rewards.py-guard-L141 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.17s
032 | rewards.py-guard-L143 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.17s
033 | rewards.py-guard-L145 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.16s
034 | rewards.py-guard-L163 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.17s
035 | rewards.py-guard-L165 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity | 1 failed, 41 passed in 0.17s
036 | rewards.py-guard-L167 | python/owl/kaggriculture/rewards.py | 0 | SURVIVED | 42 passed in 0.18s
037 | rewards.py-guard-L51 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.19s
038 | rewards.py-guard-L65 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients8-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients8-False] | 2 failed, 40 passed in 0.17s
039 | rewards.py-guard-L76 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
040 | rewards.py-guard-L54 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients3-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients3-False] | 2 failed, 40 passed in 0.19s
041 | rewards.py-guard-L58 | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients2-False] ; FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients4-False] ; FAILED tests/kaggriculture/test_rewards.py::test_reward_admission_predicate_cases[coefficients10-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients2-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients4-False] ; FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients10-False] | 6 failed, 36 passed in 0.18s
042 | game.py-guard-L43 | python/owl/game.py | 0 | SURVIVED | 27 passed in 0.04s
043 | game.py-guard-L45 | python/owl/game.py | 0 | SURVIVED | 27 passed in 0.04s
044 | game.py-guard-L47 | python/owl/game.py | 1 | FAILED tests/kaggriculture/test_game.py::test_factory_invalid_rank_streams[patch4] ; FAILED tests/kaggriculture/test_game.py::test_factory_invalid_rank_streams[patch5] | 2 failed, 25 passed in 0.05s
045 | game.py-guard-L49 | python/owl/game.py | 0 | SURVIVED | 27 passed in 0.04s
046 | game.py-guard-L51 | python/owl/game.py | 0 | SURVIVED | 27 passed in 0.04s
047 | game.py-guard-L41 | python/owl/game.py | 1 | FAILED tests/kaggriculture/test_game.py::test_factory_invalid_rank_streams[patch6] ; FAILED tests/kaggriculture/test_game.py::test_factory_invalid_rank_streams[patch15] | 2 failed, 25 passed in 0.05s
048 | codec.py-guard-L59 | python/owl/kaggriculture/codec.py | 0 | SURVIVED | 21 passed in 0.07s
049 | codec.py-guard-L73 | python/owl/kaggriculture/codec.py | 1 | FAILED tests/kaggriculture/test_codec.py::test_codec_batch_rejects_mismatched_environment_count | 1 failed, 20 passed in 0.08s
050 | codec.py-guard-L77 | python/owl/kaggriculture/codec.py | 0 | SURVIVED | 21 passed in 0.07s
051 | codec.py-guard-L107 | python/owl/kaggriculture/codec.py | 0 | SURVIVED | 21 passed in 0.07s
052 | codec.py-guard-L109 | python/owl/kaggriculture/codec.py | 0 | SURVIVED | 21 passed in 0.07s
053 | tables-type(version) is not int | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[bool-version] | 1 failed, 21 passed in 0.03s
054 | tables-version != 1 | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[version] | 1 failed, 21 passed in 0.03s
055 | tables-not isinstance(names, tuple) | python/owl/kaggriculture/gpu_grammar.py | 0 | SURVIVED | 22 passed in 0.03s
056 | tables-names != kt.SLOT_NAMES | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[name] ; FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[name-order] | 2 failed, 20 passed in 0.03s
057 | tables-not isinstance(widths, tuple) | python/owl/kaggriculture/gpu_grammar.py | 0 | SURVIVED | 22 passed in 0.02s
058 | tables-any(type(width) is not int for width in widths) | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[float-widths] | 1 failed, 21 passed in 0.03s
059 | tables-widths != kt.SLOT_WIDTHS | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[width] ; FAILED tests/kaggriculture/test_native_tables.py::test_native_constants_rejected_before_tables_are_requested[arity] | 2 failed, 20 passed in 0.03s
060 | gpu_grammar.py-guard-L253 | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_arrays_reject_corruption[missing-key] | 1 failed, 21 passed in 0.03s
061 | gpu_grammar.py-guard-L260 | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_arrays_reject_corruption[non-array] | 1 failed, 21 passed in 0.03s
062 | gpu_grammar.py-guard-L262 | python/owl/kaggriculture/gpu_grammar.py | 0 | SURVIVED | 22 passed in 0.02s
063 | gpu_grammar.py-guard-L264 | python/owl/kaggriculture/gpu_grammar.py | 0 | SURVIVED | 22 passed in 0.02s
064 | gpu_grammar.py-guard-L269 | python/owl/kaggriculture/gpu_grammar.py | 1 | FAILED tests/kaggriculture/test_native_tables.py::test_native_arrays_reject_corruption[fortran] ; FAILED tests/kaggriculture/test_native_tables.py::test_native_arrays_reject_corruption[strided] | 2 failed, 20 passed in 0.03s
065 | types.py-guard-L141 | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
066 | types.py-guard-L148 | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
067 | game-count-bool-coercion | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
068 | game-count-nonintegral-coercion | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
069 | KaggricultureGameConfig-board_size-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
070 | KaggricultureGameConfig-board_size-le | python/owl/kaggriculture/types.py | 0 | SURVIVED | 27 passed in 0.04s
071 | KaggricultureGameConfig-episode_steps-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
072 | KaggricultureGameConfig-episode_steps-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
073 | KaggricultureGameConfig-starting_money-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
074 | KaggricultureGameConfig-starting_money-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
075 | KaggricultureGameConfig-max_market_orders_per_turn-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
076 | KaggricultureGameConfig-max_market_orders_per_turn-le | python/owl/kaggriculture/types.py | 0 | SURVIVED | 27 passed in 0.04s
077 | KaggricultureGameConfig-turns_per_day-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
078 | KaggricultureGameConfig-turns_per_day-le | python/owl/kaggriculture/types.py | 0 | SURVIVED | 27 passed in 0.04s
079 | KaggricultureGameConfig-shed_capacity-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
080 | KaggricultureGameConfig-shed_capacity-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
081 | KaggricultureGameConfig-weed_spawn_chance-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
082 | KaggricultureGameConfig-town_shop_unlock_interval-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
083 | KaggricultureGameConfig-town_shop_unlock_interval-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
084 | KaggricultureGameConfig-town_shop_sell_interval-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
085 | KaggricultureGameConfig-town_shop_sell_interval-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
086 | KaggricultureGameConfig-town_center_sell_interval-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
087 | KaggricultureGameConfig-town_center_sell_interval-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
088 | KaggricultureGameConfig-farm_hand_cost_mult-ge | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
089 | KaggricultureGameConfig-farm_hand_cost_mult-le | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
090 | KaggricultureGameConfig-market_params-max_length | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
091 | KaggricultureEnvConfig-n_envs-ge | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
092 | KaggricultureEnvConfig-n_envs-strict | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
093 | KaggricultureEnvConfig-seed-ge | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
094 | KaggricultureEnvConfig-seed-le | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
095 | KaggricultureEnvConfig-seed-strict | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
096 | KaggricultureEnvConfig-native_threads-ge | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
097 | KaggricultureEnvConfig-native_threads-strict | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.05s
098 | KaggricultureRewardConfig-econ_shaping-ge | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.18s
099 | KaggricultureRewardConfig-econ_shaping-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
100 | KaggricultureRewardConfig-econ_shaping-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.16s
101 | KaggricultureRewardConfig-econ_starvation_weight-ge | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.20s
102 | KaggricultureRewardConfig-econ_starvation_weight-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
103 | KaggricultureRewardConfig-econ_starvation_weight-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
104 | KaggricultureRewardConfig-econ_drought_weight-ge | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
105 | KaggricultureRewardConfig-econ_drought_weight-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.18s
106 | KaggricultureRewardConfig-econ_drought_weight-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
107 | KaggricultureRewardConfig-econ_cap-ge | python/owl/kaggriculture/rewards.py | 0 | SURVIVED | 42 passed in 0.16s
108 | KaggricultureRewardConfig-econ_cap-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
109 | KaggricultureRewardConfig-econ_cap-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.18s
110 | KaggricultureRewardConfig-econ_ineffective_weight-ge | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
111 | KaggricultureRewardConfig-econ_ineffective_weight-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
112 | KaggricultureRewardConfig-econ_ineffective_weight-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.16s
113 | KaggricultureRewardConfig-econ_ineffective_cap-ge | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
114 | KaggricultureRewardConfig-econ_ineffective_cap-allow_inf_nan | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
115 | KaggricultureRewardConfig-econ_ineffective_cap-strict | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.17s
116 | game-nonfinite-envelope | python/owl/kaggriculture/types.py | 1 | FAILED tests/kaggriculture/test_game.py::test_game_envelope - Failed: DID NOT... | 1 failed, 26 passed in 0.05s
117 | rust-raw-weights | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients4-False] ; FAILED tests/kaggriculture/test_native_env.py::test_reward_admission_shared_binary64_predicate[values4-False] | 2 failed, 59 passed, 326 deselected in 0.10s
118 | rust-combined-rule | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients10-False] ; FAILED tests/kaggriculture/test_native_env.py::test_reward_admission_shared_binary64_predicate[values6-False] | 2 failed, 59 passed, 326 deselected in 0.10s
119 | rust-no-death-cap | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients3-False] ; FAILED tests/kaggriculture/test_native_env.py::test_reward_admission_shared_binary64_predicate[values3-False] | 2 failed, 59 passed, 326 deselected in 0.09s
120 | rust-no-ineffective-cap | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_rewards.py::test_python_and_native_reward_admission_agree[coefficients8-False] ; FAILED tests/kaggriculture/test_native_env.py::test_reward_admission_shared_binary64_predicate[values9-False] | 2 failed, 59 passed, 326 deselected in 0.09s
121 | rust-cap-equality | src/kaggriculture/reward.rs | 0 | SURVIVED | 61 passed, 326 deselected in 0.08s
122 | rust-nonfinite-negative | src/kaggriculture/reward.rs | 0 | SURVIVED | 61 passed, 326 deselected in 0.08s
123 | required-econ_shaping | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.48s
124 | required-econ_starvation_weight | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.46s
125 | required-econ_drought_weight | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.49s
126 | required-econ_cap | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.47s
127 | required-econ_ineffective_weight | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.47s
128 | required-econ_ineffective_cap | python/owl/kaggriculture/rewards.py | 1 | FAILED tests/kaggriculture/test_rewards.py::test_reward_coefficients_required_and_bounded | 1 failed, 41 passed in 0.47s
129 | required-native-threads | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.16s
130 | required-reward-shaping | python/owl/kaggriculture/config.py | 1 | FAILED tests/kaggriculture/test_game.py::test_hire_limit_has_one_owner - Fail... | 1 failed, 26 passed in 0.15s
131 | rust-cap-equality-expanded | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_native_env.py::test_reward_dict_rejects_invalid_config[caps] | 1 failed, 69 passed, 317 deselected in 0.52s
132 | rust-nonfinite-negative-expanded | src/kaggriculture/reward.rs | 1 | FAILED tests/kaggriculture/test_native_env.py::test_reward_dict_rejects_invalid_config[nan] ; FAILED tests/kaggriculture/test_native_env.py::test_reward_dict_rejects_invalid_config[negative] | 2 failed, 68 passed, 317 deselected in 0.53s

Survivor interpretation:
- Real coverage gaps: env direct allocator/constructor admission; codec seat-count and decode tensor checks; game board upper bound and order upper bound with nonconflicting day product; inactive death-cap nonnegativity.
- Redundant/native-guaranteed: factory ranges also rejected by adapter/native; table tuple type/dtype/shape also rejected by equality or converter; adapter reward mode also rejected by serializer; diagnostic/decoded JSON object guards defend native return contracts; turns_per_day upper bound follows positive orders and product<=240.
- Unreachable under admitted inputs: finite economic-penalty output guard.
- CPU-only coverage limitation: reward cross-device guards.

Findings:
P3 tests/kaggriculture/test_codec.py:260 — Existing codec tests omit invalid seat counts and malformed decode batches; removing guards at codec.py:77,107,109 survives. Add too-short/too-long seat pairs and independently malformed tokens/lengths dtype, shape, stride and device; assert rejection before native decoding. Float lengths and extra length columns are useful discriminators.
P3 tests/kaggriculture/test_game.py:87 and tests/kaggriculture/test_rewards.py:67 — Boundary tests mask independent constraints. Add boardSize=11; maxMarketOrdersPerTurn=11 with turnsPerDay=1; econ_shaping=0 with econ_cap=-0.01. These distinguish the three surviving upper/nonnegative guard removals.
P3 tests/kaggriculture/test_env.py:126 — Direct allocator/constructor argument admission lacks invalid cases. Add allocate_observation_buffers(0,pin_memory=False) and direct-constructor boolean/noninteger/negative/out-of-i64 arguments with allocation/native sentinels to enforce fail-fast validation. Removing env.py:50 and :175 survives the current env suite.
P3 cookbook/references/index.md:27 (also :8,:10) — Current descriptions still say synthetic tables/native eval construction await Task 1.4. Reconcile the linked grammar-head Reference (:4,:24,:69), evaluation Reference (:4,:42,:73) and configs Reference (:55), retaining historical evidence while naming Task 3.1 rollout/action/evaluation mapping as the remaining blocker. Current required native/reward References and API/README/model docs are otherwise accurate.
No P1/P2 production finding.

Residual risks: Task 3.1 rollout storage and action/observation mapping are absent, so canonical Kaggriculture training and policy evaluation remain deliberately blocked. Pod test test_step_does_not_overwrite_pending_dma was inspected, not executed; it uses the real binding, preallocated DMA destinations, all 35 buffers and a fence-removal control. Two pinned-observation tests are likewise CUDA-only. Early two-rank smoke remains pending Task 3.1. No GPU, learning or throughput qualification is inferred.

Final git status --short: empty. All scratch/report artifacts are under /tmp/kg-task15-verify.
VERDICT: APPROVE WITH EDITS
