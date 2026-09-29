# Complete Stage 2 command ledger

All commands use the offline/thread environment in results.md. Counts are per invocation, not additive; repeated red/green, umbrella and component runs overlap.

| Receipt | Command | Actual exit | Test counts / limits |
| --- | --- | ---: | --- |
| rust-mutant-old-table-compile.log | `rustc --edition=2021 --test /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/harness.rs -o /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests` | 0 | not a test-count command; see log diagnostics |
| rust-mutant-old-table.log | `/var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests reward_admission_predicate_cases --exact --nocapture` | 0 | test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| rust-mutant-strengthening-red-compile.log | `rustc --edition=2021 --test /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/harness.rs -o /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests` | 0 | not a test-count command; see log diagnostics |
| rust-mutant-strengthening-red.log | `/var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests reward_admission_predicate_cases --exact --nocapture` | 101 | test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| rust-restored-strengthening-green-compile.log | `rustc --edition=2021 --test /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/harness.rs -o /var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests` | 0 | not a test-count command; see log diagnostics |
| rust-restored-strengthening-green.log | `/var/folders/s_/d3vy5t5s4dv607gfkpx495000000gn/T/stage2-reward-1tz9leb7/reward-tests reward_admission_predicate_cases --exact --nocapture` | 0 | test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| reward-mutation-driver.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_reward_mutant.py` | 0 | test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s; test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s; test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| cargo-reward-green.log | `cargo test --locked --lib reward_admission_predicate_cases` | 0 | test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 278 filtered out; finished in 0.00s |
| abi-contract.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_contract.py` | 1 | not a test-count command; see log diagnostics |
| docs-status-red.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_docs.py` | 1 | not a test-count command; see log diagnostics |
| native-skip-marker.log | `grep -rn 'needs Task 1.4 binding' tests python scripts` | 1 | not a test-count command; see log diagnostics |
| docs-status-green.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_docs.py` | 0 | not a test-count command; see log diagnostics |
| abi-contract-final.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_contract.py` | 0 | not a test-count command; see log diagnostics |
| just-prepare.log | `uvx --offline --from rust-just just prepare` | -9 | budget terminated before a complete test summary |
| just-prepare-serial.log | `env RUST_TEST_THREADS=1 uvx --offline --from rust-just just prepare` | -9 | test result: ok. 274 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 33.05s; test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s; test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.79s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| py-prepare.log | `uvx --offline --from rust-just just py-prepare` | -9 | budget terminated before a complete test summary |
| kaggriculture-suite.log | `uv run --offline pytest tests/kaggriculture --ignore=tests/kaggriculture/test_base_generics_typing.py -q -rs` | -9 | budget terminated before a complete test summary |
| typing-isolated.log | `uv run --offline pytest --noconftest tests/kaggriculture/test_base_generics_typing.py -q -rs` | 0 | 1 passed in 6.81s |
| batched/tests-kaggriculture-test_base_generics_typing.log | `uv run --offline pytest tests/kaggriculture/test_base_generics_typing.py -q -rs --noconftest` | 0 | 1 passed in 6.57s |
| batched/tests-kaggriculture-test_codec.log | `uv run --offline pytest tests/kaggriculture/test_codec.py -q -rs` | 0 | 21 passed in 0.21s |
| batched/tests-kaggriculture-test_compile_gemm_backends.log | `uv run --offline pytest tests/kaggriculture/test_compile_gemm_backends.py -q -rs` | 0 | 55 passed in 0.49s |
| batched/tests-kaggriculture-test_configs.log | `uv run --offline pytest tests/kaggriculture/test_configs.py -q -rs` | 0 | 39 passed in 0.43s |
| batched/tests-kaggriculture-test_env.log | `uv run --offline pytest tests/kaggriculture/test_env.py -q -rs` | 0 | 39 passed in 0.13s |
| batched/tests-kaggriculture-test_env_cuda_fence.log | `uv run --offline pytest tests/kaggriculture/test_env_cuda_fence.py -q -rs` | 0 | 1 skipped in 0.01s |
| batched/tests-kaggriculture-test_env_reference.log | `uv run --offline pytest tests/kaggriculture/test_env_reference.py -q -rs` | 0 | 1 passed in 10.22s |
| batched/tests-kaggriculture-test_evaluation.log | `uv run --offline pytest tests/kaggriculture/test_evaluation.py -q -rs` | 0 | 13 passed in 0.01s |
| batched/tests-kaggriculture-test_game.log | `uv run --offline pytest tests/kaggriculture/test_game.py -q -rs` | 0 | 27 passed in 0.15s |
| batched/tests-kaggriculture-test_gpu_grammar.log | `uv run --offline pytest tests/kaggriculture/test_gpu_grammar.py -q -rs` | 0 | 24 passed in 0.02s |
| batched/tests-kaggriculture-test_model_compile.log | `uv run --offline pytest tests/kaggriculture/test_model_compile.py -q -rs` | 0 | 11 passed in 0.18s |
| batched/tests-kaggriculture-test_model_encoder.log | `uv run --offline pytest tests/kaggriculture/test_model_encoder.py -q -rs` | 0 | 43 passed in 0.96s |
| batched/tests-kaggriculture-test_model_heads.log | `uv run --offline pytest tests/kaggriculture/test_model_heads.py -q -rs` | -9 | budget terminated before a complete test summary |
| batched/tests-kaggriculture-test_model_registration.log | `uv run --offline pytest tests/kaggriculture/test_model_registration.py -q -rs` | 0 | 7 passed in 0.05s |
| batched/tests-kaggriculture-test_native_env.log | `uv run --offline pytest tests/kaggriculture/test_native_env.py -q -rs` | 0 | 345 passed in 1.90s |
| batched/tests-kaggriculture-test_native_grammar_bindings.log | `uv run --offline pytest tests/kaggriculture/test_native_grammar_bindings.py -q -rs` | 0 | 43 passed in 0.23s |
| batched/tests-kaggriculture-test_native_tables.log | `uv run --offline pytest tests/kaggriculture/test_native_tables.py -q -rs` | 0 | 22 passed in 0.03s |
| batched/tests-kaggriculture-test_observe.log | `uv run --offline pytest tests/kaggriculture/test_observe.py -q -rs` | 0 | 55 passed, 2 skipped in 2.44s |
| batched/tests-kaggriculture-test_rewards.log | `uv run --offline pytest tests/kaggriculture/test_rewards.py -q -rs` | 0 | 42 passed in 0.47s |
| batched/tests-kaggriculture-test_training_semantics.log | `uv run --offline pytest tests/kaggriculture/test_training_semantics.py -q -rs` | 0 | 23 passed in 0.05s |
| batched/tests-kaggriculture-test_types.log | `uv run --offline pytest tests/kaggriculture/test_types.py -q -rs` | 0 | 126 passed in 0.43s |
| batched/tests-owl-agent-test_agent.log | `uv run --offline pytest tests/owl/agent/test_agent.py -q -rs` | 0 | 53 passed in 0.99s |
| batched/tests-owl-agent-test_kaggle_engine_sanitization.log | `uv run --offline pytest tests/owl/agent/test_kaggle_engine_sanitization.py -q -rs` | 0 | 4 passed in 0.57s |
| batched/tests-owl-agent-test_kaggle_observation.log | `uv run --offline pytest tests/owl/agent/test_kaggle_observation.py -q -rs` | 0 | 3 passed in 0.01s |
| batched/tests-owl-model-test_attn.log | `uv run --offline pytest tests/owl/model/test_attn.py -q -rs` | 0 | 5 passed, 2 skipped in 0.01s |
| batched/tests-owl-model-test_model_config_files.log | `uv run --offline pytest tests/owl/model/test_model_config_files.py -q -rs` | 0 | 47 passed in 0.21s |
| batched/tests-owl-model-test_recurrent_transformer_v1.log | `uv run --offline pytest tests/owl/model/test_recurrent_transformer_v1.py -q -rs` | 0 | 14 passed in 0.09s |
| batched/tests-owl-model-test_stateless_transformer_v1.log | `uv run --offline pytest tests/owl/model/test_stateless_transformer_v1.py -q -rs` | 0 | 126 passed in 0.68s |
| batched/tests-owl-model-test_teacher_targets.log | `uv run --offline pytest tests/owl/model/test_teacher_targets.py -q -rs` | 0 | 28 passed in 0.02s |
| batched/tests-owl-test_checkpoint_quantization.log | `uv run --offline pytest tests/owl/test_checkpoint_quantization.py -q -rs` | 0 | 44 passed in 0.04s |
| batched/tests-owl-test_config.log | `uv run --offline pytest tests/owl/test_config.py -q -rs` | 0 | 9 passed in 0.02s |
| batched/tests-owl-test_int8_emulation.log | `uv run --offline pytest tests/owl/test_int8_emulation.py -q -rs` | 0 | 3 passed, 1 skipped in 0.02s |
| batched/tests-owl-test_replay.log | `uv run --offline pytest tests/owl/test_replay.py -q -rs` | 0 | 3 passed in 0.01s |
| batched/tests-owl-test_rl.log | `uv run --offline pytest tests/owl/test_rl.py -q -rs` | 0 | 78 passed in 0.53s |
| batched/tests-owl-train-test_advantages.log | `uv run --offline pytest tests/owl/train/test_advantages.py -q -rs` | 0 | 14 passed in 0.02s |
| batched/tests-owl-train-test_config.log | `uv run --offline pytest tests/owl/train/test_config.py -q -rs` | 0 | 68 passed in 0.08s |
| batched/tests-owl-train-test_distributed.log | `uv run --offline pytest tests/owl/train/test_distributed.py -q -rs` | 0 | 19 passed in 0.02s |
| batched/tests-owl-train-test_logging.log | `uv run --offline pytest tests/owl/train/test_logging.py -q -rs` | 0 | 2 passed in 0.01s |
| batched/tests-owl-train-test_loss.log | `uv run --offline pytest tests/owl/train/test_loss.py -q -rs` | 0 | 13 passed in 0.03s |
| batched/tests-owl-train-test_metrics.log | `uv run --offline pytest tests/owl/train/test_metrics.py -q -rs` | 0 | 2 passed in 0.02s |
| batched/tests-owl-train-test_optimizer.log | `uv run --offline pytest tests/owl/train/test_optimizer.py -q -rs` | 0 | 11 passed in 0.03s |
| batched/tests-owl-train-test_ppo.log | `uv run --offline pytest tests/owl/train/test_ppo.py -q -rs` | 0 | 75 passed in 0.27s |
| batched/tests-owl-train-test_ppo_observation_mapping.log | `uv run --offline pytest tests/owl/train/test_ppo_observation_mapping.py -q -rs` | 0 | 171 passed in 0.09s |
| batched/tests-scripts-test_benchmark_checkpoints.log | `uv run --offline pytest tests/scripts/test_benchmark_checkpoints.py -q -rs` | 0 | 37 passed in 0.04s |
| batched/tests-scripts-test_download_replays.log | `uv run --offline pytest tests/scripts/test_download_replays.py -q -rs` | 0 | 4 passed in 0.01s |
| batched/tests-scripts-test_extract_model_weights.log | `uv run --offline pytest tests/scripts/test_extract_model_weights.py -q -rs` | 0 | 17 passed in 0.03s |
| batched/tests-scripts-test_kaggriculture_parity.log | `uv run --offline pytest tests/scripts/test_kaggriculture_parity.py -q -rs` | -9 | budget terminated before a complete test summary |
| batched/tests-scripts-test_kaggriculture_select_replays.log | `uv run --offline pytest tests/scripts/test_kaggriculture_select_replays.py -q -rs` | 0 | 17 passed in 0.24s |
| batched/tests-scripts-test_roundtrip_checkpoint_quantization.log | `uv run --offline pytest tests/scripts/test_roundtrip_checkpoint_quantization.py -q -rs` | 0 | 18 passed in 0.04s |
| batched/tests-scripts-test_run_ppo.log | `uv run --offline pytest tests/scripts/test_run_ppo.py -q -rs` | 0 | 107 passed in 0.25s |
| batched/tests-tools-test_check_engine_trim.log | `uv run --offline pytest tests/tools/test_check_engine_trim.py -q -rs` | 0 | 79 passed in 2.73s |
| batched/tests-tools-test_observation_oracle_custody.log | `uv run --offline pytest tests/tools/test_observation_oracle_custody.py -q -rs` | 0 | 46 passed in 0.46s |
| batched/tests-tools-test_record_grammar_reference.log | `uv run --offline pytest tests/tools/test_record_grammar_reference.py -q -rs` | 0 | 25 passed in 9.56s |
| batched/tests-tools-test_record_kaggriculture_env_reference.log | `uv run --offline pytest tests/tools/test_record_kaggriculture_env_reference.py -q -rs` | 0 | 60 passed in 6.49s |
| batched/tests-tools-test_replay_viewer.log | `uv run --offline pytest tests/tools/test_replay_viewer.py -q -rs` | 0 | 2 passed in 0.05s |
| heads-remaining.log | `uv run --offline pytest tests/kaggriculture/test_model_heads.py -k 'not hire_coupling_enumeration_equals_replayed_density' -q -rs` | 0 | 60 passed, 5 deselected in 7.84s |
| heads-budget10.log | `uv run --offline pytest 'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density[10]' -q -rs` | 0 | 1 passed in 0.18s |
| heads-budgets0to3.log | `uv run --offline pytest 'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density[0]' 'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density[1]' 'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density[2]' 'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density[3]' -q -rs` | 0 | 4 passed in 0.73s |
| parity-isolated.log | `uv run --offline pytest --noconftest tests/scripts/test_kaggriculture_parity.py -q -rs` | 0 | 18 passed in 5.95s |
| rs-prepare.log | `env RUST_TEST_THREADS=1 uvx --offline --from rust-just just rs-prepare` | 0 | test result: ok. 274 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 30.53s; test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s; test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.57s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| cargo-fmt-check.log | `cargo fmt --check` | 0 | not a test-count command; see log diagnostics |
| final-docs-lint.log | `uvx --offline pymarkdownlnt scan README.md docs/rl-api-specs.md docs/model-architecture.md` | 0 | not a test-count command; see log diagnostics |
| final-docs-fresh.log | `uvx --offline --from rust-just just docs-fresh` | 0 | not a test-count command; see log diagnostics |
| final-abi-contract.log | `python3 ops/rebuild-2026-09-29/stage2-adapter/check_contract.py` | 0 | not a test-count command; see log diagnostics |
| final-missing-binding-grep.log | `grep -rn 'needs Task 1.4 binding' tests python scripts` | 1 | not a test-count command; see log diagnostics |
| final-orbit-diff.log | `git diff --stat 558ac3c -- tests/owl tests/tools` | 0 | not a test-count command; see log diagnostics |
| final-diff-check.log | `git diff --check` | 0 | not a test-count command; see log diagnostics |
| native/unskip-red.log | `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py -q` | 1 | 1 failed, 102 passed, 1 skipped in 0.25s |
| native/native-expanded-red.log | `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py -q` | 1 | 1 failed, 128 passed, 1 skipped in 0.41s |
| native/format-lint-0.log | `uvx --offline ruff format python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` | 0 | not a test-count command; see log diagnostics |
| native/format-lint-1.log | `uvx --offline ruff check python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` | 1 | not a test-count command; see log diagnostics |
| native/format-green.log | `uvx --offline ruff format python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` | 0 | not a test-count command; see log diagnostics |
| native/lint-green.log | `uvx --offline ruff check python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` | 0 | not a test-count command; see log diagnostics |
| native/env-green.log | `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_env_cuda_fence.py -q` | 0 | 39 passed, 1 skipped in 0.09s |
| native/rewards-green.log | `uv run --offline pytest tests/kaggriculture/test_rewards.py -q` | 0 | 42 passed in 0.19s |
| native/game-green.log | `uv run --offline pytest tests/kaggriculture/test_game.py -q` | 0 | 27 passed in 0.05s |
| native/codec-green.log | `uv run --offline pytest tests/kaggriculture/test_codec.py -q` | 0 | 21 passed in 0.08s |
| tables/format.log | `uv run --offline ruff format python/owl/kaggriculture/gpu_grammar.py python/owl/model/kaggriculture.py tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py` | 0 | not a test-count command; see log diagnostics |
| tables/green.log | `uv run --offline pytest tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py -q` | 0 | 46 passed in 0.05s |
| tables/lint.log | `uv run --offline ruff check python/owl/kaggriculture/gpu_grammar.py python/owl/model/kaggriculture.py tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py` | 0 | not a test-count command; see log diagnostics |
| tables/red.log | `uv run --offline pytest tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py -q` | 1 | 22 failed, 24 passed in 0.12s |
| eval/diff_check.log | `git diff --check` | 0 | not a test-count command; see log diagnostics |
| eval/format.log | `uvx --offline ruff format scripts/run_ppo.py tests/scripts/test_run_ppo.py` | 0 | not a test-count command; see log diagnostics |
| eval/green.log | `uv run --offline pytest tests/scripts/test_run_ppo.py -k 'create_eval_env or kaggriculture_eval or kaggriculture_native_eval or policy_evaluation or main_loads_kaggriculture or resume_startup_checks' -q` | 1 | 1 failed, 9 passed, 97 deselected in 0.11s |
| eval/green2.log | `uv run --offline pytest tests/scripts/test_run_ppo.py -k 'create_eval_env or kaggriculture_eval or kaggriculture_native_eval or policy_evaluation or main_loads_kaggriculture or resume_startup_checks' -q` | 1 | 1 failed, 9 passed, 97 deselected in 0.21s |
| eval/green3.log | `uv run --offline pytest tests/scripts/test_run_ppo.py -k 'create_eval_env or kaggriculture_eval or kaggriculture_native_eval or policy_evaluation or main_loads_kaggriculture or resume_startup_checks' -q` | 0 | 10 passed, 97 deselected in 0.18s |
| eval/lint.log | `uvx --offline ruff check scripts/run_ppo.py tests/scripts/test_run_ppo.py` | 1 | not a test-count command; see log diagnostics |
| eval/lint2.log | `uvx --offline ruff check scripts/run_ppo.py tests/scripts/test_run_ppo.py` | 0 | not a test-count command; see log diagnostics |
| eval/mypy.log | `uv run --offline mypy scripts/run_ppo.py` | 0 | not a test-count command; see log diagnostics |
| eval/red.log | `uv run --offline pytest tests/scripts/test_run_ppo.py -k 'create_eval_env or kaggriculture_eval or kaggriculture_native_eval or policy_evaluation or main_loads_kaggriculture or resume_startup_checks' -q` | 1 | 9 failed, 1 passed, 97 deselected in 0.38s |

The successful `rs-prepare` recipe also records these nested commands, each exit 0 (just would stop on any nonzero exit):

- `uv run python scripts/check_engine_trim.py`
- `cargo fmt`
- `cargo fmt --manifest-path engine_rs/Cargo.toml --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo clippy --manifest-path engine_rs/Cargo.toml --all-targets --locked -- -D warnings -A clippy::too_many_arguments -A clippy::collapsible_if -A clippy::needless_range_loop`
- `cargo test`: 274 passed, 5 ignored, 0 failed
- `cargo test --manifest-path engine_rs/Cargo.toml --locked`: 69 passed, 0 failed, 0 ignored
- `uv run python scripts/check_doc_freshness.py`

The preparation attempts also completed (exit 0): `uv run maturin develop`; `uvx ruff check python/ scripts/ tests/ --select I --fix`; `uvx ruff format python/ scripts/ tests/`; `uv run python scripts/check_python_311_syntax.py`; `uvx ruff check python/ scripts/ tests/`; `uvx pymarkdownlnt scan *.md`; `uvx pymarkdownlnt scan --recurse python/ scripts/ tests/ src/ docs/`; `uv run mypy python/ scripts/` (69 source files). `uv run pytest tests/` / `uv run pytest tests/ -m "not slow"` did not finish before the watchdog killed their process group.
