# Independent BC mutation verification

100 unique mutants: **27 killed, 73 survived, 0 setup errors, 0 timeouts**. Two survivors were rerun against all 18 BC tests: 102 executions total, 27 killed and 75 survived. Counts are test sensitivity observations, not a correctness score.

Baseline and restored baseline: **18 passed** each (2.29 s before, 2.20 s after). Original tracked files were never mutated. Every scratch mutation was restored byte-for-byte with SHA-256 verification. `final-custody.json` confirms pre/post source equality and `git diff --exit-code HEAD --` returned 0. `source-resolution.log` verifies imports came from the scratch tree.

The runner initially contained a syntax error before any mutation was applied; it was corrected before the recorded 102 executions. This is not a source mutation or killed mutant.

## Surviving changes

- Removing `_train_step` optimizer.step leaves all18 passing. The toy loss-decrease test invokes the optimizer directly outside the BC loop; integrated tests do not require model updates.
- Negating both `bc_loss` terms survives its targeted test and all18. Increasingly negative values satisfy final loss < half initial loss, so the assertion does not establish objective sign.
- Omitting program-length normalization survives all18: no numerical expected objective checks turn weighting.
- Removing rank from the permutation seed survives all18. Rank partitions remain disjoint; different partition sizes in the fixture cause different permutations even without rank. This changes order but does not create overlap.
- Removing autocast, compile, DDP, compile-stack/workload checks, accumulation division, scheduler stepping, clipping/nonfinite checking, train preflight and runtime cap survives. CPU tests use FP32, no compile, one rank, accumulation1 and no runtime cap. These are coverage limits, not evidence that the original implementation fails.
- Most explicit input guards and Pydantic schema/bounds survive because fixtures provide valid inputs. Several removed guards still have downstream failure mechanisms: file reads, index_select, method calls on absent logits or keyed access/unpacking of malformed state. Survivors do not prove unchanged code accepts invalid input.

## Kill interpretation

Most kills are intended assertion or expected-exception failures. The following need narrower interpretation:

- int64 restore: the observation contract dtype check rejects int32.
- rank partition: a row-count mismatch fails tensor assignment before disjointness assertions.
- old schema: exact-array-name validation still rejects the shard; the test fails only because the diagnostic no longer matches the schema regex.
- resume step reset: evaluation-step advancement rejects the repeated step.
- patience stop: direct selection assertion fails; the scripted integration evaluator also exhausts its iterator.
- zero checkpoint weights: saved-weight snapshot comparison fails. The PPO loader pair itself passes because both models load identical zero weights.

## Scope and evidence

Mutated every new test (18), explicit validation guard, BCConfig bound (10), manifest field/schema guard (9), and exclusive writer mode (2), plus objective/training/reuse mechanisms. Reused observation-contract internals and optimizer validators were not individually mutated. No real-data, GPU, distributed, compiled or W&B training was launched.

Exact substitutions, selectors and restored hashes are in the two manifest/results pairs. Full stdout/stderr is in the linked logs. A selector of all18 means the complete tests/kaggriculture/test_bc.py suite.

| Run | Mutation | Result | Selector |
| --- | --- | --- | --- |
| [001](001-round_trip_restore_int64.log) | `round_trip_restore_int64` | KILLED | shards_round_trip |
| [002](002-tampered_hash.log) | `tampered_hash` | KILLED | tampered |
| [003](003-duplicate_episode.log) | `duplicate_episode` | KILLED | tampered |
| [004](004-old_schema.log) | `old_schema` | KILLED | foreign_shard |
| [005](005-missing_split.log) | `missing_split` | KILLED | both_splits |
| [006](006-unpaired_rows.log) | `unpaired_rows` | KILLED | both_splits |
| [007](007-integer_range.log) | `integer_range` | KILLED | range_checks |
| [008](008-rank_partition.log) | `rank_partition` | KILLED | fake_ranks |
| [009](009-winner_draw.log) | `winner_draw` | KILLED | winner_targets |
| [010](010-winner_seat_order.log) | `winner_seat_order` | KILLED | winner_targets |
| [011](011-loss_sign.log) | `loss_sign` | SURVIVED | loss_decreases |
| [012](012-critic_omitted.log) | `critic_omitted` | KILLED | every_parameter |
| [013](013-rank_seed.log) | `rank_seed` | SURVIVED | sharding_is_deterministic |
| [014](014-epoch_seed.log) | `epoch_seed` | KILLED | sharding_is_deterministic |
| [015](015-small_partition.log) | `small_partition` | KILLED | sharding_is_deterministic |
| [016](016-min_delta.log) | `min_delta` | KILLED | held_out_selection |
| [017](017-eval_step_advance.log) | `eval_step_advance` | KILLED | held_out_selection |
| [018](018-finite_nll.log) | `finite_nll` | KILLED | held_out_selection |
| [019](019-patience_stop.log) | `patience_stop` | KILLED | held_out_selection or training_keeps |
| [020](020-best_save_unconditional.log) | `best_save_unconditional` | KILLED | training_keeps |
| [021](021-resume_step_reset.log) | `resume_step_reset` | KILLED | restart_from_state |
| [022](022-resume_manifest_guard.log) | `resume_manifest_guard` | KILLED | resume_rejects |
| [023](023-resume_world_guard.log) | `resume_world_guard` | KILLED | resume_rejects |
| [024](024-ppo_checkpoint_schema.log) | `ppo_checkpoint_schema` | KILLED | best_checkpoint_loads |
| [025](025-ppo_checkpoint_optimizer_step.log) | `ppo_checkpoint_optimizer_step` | KILLED | best_checkpoint_loads |
| [026](026-bc_workload_validation_removed.log) | `bc_workload_validation_removed` | KILLED | two_rank_bc_config |
| [027](027-script_provenance_ignored.log) | `script_provenance_ignored` | KILLED | script_runs |
| [028](028-observation_field_removed.log) | `observation_field_removed` | KILLED | contract_shapes |
| [029](029-writer_paired_shape.log) | `writer_paired_shape` | SURVIVED | all18 |
| [030](030-loader_rank.log) | `loader_rank` | SURVIVED | all18 |
| [031](031-loader_manifest_exists.log) | `loader_manifest_exists` | SURVIVED | all18 |
| [032](032-banks_finite.log) | `banks_finite` | SURVIVED | all18 |
| [033](033-shard_exists.log) | `shard_exists` | SURVIVED | all18 |
| [034](034-shard_bytes.log) | `shard_bytes` | SURVIVED | all18 |
| [035](035-shard_arrays.log) | `shard_arrays` | SURVIVED | all18 |
| [036](036-shard_admitted_shape.log) | `shard_admitted_shape` | SURVIVED | all18 |
| [037](037-turn_layout.log) | `turn_layout` | SURVIVED | all18 |
| [038](038-turn_increasing.log) | `turn_increasing` | SURVIVED | all18 |
| [039](039-action_layout.log) | `action_layout` | SURVIVED | all18 |
| [040](040-observation_contract.log) | `observation_contract` | SURVIVED | all18 |
| [041](041-length_bounds.log) | `length_bounds` | SURVIVED | all18 |
| [042](042-tokens_nonnegative.log) | `tokens_nonnegative` | SURVIVED | all18 |
| [043](043-shard_path.log) | `shard_path` | SURVIVED | all18 |
| [044](044-gather_nonempty.log) | `gather_nonempty` | SURVIVED | all18 |
| [045](045-gather_range.log) | `gather_range` | SURVIVED | all18 |
| [046](046-obs_field_tensor.log) | `obs_field_tensor` | SURVIVED | all18 |
| [047](047-config_game.log) | `config_game` | SURVIVED | all18 |
| [048](048-sharding_step_micro.log) | `sharding_step_micro` | SURVIVED | all18 |
| [049](049-sharding_rank.log) | `sharding_rank` | SURVIVED | all18 |
| [050](050-winner_shape.log) | `winner_shape` | SURVIVED | all18 |
| [051](051-winner_log_probs_present.log) | `winner_log_probs_present` | SURVIVED | all18 |
| [052](052-evaluation_nonempty.log) | `evaluation_nonempty` | SURVIVED | all18 |
| [053](053-resume_state_keys.log) | `resume_state_keys` | SURVIVED | all18 |
| [054](054-resume_selection_mapping.log) | `resume_selection_mapping` | SURVIVED | all18 |
| [055](055-resume_lr_schedule.log) | `resume_lr_schedule` | SURVIVED | all18 |
| [056](056-trainer_rank_context.log) | `trainer_rank_context` | SURVIVED | all18 |
| [057](057-history_exists.log) | `history_exists` | SURVIVED | all18 |
| [058](058-script_provenance_mapping.log) | `script_provenance_mapping` | SURVIVED | all18 |
| [059](059-script_override_syntax.log) | `script_override_syntax` | SURVIVED | all18 |
| [060](060-script_override_duplicate.log) | `script_override_duplicate` | SURVIVED | all18 |
| [061](061-script_fresh_config_exists.log) | `script_fresh_config_exists` | SURVIVED | all18 |
| [062](062-script_resume_dir.log) | `script_resume_dir` | SURVIVED | all18 |
| [063](063-script_resume_overrides.log) | `script_resume_overrides` | SURVIVED | all18 |
| [064](064-script_resume_source_override.log) | `script_resume_source_override` | SURVIVED | all18 |
| [065](065-script_runtime_positive.log) | `script_runtime_positive` | SURVIVED | all18 |
| [066](066-config_seed.log) | `config_seed` | SURVIVED | all18 |
| [067](067-config_rows_per_rank.log) | `config_rows_per_rank` | SURVIVED | all18 |
| [068](068-config_gradient_accumulation_steps.log) | `config_gradient_accumulation_steps` | SURVIVED | all18 |
| [069](069-config_max_grad_norm.log) | `config_max_grad_norm` | SURVIVED | all18 |
| [070](070-config_value_coef.log) | `config_value_coef` | SURVIVED | all18 |
| [071](071-config_eval_interval_steps.log) | `config_eval_interval_steps` | SURVIVED | all18 |
| [072](072-config_patience_evals.log) | `config_patience_evals` | SURVIVED | all18 |
| [073](073-config_min_delta.log) | `config_min_delta` | SURVIVED | all18 |
| [074](074-config_max_steps.log) | `config_max_steps` | SURVIVED | all18 |
| [075](075-config_validation_rows_per_forward.log) | `config_validation_rows_per_forward` | SURVIVED | all18 |
| [076](076-policy_length_normalization.log) | `policy_length_normalization` | SURVIVED | all18 |
| [077](077-autocast_seam.log) | `autocast_seam` | SURVIVED | all18 |
| [078](078-compile_seam.log) | `compile_seam` | SURVIVED | all18 |
| [079](079-ddp_seam.log) | `ddp_seam` | SURVIVED | all18 |
| [080](080-workload_guard_seam.log) | `workload_guard_seam` | SURVIVED | all18 |
| [081](081-compile_stack_seam.log) | `compile_stack_seam` | SURVIVED | all18 |
| [082](082-optimizer_step_removed.log) | `optimizer_step_removed` | SURVIVED | all18 |
| [083](083-gradient_accumulation_scaling.log) | `gradient_accumulation_scaling` | SURVIVED | all18 |
| [084](084-gradient_clipping_disabled.log) | `gradient_clipping_disabled` | SURVIVED | all18 |
| [085](085-nonfinite_gradient_guard.log) | `nonfinite_gradient_guard` | SURVIVED | all18 |
| [086](086-scheduler_step_removed.log) | `scheduler_step_removed` | SURVIVED | all18 |
| [087](087-preflight_disabled.log) | `preflight_disabled` | SURVIVED | all18 |
| [088](088-runtime_cap_removed.log) | `runtime_cap_removed` | SURVIVED | all18 |
| [089](089-best_checkpoint_weights_zero.log) | `best_checkpoint_weights_zero` | KILLED | best_checkpoint_loads or training_keeps |
| [extra/001](extra/001-manifest_episode_id_nonempty.log) | `manifest_episode_id_nonempty` | SURVIVED | all18 |
| [extra/002](extra/002-manifest_admitted_nonnegative.log) | `manifest_admitted_nonnegative` | SURVIVED | all18 |
| [extra/003](extra/003-manifest_shard_path_nonempty.log) | `manifest_shard_path_nonempty` | SURVIVED | all18 |
| [extra/004](extra/004-manifest_sha256_format.log) | `manifest_sha256_format` | SURVIVED | all18 |
| [extra/005](extra/005-manifest_shard_bytes_positive.log) | `manifest_shard_bytes_positive` | SURVIVED | all18 |
| [extra/006](extra/006-manifest_schema_literal.log) | `manifest_schema_literal` | SURVIVED | all18 |
| [extra/007](extra/007-manifest_observation_schema_literal.log) | `manifest_observation_schema_literal` | SURVIVED | all18 |
| [extra/008](extra/008-manifest_contract_document_literal.log) | `manifest_contract_document_literal` | SURVIVED | all18 |
| [extra/009](extra/009-manifest_split_literal.log) | `manifest_split_literal` | SURVIVED | all18 |
| [extra/010](extra/010-writer_exclusive_shard.log) | `writer_exclusive_shard` | SURVIVED | all18 |
| [extra/011](extra/011-writer_exclusive_manifest.log) | `writer_exclusive_manifest` | SURVIVED | all18 |
| [extra/012](extra/012-rank_seed_full18_confirmation.log) | `rank_seed_full18_confirmation` | SURVIVED | all18 |
| [extra/013](extra/013-loss_sign_full18_confirmation.log) | `loss_sign_full18_confirmation` | SURVIVED | all18 |
