# Independent native mutation verification of e197528

All mutations were made only in `.codex-tmp/verify-e197528-native`, exported from HEAD, using a separate Cargo target. No tracked worktree bytes changed.

Initial complete scratch native baseline: 97 passed, 2 ignored (159 other root tests filtered). An earlier setup run omitted the oracle driver; after copying the tracked driver/python/config files, the complete baseline passed.

Each of 31 source mutants compiled and caused its named test to fail. Every edited file was restored in `finally`, with equal SHA-256 before/after. Worktree inventories before/after matched across all 1,629 tracked files.

| Mutation | Test filter | Result |
|---|---|---|
| buffer-length | `shape_every_field_rejects_short_and_long_without_writes` | Killed; exit 101 |
| buffer-zero-env | `shape_zero_environments_is_an_error` | Killed; exit 101 |
| buffer-product | `shape_product_overflow_is_an_error` | Killed; exit 101 |
| buffer-publication | `shape_publish_mismatched_environments_leaves_all_fields_unchanged` | Killed; exit 101 |
| buffer-clear | `shape_clear_overwrites_every_field_with_positive_zero_or_false` | Killed; exit 101 |
| config-envelope | `config_rejects_envelope_violations_without_output_writes` | Killed; exit 101 |
| config-day-bound | `config_day_order_product_has_exact_240_boundary_and_checked_overflow` | Killed; exit 101 |
| finite-f32 | `config_weed_requires_finite_nonnegative_numeric_f32_without_unit_cap` | Killed; exit 101 |
| hire-own-rival | `hire_cost_uses_each_public_count` | Killed; exit 101 |
| tile-coordinate | `tile_asymmetric_coordinates_and_every_channel` | Killed; exit 101 |
| actor-coordinate | `actors_public_positions_roles_and_all_private_channels` | Killed; exit 101 |
| seat-privacy | `privacy_private_values_and_key_order_do_not_cross_seats` | Killed; exit 101 |
| inventory-rank | `actors_rank_remove_reinsert_preserves_counts_and_replacement_preserves_rank` | Killed; exit 101 |
| storage-rank | `storage_rank_remove_reinsert_changes_order_not_counts` | Killed; exit 101 |
| row-diagnostic | `context_check_row_rejects_corruptions_in_every_domain` | Killed; exit 101 |
| one-snapshot | `snapshot_both_seats_acquire_once_and_keep_staging_allocations` | Killed; exit 101 |
| relocated-root-grammar | `decoded_command_matrices_execute_both_seats` | Killed; exit 101 |
| reconstruction-scalar | `reconstruction_hand_anchors_cover_every_range_and_boundary` | Killed; exit 101 |
| bitwise-signed-zero | `reconstruction_comparator_names_record_seat_group_offset_and_signed_zero` | Killed; exit 101 |
| reconstruction-duplicate | `reconstruction_coverage_rejects_duplicate_offset` | Killed; exit 101 |
| reconstruction-zero-hole | `reconstruction_coverage_rejects_unwritten_zero_hole` | Killed; exit 101 |
| byteplane-order | `reconstruction_byteplane_decode_preserves_bits_and_rejects_bad_lengths` | Killed; exit 101 |
| record-strict-header | `producer_strict_record_and_quota_guard_reject_bad_inputs` | Killed; exit 101 |
| legacy-domain | `producer_legacy_domain_rejects_old_oracle_limits_without_narrowing_v3` | Killed; exit 101 |
| full-corpus-market | `compare_observation_oracle` | Killed; exit 101 |
| l4-feature-unification | `arbitrary_precision_uniform_fixture_float` | Killed; exit 101 |
| corpus-bitwise-coherent-market | `compare_observation_oracle` | Killed; exit 101 |
| recorded-redundancy-guard | `reconstruction_recorded_constants_and_duplicates_are_checked_independently` | Killed; exit 101 |
| decoded-exact-eof | `reconstruction_feature_streams_support_both_formats_and_exact_eof` | Killed; exit 101 |
| producer-policy-burst | `producer_policy_v2_appends_hire_burst_within_order_limit` | Killed; exit 101 |
| action-frame-mask | `context_can_act_boundaries_and_terminal_snapshots_are_live_rows` | Killed; exit 101 |

The coherent market mutation increments both exact and scaled channels. Source-bound custody and row validation passed; the independent corpus comparison failed at `record=official:95324500:0 seat=0 offset=889`: actual `1.0001`, recorded `1.0`. The root grammar mutation fails `decoded_command_matrices_execute_both_seats`, exercising the relocated kernel route. Removing one L4 `fixture_float` override fails with `invalid type: map, expected f64`.

Findings: none in the reviewed native merge surface. Pinned memory, CUDA, Python admission and watchdog/source-custody guards are verified separately by the parent. This is correctness evidence, not a throughput claim.

Restored complete root scratch suite: **254 passed, 4 ignored** in 13.66 seconds. The first complete-root scratch run lacked the local ignored Orbit fixtures (247 passed, 7 fixture-absence failures, 4 ignored); after copying all three fixture files with matching hashes, all tests passed. No fixture regeneration or source alteration was needed. Receipt: `restored-root-with-local-fixtures.log`; local fixture custody: `local-ignored-fixture-copy.json`.
