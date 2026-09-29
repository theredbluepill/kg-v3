# Independent native verification of b8747b6

Scope: Task 1.3 encoder/buffers/oracle/relocated grammar route against b51b0c0 (three-dot), with a fresh HEAD source export. No tracked worktree edits. Native mutations used `.codex-tmp/verify-b8747b6-native` and its own Cargo target.

## Findings

No actionable native implementation finding identified. Public/private role routing, constructor-shaped tiles, exact counts/ranks, checked arithmetic, one-snapshot preparation and complete-batch preflight before caller-buffer writes agree with the reviewed contract. The retained Orbit change is test-only Number decoding for arbitrary-precision serde feature unification; disabling it fails the retained fixture regression.

## Mutation checks

41 source mutants compiled and failed their targeted test (exit 101 with `test result: FAILED`, no compiler error). Every mutated file was restored in `finally`; before/after SHA-256 values match. The entire 188-file scratch export matches HEAD byte-for-byte afterward. Before/after inventories of all 1,633 tracked worktree files also match.

| Source mutation | Targeted test filter | Result |
|---|---|---|
| buffer-length | `shape_every_field_rejects_short_and_long_without_writes` | Killed; restored |
| buffer-zero-env | `shape_zero_environments_is_an_error` | Killed; restored |
| buffer-product | `shape_product_overflow_is_an_error` | Killed; restored |
| buffer-publication | `shape_publish_mismatched_environments_leaves_all_fields_unchanged` | Killed; restored |
| buffer-clear | `shape_clear_overwrites_every_field_with_positive_zero_or_false` | Killed; restored |
| config-envelope | `config_rejects_envelope_violations_without_output_writes` | Killed; restored |
| config-day-bound | `config_day_order_product_has_exact_240_boundary_and_checked_overflow` | Killed; restored |
| finite-f32 | `config_weed_requires_finite_nonnegative_numeric_f32_without_unit_cap` | Killed; restored |
| hire-own-rival | `hire_cost_uses_each_public_count` | Killed; restored |
| tile-coordinate | `tile_asymmetric_coordinates_and_every_channel` | Killed; restored |
| actor-coordinate | `actors_public_positions_roles_and_all_private_channels` | Killed; restored |
| seat-privacy | `privacy_private_values_and_key_order_do_not_cross_seats` | Killed; restored |
| inventory-rank | `actors_rank_remove_reinsert_preserves_counts_and_replacement_preserves_rank` | Killed; restored |
| storage-rank | `storage_rank_remove_reinsert_changes_order_not_counts` | Killed; restored |
| row-diagnostic | `context_check_row_rejects_corruptions_in_every_domain` | Killed; restored |
| one-snapshot | `snapshot_both_seats_acquire_once_and_keep_staging_allocations` | Killed; restored |
| relocated-root-grammar | `decoded_command_matrices_execute_both_seats` | Killed; restored |
| reconstruction-scalar | `reconstruction_hand_anchors_cover_every_range_and_boundary` | Killed; restored |
| bitwise-signed-zero | `reconstruction_comparator_names_record_seat_group_offset_and_signed_zero` | Killed; restored |
| reconstruction-duplicate | `reconstruction_coverage_rejects_duplicate_offset` | Killed; restored |
| reconstruction-zero-hole | `reconstruction_coverage_rejects_unwritten_zero_hole` | Killed; restored |
| byteplane-order | `reconstruction_byteplane_decode_preserves_bits_and_rejects_bad_lengths` | Killed; restored |
| record-strict-header | `producer_strict_record_and_quota_guard_reject_bad_inputs` | Killed; restored |
| legacy-domain | `producer_legacy_domain_rejects_old_oracle_limits_without_narrowing_v3` | Killed; restored |
| full-corpus-market | `compare_observation_oracle` | Killed; restored |
| l4-feature-unification | `arbitrary_precision_uniform_fixture_float` | Killed; restored |
| corpus-bitwise-coherent-market | `compare_observation_oracle` | Killed; restored |
| recorded-redundancy-guard | `reconstruction_recorded_constants_and_duplicates_are_checked_independently` | Killed; restored |
| decoded-exact-eof | `reconstruction_feature_streams_support_both_formats_and_exact_eof` | Killed; restored |
| producer-policy-burst | `producer_policy_v2_appends_hire_burst_within_order_limit` | Killed; restored |
| action-frame-mask | `context_can_act_boundaries_and_terminal_snapshots_are_live_rows` | Killed; restored |
| config-positive-integers | `config_rejects_envelope_violations_without_output_writes` | Killed; restored |
| public-clock | `config_rejects_farm_count_clock_and_nonfinite_derived_context_transactionally` | Killed; restored |
| strict-tile-keys | `tile_strict_constructor_keys_reject_stale_null_missing_and_unknown_fields` | Killed; restored |
| tile-sentinel-integers | `tile_integer_fields_reject_fractional_wrong_types_ranges_and_negative_counts` | Killed; restored |
| actor-inventory-cardinality | `actors_reject_malformed_positions_cardinality_and_private_counts_atomically` | Killed; restored |
| private-negative-count | `storage_rejects_unknown_items_negative_counts_and_bad_quadrants_atomically` | Killed; restored |
| duplicate-quadrants | `storage_rejects_unknown_items_negative_counts_and_bad_quadrants_atomically` | Killed; restored |
| market-integer-representation | `context_market_rejects_missing_unknown_fractional_and_wrong_type_values` | Killed; restored |
| producer-order-comparator | `producer_comparator_detects_values_nested_orders_and_money_bits` | Killed; restored |
| producer-gzip-exact-eof | `producer_gzip_stream_requires_eof_and_success` | Killed; restored |

The coherent market mutation increments exact and scaled channels together. Row validation and source custody still pass, then the independent frozen corpus fails at `record=official:95324500:0 seat=0 offset=889`: actual 1.0001, reference 1.0. This distinguishes real comparison coverage from merely rejecting an inconsistent row. The relocated-root-grammar mutant changes decoded commands to PASS and fails the root kernel execution route.

Eight additional malformed-input probes use the freshly rebuilt current root `owl.rs`: shape, dtype, foreign endian, stride, read-only, alignment, aliasing across buffers, and a late environment header error. Each raises a field-named ValueError before output writes; all supplied-buffer bytes and pointers stay unchanged. Replacing the bad input with the original valid batch restores bitwise-identical output. Probe source itself is a scratch copy restored byte-for-byte. These are input mutations, separately identified from source mutation tests.

## Counts and limits

Initial scratch Kaggriculture baseline: 97 passed, 2 ignored, 159 root tests filtered. Restored scratch Kaggriculture baseline: 97 passed, 2 ignored, 159 filtered; 25.13 s (`restored.log`). The parent owns the requested complete engine/root/Python/mypy runs and source-custody mutations. No optimized timing, CUDA, pinned-memory, throughput or training qualification is claimed. Existing oracle identity omissions predating the repair remain documented; these native checks do not reconstruct missing historical source custody.

Evidence: `combined-results.json`, per-mutation logs, `scratch-restoration.json`, `tracked-before.json`, `tracked-after.json`, `baseline.log`, `restored.log`, and `boundary-probes.json`.
