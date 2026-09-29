# Teacher merge guard mutations

Baseline: 8 repository tests and 19 supplemental scratch cases passed; restored union: 27 passed in 0.85s. The added invalid-slot helper guard baseline and restored rerun each pass 5 cases.

36 isolated mutants: 35 killed, 1 survived, 0 setup errors.

Every source mutation replaced one condition with `False` or one admission call with `pass`; all scratch source files were restored to their captured bytes in finally blocks and matched the tracked-source SHA-256 values. `import_paths.log` confirms runtime imports use scratch modules. No tracked file was changed.

The sole survivor is `combined_grammar_signature`: removing the construction-time digest comparison still rejects the changed grammar through exact table comparisons before any encode. The independent `combined_tables` mutant is killed by the in-place table-edit case whose digest remains unchanged. This is redundant defense, not an accepted mismatch.

Mutant failures distinguish missing expected errors, the wrong error type/text, and reaching an explicit no-kernel/no-student-policy sentinel. Full assertion traces are in each named log, and exact replacement/test selections are in summary.json and supplement_summary.json.

| Mutant | Result | Observed test result |
|---|---|---|
| `concat_empty` | KILLED | 1 failed in 0.24s |
| `concat_slot_presence` | KILLED | 1 failed in 0.22s |
| `concat_slot_keys` | KILLED | 1 failed in 0.21s |
| `concat_winner_presence` | KILLED | 1 failed in 0.21s |
| `concat_grammar` | KILLED | 1 failed in 0.22s |
| `cache_type` | KILLED | 1 failed in 0.19s |
| `cache_value_presence` | KILLED | 1 failed in 0.19s |
| `cache_value_shape` | KILLED | 1 failed in 0.21s |
| `cache_slot_presence` | KILLED | 1 failed in 0.19s |
| `cache_grammar` | KILLED | 2 failed in 0.25s |
| `cache_slot_keys` | KILLED | 1 failed in 0.19s |
| `cache_slot_dtype` | KILLED | 1 failed in 0.20s |
| `cache_slot_shape` | KILLED | 1 failed in 0.19s |
| `stateless_hidden_state` | KILLED | 1 failed in 0.20s |
| `stateless_dones` | KILLED | 1 failed in 0.19s |
| `combined_type` | KILLED | 1 failed in 0.13s |
| `combined_action_spec` | KILLED | 1 failed in 0.12s |
| `combined_grammar_signature` | SURVIVED | 1 passed in 0.11s |
| `combined_tables` | KILLED | 1 failed in 0.15s |
| `teacher_replay` | KILLED | 1 failed in 0.13s |
| `orbit_foreign_targets` | KILLED | 1 failed in 0.30s |
| `obs_kaggriculture_equality` | KILLED | 1 failed in 0.12s |
| `obs_cross_game` | KILLED | 1 failed in 0.13s |
| `wrapper_cached_stateless` | KILLED | 1 failed in 0.20s |
| `wrapper_combined_stateless` | KILLED | 1 failed in 0.23s |
| `ce_shape` | KILLED | 2 failed in 0.02s |
| `precompute_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `cached_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `combined_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `cached_stateless_call` | KILLED | 2 failed, 2 passed in 0.08s |
| `combined_stateless_call` | KILLED | 2 failed, 2 passed in 0.07s |
| `cached_student_replay` | KILLED | 1 failed in 0.03s |
| `combined_teacher_replay` | KILLED | 1 failed in 0.04s |
| `policy_core_field_presence` | KILLED | 1 failed in 0.01s |
| `concat_internal_presence` | KILLED | 1 failed in 0.01s |

| `slot_frames_invalid_slot` | KILLED | 5 failed in 0.01s |

All 136 tracked inputs copied into scratch match their source SHA-256 after restoration; `all_scratch_source_restoration.json` records this. The generated scratch tree was removed after verification, leaving reproducer scripts and test sources with the logs.

No guard-related correctness finding. Supplemental probes extend coverage only in the scratch area; they are not repository test additions.
