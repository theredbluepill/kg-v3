# Task 5.2 independent mutation verification, r3

Reviewed source: `b626f24e203205dadaacf32018c84edc06030522`. Each mutant ran all **30** current BC tests using the worktree virtualenv Python, copied source/tests/configs and an explicit scratch `PYTHONPATH`. No production training was launched. The existing tests include small synthetic CPU update loops.

**137 unique substitutions: 68 killed, 69 survived; zero setup errors or timeouts.** The comparable prior inventory has 120 cases: **53 killed, 67 survived**, versus 47/73 in r2. The 17 new unique cases have **15 killed, 2 survived**. One duplicate execution of case058 was detected during aggregation and excluded, with its receipt preserved as `excluded-duplicate-130.*`; replacement case130 independently changes the unscheduled patience reference. Thus 138 substitution executions produced 137 unique cases.

## Prior survivors now killed

| ID | Mutation | Failing test |
| --- | --- | --- |
| 013 | `rank_seed` | `test_equal_rank_partitions_draw_different_permutations` |
| 058 | `script_provenance_mapping` | `test_attempt_records_reject_malformed_lineage` |
| 114 | `fresh_attempt_existing_guard` | `test_attempt_records_reject_malformed_lineage` |
| 115 | `resume_needs_attempts_guard` | `test_attempt_records_reject_malformed_lineage` |
| 116 | `read_attempts_file_guard` | `test_attempt_records_reject_malformed_lineage` |
| 117 | `read_attempt_source_type_guard` | `test_attempt_records_reject_malformed_lineage` |

All other original dispositions remain unchanged. Rank-seed dependence and the attempt admission branches are now asserted. The remaining prior survivors include malformed shard/config guards and shared autocast/compile/DDP seams, accumulation scaling, clipping, nonfinite-gradient checking, scheduler, preflight, and runtime-cap wiring. Survival is evidence of missing assertions, not proof that invalid inputs or configurations reach training.

## Added cases

| ID | Mutation | Outcome | Failing tests |
| --- | --- | --- | --- |
| 121 | `patience_counts_unscheduled` | KILLED | `test_an_unscheduled_evaluation_selects_but_leaves_patience_alone`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]` |
| 122 | `unscheduled_skips_selection` | KILLED | `test_an_unscheduled_evaluation_selects_but_leaves_patience_alone`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls1-1]` |
| 123 | `unscheduled_skips_last` | KILLED | `test_an_unscheduled_evaluation_selects_but_leaves_patience_alone` |
| 124 | `terminal_evaluation_scheduled` | KILLED | `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]` |
| 125 | `schedule_flag_not_propagated` | KILLED | `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]` |
| 126 | `initial_evaluation_unscheduled` | KILLED | `test_best_checkpoint_is_the_minimum_even_inside_min_delta`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls1-1]` |
| 127 | `periodic_evaluation_unscheduled` | KILLED | `test_best_checkpoint_is_the_minimum_even_inside_min_delta`, `test_training_keeps_the_best_checkpoint_and_stops_on_degradation`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls1-1]` |
| 128 | `schedule_log_flag_inverted` | SURVIVED | None |
| 129 | `schedule_log_flag_removed` | SURVIVED | None |
| 130 | `unscheduled_patience_reference_updated` | KILLED | `test_an_unscheduled_evaluation_selects_but_leaves_patience_alone` |
| 131 | `attempt_index_guard_removed` | KILLED | `test_attempt_records_reject_malformed_lineage` |
| 132 | `attempt_object_type_guard_removed` | KILLED | `test_attempt_records_reject_malformed_lineage` |
| 133 | `attempt_source_type_accepts_integer` | KILLED | `test_attempt_records_reject_malformed_lineage` |
| 134 | `attempt_source_absence_ignored` | KILLED | `test_attempt_records_reject_malformed_lineage` |
| 135 | `terminal_evaluation_removed` | KILLED | `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls0-0]`, `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls1-1]` |
| 136 | `selection_defaults_unscheduled` | KILLED | `test_held_out_selection_keeps_the_lowest_nll_and_stops_after_patience`, `test_an_unscheduled_evaluation_selects_but_leaves_patience_alone`, `test_held_out_patience_ignores_a_trickle_below_min_delta` |
| 137 | `unscheduled_best_checkpoint_not_saved` | KILLED | `test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence[interrupted_nlls1-1]` |

The only new survivors invert or remove the `bc/scheduled_evaluation` history field. Current source records the correct flag, but no BC test checks its presence/value. Both new off-cadence integration variants fail targeted substitutions: the non-improving terminal case catches resumed stopping-cadence changes; the lower terminal NLL case catches omitted off-cadence selection or best-checkpoint writes.

## Custody and repeatability

`cases.json` contains exact substitutions. `results.json`, the individual logs, `prior-dispositions.json` and `test-mutation-map.json` map every outcome and failing test. `run.py` reproduces the final 137-case inventory. The archived `progress.log` reflects the original execution order; the excluded duplicate receipt and final case130 log explain the replacement.

Three import-resolution probes confirm copied `bc.py`, `bc_data.py`, `test_bc.py` and `train_bc.py` were executed. Each substitution was restored in `finally` and byte-compared; source hashes are in worker custody/restoration files and test hashes in `test-custody.json`. All three restored baselines pass **30/30**; worker0 also passes **30/30** after replacement case130. Final restored source/test hashes match the repository. Scratch copies were then removed (`cleanup.json`). No tracked file was edited by these checks.
