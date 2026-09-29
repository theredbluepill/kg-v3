# Teacher merge guard mutations, verification r2

Verified HEAD `2390c8e239a4d57770bdafd621c05abbcb326739` on CPU with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. No tracked file was edited.

Baseline: 8 selected repository cases plus 19 supplemental scratch cases and 5 invalid-slot helper cases passed. After mutation restoration the complete union passes: `32 passed in 0.84s`.

36 isolated mutations: 35 killed, 1 survived, 0 setup errors. Every new teacher admission guard is attempted: target concatenation (empty/presence/keys/grammar), cached target type and required value/slot shapes/dtypes/keys/signature, stateless inputs and both wrapper branches, combined teacher type/action-spec/signature/tables, replay admission in precompute/combined/student replay, observation-spec cross-game/exact equality, three teacher action-layout calls, CE shape, internal requested-output/validated-chunk presence, and invalid policy-slot lookup. Existing stateless helpers are included to verify their new callers.

The sole survivor, `combined_grammar_signature`, removes the construction-time digest comparison. The changed constructor tables still trigger the separate exact per-table comparison before encoding, so the constructor mismatch is rejected by redundant defense. The independent `combined_tables` mutant is killed by the in-place table-edit case, which retains the old construction-time digest. The cached-signature mutant is killed, including a mismatch that would otherwise admit the replayed program. This survivor does not demonstrate an accepted incompatible grammar.

Supplemental test quality: malformed action layouts and temporal inputs encounter an explicit encode sentinel if their admission call is deleted; combined-teacher replay encounters a no-student-policy sentinel when admission is removed; cached-student replay no longer raises. CE malformed shapes intentionally broadcast if the explicit shape guard is removed, so both cases then fail to raise. Presence helpers and invalid-slot probes explicitly require the contract's error type/message. Repository guard mutants fail on missing errors, wrong error type/message, or expected guard order. Those latter kills demonstrate informative-error/early-admission contracts rather than acceptance alone.

Every modified scratch source is restored in a finally block and checked immediately. All 136 copied tracked inputs match captured source, current source, and restored scratch SHA-256 values; see `all_scratch_source_restoration.json`. `import_paths.log` confirms Python loaded scratch modules. The generated scratch tree was removed after recording verification; reproducer scripts, added scratch test sources, selections, exact replacement strings, and individual failure traces remain.

No guard-related correctness finding. These supplemental probes are verifier evidence only, not repository test changes. CUDA execution and real compilation are outside this CPU guard check; the separate numerical verifier covers the new teacher routes through the integration's GEMM guard.

| Mutation | Result | Observed pytest result |
|---|---|---|
| `concat_empty` | KILLED | 1 failed in 0.26s |
| `concat_slot_presence` | KILLED | 1 failed in 0.21s |
| `concat_slot_keys` | KILLED | 1 failed in 0.21s |
| `concat_winner_presence` | KILLED | 1 failed in 0.22s |
| `concat_grammar` | KILLED | 1 failed in 0.23s |
| `cache_type` | KILLED | 1 failed in 0.20s |
| `cache_value_presence` | KILLED | 1 failed in 0.19s |
| `cache_value_shape` | KILLED | 1 failed in 0.20s |
| `cache_slot_presence` | KILLED | 1 failed in 0.18s |
| `cache_grammar` | KILLED | 2 failed in 0.26s |
| `cache_slot_keys` | KILLED | 1 failed in 0.20s |
| `cache_slot_dtype` | KILLED | 1 failed in 0.22s |
| `cache_slot_shape` | KILLED | 1 failed in 0.20s |
| `stateless_hidden_state` | KILLED | 1 failed in 0.21s |
| `stateless_dones` | KILLED | 1 failed in 0.21s |
| `combined_type` | KILLED | 1 failed in 0.12s |
| `combined_action_spec` | KILLED | 1 failed in 0.13s |
| `combined_grammar_signature` | SURVIVED | 1 passed in 0.12s |
| `combined_tables` | KILLED | 1 failed in 0.16s |
| `teacher_replay` | KILLED | 1 failed in 0.12s |
| `orbit_foreign_targets` | KILLED | 1 failed in 0.29s |
| `obs_kaggriculture_equality` | KILLED | 1 failed in 0.12s |
| `obs_cross_game` | KILLED | 1 failed in 0.15s |
| `wrapper_cached_stateless` | KILLED | 1 failed in 0.23s |
| `wrapper_combined_stateless` | KILLED | 1 failed in 0.24s |
| `ce_shape` | KILLED | 2 failed in 0.02s |
| `precompute_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `cached_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `combined_layout` | KILLED | 3 failed, 6 passed in 0.13s |
| `cached_stateless_call` | KILLED | 2 failed, 2 passed in 0.07s |
| `combined_stateless_call` | KILLED | 2 failed, 2 passed in 0.07s |
| `cached_student_replay` | KILLED | 1 failed in 0.03s |
| `combined_teacher_replay` | KILLED | 1 failed in 0.03s |
| `policy_core_field_presence` | KILLED | 1 failed in 0.01s |
| `concat_internal_presence` | KILLED | 1 failed in 0.01s |
| `slot_frames_invalid_slot` | KILLED | 5 failed in 0.01s |
