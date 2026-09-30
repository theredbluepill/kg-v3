Scope read: e197528..832b836, all five first-parent commits, brief 1.5 including overriding reviews/deviations, brief 1.4’s ABI/rewards, contract/API/model documentation, owner rules and the Stage 2 prompt.

No production defect found. Buffer ownership, fencing, zero-copy inputs, seed streams, reward arithmetic, native tables, evaluation seeding and deliberate trainer stops match the accepted specification. The stub declarations are unique and AST-identical to brief 1.4. Both merge parents’ cookbook history is preserved.

Checks:

- Focused pytest: exit 0; 1,109 passed, 0 failed, 3 skipped.
- cargo test --locked --lib reward_admission: exit 0; 1 passed, 0 failed.
- Restored focused pytest: exit 0; 1,109 passed, 3 skipped.
- Restored reward/native suite: exit 0; 387 passed.
- Additional boundary probes: exit 0; 10 passed.
- Full offline just prepare: exit 0. Python 2,224 passed, 6 skipped; root Rust 274 passed, 5 ignored; engine 69 passed. Formatting, lint, mypy and documentation freshness passed.
- Missing-binding grep: exit 1, empty as required.
- Protected-path diff: engine_rs, tests/owl and python/owl/train unchanged. Two tests/tools changes belong to merged Task 1.4.
- Final git diff --exit-code, git diff --check and git status --short: exit 0, empty.

Verification error: two initial automatic uv rebuilds used Maturin’s implicit release/LTO profile. I interrupted the next build, restored the source and reran every Rust mutation and restoration with explicit dev builds. This violated the no-release/LTO constraint. No training, GPU or throughput runs occurred.

Mutation results: 131 distinct mutants; 108 caught, 23 SURVIVED. Every mutation was restored byte-exactly. The complete plain-text report contains every attempt’s failing tests, counts, exit code and log:

/tmp/kg-task15-verify/report.txt

Required mutation table:

Mutation | Failing test
Fence without pinned condition | test_fence_condition[cuda-False-0]
Fence without CUDA condition | test_fence_condition[cpu-True-0]
Fence after reset, step or truncate—three separate mutants | test_fence_precedes_every_native_write
Python raw-weight predicate | test_reward_admission_predicate_cases[coefficients4-False]
Python combined predicate | test_reward_admission_predicate_cases[coefficients10-False]
Rust raw-weight predicate | test_python_and_native_reward_admission_agree[coefficients4-False]
Rust combined predicate | test_python_and_native_reward_admission_agree[coefficients10-False]
Rust missing death/ineffective cap checks | test_python_and_native_reward_admission_agree[coefficients3-False/coefficients8-False]
Rust cap equality/nonfinite admission | test_reward_dict_rejects_invalid_config[caps/nan/negative]
Factory missing rank or stride=1 | test_factory_rank_streams; test_real_binding_seed_streams
Native constants version/type/names/width checks | test_native_constants_rejected_before_tables_are_requested
Native key/array/layout checks | test_native_arrays_reject_corruption
Native dtype/shape checks removed individually | SURVIVED; converter repeats validation
Model synthetic-table default | test_model_default_loads_native_tables_once
Codec premature batch publication | test_codec_batch_failure_never_publishes_partial_output
Evaluation unmixed seed | test_kaggriculture_eval_factory_arguments
Main Kaggriculture stop removed | test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation
Policy-evaluation stop removed | test_kaggriculture_policy_evaluation_names_remaining_mapping_blocker

Two Rust admission mutants initially SURVIVED an overly narrow test selection; both failed after including the native invalid-config tests. The exhaustive table records both attempts. Other survivors comprise the coverage gaps below, redundant validation, native-return guarantees and CPU-only device checks.

Findings:

P3 — tests/kaggriculture/test_codec.py:260
Malformed seat pairs and decode batches lack negative tests. Removing codec.py’s seat-count, decode dtype/device and shape/contiguity checks survives. Add short/long seat pairs and independently malformed tokens/lengths, asserting rejection before native decoding. Float lengths and extra length columns are discriminating cases.

P3 — tests/kaggriculture/test_game.py:87; tests/kaggriculture/test_rewards.py:67
Some boundary tests trigger another guard first. Add rejection cases for boardSize=11; maxMarketOrdersPerTurn=11 with turnsPerDay=1; and econ_shaping=0 with econ_cap=-0.01. Their corresponding guard removals survived.

P3 — tests/kaggriculture/test_env.py:126
Direct allocator/constructor admission lacks negative cases. Add zero-environment allocation and direct constructor boolean, noninteger, negative and out-of-i64 arguments, with sentinels proving rejection precedes allocation/native construction.

P3 — cookbook/references/index.md:27, also :8 and :10
Current descriptions still claim native tables/environment construction are missing. Reconcile the linked grammar-head, evaluation and configuration References with Stage 2, preserving historical evidence and naming Task 3.1 mapping/storage as the remaining blocker.

Residual risks: Task 3.1 rollout storage and observation/action mapping remain absent. The pod DMA test was inspected, not run; it uses the real binding and includes a fence-removal control. The early two-rank smoke remains pending. No GPU, learning or throughput qualification is established.

VERDICT: APPROVE WITH EDITS