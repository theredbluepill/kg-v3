# Independent merge preservation audit

Scope: `ca37089` (`kg/merge-8rank`) against `0b8cf98ef57fc49a329dca4c8368c630586c4752...HEAD`. The merge is `a3a5cb8`, with parents `0b8cf98` and `7ad45fc`; their common ancestor is `b51b0c0`. This audit inspected git objects directly, independently of the prior verifier's inventories. No tracked file was written.

## Findings

No new actionable finding. Both parents' relevant changes are retained, and the three manual conflict resolutions are semantically correct.

## Source and test preservation

- All 1,921 integration-parent paths remain. The 14 changed existing paths are the intended config comments, documentation/plan/cookbook edits, and the two expanded test files. There is no net production Python or Rust change against integration. The integration's engine, root Rust, production Python and native bindings are retained byte-for-byte.
- An AST inventory of test function definitions (Python) and attribute-marked test functions (Rust), excluding historical `ops/` copies, counts integration at 926 Python + 327 Rust definitions and HEAD at 930 Python + 327 Rust. No integration definition is absent. Four Python test functions are added. These are source-definition counts, not collected-case or executed-test counts.
- The branch parent has 859 Python + 244 Rust definitions. Its ten apparently absent qualified names are all preexisting integration changes: nine grammar-kernel tests moved from `engine_rs/tests/grammar_kernel.rs` to `src/kaggriculture/grammar_kernel_tests.rs`, and `test_task_authored_inventory_accepts_two_tests_and_generated_manifest` became `test_task_authored_inventory_accepts_replay_test_and_generated_manifest` to reflect that bridge retirement. The moved tests retain their bodies apart from import/path plumbing and formatting; the renamed inventory test now checks the correct authored set. Both changes predate this merge and are already in `0b8cf98`.
- Likewise, deletion of the branch parent's `src/kaggriculture.rs` is the integration's prior module split, not a loss introduced here. All integration source blobs survive the current diff.
- `configs/kaggriculture_2rank.yaml`, `configs/kaggriculture_4rank.yaml`, `configs/kaggriculture_8rank.yaml`, `tests/kaggriculture/test_configs.py` and `tests/scripts/test_run_ppo.py` are byte-identical to branch parent `7ad45fc`.
- Independently collected the two changed Python test files and the two tests whose parameter lists glob training YAMLs. The integration snapshot collected 233 cases, current HEAD 250: no removed case and 17 additions. Parent configs and those test sources were exported to a scratch directory; collection used current production Python, which is byte-identical to integration, and the current unchanged scripts directory. All original config files remain; the numbered config-path IDs alone are not treated as proof of config identity.

## Resolutions

- `cookbook/log.md`: all 113 integration headings and all 92 branch headings survive with their whole blocks unchanged; HEAD has no duplicate headings. Only the merge entry and the r1 correction entry are new.
- `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md`: the integration's 11 sources followed by the branch's 13 new sources form the exact 24-entry ordered union, without duplicates. The integration body, including the allocated/reserved memory distinction and 94.97 GiB versus 97,887 MiB qualification, remains the exact prefix. The branch's entire eight-rank section remains byte-identical.
- `cookbook/references/index.md`: every integration line remains except the configs Reference line, which is replaced by the branch's line adding 8-rank 32/2. The model-only ceiling, pod environment, teacher and observation entries remain.
- Recreated ordinary three-way merges from the two parents and `b51b0c0` for `README.md`, `cookbook/decisions/index.md`, `docs/model-architecture.md` and `ops/rebuild-2026-09-29/plan.md`; each merged cleanly and its result matches current HEAD byte-for-byte. Inspection confirms the plan retains integration's memory metric and adds the branch's role, world-size-8 seed test, Task 6.3b and full-recipe versus batch-shape distinction.
- `docs/rules-parity-coverage.md` differs from the integration-only history intentionally: it inserts the current merge counts and receipt without removing the preceding Task 1.3 or teacher counts. It is reconciled, not lost.
- The configs Reference differs from the branch-only history intentionally: `ca37089` credits the already merged GPU timing while preserving CPU-only config qualification and the numerical correctness/production teacher gaps. It is reconciled, not lost.
- `engine_rs/TRIM_MANIFEST.json` is exactly integration's blob `8c1b7734a4a99f8e70ceaacf17d0f86f1ae24e5a`, unchanged by this merge or follow-up. The branch/fork blob is `a7e84a75216aa255de1d70c3fc79e1b3992a4c9d`: integration had already removed the retired grammar bridge and documented its replacement during Task 1.3. Therefore the supplied resolution phrase “not touched by either side” is accurate only for the current merge's net change, not both histories since the fork. Retaining integration's manifest is correct; reverting to the branch/fork manifest would be wrong. This is a clarification of the resolution description, not a repository defect.

## Evidence

`inventory.json` contains all source-level test IDs; `summary.json` contains parent comparisons and blob identities; `resolutions.json` contains exact conflict and clean-merge checks; `selected-collection.json` contains collected IDs and the 17 additions. The two selected-collection logs and exported source inputs accompany them. Execution checks and mutation results are performed by the coordinating verifier/other audit lane and are not claimed as this subtask's own checks.
