# Independent test inventory and documentation audit

Target: `84dd76ad570bf35b0947df52f3697f80339098e6`; parents `e1458d2a717d9d731a367cbb78b98616ee6649f4` and `7877c46e39687e4332fe3ad5fc65d3c76feb2125`.

## Finding

**P3 — update the remaining native-table handoff descriptions.** `cookbook/references/index.md:23`, `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md:4` and `docs/model-architecture.md:752` still say synthetic tables stand in until Task 1.2. This merge completes Task 1.2's Rust tables; the remaining PyO3 binding belongs to Task 1.4, as the newly updated native Reference (`:94`–`:100`), coverage document (`:435`–`:439`), and `python/owl/kaggriculture/gpu_grammar.py:229`–`:243` correctly state. Reconcile those descriptions/current prose with the actual remaining boundary. Historical Task 2.3 check counts can remain scoped to that episode. This is a documentation-only finding; no behavioral change is required.

## Inventory result

No test dropped after accounting for the authored-set accept/exact-test consolidation. `inventory.py` parses every tracked Python source in each tree using the project's Python 3.12 AST, with zero parse errors, including function decorators and class paths. It scans every tracked Rust source for `#[test]`; there are no unfamiliar test-attribute forms. Full name inventories and results are retained beside this report.

| Tree | Python test definitions outside ops | Rust test definitions outside ops | Rust definitions in ops artifacts |
| --- | ---: | ---: | ---: |
| Integration parent | 746 | 226 | 7 |
| Task 1.2 parent | 640 | 234 | 8 |
| Merge | 755 | 244 | 8 |

These are source definitions, not pytest parameter instances or Cargo invocations. The shared nine grammar tests are compiled by two crates; the Rust source count includes them once. The merge has 166 root source tests (including two ignored) and 78 definitions under engine_rs, with nine shared grammar definitions included by engine_rs at runtime.

For each parent the only missing names are its two authored-inventory accept/exact tests, replaced by `test_task_authored_inventory_accepts_two_tests_and_generated_manifest` and `test_task_authored_inventory_requires_exact_authored_set` at `tests/tools/test_check_engine_trim.py:67` and `:110`. The exact-set decorator grows from 2 integration cases / 4 Task 1.2 cases to 5 merge cases: empty set, each of the three single omissions, and an extra file. No assertion intent is removed.

All other surviving Python test ASTs equal integration. Relative to Task 1.2 the only changed surviving AST is `tests/owl/model/test_model_config_files.py::test_model_config_files_load`, inherited byte-for-byte from integration: it adds explicit loading for the new Kaggriculture model config and retains the prior loader for other models. All Rust test-bearing files already in integration are byte-identical. The sole Rust test-bearing file changed relative to Task 1.2 is `engine_rs/tests/replay_parity.rs`, identical to integration; all nine preexisting test function bodies are byte-identical and ten tests were added. No weakened test was found.

## Docs, log and merged-file result

- Log headings: integration 68; Task 1.2 55 (53 shared, two unique); merge 71. All headings are unique. The complete merge sequence is exactly one new merge entry, the two unique Task 1.2 entries in original order, then the entire integration sequence in original order. No parent heading is missing.
- The complete Live Differential Parity section and nested Known Divergences section from integration are byte-equivalent after trimming only surrounding whitespace. Native Grammar scope is present, with pre-merge 77-engine/164-root/1,045-Python evidence clearly separated from post-merge 87/164/1,337.
- The committed `ops/rebuild-2026-09-29/merge-1.2/prepare.log` reports root 164 passed / 2 ignored at line 291; engine sections 41 + 18 + 9 + 19 = 87 passed / none ignored at lines 340, 364, 379, 404; Python 1,337 passed / 4 skipped at line 482. These are inspected committed receipts, not fresh full-suite runs by this sub-agent. Parent verifier runs the requested independent suites and fresh native-table cross-check.
- Cargo.toml's existing serde_json promotion and src/lib.rs's module registration match Task 1.2 exactly; docs/rl-api-specs.md preserves the complete Task 1.2 native boundary plus integration's Task 2.3 model-output contract. plan.md preserves both the native grammar completion/remaining first-edge work and integration's GEMM/FlashAttention work.
- Native Reference's current claims are correctly separated from historical reference-branch adapter/GPU evidence. Newly listed repository sources exist. Binding absence is visible in source and documented as an open Task 1.4 edge.

No repository file was written by this sub-agent. Scratch evidence is exclusively under `/private/tmp/kg-merge84-inventory-docs`. Runtime qualification, mutations, manifest regeneration and vendored-byte checks are owned by the parent and other verifier subtasks.
