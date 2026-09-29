# Independent Python mutation verification at b8747b6

No production defects found by this bounded mutation matrix. All source mutations ran on copied files in `scratch/`; repository source bytes remained unchanged. Every mutation baseline passed, mutant failed, and restored rerun passed. `restoration.json` contains SHA-256 inventories for every copied repository input.

| Mutation | Baseline exit | Mutant exit | Restored exit | Byte restoration |
| --- | ---: | ---: | ---: | --- |
| seat_bytes | 0 | 1 | 0 | True |
| header_order | 0 | 1 | 0 | True |
| schema_version | 0 | 1 | 0 | True |
| duplicate_record | 0 | 1 | 0 | True |
| source_header_step | 0 | 1 | 0 | True |
| quota | 0 | 1 | 0 | True |
| byteplanes | 0 | 1 | 0 | True |
| source_snapshot | 0 | 1 | 0 | True |
| archive_custody | 0 | 1 | 0 | True |
| recorder_copy | 0 | 1 | 0 | True |
| caller_baseline | 0 | 1 | 0 | True |
| caller_growth | 0 | 1 | 0 | True |
| shared_deadline | 0 | 1 | 0 | True |
| retired_authored_set | 0 | 1 | 0 | True |
| bridge_retirement | 0 | 1 | 0 | True |
| pinned_availability | 0 | 1 | 0 | True |
| engine_cargo_source_omitted | 0 | 1 | 0 | True |
| engine_random_source_omitted | 0 | 1 | 0 | True |
| engine_econ_source_omitted | 0 | 1 | 0 | True |
| engine_declaration_capture | 0 | 1 | 0 | True |
| engine_declaration_recheck | 0 | 1 | 0 | True |
| header_order_independent | 0 | 1 | 0 | True |

The reused header-order target originally killed a bypass by observing the later coverage rejection instead of the expected header-hash error. The independent supplemental probe recomputes coverage after changing the header key order, and then the same guard bypass admits the corruption (`DID NOT RAISE`). This isolates the header byte-order custody check.

New engine custody coverage individually omits `engine_rs/Cargo.toml`, `engine_rs/src/py_random.rs`, and `engine_rs/src/econ_attrib.rs` from captured inputs. All three omissions are caught by phase-drift tests. Separate direct probes disable declaration checks only at source capture or only at snapshot recheck; both are killed. Capture rejection occurs before any command execution.

Pinned admission is tested by evaluating the actual `_CUDA_PINNED` AST with independently mocked availability true/false; no CUDA allocation or Torch import is attempted. This verifies platform admission, not CUDA pinning behavior.

Limits: regeneration phase tests use the repository custody harness with mocked executable runs. They establish guard sensitivity, not successful corpus regeneration, GPU behavior, or full training behavior. The parent verification performs real source-snapshot probes and full suites separately.
