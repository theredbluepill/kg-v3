# Independent parent-retention and resolution audit

Audited HEAD `2390c8e239a4d57770bdafd621c05abbcb326739` against integration
`b8747b6e8acece5f561d09a75bb914364a60ac05` and teacher
`8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`. Parent common ancestor is
`4cac1a18f2209c54d40bef80d44755334a1a1ed2`.

No new finding. The sole P3 in the requested external r1 report is **RESOLVED**.
That external report is byte-identical to the committed worktree copy (SHA-256
`a4b01f0dc7cd620ba9859de80ae113344b118fe85398309704c4cf8cfd2096d3`).
The current teacher Reference reconciles the launch-pair explanation at line 131,
T19 split at line 142, and open-dependency inventory at lines 151–155. Configs
and Task 3.2 guards are completed prerequisites; Task 3.1 trainer/action mapping,
run_ppo/native-env seams and Task 4.4 remain explicitly open. The description,
references index and newest log entry agree. Historical receipts remain scoped
as history.

## Parent retention

Source inventories use Python AST on tracked `tests/**/*.py` (qualified names,
including class scopes), and Rust `#[test]` function declarations in `src/`
and `engine_rs/`. These count definitions, not parametrized runtime cases.

| Inventory | Integration | Teacher | HEAD |
| --- | ---: | ---: | ---: |
| Python test definitions | 893 | 797 | 926 |
| Rust test definitions | 327 | 226 | 327 |
| Cookbook log section titles | 100 | 75 | 106 |

- No tracked path from either parent is absent.
- Every integration Python test name remains; only the value-CE test changes
  AST, to call the relocated model-owned method with the same numerical assertion.
- The sole absent teacher Python name is
  `test_task_authored_inventory_requires_exact_replay_test`; integration already
  renamed/expanded it to `test_task_authored_inventory_requires_exact_authored_set`,
  adding generated-manifest and retired-bridge cases. It remains in HEAD.
- All 33 teacher test function bodies are AST-identical to the teacher parent.
  Two decorators use the renamed current skip reason; skip strings identify the
  demonstrated remaining seams. Other teacher-parent test differences are
  preserved integration compile/evaluation changes.
- No Rust test names are missing, and the complete `src/` and `engine_rs/` trees
  are byte-identical to integration, including `TRIM_MANIFEST.json`.
- All log sections from both parents retain their bodies (section-boundary
  whitespace stripped). HEAD has 106 unique titles, with no duplicates.

Full inventories and differences: `inventory.json`, `summary.json`,
`semantic_checks.json`, `parent_file_retention.json`.

## Reconstruction and semantic review

A fresh three-way `git merge-file -p` reconstruction in scratch files examines
all twelve paths changed by both parents. Eight reconstruct byte-for-byte as
HEAD; four conflict exactly where reported: run_ppo imports, RL API docs,
cookbook log and references index. The remaining one-parent-only changes are
limited to the disclosed coverage counts, teacher skip reasons and teacher
Reference reconciliation. No previously unchanged parent path changes in HEAD.
`reconstructed_merges.json` records the results; conflict reconstructions are
saved next to it.

- `scripts/run_ppo.py` keeps `terminal_seat_banks` (used at line 1637) and
  `KaggricultureObsConfig`. Its teacher obs dispatch preserves Orbit max-entity
  adaptation, requires exact Kaggriculture spec equality, and rejects cross-game
  teacher/student pairs.
- RL API docs retain the teacher-target bullet and the integration Environment
  description in full, followed by both native sections. The discarded shorter
  Environment line contains no distinct semantics.
- The references index retains both topic sets and integration's completed
  model-registration statement. Log section bodies from both parents survive.
- All three teacher model entry paths call `encode_observations`, which calls
  `_run_trunk`; its retained `require_compiled_gemm_backends` check precedes
  compiled dispatch. The teacher side never replaces trunk dispatch or bypasses
  the integration's cuBLAS-only policy.
- Teacher-specific PPO changes coexist with the integration's truncation
  dispatch: protocol typing/cache metrics and model-owned CE do not overwrite
  the retained economic-reward/bootstrap behavior.
- Base value CE matches the old free function, and removal of `.view_as` does
  not alter Orbit's segment-major layout. Kaggriculture's override computes one
  live-seat mean per state, aligning with the existing state-weighted loss.

This audit performs no tracked modifications. Runtime and mutation checks are
being independently handled by the root verifier; this report does not claim
those outcomes. Phase 4 is still incomplete at the disclosed trainer/native-env
and Task 4.4 seams; this audit assesses the staging merge's implemented scope.
