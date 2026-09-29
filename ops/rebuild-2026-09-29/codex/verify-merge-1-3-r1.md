**P2 — Oracle source custody omits live engine inputs.** At [regenerate.py:975](/Users/poonszesen/kg-v3-m-1-3/scripts/kaggriculture_observation_oracle/regenerate.py:975), snapshots omit `engine_rs/src/py_random.rs` and `engine_rs/Cargo.toml`. Mutations to either survive the real source checks; regeneration probes install output despite that drift. **Fix:** capture/recheck all live engine inputs and add both regression cases.

Parent preservation passed: all nine relocated kernel tests survive, cuBLAS-only and 3.x code remain intact, and the manifest generator reproduces merged bytes exactly. Docs/cookbook preserve both parents’ records.

Fresh checks:

| Check | Result |
|---|---|
| Engine Cargo tests | 69 passed |
| Root Cargo tests | 254 passed, 4 ignored |
| Trim checker | Passed |
| Pytest | 1,618 passed, 7 skipped |
| Mypy | Passed, 62 files |

Pytest completed in 45 shards after the combined run hit the memory guard. **47 source mutants were detected**, plus 8 rejected boundary-input mutations; the two custody-drift probes exposed the finding.

All scratch edits were restored byte-for-byte. All **1,629 tracked files remain unchanged**.

[Full report and evidence](/Users/poonszesen/kg-v3-m-1-3/ops/rebuild-2026-09-29/verify-e197528-independent/review.md)

VERDICT: REJECT