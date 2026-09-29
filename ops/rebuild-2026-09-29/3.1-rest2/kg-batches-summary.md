# Complete bounded Kaggriculture validation

All 1,095 collected node IDs occur exactly once across 15 successful batches: **1,092 passed, 3 skipped**, no missing, extra or duplicated node IDs. `kg-coverage-manifest.json` contains exact commands, node outcomes, printed summaries and resource receipts. The failed full-teacher attempt is excluded from that coverage; it stopped at 952,156,160 bytes during the brute-force hire case, then the same test passed alone.

Every batch command used `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 PYTHONPATH=ops/rebuild-2026-09-29/3.1-rest2 PYTEST_PLUGINS=bounded_pytest UV_OFFLINE=1`. The ops-only plugin limits each pytest process to 900 MiB and 115 seconds. Its RSS scope is the pytest process only, sampled each 10 ms; process enumeration is denied by the sandbox. The static typing probe uses `--noconftest` to avoid loading Torch/project fixtures into the pure mypy test; the test body is unchanged. All other batches use normal conftests.

| Batch | Printed result | Pytest sampled RSS bytes |
| --- | --- | --- |
| `typing` | 1 passed in 7.05s | 760725504 |
| `native` | 757 passed, 3 skipped in 3.74s | 417464320 |
| `grammar_configs` | 128 passed in 1.06s | 324845568 |
| `model_encoder` | 43 passed in 1.61s | 407175168 |
| `model_heads` | 60 passed, 5 deselected in 5.18s | 476856320 |
| `heads_budget_0` | 1 passed in 0.13s | 495779840 |
| `heads_budget_1` | 1 passed in 0.17s | 594624512 |
| `heads_budget_2` | 1 passed in 0.17s | 612499456 |
| `heads_budget_3` | 1 passed in 0.18s | 613662720 |
| `heads_budget_10` | 1 passed in 0.18s | 613695488 |
| `model_compile_registration` | 18 passed in 0.28s | 315752448 |
| `teacher_non_bruteforce` | 50 passed, 2 deselected in 4.66s | 548618240 |
| `teacher_bruteforce_base` | 1 passed in 0.12s | 466993152 |
| `teacher_bruteforce_hire` | 1 passed in 0.31s | 814465024 |
| `training` | 28 passed in 0.13s | 322142208 |

No production or test source files changed for these runs. Existing monolithic suite/py-prepare resource stops remain visible in the root receipts; complete isolated coverage does not claim the monolithic command passed.
