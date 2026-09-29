# Independent mutation verification, review r2

Source: `ca370893eff25f98a1a00a0bbcde82750eff1c4c`, three-dot base
`0b8cf98ef57fc49a329dca4c8368c630586c4752`.

All **20 mutations were detected** by the intended oracle with pytest exit 1;
none survived or failed at collection/import. Baseline and restored runs each
passed **53 tests** (all config tests plus all three ranked startup cases).
`summary.json`, `results.json` and the individual logs record the exact commands,
failure observations and SHA-256 hashes. `run_mutations.py` is the independently
written driver; it does not reuse the r1 receipts or execution scripts.

The physical `scratch/` copy contains configs, the Python package, relevant tests
and conftests, `scripts/run_ppo.py`, and pytest configuration. The worktree's
virtualenv interpreter runs with scratch `PYTHONPATH`, no bytecode/cache writes,
and `OMP_NUM_THREADS=2`. All mutations affect only physical scratch files. Every
target is restored in `finally`, and all **118 copied files** were then verified
byte-for-byte against both the original copies and live source. The full hash
manifest is `restored-sha256.json`. Tracked source was never edited.

## Mutation-to-oracle map

Test names below are in `tests/kaggriculture/test_configs.py` except startup.
Expanded parametrized tests use the exact eight-rank node ID, avoiding accidental
matches against the scratch directory name.

| Mutation receipt | Intended oracle and observed failure |
| --- | --- |
| `per-rank-spm.log` | New divided-shape test: eight-rank spm 2 → 4 differs from 16/8. |
| `per-rank-envs.log` | New divided-shape test: envs 32 → 64 differs from 256/8 and changes clamped teacher slice. |
| `fixed-teacher-setting.log` | New divided-shape test: teacher setting 128 → 64 leaves slice 32 but fails fixed-setting assertion. |
| `env-divisibility-guard.log` | New helper's env divisibility guard bypassed: world size 3 produces the wrong named error and fails expected regex. |
| `spm-divisibility-guard.log` | New helper's spm divisibility guard bypassed: world size 32 produces ZeroDivisionError rather than explicit ValueError. |
| `partial-minibatch-guard.log` | New helper's partial-minibatch guard bypassed: accumulation 3 no longer raises. |
| `partial-teacher-guard.log` | New helper's partial-teacher guard bypassed: teacher slice 24 at world size 8 no longer raises. |
| `cross-rank-optimizer.log` | Expanded rank comparison: eight-rank Muon LR .002 → .003 fails optimizer equality. |
| `cross-rank-env.log` | Expanded rank comparison: eight-rank native threads 2 → 3 fails env equality. |
| `cross-rank-ppo.log` | Expanded rank comparison: eight-rank entropy coefficient 1e-6 → 2e-6 fails PPO equality. |
| `cross-rank-model.log` | Expanded rank comparison: eight-rank model depth 8 → 7 fails model equality. |
| `rollout-seat-rows.log` | New eight-rank headroom test: omit rollout seat multiplier, 32 rows instead of 64. |
| `minibatch-seat-rows.log` | New eight-rank headroom test: omit minibatch seat multiplier, 128 rows instead of 256. |
| `teacher-clamp-and-call-counts.log` | New eight-rank headroom test: remove teacher clamp, yielding (16384 rows, 3 trunk, 2 head calls) instead of (4096, 1, 1). |
| `evaluation-seat-rows.log` | New eight-rank headroom test: omit evaluation seat multiplier, 32 rows instead of 64. |
| `startup-workload-caller.log` | New eight-rank `test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation` case in `tests/scripts/test_run_ppo.py`: remove main's caller, required headroom lines become empty. |
| `global-workload.log` | Expanded global-workload test: envs 32 → 64 yields 512 global envs instead of 256. |
| `scaling-optimizer.log` | Expanded recipe equality test: Muon LR .002 → .003 differs from scaling_6m. |
| `reward-recipe.log` | Expanded env/cross-section test: economic shaping .2 → .1 differs from required reward recipe. |
| `model-factory.log` | Expanded loader/construction test: disable Kaggriculture dispatch, factory reaches `assert_never`. |

Scope is CPU oracle sensitivity for the new and extended tests. Production guards
are unchanged in this diff; their copied implementations are mutated only to
exercise the extended tests. This does not qualify GPU, DDP or training behavior.
The guard bypass yielding ZeroDivisionError is specifically an explicit-error
contract failure, not a numerical assertion failure.
