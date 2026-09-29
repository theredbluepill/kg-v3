# Fresh mutation verification of b51b0c0...7ad45fc

Target: the new/expanded config oracles and eight-rank startup case. Success required baseline success, intended test failure for each mutation, restoration after every case, and complete source-byte identity after the final baseline. No training or GPU execution was attempted.

Fresh files were copied from the current worktree, not the earlier scratch. Source identity and native-extension SHA-256 are in `source-identity.json`; exact mutation text, pytest selectors, commands and per-file hashes are in `results.json`. Every command used the worktree virtualenv Python with scratch `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2`, and pytest's cache provider disabled.

- Baseline: **17 passed, 137 deselected**.
- Mutations: **15/15 killed**; **22 expected test failures** in total, no collection/import/resource errors.
- Restored baseline: **17 passed, 137 deselected**.
- All **187** scratch files, including the copied native extension, match their original bytes and SHA-256 hashes after restoration. File inventory also matches. The corresponding workspace files still match the initial manifest.

| Mutation | Selected oracle and observed failure |
| --- | --- |
| 8-rank spm 2 to 4 | Global-workload equality, per-rank division, fixed cross-rank shape and eight-rank headroom all fail (4 failures). |
| 8-rank envs 32 to 64 | Same four oracles plus startup's printed rollout row count fail (5 failures). |
| Teacher constant 128 to 64 | The new per-rank oracle independently catches the literal teacher setting even though the effective 32-env clamp is unchanged. |
| Remove env divisibility guard | World 3 loses the required n_envs diagnostic and reports the later spm diagnostic; the exact-message test fails. |
| Remove spm divisibility guard | World 32 floors spm to zero and produces ZeroDivisionError instead of the required informative ValueError; the test fails. |
| Remove minibatch divisibility guard | Accumulation 3 no longer raises the required ValueError. |
| Remove teacher divisibility guard | Teacher chunk 24 at world 8 no longer raises the required ValueError. |
| Wrong optimizer muon_lr | Cross-rank optimizer equality fails. |
| Wrong env native_threads | Cross-rank env equality fails. |
| Wrong model preset | Cross-rank model equality fails. |
| Wrong PPO ent_coef | Cross-rank PPO equality fails. |
| Remove startup _check_model_workload call | The eight-rank main test fails because all four required headroom lines disappear. |
| Replace teacher clamp min with max | Only the new eight-rank headroom test runs; teacher rows/calls become (16384, 3, 2), expected (4096, 1, 1). |
| Add one to computed trunk calls | Only the new eight-rank headroom test runs; each workload has two trunk calls, expected one. |
| Add one to computed head calls | Only the new eight-rank headroom test runs; each workload has two head calls, expected one. |

The four divisibility guards are test-helper checks, not runtime guards. The startup test uses the real YAML loader but stubs distributed/torch setup and stops at the explicit Kaggriculture-env-not-wired error before allocation. These results demonstrate the CPU oracles' sensitivity; they establish no eight-process launch, GPU memory, DDP, throughput, learning or end-to-end claim. No finding arose from these checks.
