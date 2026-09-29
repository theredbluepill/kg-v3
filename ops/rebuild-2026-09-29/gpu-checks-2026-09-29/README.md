# GPU checks bundle — evidence (2026-09-29, pod `w7ia3zvxqsvs3g`, GPUs 0 and 1)

Run statement: `../run-statements/gpu-checks-bundle.md`. It was committed with the scripts in `5f2ee2d` (09:14:26Z), before any launch. Amendment 1 was committed in `a4c75e0` (09:20:45Z), after attempt 1 stopped and before attempt 2. The results section is "GPU checks bundle (component)" in `../results.md`.

## Identity

- **Source:** `kg/rebuild-trainer-model` @ `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`, tree `70d50fa37e66aa322e8db00c9359d3f9e3031499`.
- **Pod checkout:** `/workspace/kg-v3-rebuild` went from `e1458d2` (tree `16b79ba5…`, porcelain empty) to detached `8fde43c` via a git bundle (sha256 `ecb953dc…9e4e`; `git bundle verify` ok) plus fetch and checkout at 09:15:04Z. Receipt: `pod/attempt1/receipts/git_update.txt`.
  - `origin` is unchanged (the earlier bundle path), and nothing was pushed.
  - Porcelain was empty after the update, at both launches (`git_pre.txt`), after both runs (`git_post.txt`) and at 09:29:44Z (`post-run/idle_after_copy.txt`).
- **Rust extension:** not rebuilt. No `*.rs`, `Cargo.*`, `build.rs`, `pyproject.toml` or `uv.lock` change between `e1458d2` and `8fde43c`. `python/owl/rs.abi3.so` sha256 was `35239d1b…93b8` before and after.
- **Stack** (`receipts/versions.txt`): torch 2.9.0+cu128 (git `0fabc3ba…`), triton 3.5.0, flash-attn 2.8.3, CUDA runtime 12.8, driver 595.91.07, 2× RTX PRO 6000 Blackwell Server Edition (97,887 MiB each). No `TORCHINDUCTOR_*`/`TRITON_*` env vars were set. `which nsys` exits 1.
- **Backend records:** every `*.backend.json` has `value_at_start == value_at_end ==` the requested setting (`ATEN` or `ATEN,TRITON,CPP`). Every process record shows HEAD `8fde43c`, an empty porcelain, `kg_file` under `/workspace/kg-v3-rebuild/python`, and its `CUDA_VISIBLE_DEVICES`.

## Attempts

| attempt | driver wall | exit | what ran |
|---|---|---|---|
| 1 (`pod/attempt1/`) | 09:15:32–09:16:28Z, 55.4 s | 3 | `c3_mid_aten` pass. `c2_aten_bwd` **FAIL_stop**, only on `blocks.0.attn.k.bias`: gradient rel_max 1.155 / 0.807 / 0.858 at the warm point / 4,194,305 / 4,198,400. The running `c3_dense_aten` was terminated (rc −15) and nothing else started. |
| 2 (`pod/attempt2/`) | 09:21:13–09:29:13Z, 480.0 s | 0 | Every stage passed. The control reproduced its expected failure. |

The aggregate driver wall was 535 s, against the 60-min limit. Attempt 1 stopped on a degenerate criterion. The key-bias gradient is analytically zero (softmax shift invariance): fp64 CPU check 3.9e-14 vs 30.8 for the query bias, in `post-run/kbias_analytic_zero_cpu_check.txt`. Amendment 1 changed only that parameter's criterion, and attempt 2 reran the whole bundle. Not every c2 number reproduced across the attempts. The selected maxima did: output max |Δ|, dX `rel_max` and max token rel-L2, and the largest parameter-gradient `rel_max` excluding `k.bias`, at all three points. Other metrics changed:

- key-bias `rel_max` 1.155399 → 1.125237 (warm), 0.806780 → 0.796601 (4,194,305) and 0.858145 → 0.868495 (4,198,400);
- dX Frobenius-relative error, slightly: 0.004453830 → 0.004453535 (warm), 0.004454483 → 0.004454478 and 0.004454899 → 0.004454910;
- the eager-vs-eager floor at the warm point (dX 0.00133 → 0.00266, params 0.00172 → 0.00345), since flash backward is nondeterministic.

## Placement and timeline, attempt 2 (`pod/attempt2/driver.jsonl`)

- **GPU 1:** `c2_aten_bwd` 09:21:13–09:22:10 pass. Then `c2_ctl_default_bwd` until 09:22:47, control_reproduced: rc 1, illegal memory access during default-backend autotuning at 4,194,305. Then `c1_aten` until 09:23:19 pass, and `c1_default` until 09:23:51 pass.
- **GPU 0:** `c3_mid_aten` until 09:22:11, `c3_dense_aten` until 09:23:07, `c3_mid_default` until 09:24:16 and `c3_dense_default` until 09:25:27, all pass.
- **Phase 2, GPU 0 alone, GPU 1 idle:** `c4_mid_aten` 09:25:27–09:27:10 pass, then `c4_dense_aten` until 09:29:13 pass.
- **Idle:** GPUs 0 and 1 showed 0 MiB, 0 % and no compute processes at pre and prelaunch for both attempts, and at post and 09:29:44Z (`post-run/idle_after_copy.txt`). The pod was left running and idle.

## Files

- `scripts/`: the committed scripts. All are the attempt 2 versions except `driver.py` and `launch.sh`, which were revised after the run (see "Post-run corrections"), and `test_driver_cleanup.py`, which was added then. `pod/attempt{1,2}/` hold the scripts as run (sha256 in `receipts/scripts.sha256`), the logs, JSONL records, backend records, `template_census.*` and receipts.
- `pod/attempt2/c1_*.outliers.json`: every outlier coordinate (row, token, channel) per density and path.
- `summarize.py` → `summary.json`: every number quoted in `results.md`, computed from the retained records. The `c1_channels` and `c4_scaling` sections were added after the Codex review; the earlier sections are unchanged.
- `post-run/`: the key-bias CPU check, the post-copy idle/git capture and the local driver-cleanup test output (`driver_cleanup_test_local.txt`).
- Not copied: the Inductor/Triton caches (the pod run dir is 977 MB) and attempt 2's `receipts/compile_caches.sha256`, which is 4,123,148 B and over the 1 MB rule (sha256 `52b8e71ad24197ccb3d0b8b4fc55b2dd7efcfd8d6443c3a4cb2c30e7a5323eb9`, 22,125 lines). Both remain on the pod under `/workspace/kg-v3-rebuild/runs/gpu-checks-2026-09-29/`. Attempt 1's cache hash list (865 kB) is included.
- `MANIFEST.sha256`: sha256 of every file here.

## Post-run corrections (2026-09-29, after Codex reviews `verify-gpu-bundle-r1` and `verify-gpu-bundle-r2`)

- **Executed value-loss coefficient.** c3 and c4 computed `v_loss = 0.5·mean((v − ret)²)` and added `0.5 · v_loss`, so the executed value term was **0.25·MSE**, not the 0.5·MSE the run statement declares. The run statement carries a dated addendum. The c4 component timings remain usable, since a scalar loss coefficient changes no kernel or shape. The as-run scripts are left unchanged as evidence.
- **Outer-timeout cleanup.** The as-run `driver.py` started every stage in its own session (`start_new_session=True`) and had no signal handler. So `launch.sh`'s `timeout -k 20 3540` would have killed only the driver, leaving running stages alive. **Both recorded attempts used that driver, and neither overran:** attempt 2's driver took 480 s, and attempt 1 stopped normally with its running stage terminated (rc −15).
  - `scripts/driver.py` now installs SIGTERM/SIGINT/SIGHUP handlers and an `atexit` hook. They signal every started stage's process group: SIGTERM, up to 8 s wait, SIGKILL, up to 4 s wait. It exits 128 + signum.
  - `scripts/launch.sh` uses `timeout -s TERM -k 20 3540`, so the driver gets SIGTERM first and finishes cleanup well inside the 20 s grace.
  - `scripts/test_driver_cleanup.py` tests this locally on macOS with dummy stages (Python sleep processes with grandchildren, one stage ignoring SIGTERM; no GPU, no torch) under an emulation of GNU `timeout -s TERM -k 20 3`, since GNU `timeout` is not installed on the Mac.
  - With the revised driver, a timeout mid-stage in Phase 1 (worker threads) and in Phase 2 (main thread) left 0 surviving stage processes. The driver exited with 143 about 8.1 s after the signal (SIGKILL escalation for the TERM-ignoring stage), and `timeout` never needed its SIGKILL. An ordinary run exited 0 with no cleanup.
  - The as-run driver (`pod/attempt2/driver.py`) left all 4 stage processes alive, which shows the test discriminates.
  - **Spawn-window race (second revision, after Codex review `verify-gpu-bundle-r2` finding 3).** In Phase 2 (main thread), a signal after a stage's `Popen` had started the child but before the driver registered it ran cleanup without that stage and then raised `SystemExit`, so the stage survived. `driver.py` now starts and registers every stage in `_spawn`. While the main thread is in that call, the handler only records the signal, then handles it once the stage is registered. That covers a signal during `Popen` too. Phase 1 worker threads already register under the lock that cleanup takes, and kill the stage themselves if cleanup has run.
    - Signals are deferred rather than blocked with `signal.pthread_sigmask`: a local check showed that a child started while SIGTERM is blocked inherits the blocked mask (Python 3.9.6 and 3.14.5 on macOS), so the stage would ignore cleanup's SIGTERM.
    - `test_driver_cleanup.py` adds two deterministic scenarios: `new_spawn_after_popen` (a trace hook sends SIGTERM on the first driver line where the new stage exists unregistered) and `new_spawn_in_popen` (SIGTERM sent from inside the `Popen` call after the child started). Both check that the stage was unregistered when signalled, the driver logged a cleanup of it, exited 143 and left 0 survivors, and the next stage never launched.
    - Against the previous revision (driver sha256 `e0876f73…182a`), both new scenarios FAIL with 2 surviving stage processes each; the other four pass. With the current driver (`bc89e861…9ba3`), all six pass (`post-run/driver_cleanup_test_local.txt`, test sha256 `d6e3c601…f4ae5e`). Phase 1 worker-thread spawn races have no dedicated deterministic scenario.
  - Output: `post-run/driver_cleanup_test_local.txt`. The revised driver has not run on the pod or under GNU `timeout`.
