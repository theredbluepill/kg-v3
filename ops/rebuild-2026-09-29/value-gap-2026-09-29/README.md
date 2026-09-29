# Value-gap diagnostic — evidence (2026-09-29, pod `w7ia3zvxqsvs3g`, GPUs 0 and 1)

Run statement: `../run-statements/value-gap-diagnostic.md`, with Amendments 1–3, each committed before its relaunch. Results: `../results.md` "Value gap diagnostic". Every number there comes from `summary.json` (`python3 summarize.py pod pod/frerun`).

## Identity

- **Source under test.** Pod checkout `/workspace/kg-v3-rebuild` at `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed` (tree `70d50fa3…`).
  - HEAD and empty porcelain were recorded at every launch (`pod/*/receipts/git_pre.txt`), after every driver (`git_post.txt`) and at 11:08:15Z (`post-run/idle_and_git_after_copy.txt`). `python/owl/rs.abi3.so` sha256 `35239d1b…` was unchanged throughout.
  - No transfer, checkout change, rebuild or install. `/workspace/kg-v3` and both venvs were not touched.
- **Stack** (`pod/receipts/versions.txt`): torch 2.9.0+cu128 (git `0fabc3ba…`), triton 3.5.0, flash-attn 2.8.3, driver 595.91.07, 2× RTX PRO 6000 Blackwell Server Edition.
- **nsys:** absent (`pod/receipts/nsys_check.txt`, `which` rc 1). No timing was taken.
- **Branch commits:** statement `8bfd1c4`, scripts `e97c7a7`, Amendment 1 `96ae784`, Amendment 2 `c9b6cde`, Amendment 3 `6afead3`. Each was committed before the launch it governs.

## Attempts (aggregate driver wall 449.1 s of the 45-min limit, ≈ $0.52 at $4.18/h)

| attempt | driver wall | exit | outcome | files |
|---|---|---|---|---|
| 1 (10:47:44–10:49:20Z) | 95.9 s | 3 | `kgC_fp32_comp_aten`: Inductor compile error in the fp32 training graph (`analyze_memory_coalescing`, sympy `is_constant`). Amendment 1 | `pod/attempt1/` |
| 2 (10:52:29–10:53:41Z) | 72.1 s | 3 | `kgC_fp32_eager`: CUDA OOM at 1,024 rows (92.4 GiB allocated). Amendment 2 | `pod/attempt2/` |
| 3 (10:58:27–11:02:23Z) | 236.5 s | 0 | all 9 stages passed. The compiled F stages hit Dynamo's recompile limit through the `is_gap.py` flash wrapper, so their 1,024-row cells ran eagerly. Amendment 3 | `pod/` |
| 3-F rerun (11:06:06–11:06:50Z) | 44.6 s | 0 | `isF_comp_default` and `isF_comp_aten` passed, on attempt 3's states (operator-reported; see the run statement's post-run addendum), with one `grad_mode` recompile each and no limit | `pod/frerun/` |

- The idle gate passed at 0 s at every launch. GPUs were idle after every run and at 11:08:15Z, at 0 MiB, 0 % and with no compute processes (`receipts/idle_nvidia_smi_*`, `post-run/`).
- The pod was left running and idle.
- Amendment 2 quotes attempt 2 as 72.3 s; the receipt says 72.1 s (`pod/attempt2/driver.jsonl`).
- The driver's process-group cleanup was not exercised by a signal on the pod. Each driver ended normally, and its atexit cleanup found no live group. The stops in attempts 1 and 2 terminated the other stream's stage (rc −15). Local test outputs are in `pre-launch/driver_cleanup_test_local*.txt`.

## Layout

- `scripts/` holds the final scripts (Amendment 3). The scripts as run in each attempt are in `pod/<attempt>/` and hashed in its `receipts/scripts.sha256`.
- Per stage: `pod/**/<stage>.jsonl` (process and result records), `.log` (stdout/stderr including `TORCH_LOGS=recompiles`) and `.backend.json` (the GEMM-backend record).
- `driver.jsonl` and `launch.out` per attempt.
- `summarize.py` and `summary.json` (the table and the pre-declared prediction verdicts).
- `pre-launch/`: local cleanup tests and the dry-run note (not evidence).
- `pod/large_files.sha256`: the excluded files, which stay on the pod: Orbit Wars observation files (`obs/orbit_obs_{256,1024}.pt`, 30.6 MB / 122.3 MB per attempt), the per-attempt cache hash lists `receipts/caches_and_obs.sha256` (0.9–2.8 MB), and Inductor/Triton cache file counts. The file is kept as copied from the pod. Its lines 13–14 (`fc1a0b35…` and the `ls -l` line dated 11:06) name `receipts/caches_and_obs.sha256` without a directory, like line 7, but they belong to the rerun, `frerun/receipts/caches_and_obs.sha256`. They follow the attempt-2 cache count, precede the `frerun` cache count, and are dated inside the rerun's launch window (11:06:03–11:06:56Z, `pod/frerun/launch.out`). Line 7 is attempt 3's list.
- `MANIFEST.sha256`: sha256 of every file in this directory.
