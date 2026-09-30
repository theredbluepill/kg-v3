# GPU bundle confirmation — r2

Reviewed branch `kg/rebuild-gpu-checks`, HEAD `63921603e95ce8065f082c4a4ee031df4003dbb4`, against its parent and the six findings in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-gpu-bundle-r1.md`. This is a retained-data/source review plus CPU-only subprocess tests. No GPU run or tracked edit was made.

| r1 finding | Status | Confirmation |
|---|---|---|
| 1. fp32/channel attribution | **RESOLVED** | Corrected channel shares reproduce from retained coordinates. Reporting now preserves the unexplained shared concentration and the narrower aggregate-threshold conclusion. |
| 2. Scaling explanation | **RESOLVED** | CUDA-event medians reproduce the A/B/C/D decomposition and the 2×/4× ideal scaling denominators. |
| 3. Outer-timeout cleanup | **PARTIAL** | Registered-stage cleanup passes both supplied timeout scenarios, but a signal during Phase 2 launch before registration still leaves a live child. |
| 4. Cross-attempt metric identity | **RESOLVED** | README correctly distinguishes reproduced selected maxima from changed key-bias and dX Frobenius-relative errors. |
| 5. Executed value coefficient | **RESOLVED** | C3 and C4 execute **0.25·MSE**; results, README and the dated run-statement addendum disclose it. Original declarations and as-run scripts are preserved. |
| 6. Mid replay maximum | **RESOLVED** | Raw mid-default maximum is **0.001983642578125**, now reported as approximately **2.0e−3**. |

## Remaining edit: finding 3

At `gpu-checks-2026-09-29/scripts/driver.py:356–359`, Phase 2 starts its child in the driver's main thread, then registers it in `self.procs`. A signal after `Popen` returns but before registration invokes cleanup without that child in its snapshot. `_on_signal` raises `SystemExit` at line 322, so execution never reaches registration or the subsequent `late` SIGKILL branch. An `atexit` hook cannot recover it because cleanup has already set `cleaned` and has no registered child.

A deterministic CPU-only trace injection delivered SIGTERM at line 358, after a separate-session sleep child existed. The unmodified driver exited **143**, `registered_groups` was empty, and the child remained alive. The harness killed that specific owned process group and verified its PID absent. Evidence: [boundary result](verify-gpu-bundle-r2-cleanup-boundary/result.json), [reproducible harness](verify-gpu-bundle-r2-cleanup-boundary/probe.txt). Driver SHA-256: `e0876f73f26598b8259c2d4e820888e8dfbaee4366ac4c8b761270ef51b3182a`.

Protect the entire spawn/registration operation against signal-driven unwinding, including signals while `Popen` is executing, or defer cleanup/unwinding until the newly created group is registered. Add a CPU regression at this boundary before asserting cleanup of every started stage. The current mid-stage tests fire after registration and do not cover it. This does not invalidate either historical GPU attempt: neither overran.

## Recomputed channel-229 shares

All 18 coordinate lists were counted independently. Coordinates are unique, every list matches the raw outside-tolerance count, and none reached the 50,000 retention cap. Each channel represents **1/256 = 0.390625%** of present-token elements.

| Density | Eager (both backends) | Padded (both backends) | Compiled ATEN | Compiled default |
|---|---:|---:|---:|---:|
| Mid | 700/5,305 = 13.195099% | 663/5,248 = 12.633384% | 176/1,171 = 15.029889% | 138/910 = 15.164835% |
| Mixed | 515/5,917 = 8.703735% | 482/5,943 = 8.110382% | 115/1,110 = 10.360360% | 107/842 = 12.707838% |
| Dense | 238/6,086 = 3.910615% | 222/6,061 = 3.662762% | 11/821 = 1.339829% | 6/597 = 1.005025% |

Source: `pod/attempt2/c1_{aten,default}.outliers.json`, cross-checked against their JSONL counts. Corrected reporting: `../results.md:322,342–345`. The maximum mean-error ratio to the median is 1.0004812 and maximum outside-fraction ratio is 1.0108613; neither exceeds the predeclared 2× criterion. BF16 rounding is supported; the concentration remains unattributed.

## Recomputed scaling decomposition

Recomputed medians from each retained `cuda_event_ms` array. Wall is `64·A + 16·B + C + D`; excess below is `f·wall(ranks) − wall(2)`, where `f = ranks/2`. Efficiency is `wall(2)/(f·wall(ranks))`, equivalently global SPS divided by the 2× or 4× ideal rate.

| Density/ranks | A excess, s | B excess, s | C excess, s | D excess, s | Total excess, s | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Mid/4 | −0.038175720 | +0.143789551 | −0.000802246 | −0.000994607 | +0.103816977 | 97.960545% |
| Mid/8 | +0.295025635 | +0.300479370 | −0.026266754 | −0.000173488 | +0.569064763 | 89.757055% |
| Dense/4 | −0.059794495 | +0.148553955 | −0.004346069 | −0.001214959 | +0.083198432 | 99.085849% |
| Dense/8 | −0.062792786 | +0.571544922 | −0.035078064 | −0.001966096 | +0.471707976 | 95.029254% |

Two/four/eight-rank-shape walls are **4.986610763 / 2.545213870 / 1.388918882 s** (mid) and **9.017973435 / 4.550585934 / 2.372420353 s** (dense). They match the retained derived records and corrected `results.md:400–404`. These are component estimates from one GPU at per-rank shapes, excluding engine, copies, all-reduce, GAE and logging.

Loading `summarize.py` under a non-main name and invoking its six reader functions reproduced every stored `summary.json` section exactly, without executing the writer.

## Other numerical confirmations

- Key-bias `rel_max` changed across attempts: **1.155399084 → 1.125236869**, **0.806780159 → 0.796601295**, **0.858144999 → 0.868495226**. The four selected maxima claimed in the corrected README match exactly at all three points; the stated dX Frobenius-relative changes also reproduce.
- Retained `c3_smoke.py:225,231` and `c4_timing.py:203,209` compute half-MSE then apply another 0.5 coefficient. Executed value loss is **0.25·MSE**.
- Mid default's two replay batch sizes both give maximum absolute log-ratio **0.001983642578125**; mid ATEN gives **0.0018463134765625**.

## CPU cleanup tests and custody

The supplied test initially failed at the old-driver control because the sandbox denied `ps`; `pgrep` also cannot enumerate processes. The failure transcript is [retained](verify-gpu-bundle-r2-cleanup.txt). The test was rerun with only process-observation helpers substituted: `os.kill(pid, 0)` checks all recorded leader/grandchild PIDs keyed to each scenario, instead of `ps`/`pgrep`. The driver, dummy stages, timeout emulation, signals, assertions and control cleanup stayed unchanged. [Rerun output](verify-gpu-bundle-r2-cleanup-pids.txt):

- Phase 1: **PASS**, four recorded stage/child PIDs absent; driver exit 143 **8.21 s** after SIGTERM; outer SIGKILL unnecessary.
- Phase 2: **PASS**, two recorded stage/child PIDs absent; driver exit 143 **8.09 s** after SIGTERM; outer SIGKILL unnecessary.
- Ordinary completion: **PASS**, exit 0, no signal/cleanup event.
- Old-driver control: **PASS as a discriminating control**; four survivors before harness cleanup, zero after its SIGKILL cleanup.
- Boundary regression described above: confirms the remaining defect; its child was absent after harness cleanup.

This is local macOS/CPU evidence with emulated GNU timeout, not a pod or actual GNU-timeout rerun.

The first, unadapted test's control left four 300-second sleep processes when `ps` failed before its cleanup block. A later sandbox invocation could not signal them, so their bounded lifetimes were allowed to expire. At **09:55:45Z**, `os.kill(pid, 0)` returned `ProcessLookupError` for all four, and for every additional PID retained from the rerun and boundary probe. Together with the per-scenario checks above, no test child remains alive. [Final PID receipt](verify-gpu-bundle-r2-final-pids.json). Process-wide enumeration was unavailable; verification used the test's recorded children.

Custody checks at HEAD:

- **110/110 manifest checksums pass**, with complete local coverage except the manifest itself.
- Relative to the parent's 108 entries: **103 unchanged**, five updated files (README, driver, launcher, summarizer, summary), two additions (cleanup test and its local receipt), no deletions.
- All **94 raw pod files**—34 in attempt 1, 60 in attempt 2—are byte-identical to the parent commit and have unchanged manifest hashes.
- Both historical script receipts verify **9/9**. Attempt 1's nine scripts match prelaunch `5f2ee2d`; attempt 2's nine match amendment `a4c75e0`.
- The manifest's remote-only attempt-2 cache-hash-list comment is not a local entry. That remote file was not reverified in this review.

Final manifest recheck again passed **110/110**. Both `git diff --exit-code` and `git diff --cached --exit-code` returned **0**, with no tracked changes. This report and its new local test receipts are untracked review artifacts.

VERDICT: APPROVE WITH EDITS
