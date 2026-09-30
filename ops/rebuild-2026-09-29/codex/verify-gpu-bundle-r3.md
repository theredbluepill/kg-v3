# GPU bundle confirmation — r3

Confirmed branch `kg/rebuild-gpu-checks`, HEAD `24380f3eb0e9c7090d874f8f1a45c0d4e3477f40`, against parent `63921603e95ce8065f082c4a4ee031df4003dbb4` and [r2 finding 3](verify-gpu-bundle-r2.md). **F3 is RESOLVED. No blocking findings or requested edits.** This review made no tracked modification and ran no GPU work.

The bounded question was whether SIGTERM after child creation but before registration can still unwind Phase 2 and orphan the stage. Completion required the current test to pass on the fix, fail specifically at this boundary on the parent, leave no recorded test survivors after control cleanup, and verify retained artifact custody.

[`Driver._spawn`](../gpu-checks-2026-09-29/scripts/driver.py) lines 358–372 sets main-thread deferral before entering `Popen`, registers the process under the cleanup lock, then drains the pending signal. `_on_signal` lines 331–336 records the signal and returns while deferral is active. Consequently, cleanup sees the newly created group before `SystemExit` can unwind the main thread. A signal after deferral clears also sees the registered process. Worker-thread registration retains its existing lock/late-cleanup handling.

The regression targets the missing window precisely. In [`test_driver_cleanup.py`](../gpu-checks-2026-09-29/scripts/test_driver_cleanup.py), the trace hook at lines 179–189 requires a bound `Popen` for `c4_race` which is absent from `driver.procs`. The second hook at lines 170–176 sends SIGTERM after child creation while the `Popen` constructor has not returned to the driver. Lines 312–320 require an unregistered boundary, cleanup of `c4_race`, exit 143 without outer timeout, zero observed survivors, and no launch of `c4_after`.

| CPU scenario | Fixed driver | Parent driver |
|---|---|---|
| Phase 1 timeout, including TERM-ignoring stage/grandchild | PASS; exit 143, 8.18 s after signal, no survivors | PASS |
| Phase 2 timeout, including TERM-ignoring stage/grandchild | PASS; exit 143, 8.14 s after signal, no survivors | PASS |
| Ordinary completion | PASS; exit 0, no cleanup | PASS |
| Signal after `Popen`, before registration | PASS; exit 143, no survivors | FAIL; exit 143, two surviving processes |
| Signal inside `Popen`, after child creation | PASS; exit 143, no survivors | FAIL; exit 143, two surviving processes |
| Historical as-run driver negative control | PASS as control; four survivors, then harness cleanup | PASS as control |

The fixed suite passes **6/6, exit 0**. The same suite selecting the parent driver fails **exactly the two new boundary scenarios, exit 1**. For the post-return injection, retained boundary records identify fixed `_spawn` line 364 and parent `run_stage` line 358. All four boundary records have `registered_at_signal: false`. Both fixed-driver cleanup records contain `c4_race`, `sigkill: false`, `survived: false`; the parent has no cleanup record for it. No following stage launches. The parent’s surviving leader/grandchild pairs were killed by the test harness.

Evidence: [fixed transcript](verify-gpu-bundle-r3-evidence/fixed-cleanup.txt), [parent transcript](verify-gpu-bundle-r3-evidence/parent-cleanup.txt), [all scenario results](verify-gpu-bundle-r3-evidence/scenario-results.json), [reproducible review harness](verify-gpu-bundle-r3-evidence/run-review.txt), [harness result](verify-gpu-bundle-r3-evidence/harness.txt). Scenario directories are retained alongside these receipts.

This rerun used macOS/Python 3.9.6, CPU dummy stages and the committed GNU-timeout emulation. The sandbox denies `ps` and prevents `pgrep` enumeration. Only parent-side process-observation helpers were replaced with conservative `os.kill(pid, 0)` checks over recorded leader, grandchild and boundary PIDs; permission errors count as alive. The driver, child-side signal injection, timeout logic and test assertions remained unchanged. Review instrumentation also retained temporary directories and selected an exact `git show 24380f3^:.../driver.py` export for the parent arm. All **26 distinct recorded PIDs were absent**, already before the review harness’s final safety-cleanup step; [final PID receipt](verify-gpu-bundle-r3-evidence/final-pids.json). This is recorded-PID verification, not process-wide enumeration. Actual GNU timeout/pod execution and deterministic Phase 1 spawn-boundary injection remain untested, as the bundle README discloses.

Custody checks pass ([receipt](verify-gpu-bundle-r3-evidence/custody.json)):

- **110/110 manifest entries verify**, covering all 111 tracked bundle files except the manifest itself. The local manifest has no duplicate, missing or mismatched entries.
- Versus the parent: the same 110 paths, **106 unchanged hashes**, and four updated entries: README, driver, regression and local cleanup receipt.
- All **94 raw attempt files**—34 in attempt 1 and 60 in attempt 2—are byte-identical to both the parent and the original bundle-results commit `73b822a`, with no path additions or removals.
- Both historical script receipts verify **9/9**. The deliberately uncopied remote compile-cache listing was not reverified.
- Fixed driver SHA-256: `bc89e8615cc8820b88721a5e7061cbfa32e866a48904d259b40e7676fde49ba3`; parent: `e0876f73f26598b8259c2d4e820888e8dfbaee4366ac4c8b761270ef51b3182a`; regression: `d6e3c6013b5b4e1da1fd74dca12cf0fe215d9939b4725c40c2485c2838f4ae5e`.

`git diff --check 24380f3^ 24380f3` passes. Final working-tree and index diffs are empty; all new review evidence is untracked. Earlier review artifacts were preserved.

VERDICT: APPROVE
