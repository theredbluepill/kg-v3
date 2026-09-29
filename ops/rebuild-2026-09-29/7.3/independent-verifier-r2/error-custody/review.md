# Independent live-evaluation custody probes

**P2 — Aborted evaluation loses all active selected-game custody.**
`python/owl/kaggriculture/native_evaluation.py:119-141` calls the policy,
selected-game decoder and native `env.step` without an exception handler that
closes active recorder entries. `fail_game` is used only for the narrower bank
mismatch at lines 157–163. The Task 7.3 brief, export contract item 6, requires
native runtime errors or failed transactions to produce explicit incomplete/error
records.

Two bounded live probes each commit one turn, then submit length zero on the
second call. The first rejects in the selected-game decoder (line 129), leaving
both selected games active and zero files. The second corrupts only the
unselected env so selected decoding succeeds and the batch transaction rejects
at line 141; the selected game remains active with zero files. In both cases the
native transaction correctly preserves step 1, but its one committed action and
captured state survive only in the recorder's memory. The receipt distinguishes
the native rejection from the custody defect.

**Fix:** add exception cleanup for all active selected games, persist their
committed tapes as error custody, then re-raise the original failure. Cover both
selected decoder failure and native transaction failure; ensure one failing
game does not leave other active selected games without custody.

Command (exit 0; probe records the defect without expecting patched behavior):

```sh
uv run --no-sync python ops/rebuild-2026-09-29/7.3/independent-verifier-r2/error-custody/probe.py ops/rebuild-2026-09-29/7.3/independent-verifier-r2/error-custody/receipt.json
```

`probe.log` and `receipt.json` retain the observed exception text and actual vs
expected sidecar counts. No tracked source was changed. Temporary replay
directories were deleted on probe completion.
