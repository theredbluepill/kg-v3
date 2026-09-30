Confirmed `kg/rebuild-gpu-checks` at `24380f3`. **F3 is resolved.**

- Spawn/registration is protected against cleanup signals.
- Both regressions hit the exact window and fail on the parent, each exposing two survivors.
- Fixed CPU suite: **6/6 pass**. All **26 recorded test PIDs** are absent after cleanup.
- Manifest: **110/110 verified**.
- Raw attempts: **94/94 unchanged**.
- No tracked modifications remain.

Testing used CPU dummy stages and emulated timeout; sandbox restrictions required recorded-PID liveness checks.

[Full verification report](/Users/poonszesen/kg-v3-gpuchecks/ops/rebuild-2026-09-29/codex/verify-gpu-bundle-r3.md)

VERDICT: APPROVE