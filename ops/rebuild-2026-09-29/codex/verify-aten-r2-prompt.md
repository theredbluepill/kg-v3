You are Codex, INDEPENDENT VERIFIER of branch kg/rebuild-aten at HEAD 3b735c064ce3b11c743736d89669edbe0ea92248 (this worktree, /Users/poonszesen/kg-v3-aten) relative to integration f4ecd90ec15cf09adcd20570aedf636324ffac30. Leave no tracked modification.

Task under review: cuBLAS-only GEMM setting for Kaggriculture compiled regions (evidence: ops/rebuild-2026-09-29/results.md 'ATEN-only GEMM A/B').

Verify:
- the setting is applied before torch.compile only for Kaggriculture max-autotune compiles;
- Orbit paths are untouched;
- process-global conflicts fail loudly;
- the version check is correct and honest about the driver;
- guards are retained;
- the Decision does not overclaim owner adoption or immunity (FlexAttention/static/bmm/backward limits);
- the cookbook is consistent.

Also confirm every finding in /Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-aten-r1.md is resolved.

Run:
  uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q
  uv run mypy python/owl scripts
and report counts.

End with VERDICT: APPROVE / APPROVE WITH EDITS / REJECT, with findings (severity, file:line, fix).
