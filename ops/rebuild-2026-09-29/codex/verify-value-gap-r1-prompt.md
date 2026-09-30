You are Codex. READ-ONLY review of the value-gap diagnostic on branch kg/rebuild-value-gap (worktree /Users/poonszesen/kg-v3-valuegap): run statement, ops/rebuild-2026-09-29/value-gap-2026-09-29/, results.md section 'Value gap diagnostic'.

Summary as reported by the executing agent (verify every claim against the files; do not trust it):

- ran: true
- source_commit: 8fde43cd7408c9c4f9147eeb8f916bd08f8266ed (pod checkout /workspace/kg-v3-rebuild, tree 70d50fa3…; HEAD and empty porcelain recorded before and after every launch and again at 11:08:15Z)
- run_statement: /Users/poonszesen/kg-v3-valuegap/ops/rebuild-2026-09-29/run-statements/value-gap-diagnostic.md on branch kg/rebuild-value-gap. Commit history:
  - 8bfd1c4: statement committed first, before any pod execution. It holds the question, H1/H2, numeric pre-declared predictions 0–5, inputs, stages, the first-unexpected-failure stopping rule, the 45-min aggregate limit with signal-handled process-group cleanup, the idle rule and the artifacts.
  - e97c7a7: scripts, the local cleanup test (ALL PASS) and the dry run.
  - 96ae784: Amendment 1, fp32 compiled stages run with coalesce_tiling_analysis off and are moved last.
  - c9b6cde: Amendment 2, fp32 stages run at 256 rows only.
  - 6afead3: Amendment 3, rerun of the compiled Isaiah stages after an instrumentation defect.
  - bd31cb6: post-run addendum.
  Each amendment was committed before its relaunch.
  The GPU-checks driver's fix had landed only partly: Codex r2 finding 3 is a spawn/registration race. The new driver closes it structurally: only worker threads spawn, and spawning plus registration share a non-reentrant lock with the signal handler.

Results:
- Mechanism: every compiled stage logged exactly one Dynamo recompile, 'GLOBAL_STATE changed: grad_mode', so no-grad sampling and grad-enabled replay run different compiled trunk graphs. Within one grad mode everything was bit-identical: evaluate_actions under no_grad equals S, compute_value under no_grad equals S, repeats are identical, and the trunk input is identical.
- P0 (reproduction, Kaggriculture BF16 compiled): value gap max 0.0195 at both 256 and 1,024 rows with ATEN-only GEMMs, 0.0156 and 0.0233 with default backends, mean 0.0041-0.0044. Log-prob gap max 3.6e-4. Reproduced.
- P1 (eager BF16, B): value gap 0 at both row counts, and hidden states bit-identical. Supports H1.
- P2 (fp32, C, 256 rows): compiled 8.3e-7 (ATEN) and 1.2e-6 (default), eager 0. Supports H1.
- P3 (actor .out gain 1.0, D): log-prob gap rose from 3.6e-4 to 0.032-0.037, a factor of 86-101x, about the 100x gain ratio, under both backends. Eager stayed at 0. Per-row joint log-ratio max was 0.16-0.27 nats. Supports H1.
- P4 (hidden states, E): at the critic tokens, grad-vs-no-grad mean |d| was 0.0048-0.0052. The no-grad path's error vs an fp32 reference was 0.0054 and the grad path's 0.0060. The critic-token relative error equals the all-token, plan and actor-token values (0.0061). The same head applied to each path's hidden states reproduces the full gap. Against the fp32-reference value, the no-grad and grad values each err by 0.013-0.015 max, so neither path is the wrong one. BF16 head rounding alone contributes 0.006-0.009. Supports H1 at both backends and both row counts.
- P5 (Isaiah stateless_transformer_6m, real Orbit Wars states from his Rust VectorizedEnv, compiled as his trainer does; Amendment 3 rerun): head-swap and pair value gap 0.0145 / 0.0191 (default) and 0.0117 / 0.0161 (ATEN) at 256 / 1,024 rows, which is 0.60-0.98x Kaggriculture's 0.0195. His eager gap is 0. Head gains from source are actor 0.01 and critic 1.0 (stateless_transformer_v1.py:125-126, Isaiah commit 99d995d1). Supports H1.
- Unplanned: Isaiah's compiled log-prob gap is much larger than Kaggriculture's, even though both actor .out gains are 0.01. His per-entity gap is 0.044-0.057 nats and his per-player joint log-ratio max is 0.053-0.075 (means -9e-4 to +1e-4), against 3.6e-4 for Kaggriculture. So the 'actor heads suppress the noise ~100x' part of H1 holds for Kaggriculture's grammar heads but not for Isaiah's discrete_targets actor. The cause was not investigated.
- Evidence is in /Users/poonszesen/kg-v3-valuegap/ops/rebuild-2026-09-29/value-gap-2026-09-29/ (README.md, summary.json, summarize.py, pod/, pod/attempt1, pod/attempt2, pod/frerun, pre-launch/, post-run/, large_files.sha256, MANIFEST.sha256). The results.md section is 'Value gap diagnostic'.

Attribution (as reported): H1 is supported and H2 is not.
- Cause of the gap. Dynamo guards on grad mode, so no-grad sampling and grad-enabled replay run different compiled trunk graphs. Those graphs round differently in BF16. The gap vanishes in eager BF16 (bit-identical); vanishes in fp32, compiled or eager (at most 1.2e-6); is the same size under ATEN-only and default GEMM backends.
- Path to the value. The trunk-level difference is BF16-scale and uniform across token groups. The gain-1.0 critic head carries it into the value: the head swap reproduces all of it.
- Actor side. Raising the actor head gain to 1.0 grows Kaggriculture's log-prob gap by the gain ratio (86–101x).
- Isaiah control. Isaiah's model on real Orbit Wars states shows the same value gap, 0.6–1.0x of Kaggriculture's.
- No Kaggriculture-specific cause found. The critic tokens are n[summary truncated in the orchestrator's relay; read the full attribution in the run statement addendum and results.md].

Check:
1. The statement was committed before launch with pre-declared thresholds (verify via git log/show order and content of 8bfd1c4 vs. pod timestamps); each amendment was committed before its relaunch.
2. The control really runs Isaiah's model on real Orbit observations, compiled as his trainer does (inspect the scripts and pod logs).
3. The eager/fp32/gain-swap/hidden-state comparisons use identical weights and inputs.
4. The attribution follows from the pre-declared criteria and does not overclaim (including the unplanned Isaiah log-prob finding and the partial H1 claim about actor suppression).
5. Limits are stated; the reported numbers match summary.json and the raw pod outputs; manifests/checksums are consistent.

Do not modify any file. End with a line `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS`, or `VERDICT: REJECT`, preceded by numbered findings with file:line references.
