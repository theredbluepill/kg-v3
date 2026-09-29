# Documentation merge audit

HEAD: `49a48350cdd72d6b546320441d677fbcbae2d893`; integration parent: `666deec`; adapter parent: `8699ca9`; common ancestor: `e197528`. Read-only audit; only this receipt directory was written. No tests, training or GPU work run by this agent.

## Verified resolutions

- `cookbook/log.md`: 131 dated headings at HEAD, 120 in integration and 109 in adapter. Every parent heading and its body survives unchanged (comparison strips only blank separators around each body), no duplicate headings, dates nonincreasing. The new merge entry precedes the integration-exclusive entries, then adapter-exclusive entries, then the common history. Same-date entries have no finer timestamp in the document; the stated integration-first policy is honored.
- `cookbook/references/index.md`: 21 distinct reference targets at HEAD, retaining every target from either parent (21 integration, 18 adapter). The integration's SPS-ceiling, flash-attn, teacher and observation entries are byte-identical, as are the adapter's native-semantics and reward-reuse entries. The configs entry retains all three ranked shapes and the corrected Task 3.1 stop reason. Current/native/reward descriptions agree with their notes. One inherited teacher-status defect remains below.
- Configs Reference: its 30 sources are exactly the union of the integration's 29 and adapter's 26; no omission or extra. The 8-rank description/table, timed-GPU-chunk caveat, Task 3.1 stop reason and evaluation-factory wording survive. Two retained implementation claims remain stale below.
- `docs/model-architecture.md`: the integration's teacher API/targets detail and 8-rank one-trunk/one-head workload survive, as do native table loading and the adapter's explicit Task 3.1 startup stop. Diff against integration changes only the native tables paragraph and startup stop sentence; diff against adapter adds the integration teacher detail and 8-rank sentence.
- `docs/rl-api-specs.md`: the integration's Teacher targets bullet survives immediately before the adapter Environment bullet. Everything from the complete `Python adapter, factory and reward configuration (Task 1.5 Stage 2)` section through EOF is byte-identical to adapter. A minor inherited future-tense sentence is noted below.
- `README.md`: integration's 8-rank shapes and teacher protocol/cache details survive, together with adapter's corrected Task 3.1 stop and seeded native evaluation factory.
- `docs/rules-parity-coverage.md`: integration's Phase 4/8-rank counts and observation-custody strengthening survive; adapter's Task 1.4 lifecycle coverage survives. The two stale table-binding statements are repaired at lines 478–480 and 654–655. Lines 466–474 record engine 69, root 274/5 ignored, Python 2,319/10 skipped with the new receipt path. This agent checked that the count prose matches the claimed reconciliation; root agent independently runs the suites.
- No conflict markers in tracked Markdown under README/docs/cookbook. No behavioral test references the prose 8-rank model-doc sentence or the text of `NEEDS_RUN_PPO_GAME_SEAM`; no documentation/reason-only mutation was claimed. The existing tests exercise shapes and startup behavior; root agent owns the configuration mutation.

Mechanical details: `docs-mechanical.json`, `docs-claims.json`, `docs-stale-origins.json`.

## Findings

### P3 — Reconcile surviving native-environment blockers

Primary location: `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:160` (also lines 134, 145; its description at line 4 and index line 7 still call the startup guard a “no-env stop”). This current Limits section continues to make Task 1.4 native env an open dependency, and line 134 claims the skip reason still names that dependency. HEAD actually changes `NEEDS_RUN_PPO_GAME_SEAM` at `tests/kaggriculture/test_teacher.py:874` to say the native env exists and only Task 3.1 rollout storage/action mapping remain. The teacher Reference is inherited verbatim from integration; it was absent from adapter, so this is unreconciled status after merging Task 1.4/1.5, not lost code.

Related same-cause locations: `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md:61` still lists native env as open and names the former startup guard (identical in both parents, already stale in adapter); config header comments at `configs/kaggriculture.yaml:7`, `configs/kaggriculture_2rank.yaml:12`, `configs/kaggriculture_4rank.yaml:12`, `configs/kaggriculture_8rank.yaml:13` still say launch awaits Task 1.4 binding (first three inherited from both parents, 8-rank from integration). `docs/rl-api-specs.md:1262` says Task 1.5 adapter “will own” the buffers, inherited from adapter, while its new current-status section correctly says it is implemented.

Fix: refresh current blocker/skip descriptions to the Task 3.1 rollout/observation/action mapping work and native env already landed; update the teacher description/index together. Preserve dated prior test outcomes and probes as history. In the native lifecycle section, change the adapter buffer description to present tense without claiming the pending pod DMA test passed.

### P3 — Configs Reference still describes the superseded reward schema/startup caller

Location: `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:28` (and line 30). Line 28 says `config.py` defines the reward class and that it retains defaults including ineffective cap 0.1, then describes only the old cap/raw-positive-weight admission. Actual class is `python/owl/kaggriculture/rewards.py:23`, its six fields at lines 30–35 are all required, and its validator requires positive enabled caps and a positive binary64 death-weight product. This is exactly why the merge must explicitly add the 8-rank ineffective cap. `docs/rl-api-specs.md:912` and the reward Reference already describe the new contract correctly. The line is identical in both parents and was already stale in adapter.

Line 30's merged evaluation wording is correct, but its startup claim still says `require_orbit_env` raises immediately after headroom. In `scripts/run_ppo.py:182–188`, the new `isinstance(KaggricultureEnvConfig)` guard raises first and `require_orbit_env` is reached only for Orbit. Both parent versions of the sentence already named the old caller, and the merge retained that clause while revising the evaluation clause.

Fix: revise the current schema description to the required definition in `rewards.py` and exact admission semantics (or link the reward Reference), distinguishing explicit shipped YAML values from defaults; name the explicit Kaggriculture startup guard. Keep historical Task 3.4 checks as history.

No P1/P2 documentation or semantic-loss finding. Documentation-only recommendation: APPROVE WITH EDITS.
