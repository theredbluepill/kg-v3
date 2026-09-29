You are Codex, the INDEPENDENT VERIFIER of rebuild Task 4.4 (Kaggriculture teacher configs) on branch kg/rebuild-4-4 in this worktree (/Users/poonszesen/kg-v3-t44). Leave no tracked modification: git status must be clean at the end (write scratch files only under /tmp or /private/tmp).

Scope: verify `git diff faed717...HEAD` (three-dot; faed717 is the integration tip kg/isaiah-gap-closure). Read:
- ops/rebuild-2026-09-29/plan.md (Phase 4, "4.4 configs"; Phase 6.1 runs "with the teacher on") and ops/rebuild-2026-09-29/briefs/4-teacher.md sections "4. Configs (4.4)" and "Task 4.4".
- cookbook/decisions/recipe-choices-align-to-isaiah-without-owner-escalation.md (owner rule: recipe choices resolve toward Isaiah's scaling_6m), cookbook/decisions/the-policy-is-stateless-and-observation-only.md, cookbook/decisions/evaluation-preserves-generality-and-evidence.md (W&B under v3 identifiers).
- Isaiah's configs: configs/scaling_6m.yaml, scaling_6m_halfbatch.yaml, scaling_3m.yaml, scaling_1p5m.yaml, winner_ce_6m_4x5090.yaml.
- Changed code: configs/kaggriculture*.yaml, python/owl/model/kaggriculture_workload.py, python/owl/train/logging.py, scripts/run_ppo.py, tests/kaggriculture/test_configs.py, tests/scripts/test_run_ppo.py, tests/owl/train/test_logging.py, README.md, docs/model-architecture.md, and the cookbook changes (configs and teacher References, new cookbook/references/ppo-runs-publish-kaggriculture-telemetry-to-the-v3-wandb-project.md, references/index.md, log.md) plus ops/rebuild-2026-09-29/teacher-configs-4.4/.

The task asked for: (1) teacher config blocks in the 2/4/8-rank configs following Isaiah's scaling_6m teacher settings (coefficient, cache, refresh cadence); (2) extending the startup workload assertion or its test to cover the teacher chunk rows; (3) a config test that fails on a wrong value; (4) the teacher checkpoint path as an explicit required launch-time input with a clear error, not an invented path. The task text also said "teacher_spm divided per rank like the other per-rank quantities"; the implementation deliberately keeps teacher_segments_per_minibatch 128 undivided, citing Isaiah's per-rank configs and the reviewed brief §4. Judge independently whether that resolution is correct under the owner's align-to-Isaiah rule and whether the chunk size affects the targets' math. Separately, the owner asked to wire all v3 experiments to W&B; the branch sends Kaggriculture PPO runs to project kg-v3 and adds --wandb-mode offline. Check it is correct and scoped (the BC trainer lives on another branch and must not be touched).

Check especially:
- Every numeric claim (cache bytes 1,674,575,872 / 837,287,936 / 418,643,968; chunk rows; refresh threshold 0.7 and checkpoint_freq) against code.
- That _require_kaggriculture_teacher_source runs before any run dir/env/model, does not change Orbit launches or resumes, and that the error names the remedies; whether its placement or scope conflicts with Isaiah's last-best semantics in a way that matters.
- The W&B logger: project selection, mode forwarding, --wandb-mode validation, the offline notice, and whether any claim about wandb offline-resume behavior matches the installed wandb SDK source (.venv/lib/python3.12/site-packages/wandb/sdk/wandb_init.py).
- The stateless/observation-only policy and no opponent identity are untouched.
- Cookbook accuracy: every claim in the edited notes is supported by the diff or the cited receipts; no overclaiming (no live W&B call was made; no GPU).
- No getattr/setattr in first-class code paths, no shims, fail-fast errors.

Non-vacuity: for EACH new guard, apply at least one mutation on a scratch copy (or in place, then restore byte-for-byte and verify with sha256 / git diff), run the relevant tests, and report which tests failed. New guards: the teacher-field pin test; the cache-bytes reporting (cached_teacher_rows); the cache-smaller-than-chunk rejection; the teacher-source launch check; the teacher_init existence check; the W&B project selection; the W&B mode forwarding; the --wandb-mode/--log-mode debug rejection; the offline notice.

Run and report counts:
- OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q
- OMP_NUM_THREADS=2 uv run mypy python/owl scripts
- OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare
No training, no GPU, CPU only, keep each check under ~2 minutes and <1 GB where possible.

Report format: a findings list with severity (P1 blocking correctness / P2 must fix / P3 edit), file:line and the concrete fix; the check counts; the mutation table; a confirmation that git status is clean. End the report with exactly one final line: VERDICT: APPROVE, VERDICT: APPROVE WITH EDITS, or VERDICT: REQUEST CHANGES.
