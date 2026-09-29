Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: Task 4.4 teacher configs onto the integration (`kg/merge-4-4-c`)

VERDICT: APPROVE

## What I checked

- **Scope:** the merge `c34845a` on staging branch `kg/merge-4-4-c` (worktree `/Users/poonszesen/kg-v3-m-4-4`). First parent: integration `kg/isaiah-gap-closure` at BASE `994818b`, which is the 7.1 landing. Second parent: `kg/rebuild-4-4` `4a662ad`, which Codex `verify-4.4-r1` approved with edits; its three P3 edits are applied. The merge base is `faed717`. The landing-record commit follows it on the same branch.
- **Earlier attempt:** the staging branch `kg/merge-teacher-configs-4-4` (`0a9e04a`/`c23c1a2`) has the same two parents. It never landed because Codex hit its usage limit during its verify-merge. I restaged from the current integration tip. For the four prose conflicts I resolved by hand (README, model-architecture, configs Reference, teacher Reference), I then diffed my resolutions against `0a9e04a`. They matched except for the order of the source list and one sentence join in the teacher Reference description. I took `0a9e04a`'s resolutions for the References index, the cookbook log and `phase-status.md`, and rewrote the staging names, merge SHA and reviewer for this landing.

## Nothing lost from either parent

- **Test names.** I extracted every Python `def test_*` from files whose path contains "test", and every Rust `#[test] fn`, at `994818b`, `4a662ad` and the merge. Counts were 1,649 at the integration parent, 1,463 at the 4.4 parent and 1,663 at the merge.
  - Lost from the integration parent: none.
  - Lost from the 4.4 parent: two, `test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` and `test_grammar_bridge_is_retired_to_root_integration`. Both exist at the merge base `faed717`. The integration side removed them on purpose in Task 1.5 stage 2 (`d0d65f7`) and Task 1.4 (`8d98ea8`); these are the "two BASE tests deliberately replaced" that verify-merge-env-adapter r1 already accepted. `4a662ad` never edits either one, so dropping them is the correct three-way result.
  - New against the integration parent: exactly the 14 tests that 4.4 adds (4 in `test_configs.py`, 2 in `test_logging.py`, 8 in `test_run_ppo.py`).
- **Code delta.** In `python/`, `scripts/`, `tests/` and `configs/`, the `+/-` lines of `git diff 994818b c34845a` match those of `git diff faed717 4a662ad` line for line. The auto-merge applied 4.4's code exactly and changed nothing from the integration.
- **Engine.** `engine_rs/` is unchanged against BASE. Rerunning `ops/rebuild-2026-09-29/7.1/update_trim_manifest.py` left `TRIM_MANIFEST.json` byte-identical, and the trim check in `rs-format` passed.

## Resolutions

- `README.md` and `docs/model-architecture.md` keep the integration's current stop reason ("Task 3.1 rollout storage and action mapping are not implemented"; the native env exists since Tasks 1.4/1.5) and add 4.4's teacher cache bytes: 1,674,575,872 B at 2 ranks, 837,287,936 B at 4 and 418,643,968 B at 8. Both match the merged `run_ppo.py`. Its startup order is `_require_kaggriculture_teacher_source`, then the runtime-GPU adaptation, `_check_model_workload`, `_check_compile_stack`, and the explicit `KaggricultureEnvConfig` `RuntimeError` that names Task 3.1, all before the run directory exists.
- **Configs Reference:** 4.4's description, with its stale "no Kaggriculture env is wired yet" replaced by the integration's Task 3.1 wording. It has 37 sources, the union of both sides: 4.4's 35 plus `python/owl/kaggriculture/rewards.py` and `ops/rebuild-2026-09-29/briefs/1.5.md`, with the reference-branch sources kept last. Every `repository:` source exists in the merged tree. The body keeps the integration's reward/`rewards.py` and guard-order bullets and 4.4's "Teacher settings (rebuild Task 4.4)" section.
- **Teacher Reference:** the description keeps the integration's Task 3.1 stop wording and 4.4's "Task 4.4 has since pinned…" sentence. The body keeps the integration's T19b bullet and now says that 4.4 is merged (`4a662ad`, through `kg/merge-4-4-c`).
- **References index:** the integration's evaluation line (native adapter since Task 1.5) and configs stop reason, plus 4.4's teacher clause, W&B Reference line and teacher-line pointer. Every wikilink in the changed cookbook files resolves.
- **Cookbook log:** newest first, with no duplicate headings (144 headings, all unique). The new landing entry is at the top, the integration's 7.1 and 1.4/1.5 entries follow, and 4.4's own branch entry comes after them.
- **`plan.md` / `phase-status.md`:** 4.4 is marked merged, and the Phase 4 box stays unticked until T18/T19b run, which is consistent with the teacher Reference. Phase 4's summary reads 4/4 with the phase still open. The 4.4 row names this report and says the reviewer is a Claude substitute, not Codex. The "Merged" definition now covers the 4.4 row.
- **The skipped T19b launch/resume tests** in `tests/kaggriculture/test_teacher.py` stay consistent with 4.4's new teacher-source rule. The fresh-launch test passes `--load-model-weights`, and the resume test is exempt. So when Task 3.1 unskips them, they will not trip the new check.

## Owner rules

- `scripts/run_ppo.py` remains the only trainer, and no v2 model code enters.
- W&B project selection keys on the game (Kaggriculture goes to `kg-v3`), never on opponent identity. `--wandb-mode offline` is visible in the run. This follows the rule that telemetry outages stay visible.
- No policy state, history tokens or opponent-identity inputs are added.
- The teacher settings match `scaling_6m`, consistent with the rule to align the recipe to Isaiah.

## Checks run on the merged tree

- `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` exited 0 (`ops/rebuild-2026-09-29/merge-4-4-c/prepare.log`):
  - root Rust: 274 passed, 5 ignored; the other Rust crates all passed
  - Python: 2,422 passed, 21 skipped
  - docs-fresh: "No doc updates required", without `DOCS_CURRENT`
- The first run (`prepare-no-fixtures.log`) failed seven Orbit parity tests. The cause was environmental: the ignored generated fixtures (`tests/fixtures/generation/reference_generation.json`, `tests/fixtures/orbit_wars_replays/*`) are absent from a fresh worktree. I copied them from `/Users/poonszesen/kg-v3-int`, as earlier landings did. They are not tracked.
- The cookbook lint hook passed on the three changed References.

## Non-blocking notes

- **Stale wording (P3, not caused by this merge):** the configs Reference's Tests bullet (line 31) still says the ranked configs "stop with the explicit not-wired error". That phrase comes from the integration side; the error now names Task 3.1. I left it unchanged because the merge did not cause it.
- **Evidence left local:** the partial `verify-merge-teacher-configs-4-4-r1` prompt and transcript from the abandoned staging stay local and uncommitted.
- **Limits:** CPU only, on a Mac. There is no live W&B call and no GPU or teacher-memory measurement; those wait for Task 6.1.
