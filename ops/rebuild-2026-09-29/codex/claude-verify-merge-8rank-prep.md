Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: 8-rank prep onto the integration (label `8rank-prep`)

The landing Claude agent wrote this pass. It is not independent of the merge it checks.

- **Staging:** worktree `/Users/poonszesen/kg-v3-m-8rank-prep`, branch `kg/merge-8rank-prep-c`.
- **BASE:** the integration tip `kg/isaiah-gap-closure` `7e87f5420b3696141ea41f6453d4fafb2748eea0` (the BC W&B landing).
- **Merged:** `kg/rebuild-8rank-prep` `fee4c33` (`669737a` halve the interval, `ef515a6` recipe-J presets, `2eb0ee4` 8-rank run package, `661bb5e` r1 review fixes, `fee4c33` prepare receipt).
- **Merge commit:** `2907af1` (`--no-ff`).
- **Owner, 2026-09-30, verbatim:** "cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank".

## Checks

1. **Merge shape.** The branch was cut from BASE, so nothing conflicted. `git diff fee4c33 2907af1` is empty: the merged tree equals the branch tree. No semantic resolution was needed.
2. **Interval semantics, against the trainer code.** `scripts/run_ppo.py` `_next_periodic_checkpoint_step` returns `(env_steps // checkpoint_freq + 1) * checkpoint_freq`. The loop re-arms it after each checkpoint and evaluation. The periodic checkpoint, the last-best evaluation and promotion at `LAST_BEST_WIN_RATE_THRESHOLD` (0.7) all share that one threshold. With 16,384 env steps per iteration and 10,000,000 steps, the first interval closes at iteration 611 (10,010,624 steps). Later ones close at 1,221 and then 1,832, so the gaps are 610 or 611 iterations. This matches the Decision, the README, the config comments and `test_checkpoint_interval_is_the_owners_half_of_scaling_6m`. The owner's "about 610 iterations" is correct.
3. **Coverage of the change.**
   - Every Kaggriculture GPU config sets `checkpoint_freq: 10_000_000`: 2-, 4- and 8-rank, `kaggriculture_1gpu_eager.yaml`, and both `*_bc_finetune.yaml`.
   - `configs/kaggriculture.yaml` (CPU test cadence) and the Orbit configs keep their values, as the Decision states.
   - A grep for `20M`, `20_000_000`, `20,000,000` and `1,221` across the configs, `README.md`, `docs/`, `cookbook/`, `plan.md`, `phase-status.md` and the run package finds no stale Kaggriculture cadence claim. The remaining hits are Orbit presets, `README.md:125` (the Orbit `baseline.yaml`), historical scoped statements and the Decision's own quotation of the question.
4. **Recipe J presets.** Each preset equals its ranked config apart from `muon_lr` 0.0002 and `adamw_lr` 1e-5. A whole-`FullConfig` diff test pins this, with the global workload equal to `scaling_6m`. The 8-rank per-step LR transfer is labelled as reasoning, not a measurement. The ablation evidence is labelled pre-landing, not Codex-verified, and inside the LR warm-up.
5. **Prior branch review.** `codex/claude-verify-8rank-prep-r1.md` gave APPROVE WITH EDITS.
   - F1, `--kill=none`: in `qualify.sh`.
   - F2, `checkpoints_final.sha256` beside the append-only watchdog list: in `common.sh` and `pull_from_pod.sh`, exercised by `local-pull-check.sh`.
   - F3, the handoff Reference: the fresh-head wording is scoped to the historical launch at lines 38–50.
   - All three are applied in `661bb5e`, and the P3s are recorded in that commit and the log.
6. **Cookbook contract.**
   - The Decision quotes the owner verbatim in `decider` and the body, with the first tag `kaggriculture-v3` and repository sources.
   - The Decisions index, the References index and prepended `cookbook/log.md` entries accompany the notes.
   - No board note exists (`cookbook/decisions/the-kaggriculture-v3-board.md` absent), so there is no board edit.
7. **`just prepare` on `2907af1`.**
   - Command: `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`. It exits 0.
   - Rust `owl` lib: 274 passed, 5 ignored, plus the other crates. Python: 2,729 passed, 18 skipped. Ruff, mypy, markdown lint and docs freshness also pass.
   - Receipt: `ops/rebuild-2026-09-29/merge-8rank-prep-c/prepare.log`.
   - The first attempt exited 101. Seven Orbit parity tests found no fixtures, because the git-ignored `tests/fixtures/generation/` and `tests/fixtures/orbit_wars_replays/` do not exist in a fresh worktree. Those fixtures were copied from the main checkout (byte-identical by `diff -rq`, SHA-256 in the log header), and the rerun passed. No tracked file changed.
8. **Merge-seam mutations.** Each mutation ran `uv run pytest tests/kaggriculture/test_configs.py -q` and then restored the file. All five were killed, and the tree was restored clean. Receipt: `ops/rebuild-2026-09-29/merge-8rank-prep-c/mutations.log`.
   - MM1: the 8-rank preset at 20M.
   - MM2: the 2-rank base at 20M, with the preset kept at 10M.
   - MM3: the 8-rank preset at full `muon_lr`.
   - MM4: the 2-rank preset's `teacher_kl_coef` drifting from its base.
   - MM5: the 8-rank preset's `n_envs` drifting from its base.
9. **Scope.** This landing is code, config and docs only. Nothing ran on a GPU, and no pod was created, read or touched. `~/kaggriculture-v2` was not touched. Nothing was pushed.

## Findings

None blocking.

- **P3 (process).** A fresh staging worktree lacks the ignored Orbit parity fixtures, so `just prepare` fails there until they are copied in. Earlier landings did the same copy, but did not say so in their prepare headers. This receipt records it.
- **P3 (unverified here).** The orchestrator states J/K are already stopped. I did not check any pod.

## Not checked

- No GPU, `torchrun`, nsys or pod execution. The run package scripts (`setup.sh`, `qualify.sh`, `launch.sh`, `watchdog.py` under a real run) remain drafts that were checked on the Mac only.
- The ablation branch's (`kg/pod-ppo-prelanding`) code equivalence to the integration trainer, as the preset Reference already states.
- The evaluation's wall-time share at the doubled rate is unmeasured at 8 ranks.

VERDICT: APPROVE
