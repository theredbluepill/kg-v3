Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: BC W&B wiring onto the integration (`kg/merge-bc-wandb-c`)

**Merge.** Staging branch `kg/merge-bc-wandb-c` (worktree `/Users/poonszesen/kg-v3-m-bc-wandb`) was cut at the integration tip BASE `ff9ebcb` (the bank-metrics landing). `a3abf56` is a `--no-ff` merge of `kg/rebuild-bc-wandb` `dafa02a` (`619bf50` implementation, `dafa02a` review fix and records). The branch was cut from the same `ff9ebcb`, so the merge had no conflicts. Its tree equals `dafa02a`'s tree (`git rev-parse` of both trees match), so nothing needed a semantic resolution and no side's content was dropped. This pass is by the landing agent, which also implemented the branch. It is independent of Codex but not of the implementation; the branch's own separate review is `claude-verify-bc-wandb.md` (APPROVE WITH EDITS, P3-1 applied).

## Seams checked

- **`run_ppo` and the moved outage helper.** `run_ppo._run_training_session` now calls `owl.train.logging.announce_recorded_outage` in place of its two inline prints. The text is byte-identical, and the unused `sys` import is gone. `test_run_ppo`'s outage-receipt and offline-visibility tests pass on the merge.
- **`config_sha256`.** It now delegates to `json_sha256` with the same canonical JSON (`sort_keys`, `(",", ":")` separators), so existing receipt hashes are unchanged.
- **Other `owl.train.logging` users.** `run_ppo` and `train_bc` are the only launchers on the integration. `grep -rn` for `create_bc_logger`, `BCWandbLogger` and `BC_WANDB_PROJECT` over `python/ scripts/ tests/` finds nothing.
- **Bank telemetry (the previous landing).** Its keys ride `MetricLogger.log` in `run_ppo` and are untouched by this merge.

## Checks on the merge

- `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` on `a3abf56` exited 0 (`ops/rebuild-2026-09-29/merge-bc-wandb-c/prepare.log`). It ran format, lint, docs-lint and mypy. Rust passed 274 with 5 ignored, plus 41, 9, 22, 12, 5 and 5 in the other crates and doc targets. Python passed 2,699 with 18 skipped. docs-fresh reported "No doc updates required".
- Three merge-seam mutations ran in a scratch worktree detached at `a3abf56` (`ops/rebuild-2026-09-29/merge-bc-wandb-c/mutations.log`), and each file was restored afterwards:
  - MM1: `run_ppo` drops the recorded-outage announcement. Killed by `test_run_training_session_opens_an_offline_wandb_run_visibly`.
  - MM2: the announcement also prints for online telemetry. Killed by `test_run_training_session_records_the_attempt_and_its_telemetry_mode[wandb-online-…]`.
  - MM3: the BC settings hash ignores the PPO config. Killed by `test_bc_config_sha256_hashes_the_ppo_config_content_not_its_path`.
- The branch review's 15 mutations (14 killed, 1 equivalent) and its real wandb 0.26.1 offline CPU run apply unchanged, because the tree is identical.

## Findings

No P1, P2 or P3 finding on the merge. The branch review's recorded P3-2 to P3-4 stand as limits: a crash between the two receipt appends fails loudly, an unsynced offline attempt fails late on an online restart, and runs are named `bc-bc-`. No live online W&B call, GPU, multi-rank or real-shard BC run was made.

VERDICT: APPROVE
