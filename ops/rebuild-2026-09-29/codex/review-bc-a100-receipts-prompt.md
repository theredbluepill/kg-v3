You are Codex, reviewing Claude's receipts for the A100 behavior-cloning (BC) run. READ-ONLY: do not modify files. The owner wants BC kept lean: report BLOCKING findings only (a false, missing or misleading claim in the receipts or cookbook note that a later reader would act on). Do not report style, wording preferences or non-blocking suggestions.

Worktree: branch `kg/rebuild-bc-now` at `218a05b`. Files under review:
- `ops/rebuild-2026-09-29/run-statements/bc-a100.md` (run statement, committed in `f0b7a38`)
- `ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md`, `manifest-summary.json`, `pairing.json`, `prepare-top1.log`, `run-top1.sh` (data preparation, committed in `933d661`)
- `ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/` (training: `README.md`, `bc_history.jsonl`, `bc_result.json`, `checkpoint_bc_best.json`, `bc_attempts.jsonl`, `train.log`, `SHA256SUMS`, `launch.start`, `gpu-samples.csv`, configs, `run-bc-a100.sh`; committed in `218a05b`)
- `cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md`, its entry in `cookbook/references/index.md`, and the matching top entry of `cookbook/log.md`
- Supporting code if needed: `scripts/train_bc.py`, `scripts/kaggriculture_prepare_bc.py`, `configs/bc/kaggriculture_1gpu_eager.yaml`, `configs/kaggriculture_1gpu_eager.yaml`.

Facts Claude checked on the pod (you cannot reach the pod; take these as given):
- `sha256sum` on the pod now: `checkpoint_bc_best.pt` = `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`, `bc_state.pt` = `7644ded6a1544981495fc27390a61026ff4112cd10a1b33c3962d188ac80529f`.
- W&B: the run was launched with `--wandb-mode offline`, but it has since been synced. The pod run dir holds `wandb/offline-run-20260929_142222-kvl4rfda/run-kvl4rfda.wandb.synced` (mtime 2026-09-29 15:01:13Z); the orchestrator reports the final sync at 15:01:14Z to `https://wandb.ai/spoon/kg-v3/runs/kvl4rfda`. The receipts and note still say "not synced"; that is a known error Claude will correct. Say exactly which lines must change and give the replacement text.

Check, citing files and lines:
1. The run statement precedes the launch: compare the commit time of `f0b7a38` (`git log --format=%ci`) with `train/launch.start` and the start times in `train.log` / `bc_result.json`, and confirm `bc_attempts.jsonl` records source `f0b7a38...`.
2. Data denominators agree across `receipts.md`, `manifest-summary.json`, `prepare-top1.log`, the run statement, the train README and the note: 842 team episodes seen, 523 kept (team won), 3356 team absent, 319 team lost (4198 listed; 523 + 3356 + 319 = 4198); 507 train / 16 validation episodes; 182,273 + 5,748 = 188,021 rows; turn stride 2 with its stated reason (the `--resident-budget-gib` formula counts all listed episodes; applied to the 523 kept episodes it gives stride 2). Check the arithmetic.
3. The pairing diagnostic is reported as 5743/5752 and presented as a diagnostic, not as parity proof or label correctness proof beyond what `pairing.json` supports.
4. The NLL curve in the train README matches `bc_history.jsonl`; best step 3200 at 0.480 (0.48018); the L9 stop at step 5200 with `stop_reason` `no_held_out_improvement`; the stated patience (10 evaluations x 200 steps) is consistent with 3200 + 2000 = 5200 and with `train_bc.py` / the BC config.
5. Checkpoint custody: SHA-256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51` agrees across `SHA256SUMS`, `bc_result.json`, `checkpoint_bc_best.json`, the README and the note; the local compact-file hashes in `SHA256SUMS` match the committed files (run `shasum -a 256` on them).
6. The eager-mode deviation (driver 580.159.03 outside the probed compile stack) is declared in the run statement before launch, and the receipts limit throughput claims accordingly (no ceiling claim, no correctness claim that depends on compile).
7. W&B: every "offline only" / "not synced" claim in the receipts, README and note (and the log/index entry if present). List each line to correct.
8. The cookbook note claims selection only, not playing strength, legality in play or PPO warm-start benefit; its frontmatter (`type`, `title`, `description`, `tags` with first tag `kaggriculture-v3`, `status`, `generated`, `sources` as `repository:` paths that exist) is valid.
9. The superseded mixed-winner shards (`/workspace/kg-v3-bc-2026-09-29/shards/`) are marked as not trained on, and nothing states or implies the run used them.

FINAL REPORT: numbered BLOCKING findings only, each with file, line and the exact replacement text. If an item passes, say so in one line. End the report with exactly one line of the form `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS` or `VERDICT: REVISE`.
