1. **BLOCKING — pairing overclaim.** [receipts.md:33–34](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md:33) says labels are unaffected, but [pairing.json](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json:123) records state comparisons, not label validation. Replace lines 33–34 with:

   ```text
   5743/5752 transitions match (0.9984) over 8 episodes, kaggle_environments
   1.32.7. The nine mismatches name only private0/private1 at day-end turns.
   This is a pairing diagnostic; it does not establish parity or prove label correctness.
   ```

2. **BLOCKING — stale W&B receipt.** Replace [train/README.md:23–24](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/README.md:23) with:

   ```markdown
   - W&B: launched with `--wandb-mode offline` (no key on the pod), then synced
     to [spoon/kg-v3/kvl4rfda](https://wandb.ai/spoon/kg-v3/runs/kvl4rfda).
     The pod's `wandb/offline-run-20260929_142222-kvl4rfda/run-kvl4rfda.wandb.synced`
     marker has mtime `2026-09-29T15:01:13Z`; the orchestrator reports final sync
     at `2026-09-29T15:01:14Z`.
   ```

3. **BLOCKING — stale W&B cookbook claim.** Replace [cookbook note:30](/Users/poonszesen/kg-v3-bcnow/cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:30) with:

   ```markdown
   - W&B telemetry was collected offline, then synced at `2026-09-29T15:01:14Z` to [spoon/kg-v3/kvl4rfda](https://wandb.ai/spoon/kg-v3/runs/kvl4rfda), supported by the pod sync marker and orchestrator report.
   ```

Check 1 PASS — `git log --format=%ci` gives 14:18:50Z, 33 seconds before [launch.start:1](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/launch.start:1); [attempts:1](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/bc_attempts.jsonl:1) records `f0b7a3877eb16657f973aaf55c70779b040e5b46` and 14:22:16Z, matching [result:18](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/bc_result.json:18) and preceding the [log’s 14:22:22 W&B directory](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/train.log:3).

Check 2 PASS — All reviewed denominators agree: 523+319=842; 523+3356+319=4198; 507+16=523; 182273+5748=188021; the 297418-row budget yields stride 11 for all listed episodes and 2 for kept episodes ([receipts:38–77](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md:38)).

Check 4 PASS — Every README table value matches history rounding; best NLL is 0.4801823178678232 at 3200, then ten scheduled evaluations give 3200+10×200=5200 and `no_held_out_improvement` ([history:17](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/bc_history.jsonl:17), [config:38](/Users/poonszesen/kg-v3-bcnow/configs/bc/kaggriculture_1gpu_eager.yaml:38), [selection code:438](/Users/poonszesen/kg-v3-bcnow/python/owl/train/bc.py:438)).

Check 5 PASS — Checkpoint SHA-256 agrees across all five requested locations; both pod hashes match the supplied facts; `shasum -a 256` matches all nine local files listed in [SHA256SUMS:1–11](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/SHA256SUMS:1), whose bytes also match HEAD.

Check 6 PASS — Eager mode and driver mismatch were declared [before launch:25](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/run-statements/bc-a100.md:25); [receipts:74–75](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/README.md:74) explicitly exclude a throughput-ceiling claim; correctness does not depend on compilation.

Check 8 PASS — [Note frontmatter:1–8](/Users/poonszesen/kg-v3-bcnow/cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:1) is valid, all eight repository sources exist, and [scope:28](/Users/poonszesen/kg-v3-bcnow/cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:28) explicitly excludes playing-strength, live-legality and PPO-benefit qualification.

Check 9 PASS — Mixed-winner shards are explicitly [superseded and untrained:22–26](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md:22); the [training command:5](/Users/poonszesen/kg-v3-bcnow/ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/run-bc-a100.sh:5) uses `shards-top1`.

VERDICT: APPROVE WITH EDITS