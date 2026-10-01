# 90M Kaggle submission receipt (ref 56716929)

Owner (verbatim, 2026-10-01): "when 90m finished anchors, submit it to kaggle to spend 1 slot."

- Submitted 2026-09-30 19:26:12 UTC with
  `kaggle competitions submit -c kaggriculture -f artifacts/anchor-eval/90M-ft-on-6b196ef33b3d/submission.tar.gz -m "v3 PPO 90M (...)"`.
  Kaggle reported "2 submissions remaining today"; status right after: PENDING.
- Archive sha256 `5e460d20ddfc0514e24c072cdb6d1846c2f7eda46a1bdfaef518d9a7482d44d2` (packager verify ok, no problems).
- Checkpoint `checkpoint_00_090_086_144.pt`, sha256 `6b196ef33b3db9dab74220421f147d4bb9cdb49562115e3e7ac822947a11d19a`,
  from the 8xH200 run `earn720-r30e01w30-8xh200-from-60M-20261001` (W&B xuft2e2i); not promoted by the trainer
  (0.60 vs 80M over 20 games), promoted manually by the owner.
- Package: built by `cpu-pod/eval_ckpt.sh` via `scripts/package_checkpoint.sh --final-turn-liquidation` at this branch
  (`kg/package-60m`, cfc45f0e): rule 1 baked on, rule 2 off, clean Kaggle-feature native module (no fixed opponents).
  The packager's 720-turn full-episode Kaggle-image game was not run for this build.
- Evidence before submitting: the same archive played the 48-game fixed-shop anchor panel on the Linux CPU pod
  (py3.11, torch 2.6.0 CPU): 48/48 qualified, W-L 38-10, mean margin +1,943; vs 80M paired +2,494 ± 761 over 8 seeds,
  8/8 seeds better, 8 losses to wins, 0 wins to losses. Receipts:
  `kg-v3-int/ops/earn-money-2026-09-30/anchor-games/games-fixedshop-linux/90M-ft-on/`.
- Limits: selection panel (3 anchors, 8 seeds), not held-out; ladder rating is the real test.
