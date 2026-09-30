# 210M Kaggle submission receipt (ref 56722061)

Owner (verbatim, 2026-10-01 ~23:49Z): "沒事，打包210m, 提交，然後eval." (package 210M, submit, then evaluate).

- Submitted 2026-09-30 23:50:02 UTC with `kaggle competitions submit -c kaggriculture -f artifacts/submit-210m/submission.tar.gz`;
  Kaggle reported "0 submissions remaining today". Final slot of the day.
- Archive sha256 `45efe071e0c92954995440109cc4b0f7d034b746540ed4d63b36616084f9c36a`; packager verify ok; `main.py` has
  `final_turn_liquidation=True` (rule 1 on), rule 2 off; clean Kaggle-feature native module; built by `scripts/package_checkpoint.sh --final-turn-liquidation` at this branch.
- Checkpoint `checkpoint_00_210_009_344.pt` sha256 `790b64d80d60e8e81c0736b3a9ad925a63f072abcced8a2c100075e3677118cd` (Mac copy matches the pod),
  run `earn720-r30e01w30-8xh200-from-130M-sps-20261001` (W&B r4zqqs49). Self-play check vs last_best 180M: 0.65 over 20 games, mean margin +862 (not promoted).
- Submitted before its anchor panel, as the owner asked; the panel is run afterwards on this same archive (results appended below when done).
- Context: 180M 48-0 (+7,321), 200M 48-0 (+7,889) on the same panel.
