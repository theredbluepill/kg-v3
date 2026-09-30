# 170M Kaggle submission receipt (ref 56720629)

Owner (verbatim, 2026-10-01): "submit 170m to kaggle after anchor panel".

- Submitted right after the anchor panel finished (panel done 22:47:51Z), by
  `kaggle competitions submit -c kaggriculture -f artifacts/anchor-eval/170M-ft-on-693e30e7f79f/submission.tar.gz`.
  Kaggle reported "1 submissions remaining today".
- Archive sha256 `428a63d706cf528e2c0cad17587452e07c853a66971eb14a1ad219ca770ce3c3`; rule 1 baked on, rule 2 off, clean native module.
- Checkpoint `checkpoint_00_170_034_944.pt` from run `earn720-r30e01w30-8xh200-from-130M-sps-20261001` (W&B r4zqqs49),
  auto-promoted in self-play (0.80 vs 140M, mean margin +3.3k).
- Anchor panel with this exact archive (Linux CPU pod, 8 seeds x 2 seats x 3 anchors): 48/48 qualified, W-L 48-0, mean margin +6,904
  (smaller_market_shock +6,567 16/16, cha22 +6,452 16/16, v56 +7,694 16/16).
- Gate used before submitting: 48/48 games qualified and the package present.
- Limits: selection panel (not held-out); the packager's 720-turn Kaggle-image game was not run for this build (48 full games in the same package were).
