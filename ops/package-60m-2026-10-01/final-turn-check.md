# Rule 1 in the packaged 60M full Kaggle-image episode

Source: `artifacts/60m-r1/validation/replays/replay-seed7-self.json` (gitignored),
from `scripts/package_checkpoint.sh ... --full-episode --final-turn-liquidation`
at commit `de26cd68`. Self-play, seed 7, both seats run the same package.

- Observation step 718 (the last resolved turn), seat 0 and seat 1 identical:
  shed WHEAT 22, CARROT 7, MILK 11, FERTILIZER 4; prices CARROT 44, WHEAT 21,
  FERTILIZER 20, MILK 5; one hand (index 2) carries CARROT 3 off the shed tiles.
- Action on that observation (replay `steps[719]`): market
  `SELL CARROT 7, SELL WHEAT 22, SELL FERTILIZER 4, SELL MILK 11` (one SELL per
  shed product, full shed, highest price first, no BUY/HIRE); no DROP needed.
- Final shed: all products 0. The carried CARROT stays unsold (off the shed
  tiles; the known limit of the rule).
- Earlier turns in this game carry the model's own orders (e.g. step 717's
  `SELL WOOL 14, SELL FERTILIZER 2`), consistent with the rule acting only on
  step 718.

Checked by reading the replay JSON with a short Python snippet; not a paired A/B.
