# 80M vs 70M vs 60M on the fixed-shop anchors: rule 1 on (played), rule 1 off (derived); rule 2 not applied

Date 2026-10-01 (runs 2026-09-30 18:31Z-19:15Z). Owner, verbatim: "Let's start a cpu pod as you said and evaluate the pt." / "let anchors speak will be good".
This is the Mac fast path and repeats `../eval-70M/` exactly. Nothing was uploaded, submitted, committed or pushed. No pod was touched, no code default was flipped, and the 60M and 70M game folders were left alone.

## Package `../pkg-80M/` (local Mac, NOT a submission)
- Built like `pkg-70M`: `../pkg_local_mac.py`, the same clean detached worktree (kg/submit-08bc `4c99768a`, rule 1 only, no rule-2 code, `diff_vs_head` empty) and the same macOS arm64 native module. `source_equal`, `native_equal` and `builder_equal` are all true against pkg-70M (`pkg80M_verify.json`, written by `verify_pkg80.py`).
- Checkpoint: `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-80M/checkpoint_00_080_063_744.pt`, sha256 `ded916bd72c3697a9fd0da0a1475f1ca86cf7514fe00e47f80ec0c577a36b152` (rehashed). The config sha256 `f7a1349a…e377c` is the same file as 70M's.
- Slim model-only checkpoint sha256 `8274cc005da081056267a62f9b87b8aba4684fc28ebc6d62a1202bd9b62babcb`. All 210 tensors are torch.equal to the 80M `model` state dict, and all 210 differ from 70M.
- Manifest sha256 `c6fa0f2b3c1c2c0a5fa5a30240308e34b6c20b2fb53ac987c421bcd0d4ec307a`. Compared with `pkg-70M`, only `models/primary/checkpoint.pt` differs.

## Runs
- `run_fixedshop_80M.sh 80M-ft-on 1` wrote `../games-fixedshop/80M-ft-on/` (log `run-80M-ft-on.log`). It ran 48 games: 8 seeds 93001-93008 x 2 seats x {smaller_market_shock, cha22, v56}. Settings were the same as 70M: fixed-shop engine (`kaggriculture.py` `f73d27ce…5004`), nice 10, 5 parallel, 1 torch thread, strict agent, and `eval-70M/run_game_prerule.py` reused unchanged.
  - Interruption: the first invocation stopped after 35/48 receipts. Five v56 games were in flight and were killed without writing receipts. The script skips labels that already have a receipt, so a rerun at 19:08Z played the remaining 13 from scratch. Their earlier partial logs were overwritten.
- Spot check with rule 1 actually OFF: `80M-off-spot` played the same three labels as 70M (smaller_market_shock s93001 seat0, cha22 s93004 seat1, v56 s93007 seat0). Logs are `run-80M-off-spot-*.log`.

## Rule-1-OFF arm: derived with `derive_off.py` (the 70M script with the prefix set to 80M)
- `derive80_derivation.jsonl`/`.log`, replays in `../games-fixedshop/80M-off-derived/replays/`.
- 48/48 on each of these checks:
  - a prerule record exists;
  - the rule action equals the ON replay;
  - the rule changed the action;
  - prefix actions on obs 0-717 are equal;
  - the opponent's final action is equal;
  - our final action is the policy's own.
- 0 bad statuses. The cf.py market model matches the engine's money change in 48/48 games.
- The opponent's bank is unchanged vs ON in 29/48 games (70M: 37/48). The engine applies our final-turn sale to the market before the opponent's final sale, so the opponent's bank can move. The cf check covers this.
- No-override replay of ON matches in 3/3 games on banks, actions and observations (`replaycheck80.jsonl`).
- The 3 played rule-off games equal the derived ones in 3/3 on banks, all actions and all observations excluding remainingOverageTime (`spot_check_off.json`).

## Health
All 48 80M-ft games qualified: 719 calls each, 0 exceptions, 0 invalid raw actions, 0 default-pass fallbacks and 0 bad statuses. Every game used the pkg-80M manifest and the one engine hash. Mean wall time was 94.0 s per game.

## Results (`eval80m.py`, output in `eval80m_tables.md`, `eval80m_summary.json`, `eval80m_games.jsonl`)
SE is computed over the 8 per-seed means. Mirrored seats often play the same game: in 20/24 seed×anchor pairs, 80M-ft has an identical margin in both seats. So each W-L tally is closer to about 28 independent games than to 48.

| arm | smaller_market_shock | cha22 | v56 | all 48 | own bank | anchor bank |
|---|---|---|---|---|---|---|
| 80M ft (played) | +208 (8-8) | -1,558 (10-6) | -305 (12-4) | **-551 (30-18)** | 99,006 | 99,557 |
| 80M off (derived) | -284 (6-10) | -2,097 (6-10) | -1,006 (10-6) | -1,129 (22-26) | 98,430 | 99,558 |
| 70M ft | -1,169 (6-10) | -1,790 (4-12) | -1,702 (6-10) | -1,554 (16-32) | 97,611 | 99,164 |
| 70M off (derived) | -1,813 (6-10) | -2,308 (4-12) | -2,154 (6-10) | -2,092 (16-32) | 97,072 | 99,164 |

Paired Δ margin (mean ± SE over seed means, better on k/8 seeds, flips L->W / W->L):

| pair | smaller_market_shock | cha22 | v56 | all |
|---|---|---|---|---|
| 80M-ft vs 70M-ft | +1,377 ± 1,436, 5/8, 4/2 | +232 ± 1,036, 5/8, 6/0 | +1,398 ± 877, 4/8, 6/0 | **+1,002 ± 929, 5/8, 16/2** |
| 80M-ft vs 60M-ft | +1,424 ± 1,216, 5/8, 6/2 | +2,935 ± 1,640, 5/8, 8/0 | +3,900 ± 1,888, 6/8, 10/0 | **+2,753 ± 1,160, 5/8, 24/2** |
| 80M-off vs 70M-off (both derived) | +1,530 ± 1,396, 5/8 | +212 ± 1,096, 5/8 | +1,148 ± 945, 4/8 | **+963 ± 962, 5/8, 12/6** |
| 80M-ft vs 80M-off (rule-1 effect) | +492 ± 116, 8/8 | +539 ± 140, 8/8 | +701 ± 173, 8/8 | +577 ± 105, 8/8 |

Per-seed Δ margin, 80M-ft vs 70M-ft (all anchors): +1,801, +3,685, +245, -557, -1,552, +4,203, +3,183, -2,991. Nearly all of the gain vs 70M comes from our own bank (+1,395 ± 650); the anchor bank moves +393 ± 687.

Running table, mean margin (W-L) over 48 games, is in `eval80m_tables.md`. Rule-off margins by checkpoint:

| arm | margin (W-L) |
|---|---|
| BC | -60,598 (0-48) |
| fc6b | -41,123 (0-48) |
| f610 | -21,304 (0-48) |
| 60f2 | -13,785 (0-48) |
| 08bc | -11,943 (0-48) |
| c50 | -8,060 (6-42) |
| 60M | -4,175 (4-44) |
| 70M (derived) | -2,092 (16-32) |
| 80M (derived) | -1,129 (22-26) |

Rule-on margins: c50 ft -7,186 (6-42), 60M ft -3,304 (8-40), 70M ft -1,554 (16-32), 80M ft -551 (30-18).

## Reading and limits
- 80M continues the margin trend on all three anchors and in both rule arms. With rule 1 on, it is the first checkpoint to win more than half of these games (30-18), and its margin is about even on smaller_market_shock (+208).
- The mean margin is still negative. cha22 remains the hardest anchor (-1,558).
- The step from 70M to 80M is the same size as its noise: +1,002 ± 929, better on only 5/8 seeds. Treat it as consistent with improvement, not as established improvement. The step from 60M to 80M is clearer: +2,753 ± 1,160, with 24 L->W flips vs 2 W->L.
- Scope is limited: three anchors, 8 seeds, the fixed-shop engine and a Mac CPU. This is selection evidence, not held-out qualification. The self-play promotion (0.95 vs 70M over 20 games) is a separate signal.
