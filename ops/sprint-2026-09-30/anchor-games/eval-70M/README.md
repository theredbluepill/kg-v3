# 70M vs 60M on the fixed-shop anchors: rule 1 on (played), rule 1 off (derived); rule 2 not applied

Date 2026-10-01. Owner, verbatim: "Let's start a cpu pod as you said and evaluate the pt." Earlier: "let's not apply rule2."
This is the Mac fast path. Nothing was uploaded, submitted, committed or pushed, no pod was touched, no code default was flipped, and the 60M and c50 game folders were left alone.

## Package `../pkg-70M/` (local Mac, NOT a submission)
- Built exactly like `pkg-60M`: `../pkg_local_mac.py`, the same clean detached worktree `scratchpad/src-08bc-4c99768a` (kg/submit-08bc `4c99768a`, rule 1 only, no rule-2 code, `diff_vs_head` empty), and the same macOS arm64 native module `81b9bf3f…8507` (copied from `pkg-60M`).
- Checkpoint: `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/checkpoint_00_070_041_344.pt`, sha256 `4c8b94831dd2a7b0688646555e19fa47170da538bd37d14296f529b9745238e9` (matches SHA256SUMS). Config sha256 `f7a1349a651904e9374c39a3aff4fa2460613eaf011d9fc319017af24eae377c`; it differs from the 60M config only in `n_envs` 12->20 and `n_runtime_gpus` 4->8.
- Slim model-only checkpoint sha256 `addbdb38de0d508c71527d80f19485e60d0080ef77052e1ee4892f7b6783dfba`. All 210 tensors are torch.equal to the 70M `model` state dict, and all 210 differ from 60M (`pkg70M_verify.json`).
- Manifest sha256 `761ff250ba287f1c8815c32b483e56965b172381c3815afea9c555579c25e8df`. Compared with `pkg-60M`, only `models/primary/{checkpoint.pt,config.yaml}` differ.

## Runs
- `run_fixedshop_70M.sh 70M-ft-on 1` wrote `../games-fixedshop/70M-ft-on/` (log `run-70M-ft-on.log`, 18:00-18:15Z): 48 games, 8 seeds 93001-93008 x 2 seats x {smaller_market_shock, cha22, v56}, fixed-shop kaggle-environments (`kaggriculture.py` `f73d27ce…5004`), nice 10, 5 parallel, 1 torch thread, strict agent. Each game ran through `run_game_prerule.py`, which is `run_game.py` unchanged plus a pass-through tap on `liquidate_final_turn`. The tap wrote the policy's own validated obs-718 action and the rule's output to `prerule/<label>.json`.
- Spot check, rule 1 actually OFF: `70M-off-spot` played smaller_market_shock s93001 seat0, cha22 s93004 seat1 and v56 s93007 seat0 (logs `run-70M-off-spot-*.log`).

## Rule-1-OFF arm: derived (exact final-turn counterfactual)
`derive_off.py` builds the OFF game from these inputs:
- the ON game's recorded actions for both seats on obs 0-717;
- the opponent's recorded final action;
- our model's own pre-rule obs-718 action.

It replays them through the same engine (`make("kaggriculture", {"seed": seed})`, replay callables, no inference). The derived replays are in `../games-fixedshop/70M-off-derived/replays/`.

Checks:
- **Method on 60M.** 60M-ft-on replayed with 60M-off's obs-718 action gives the played 60M-off banks in 48/48 games (`validate60_derivation.jsonl`).
- **Engine replay is exact on 70M.** Replaying 70M ON with no override reproduced every action and every observation (excluding remainingOverageTime) in 3/3 games (`replaycheck70.jsonl`).
- **Played rule-off games match.** The 3 played rule-off games equal the derived ones in banks, all actions and all observations (`spot_check_off.json`).
- **cf.py agrees.** The `endgame/cf.py` market model reproduces the engine's final money change on 48/48 derived games.

## Analysis
`eval70m.py` follows the `eval-60M/eval60m.py` pairing. It writes `eval70m_tables.md`, `eval70m_summary.json`, `eval70m_games.jsonl` and `eval70m.log`. SE is computed over the 8 per-seed means, because mirrored seats often play the same game.
