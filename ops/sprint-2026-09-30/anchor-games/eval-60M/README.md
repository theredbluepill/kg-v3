# 60M vs c50 on the fixed-shop anchors, rule 1 off/on (rule 2 not applied)

Date 2026-10-01. Owner, verbatim: "60M is close, you can use rule1/rule2 to evaluate that locally as well." then "let's not apply rule2."
Nothing uploaded, submitted or pushed; no pod touched; no code default flipped; c50 game folders untouched.

## Package `../pkg-60M/` (local Mac, NOT a submission)
- Source: kg/submit-08bc `4c99768a` (rule 1 only; predates the rule-2 commits, so the package contains no rule-2 code), clean detached worktree `scratchpad/src-08bc-4c99768a`, `diff_vs_head` empty. Same source as `pkg-c50-ft`.
- Built with `../pkg_local_mac.py`; native module reused from `pkg-c50-ft` (the same 4c99768a macOS arm64 build, `81b9bf3f…8507`; Rust unchanged 4c99768a..6969b324).
- Checkpoint: `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-from-c50-4rank-20260930/ckpt-60M/checkpoint_00_060_018_944.pt`, sha256 `20b1f795…a91a2` (matches SHA256SUMS). Slim model-only `ad06d493ebde02c019f05f04c0398dec31c38787b3549c90627ee2e21924f6bc`; all 210 tensors torch.equal to the original `model` state dict. Config `a60a577e…e27c`.
- Manifest sha256 `2d48dd16885e16ce261a72b044cb9109442f39f2cb2e60a2af83ed185f44d58b`. Versus `pkg-c50-ft` only `models/primary/{checkpoint.pt,config.yaml}` differ.
- One package serves both arms: `KAGGRICULTURE_FINAL_TURN_LIQUIDATION=0|1`; `KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS` unset (and absent from the code).

## Runs
- `run_fixedshop_60M.sh 60M 0` -> `../games-fixedshop/60M/`; `run_fixedshop_60M.sh 60M-ft-on 1` -> `../games-fixedshop/60M-ft-on/` (logs `run-60M*.log`).
- 8 seeds 93001-93008 x 2 seats x {smaller_market_shock, cha22, v56} = 48 games per arm, fixed-shop kaggle-environments 1.32.7 (`kaggriculture.py` `f73d27ce…5004` in all 192 receipts), nice 10, 5 parallel, 1 torch thread, strict agent.

## Analysis
`eval60m.py` (reuses `endgame/ab.py` load/health/unsold/obs_at and the packaged `liquidate_final_turn`) -> `eval60m_tables.md`, `eval60m_summary.json`, `eval60m_games.jsonl`, `eval60m_rule1.jsonl`, `eval60m.log`.
SE is over the 8 per-seed means; mirrored seats are often the same game (16/24 seat pairs identical in 60M-off), so seeds are the independent unit.
