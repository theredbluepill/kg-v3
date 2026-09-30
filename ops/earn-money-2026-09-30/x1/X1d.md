# X1d: where the production chain breaks first (2026-09-30)

Item X1d of `../plan.md` section 2. It is a diagnostic only: no training and no repository code change. Run by a Claude subagent. It is not Codex-verified.

## Question and stopping condition

- **Question.** When the checkpoints are played in self-play on the same worlds, in the order BC best -> J/2 final -> hz4@10M -> M@10M -> M@20M, which econ counter moves away from BC first?
  - If investment (land, strawberry, sheep) moves first, that points to M3.
  - If upkeep deaths or command waste move first, that points to M6 or to undirected drift.
- **Stopping condition.** The run stops once all 5 checkpoints have played 8 games, or once 60 minutes have passed. All 5 finished.

## Setup (exact)

- **Pod.** abl4mvr5w1mmn4, code `/root/kg-v3-anchor` at 0f70773, `.venv`. CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19`, 2 concurrent processes x 4 torch/OMP/rayon threads, so 8 threads in total.
- **Worlds.** Config `configs/kaggriculture_4rank_vs_cha22.yaml`, with `env.n_envs=8` and `rl.dtype=float32`. The worlds come from `run_ppo._create_eval_env(env_steps=0, opponent_mix=None)`, which is two-model self-play with both seats on the same checkpoint.
  - Eval base seed 4611686018427387904 (2^62). The 8 env seeds are 2^62+0 … 2^62+7, recorded from `env.seed_state()` in every output file.
  - The same 8 worlds were used for every checkpoint.
- **Actions.** Sampled with `run_ppo.forward_learner_rows`. `torch.manual_seed(20260930)` is set per process.
  - The worlds are identical across checkpoints, but the action streams are not. Games are therefore paired by world, not by trajectory.
- **Checkpoints.**

| label | path |
|---|---|
| BC | `/root/bc-best/checkpoint_bc_best.pt` |
| J2_final | `/root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt` |
| hz4_10M | `/root/runs/J2-resume-r0208-20260930/20260930-020138/checkpoint_00_010_010_624.pt` |
| M_10M | `/root/runs/M-margin-J2-4rank-20260930/20260930-033319/checkpoint_00_010_010_624.pt` |
| M_20M | `/root/runs/M-margin-J2-4rank-20260930/20260930-033319/checkpoint_00_020_004_864.pt` |

- **Lineage (from `../evidence-runs.md`).** J/2 comes from BC. hz4 and M are **siblings**: each was resumed from J/2 final with a different reward.
  - hz4: shaping 0.2 + terminal sign 0.8.
  - M: margin 0.5 + terminal 0.5.
  - So "the ordering" is a tree, BC -> J/2 -> {hz4, M10 -> M20}, not a chain.
- **Commands.**
  - Launch on the pod: `/root/x1/run_x1d.sh`. Each call is `nice -n 19 .venv/bin/python /root/x1/x1d_selfplay.py LABEL CKPT 8 4`.
  - Analysis on the Mac: `python3 x1d_analyze.py data > x1d_analysis.txt`.
- **Wall time.** 603-632 s per checkpoint, 719 env steps each. Everything ran from 06:48 to 07:20 UTC.
- **Learner impact.** The live learner (scratch-bank-lr2e3) ran at `perf/steps_per_second` 4849 before the diagnostic and 4818-4980 during it. The drop was under 1%, well below the 10% limit.
- **Denominator.** 8 games (16 seats) per checkpoint. The unit of analysis is the per-game mean of the two seats, n = 8.
  - Tile-days and animal-days are counts of occupied tiles, summed over 30 daily snapshots (step 0, then hour 23 of each day). This definition reproduces the evidence-econ numbers exactly on its files: BC 200 strawberry and M20 5 strawberry.
  - Land purchases = unlocked quadrants at the end minus the start.
  - Deaths = starvation + drought.
- **Screening rule.** I declared this rule for this diagnostic; it is not a calibrated threshold. A counter has "moved" when both hold:
  - |Welch t| >= 2.5 on the 8 game means against BC;
  - |relative change| >= 20%.

## Results (per-seat means, 8 games each; t against BC; `*` = moved under the rule)

| counter | BC | J/2 final | hz4@10M | M@10M | M@20M |
|---|---|---|---|---|---|
| **final bank** | 60,524 | 69,739 (+15%, t+1.1) | 49,579 (-18%, t-1.6) | 48,762 (-19%, t-1.8) | **36,454 (-40%, t-3.6)\*** |
| sell cash | 74,399 | 90,830 (+22%, t+1.9) | 60,081 (-19%, t-2.1) | 56,519 (-24%)\* | 42,621 (-43%)\* |
| units sold | 727 | 880 (+21%, t+1.6) | 524 (-28%)\* | 669 (-8%) | 348 (-52%)\* |
| harvests | 233 | 313 (+34%, t+2.5) | 194 (-17%) | 253 (+9%) | 121 (-48%)\* |
| water | 675 | 819 (+21%) | 620 (-8%) | 684 (+1%) | 318 (-53%)\* |
| **strawberry tile-days** | 230 | 275 (+19%, t+0.6) | **16.2 (-93%, t-4.1)\*** | **3.8 (-98%, t-4.4)\*** | 12.1 (-95%)\* |
| land purchases (quadrants) | 1.6 | 2.3 (+48%)\* | 0.8 (-48%)\* | 1.1 (-32%)\* | 0.2 (-88%)\* |
| seats never buying land | 0/16 | 0/16 | 3/16 | 0/16 | 13/16 |
| first land snapshot (about day+1; 30 = never) | 11.8 | 8.6 | 15.9 | 8.0\* | 28.2\* |
| animal-days (cow+sheep+goose) | 176 | 251 (+43%)\* | 170 (-3%) | 140 (-20%) | 127 (-28%)\* |
| sheep-days | 89 | 82 (-9%) | 64 (-28%, t-1.8) | 44 (-51%)\* | 28 (-69%)\* |
| cow-days | 80 | 111 (+39%) | 91 | 65 | 100 |
| goose-days | 7 | 59\* | 15 | 32\* | 0\* |
| melon / wheat tile-days | 90 / 404 | 140\* / 535 | 169\* / 372 | 124\* / 604\* | 76 / 252\* |
| pasture tile-days | 23 | 68 | 113\* | 39 | 57\* |
| empty tile-days | 413 | 376 | 220\* | 302 | 249\* |
| hands (mean over snapshots) | 6.8 | 8.8\* | 8.4\* | 7.9 | 8.1 |
| **deaths (starv + drought)** | 21.8 (8.3 + 13.5) | 32.2 (6.9 + 25.3; drought +88%\*) | **1.4** (0.6 + 0.9)\* | 28.2 (2.4 + 25.8) | 17.5 (1.7 + 15.8) |
| unit commands | 5,435 | 6,889\* | 6,637\* | 6,282 | 6,310 |
| **ineffective commands** | 755 | **977 (+29%, t+3.1)\*** | 1,252 (+66%)\* | 940 (+24%, t+2.4) | 891 (+18%, t+1.7) |
| unsold at end (units) | 21.8 | **59.9 (+176%)\*** | 101.6\* | 74.0\* | 25.9 |
| money at day 4 / 9 / 14 / 19 | 218 / 2,184 / 14,076 / 29,591 | 194 / 1,848 / 14,015 / 29,514 | 75 / 1,892 / 10,768 / 22,607 | 624\* / 4,129\* / 21,522\* / 31,380 | 238 / 4,919\* / 15,769 / 22,684 |

The full table, with all 32 econ counters plus the derived counters for every checkpoint, is in `x1d_analysis.txt`. The other counters that moved at 10M include:
- fertilizer wasted: hz4 39 and M10 15, against BC 2.2;
- dug units: hz4 48, against BC 3.9;
- terminal flags: 21 and 14, against 2.9;
- expiry units: M10 31, against 5.8;
- missed growth: M10 627, against 397;
- market unfilled: fell at every checkpoint, from 868 down to 235-467.

Per-seat strawberry tile-days:

| checkpoint | per-seat values |
|---|---|
| BC | 60-495, every seat at 60 or more |
| J/2 | 26-648 |
| hz4@10M | 0-75 |
| M@10M | 0-16, with 8 of 16 seats at 0 |
| M@20M | 0-24 |

## Readout

1. **[S] The first adverse link to break is strawberry, the long-payback perennial.** It falls at the first post-J/2 checkpoint in *both* sibling branches: -93% under hz4's reward and -98% under term M.
   - At J/2 it is still intact: 275 against 230, t +0.6, and every seat still plants it.
   - Bank, sales, harvests and water are not yet significantly lower at 10M. Their t values run from -2.8 to +1.1, and only M10 sell cash (-24%) and hz4 units sold (-28%) pass the rule.
   - The breaks that follow, in order:
     - land: J/2 +48%, then 10M -32% to -48%, then M20 -88%, when 13 of 16 seats never buy;
     - sheep: -28% (not significant), then -51%, then -69%;
     - upkeep throughput only at M20: water -53%, harvests -48%, units sold -52%;
     - the bank: -40%, t -3.6, only at M20.
   - Under the plan's readout, **investment moves first, which points to M3**. The 10M money curves show the same thing: M10 holds about 2x BC's cash on days 9-14 (4.1k against 2.2k; 21.5k against 14.1k), which means capital was not spent.
2. **[S] Upkeep deaths do not lead.** hz4 cut deaths to 1.4 per seat (-93%) and still lost strawberry just as much as M10, whose deaths were unchanged at 28.2 (+29%, not significant). So strawberry loss does not come from deaths, and a drop in deaths did not protect the bank. hz4 was still -18% on bank.
3. **[S] Command waste moves earlier, but without a bank loss.** At J/2, ineffective commands (+29%), drought deaths (+88%) and unsold-at-end units (+176%) already moved. At the same checkpoint, investment, hands and commands also *rose*, and the bank was +15% (not significant).
   - After J/2, ineffective commands do not trend with the collapse: 1,252, then 940, then 891, against BC's 755.
   - Waste, then, is the first counter to move at all, but it moves together with a larger economy, not a failing one.
   - Unsold-at-end at J/2 is about 38 more units per seat than BC. At BC's average 102 cash per unit, that is at most about 4k per seat, and end-of-game glut prices would make it less **[O]**. The count reverts at M20 (26), and at M20 the collapse is in volume, not in clearing the shed.
4. **[S] The loss is not specific to term M.** hz4, under a terminal-sign-dominated reward, drops strawberry and land just as M does. Where M and hz4 differ is upkeep: hz4 has no deaths but more digging and fertilizer waste, while M has more wheat and money hoarding.
   - This does not choose between M3 readings (a) credit inversion and (c) drift removing the rarest chains. It does weaken reading (b), that term M specifically scores investment as a margin loss, as the *only* cause. hz4's reward is 0.8 terminal sign, and hz4 lost strawberry too.

## Limits

- **Sample size.** There are 8 worlds and one sampled action stream per checkpoint (n = 8 game means).
  - BC's per-seat bank ranges from 28k to 93k.
  - On the same first 4 worlds, BC averages 60.7k here against 75.6k in evidence-econ, whose action sampling was unseeded. Sampling alone moves a 4-game mean by about 15k.
  - Bank differences under about 20k are not resolved at this n.
  - The strawberry collapse (t -4.1 to -4.4, with per-seat ranges that do not overlap BC's) is robust to this.
- **Checkpoint granularity.** There is one checkpoint per branch between J/2 and 10M.
  - Whether strawberry falls before land and sheep *within* the 0-10M window is inferred from their different magnitudes at 10M: -93/-98% for strawberry against -32/-48% for land and -28/-51% for sheep. Intermediate checkpoints were not tested.
  - hz4 and M are siblings, so hz4 comes before M10 only by label.
- **What "first" means here.** "First counter to move" is judged on end-of-game aggregates under my declared screening rule. There is no per-day composition trace; only money, quadrants and hands were stored per day.
- **What was not done.** Evaluation against BC or cha22 was not run. Self-play banks here are both-seats-same-policy numbers and are not the primary metric in plan section 0.

## Artifacts

All paths are under `/Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/x1/`, mirrored on the pod in `/root/x1/`. Scripts are working artifacts.

| file | sha256 |
|---|---|
| `x1d_selfplay.py` | `ec503a7c…` |
| `run_x1d.sh` | `f1872fb3…` |
| `x1d_analyze.py` | `c60102c7…` |
| `x1d_analysis.txt` | full output |
| `data/games_BC.json` | `8c9f1081…` |
| `data/games_J2_final.json` | `f7da9f29…` |
| `data/games_hz4_10M.json` | `444d428f…` |
| `data/games_M_10M.json` | `c175c289…` |
| `data/games_M_20M.json` | `ed8c92da…` |

Each `data/games_*.json` file holds per-seat raw data: the 32 econ counters, tile-days by kind, and money, quadrants and hands by day.
