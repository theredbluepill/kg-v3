# X1a: how much does the BC teacher (leaderboard #1 team) earn?

Date 2026-09-30. Plan: `ops/earn-money-2026-09-30/plan.md` section 2, X1a. CPU only, on the Mac, niced; no pod load and no GPU use (the pod was only listed, never loaded). Live learner `scratch-bank-lr2e3-4rank-20260930` perf/steps_per_second: 4714.8 before, 4953.3 after. No repo code changed. Working artifacts are in `ops/earn-money-2026-09-30/x1/`.

## Answer

The teacher earns about **103k per game**, which is about **1.35x BC best's ~76k in self-play**, and it wastes about **9x fewer commands**. Its opponents earn about the same, so the teacher wins by thin margins at a high level, and games often have a joint bank above 200k.

| Per game, teacher seat | Teacher (Kaggle replays) | BC best self-play (evidence-econ, n=8 seats) |
|---|---|---|
| Final bank | wins 110.0k (census n=523); losses 91.0k (sample n=16); **weighted about 102.8k** | 75.6k (62-80k range) |
| Ineffective unit commands | **63** (n=24; range 20-171) | 565 |
| Starvation / drought deaths | 9 / 4 (n=24) | 13 / 14 |
| Unit commands | 7,219 | 5,704 |
| Harvest / water / feed | 524 / 1,207 / 367 | 248 / 661 / 185 |
| Sale cash | 126.1k | 92.8k |

Reading, against the plan's two branches: **the teacher earns well above BC's 62-80k, with little waste.** So under the plan's rule, part of the missing money is imitation fidelity. BC best recovers about 74% of the teacher's bank and has 9x its ineffective commands. RL is not the first lever for closing that gap. This is consistent with M6 and owner option O3, but this item does not establish the cause. What it establishes is the size of the gap, not why BC loses it.

## Sources and denominators

- **Corpus.** The BC prep manifest is `/Users/poonszesen/kg-v3-runs/bc-best/shards-top1/manifest.json` (sha256 prefix `ba5fe1c417741c58`).
  - It covers 4,198 listed public episodes, from Kaggle datasets `kaggle/kaggriculture-episodes-2026-09-{22..28}`.
  - The teacher team (TeamNames sha256 `bdf5243c...ea24`) played **842** of them: **523 won** (admitted, with `terminal_banks`) and **319 lost** (rejected as "team lost", with no banks recorded). It was absent from 3,356. There were 0 draws.
  - Teacher win rate on this corpus: 523/842 = 62.1%.
  - The shards (npz) are not on the Mac (the folder holds only the manifest plus 1 validation shard). Neither `/root/bc-archives` nor the raw ZIPs are on pod abl4mvr5w1mmn4 or the Mac.
- **Won-game census from the manifest** (`x1a-data/manifest_banks.py`). Figures are for n=523 games; teacher seat 0 in 281 games and seat 1 in 242.
  - Teacher bank: mean 110,037, median 108,714, p10 88,208, p90 132,423, min 73,035, max 215,671.
  - Opponent bank: mean 103,822, median 103,171, p10 83,287, p90 125,101.
  - Margin: mean 6,215, median 4,350. Joint bank: mean 213,858, p10 172,514, p90 257,014.
  - Teacher below 80k in 11 of 523 games; at or above 100k in 357 of 523.
  - Per-day teacher means: 118.3k, 104.6k, 111.4k, 115.4k, 108.2k, 108.8k and 109.2k (days 22-28).
- **Replayed sample for losses and econ counters** (`x1a-data/sample.tsv`). This is a deterministic sample: episodes sorted by sha256("x1a:"+id), taking the first 16 of the 319 lost and the first 8 of the 523 won.
  - The episodes were downloaded individually with `kaggle datasets download kaggle/kaggriculture-episodes-<day> -f <id>.json` (`x1a-data/fetch.sh`).
  - Each episode was replayed through the v3 Rust engine (`engine_rs` at kg-v3-int 3e89425, the crate behind `terminal_metrics`' `econ_0/econ_1`). The replay built the engine with `Game::new_with_seed_decimal(configuration, info.seed)` and stepped it with the recorded actions `steps[t][i].action` for t=1..719.
  - Tool: `x1a-replay/` (a standalone working crate, not repo code). Output: `x1a-data/replay.jsonl`, summarized by `x1a-data/summarize.py`.
  - **Parity check.** All 24 replays ran 719 steps with 0 step errors, ended done, and reproduced both recorded Kaggle banks exactly.
  - An extra episode, 113911257 (lost, fetched first as a probe and not in the sample), also matched. Teacher 99,809 vs 103,669; 63 ineffective commands; 18 starvation and 2 drought deaths.

| Replayed (means) | Teacher bank | Opp bank | Teacher ineffective | Opp ineffective | Teacher starv/drought |
|---|---|---|---|---|---|
| Lost, n=16 | 90,983 [65,171-122,791] | 95,695 | 62 | 26 | 8 / 3 |
| Won, n=8 | 113,257 [90,230-138,833] | 107,701 | 66 | 26 | 11 / 7 |
| All, n=24 | 98,408 | 99,697 | 63 | 26 | 9 / 4 |

- The won sample's mean (113.3k) is close to the won census (110.0k).
- The weighted all-games teacher mean is (523 x 110,037 + 319 x 90,983) / 842 = **102,818**. The loss half rests on only 16 of the 319 losses.

## Other observations (supported by the data above, not interpreted further)

- **Opponents.** The teacher's opponents (other ladder teams) are also rich: about 104k in the teacher's wins and 96k in its losses. Their ineffective commands are lower still, 26 on average, and 0 in 8 of 24 games. Low waste looks like the ladder norm rather than something unique to #1.
- **Joint bank against the plan's bound.** Joint banks in teacher games average 214k (won census). That is above plan section 0's 195-211k base-price demand bound. The plan states that bound excludes the wheat and egg glut. Opponents sold more units than the teacher (2,778 vs 1,821) for more cash (162k vs 126k).
- **Deaths are not zero for the teacher.** It has 13 per game against cha22's about 1. That is fewer than BC's 27, but deaths are not the main difference.

## Limits

- **Losses are sampled, wins are complete.** Losses rest on a 16-game sample; the 319 lost games have no banks in the manifest.
- **Econ counters come from 24 games** (plus 1 probe), not the full 842.
- **The two comparisons differ in conditions.**
  - The teacher's numbers come from ladder games against varied, mostly strong opponents. BC's 75.6k comes from self-play on the eval seeds, n=8 seats (evidence-econ). Opponent quality changes the shared-demand market, so this is not a paired, same-condition comparison.
  - The teacher was not run in mirror self-play, and it cannot be: there is no teacher policy, only its replays.
- **BC's own ineffective and death figures** come from evidence-econ (n=8 seats), not re-measured here.
- **Replay parity:** the engine matched the recorded final banks. Intermediate states were not compared.

## Commands

```
# manifest census
python3 ops/earn-money-2026-09-30/x1/x1a-data/manifest_banks.py
# download sample (Kaggle CLI, public datasets)
ops/earn-money-2026-09-30/x1/x1a-data/fetch.sh
# build + replay
cd ops/earn-money-2026-09-30/x1/x1a-replay && nice -n 19 cargo build --release --offline -j 4 --target-dir <scratchpad>/x1a-target
cd ../x1a-data && nice -n 19 <scratchpad>/x1a-target/release/x1a-replay bdf5243c27ea6e5caf82529c401a47e30f7b9e20731c385ea6a626ac09d8ea24 $(cut -f3 sample.tsv | sed 's/$/.json/') > replay.jsonl
python3 summarize.py
```

The 25 raw episode JSONs (about 37 MB each, about 0.9 GB total) sit in `x1/x1a-data/` as local working data. They should not be committed.
