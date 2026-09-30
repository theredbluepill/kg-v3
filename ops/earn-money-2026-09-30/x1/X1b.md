# X1b: how much two competent sellers earn (cha22 vs cha22)

Diagnostic question (plan section 2, X1b): how much do two competent sellers make together in a mirror game? That is the realistic self-play target and the denominator for "earning for real". Expected discriminating observation: the cha22 mirror joint bank compared with BC-vs-BC (about 152k) and the plan's demand bound (about 195-211k joint). Stopping condition: 8 primary games, plus a cheap extension if runtime allowed.

Status: measured 2026-09-30 by a Claude subagent. Not Codex-verified. No training was run, and no repository code was changed.

## Result

- **Primary set (8 games, seeds 0-7):** the joint bank averaged **178.6k** (SD 50.7k, SE 17.9k, median 162.2k, range 144.0k-298.6k). Per seat it was 89.3k in seat 0 and 89.4k in seat 1.
- **All 96 games:** the joint bank averaged **190.8k** (SD 46.2k, SE 4.7k, median 180.1k, range 122.1k-334.9k). Per seat it was 95.4k.
  - 30 of the 96 games (31%) finished above the 211k "demand bound". The maximum was 334.9k.
  - 21 of the 96 finished below 152k.
- **On the same worlds as BC self-play** (seeds 2^62+4..7, n = 4 games): cha22 mirror play made **170.8k joint** (85.4k per seat), against BC's **151.3k** (75.6k per seat). That is +19.6k joint on average. But the per-world differences were +98.2k, -14.1k, +6.7k and -12.5k, so **BC out-earned the cha22 mirror on 2 of the 4 worlds.** With n = 4 this is direction only.
- **Legality and completion:** 96 of 96 games completed with 719 transitions. There were 0 controller errors and 0 engine errors.
- **The mirror splits the money evenly:** 89 of 96 games ended in an exact tie. The other 7 differed by at most 1.5k.

## Reading

1. **The 195-211k "demand bound" is not a ceiling.** A third of mirror games beat it, and the largest was 335k. The drain-at-base-price estimate leaves out the scarcity premium and the glut buffers. evidence-econ itself put the capped pie at about 270k once those are counted, and three games went above even that. Do not use about 200k as the upper limit for self-play.
2. **The world seed dominates.** Across worlds the joint bank's SD is about 46k, with a range of 122k-335k, for the *same deterministic policy*. The 152k BC figure comes from 4 worlds and cannot be compared with any other number unless the seeds match. Any "for real" bank target needs a matched-seed denominator (a paired, per-world ratio), not a bare level.
3. **A realistic mirror target** from a competent scripted seller is about **95k per seat** on average (median 90k), which falls in the plan's inferred 90-100k band. On BC's own 4 worlds it is 85.4k, about +10k per seat over BC. On those worlds, BC self-play reaches about **89%** of the cha22 mirror joint (151.3k / 170.8k). That is a small gap, and it varies by world.
4. **cha22's 129-179k against weaker sellers is mostly market share, not an achievable per-seat self-play level.** In the mirror, cha22 averages 95k per seat.
5. For the plan: this gives BC more credit than section 0 does. The gap between BC self-play and two competent sellers on the same worlds is about 20k joint (10k per seat), with n = 4. Growth past that has to come from higher-value worlds or strategies (the high-joint worlds reach 250-335k), not from taking more of a fixed ~200k pie.

## Tables

The primary set, seeds 0-7, run with `play_match(Config::default(), seed, [cha22, cha22])`:

| seed | bank0 | bank1 | joint |
|---|---|---|---|
| 0 | 71,702 | 72,340 | 144,042 |
| 1 | 149,310 | 149,310 | 298,620 |
| 2 | 77,778 | 77,778 | 155,556 |
| 3 | 74,960 | 74,960 | 149,920 |
| 4 | 73,970 | 73,970 | 147,940 |
| 5 | 91,914 | 91,914 | 183,828 |
| 6 | 84,460 | 84,460 | 168,920 |
| 7 | 90,084 | 90,084 | 180,168 |

Matched worlds against BC self-play (B = 2^62 = 4611686018427387904). The BC banks come from `scratchpad/games_self.json` (the evidence-econ run, sampled actions):

| world | cha22 mirror joint (per seat) | BC self-play joint (seats) | difference |
|---|---|---|---|
| B+4 | 225,615 (112,866 / 112,749) | 127,462 (64,359 / 63,103) | +98,153 |
| B+5 | 142,756 (71,378 x 2) | 156,840 (68,782 / 88,058) | -14,084 |
| B+6 | 180,020 (90,010 x 2) | 173,326 (103,755 / 69,571) | +6,694 |
| B+7 | 134,996 (67,498 x 2) | 147,479 (88,224 / 59,255) | -12,483 |
| mean | 170,847 (85.4k) | 151,277 (75.6k) | +19,570 |

Summaries of the sets:

| set | n | joint mean | SD | SE | median | >211k | <152k |
|---|---|---|---|---|---|---|---|
| seeds 0-7 (primary) | 8 | 178,624 | 50,732 | 17,936 | 162,238 | 1 | 3 |
| seeds 8-63 | 56 | 190,291 | 42,830 | 5,723 | 181,014 | 16 | 9 |
| eval band B+0..31 | 32 | 194,610 | 51,549 | 9,113 | 195,722 | 13 | 9 |
| all | 96 | 190,758 | 46,220 | 4,717 | 180,094 | 30 | 21 |

## How the seeds were matched

- The evaluation env's base seed is `_evaluation_seed(base_seed=cfg.env.seed=0, env_steps=0)` = 2^62.
- On the pod, `_create_eval_env(n_envs=4)` then `env.reset()` gives `seed_state()`:
  - before the reset: `(B+4, (B+0..B+3))`;
  - after the reset: `(B+8, (B+4..B+7))`.
- `bc_games.py` calls `env.reset()` once, so BC self-play ran on worlds B+4..B+7.
- Engine seeding: the env uses `ObservationGame::from_seed(config, decimal)`, which calls `Game::new_with_seed_decimal`, which calls `new_with_seed(BigInt)`. `play_match` uses `Game::new(i64)`, which calls `new_with_seed(BigInt::from)`. So the same integer gives the same seeding path.
- The env config sets no engine overrides (the yaml `env:` block has reward and grammar keys only). `play_match` refuses any non-default Config.
- Not byte-verified: I did not compare step-0 snapshots between the env and `play_match`.

## Method and commands

- **Code:** pod `/root/kg-v3-anchor` at `0f7077319ed731894594e9a65f0671f81df37527`, clean (0 porcelain lines, before and after).
- **Runner:** a scratch crate `cha22pair`. Its local copy is `/Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/x1/cha22pair/` and the pod copy is `/root/x1/cha22pair`.
  - It depends by path on `/root/kg-v3-anchor/opponents_rs` and calls `kaggriculture_opponents::play_match` with `[Cha22, Cha22]`.
  - This is the crate's own native runner. `tests/hosted.rs` asserts that hosted seats in the env reproduce `play_match` actions and banks.
  - sha256: main.rs `6359270d…`, Cargo.lock `a8f3e994…`, binary `9f22124d…`.
- **Build:**

  ```
  cd /root/x1/cha22pair && CARGO_TARGET_DIR=/root/x1/target nice -n 19 ~/.cargo/bin/cargo +nightly-2026-04-18 build --release -j 8
  ```

  The toolchain was rustc 1.97.0-nightly (e9e32aca5 2026-04-17). The build downloaded registry crates, because `--offline` failed on autocfg.
- **Primary run** (8 processes, one per game):

  ```
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 19 ./target/release/cha22pair cha22 cha22 $s
  ```

  for s in 0..7. Output: `out_cha22_primary.jsonl` (sha256 `24b29528…`).
- **Extension run:**

  ```
  (seq 8 63; B+0..B+31) | nice -n 19 xargs -P 8 -I{} ./target/release/cha22pair cha22 cha22 {}
  ```

  Output: `out_cha22_ext.jsonl` (sha256 `34b99b90…`).
- **Outputs:** both JSONL files are copied next to this report. Each line holds the banks, joint, winner, errors, joint market counters and wall time.
- **Runtime:** about 3.4 s per game on one core. All 96 games used about 330 core-seconds, at no more than 8 CPU threads, on no GPU.

## Impact on the live learner

The run was `scratch-bank-lr2e3-4rank-20260930`, and I did not touch it.
- Before my work, `perf/steps_per_second` read 4715-5047.
- One logged iteration during the 96-game burst read **4222**, about 16% below the adjacent ~5010. This passes the 10% threshold for one iteration. The burst lasted about 40 s, so it had already ended and there was no load left to reduce.
- The next iterations read 4656 and then 5030, so it recovered.
- One earlier iteration, before any batch game run, read 4494, so dips of this size also occur without my load.
- Future batches of this kind should use `-P 4`.

## Limits

- cha22 is deterministic. Each world gives one outcome, and seat-swapping gives no new information: the banks are symmetric up to under 1.5k.
- The BC comparison has n = 4 worlds, and BC's actions are sampled while cha22's are deterministic. Treat the 89% ratio and the +19.6k gap as direction only.
- I did not record per-seat sale revenue or econ counters. `play_match` exposes only joint market counters, which I recorded (sell, buy, hire and land orders, and committed units).
- I did not investigate why some worlds reach 250-335k. The shop schedule and price events are the candidates.
