# Candidate vs v2 anchors: local Kaggle-harness games (2026-09-30)

**Question.** Can the promoted PPO checkpoint from `earn-bank-credit-4rank-20260930` (W&B h3lpxy6q) beat v2 anchors when it runs as a locally packaged Kaggle agent? Owner request, verbatim: "can you run a local game just to check it compete with some anchors we have in v2? 64 games will be fine (make sure you pkg first locally in mac". The BC start ran on the same seeds and seats as a baseline. Nothing was uploaded or submitted, and nothing was committed.

## Result

- **The candidate lost all 48 games against the three real anchors** (cha22, smaller_market_shock, v43), with mean margins between -36k and -44k. It won 16/16 against the official starter. Overall it went 16-48-0, a win rate of 0.25 [0.16, 0.37].
- **BC has exactly the same record** (16-48-0). The candidate is still clearly better than BC on the same seeds and seats. Its paired mean margin is +25,996 higher over 64 games, and its margin was higher in 58 of the 64 games. Per anchor the paired gain is: cha22 +24.6k (15/16 games), smaller +26.1k (15/16), v43 +44.4k (16/16), starter +8.9k (12/16).
- **Where the gap is.** The candidate banks about 89-92k against the real anchors, which is roughly what v2 anchors bank against each other (see the v2 reference "anchors split a world-sized market", where 67-116k is contested). The anchors, however, bank 125-133k against it. The candidate is not short of income. It lets the anchor take a much larger share of the market. v43, the weakest v2 anchor (0/26 against the other anchors), still wins by 44k.
- **Against starter the candidate banks only 44k**, about half of what it banks against the stronger anchors. Why is not attributed. One hypothesis is that its income depends on a trading counterpart or on market conditions the anchors create. This is unmeasured.
- **Integrity.** All 128 games qualified: 0 ERROR/TIMEOUT/INVALID statuses in either seat, 0 exceptions, 0 invalid raw actions, 0 PASS fallbacks (strict mode). Every recorded seat had 719/719 calls. Remaining overage time never dropped below 60.0 s. The steady per-turn p99 was at most 0.142 s (median game 0.084 s). The first turn, including load and warm-up, took at most 0.41 s. These times are on the Mac, not Kaggle hardware.

**candidate**

| anchor | games | W-L-D | win rate [95% Wilson] | mean own bank | mean anchor bank | mean margin | seat0 W/n (margin) | seat1 W/n (margin) | errors/timeouts (bad statuses) | exceptions | invalid | PASS fallbacks | turn p99 s (max game / median game) | max turn s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cha22 | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 88,749 | 124,789 | -36,040 | 0/8 (-35,560) | 0/8 (-36,520) | 0 | 0 | 0 | 0 | 0.142 / 0.090 | 0.241 |
| smaller_market_shock | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 92,329 | 133,492 | -41,163 | 0/8 (-40,999) | 0/8 (-41,327) | 0 | 0 | 0 | 0 | 0.101 / 0.086 | 0.154 |
| v43 | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 88,664 | 132,386 | -43,722 | 0/8 (-41,660) | 0/8 (-45,784) | 0 | 0 | 0 | 0 | 0.096 / 0.082 | 0.218 |
| starter | 16 | 16-0-0 | 1.00 [0.81, 1.00] | 43,919 | 3,535 | +40,384 | 8/8 (+40,379) | 8/8 (+40,389) | 0 | 0 | 0 | 0 | 0.102 / 0.083 | 0.243 |
| ALL | 64 | 16-48-0 | 0.25 [0.16, 0.37] | 78,415 | 98,551 | -20,135 | 8/32 (-19,460) | 8/32 (-20,810) | 0 | 0 | 0 | 0 | 0.142 / 0.084 | 0.243 |

**bc**

| anchor | games | W-L-D | win rate [95% Wilson] | mean own bank | mean anchor bank | mean margin | seat0 W/n (margin) | seat1 W/n (margin) | errors/timeouts (bad statuses) | exceptions | invalid | PASS fallbacks | turn p99 s (max game / median game) | max turn s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cha22 | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 86,919 | 147,512 | -60,593 | 0/8 (-61,642) | 0/8 (-59,545) | 0 | 0 | 0 | 0 | 0.127 / 0.087 | 0.227 |
| smaller_market_shock | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 77,096 | 144,322 | -67,226 | 0/8 (-63,462) | 0/8 (-70,990) | 0 | 0 | 0 | 0 | 0.092 / 0.083 | 0.227 |
| v43 | 16 | 0-16-0 | 0.00 [0.00, 0.19] | 67,859 | 156,014 | -88,155 | 0/8 (-88,318) | 0/8 (-87,992) | 0 | 0 | 0 | 0 | 0.117 / 0.089 | 0.365 |
| starter | 16 | 16-0-0 | 1.00 [0.81, 1.00] | 35,008 | 3,560 | +31,448 | 8/8 (+33,099) | 8/8 (+29,796) | 0 | 0 | 0 | 0 | 0.129 / 0.087 | 0.242 |
| ALL | 64 | 16-48-0 | 0.25 [0.16, 0.37] | 66,721 | 112,852 | -46,132 | 8/32 (-45,081) | 8/32 (-47,182) | 0 | 0 | 0 | 0 | 0.129 / 0.085 | 0.365 |

**candidate minus BC (same seeds and seats)**

| anchor | paired games | win-rate diff | mean own-bank diff | mean margin diff | paired margin diff (per game, mean) |
|---|---|---|---|---|---|
| cha22 | 16 | +0.00 | +1,830 | +24,553 | +24,553 |
| smaller_market_shock | 16 | +0.00 | +15,233 | +26,064 | +26,064 |
| v43 | 16 | +0.00 | +20,805 | +44,433 | +44,433 |
| starter | 16 | +0.00 | +8,911 | +8,936 | +8,936 |
| ALL | 64 | +0.00 | +11,695 | +25,996 | +25,996 |

Per-turn times are the candidate's own `Agent.act` durations for steps 1 and later. "turn p99" gives the maximum and the median of the per-game p99. Win rate uses a 95% Wilson interval. Paired values are candidate minus BC for the same (anchor, seed, seat) game.

## Setup

- **Candidate.** Pod `/root/runs/earn-bank-credit-4rank-20260930/20260930-072457/checkpoint_last_best.pt`, sha256 `fc6b123c5e5f76dc5aa5f2153531ede0291c118f273da9d5324b02421b37bacf` (written 08:33:55Z). It was absent from the Mac copy-off, so I copied it read-only with scp and verified the sha. Its config.yaml has sha256 `d232fb46…`. The local copy is under `~/kg-v3-runs/earn-bank-credit-4rank-20260930/checkpoints/20260930-072457/`.
- **BC.** `~/kg-v3-runs/bc-best/checkpoint_bc_best.pt`, sha256 `fd8545872aca…`, with its sibling config.yaml.
- **Packages** (`pkg-candidate/`, `pkg-bc/`). Staged by `pkg_local_mac.py`, which mirrors `scripts/build_kaggriculture_submission.py` on kg/rebuild-7-4-ship 619349f: main.py comes from `python/kaggriculture_main.py`, plus the owl package, a slim model-only fp32 checkpoint, the config.yaml and manifest.json. Greedy, CPU, 1 torch thread, and `load_state_dict(strict=True)` against the packaged config. The native `owl.rs` is a macOS arm64 abi3 build made with `maturin build --release` (nightly-2026-04-18) instead of the Linux .so. `kg-v3-ship` was not edited.
  - **Candidate source.** The ship branch cannot validate the run's config, because its `reward_shaping` schema lacks the `econ_bank_*` keys added in training commit 0f70773. The candidate package therefore uses a scratch no-commit merge of 619349f into 0f70773 (a detached worktree in the session scratchpad at `pkgsrc/`). The only conflict was additive, in `src/kaggriculture/observe.rs`, and both impl blocks were kept. `python/owl/model` is byte-identical to 0f70773. The config is the run's own config.yaml, unmodified, and critic_offset does not exist at this source.
  - **BC source.** The BC config predates the margin keys and fails that merged schema. My first BC attempt failed to load (every seat-0 turn ERROR). It was discarded, moved to the scratchpad, and is not counted. BC was then packaged from a clean detached checkout of 619349f (`pkgsrc-ship/`) with its own Mac native build. This is the source that Task 7.4 validated BC with.
- **Harness.** kaggle-environments 1.32.7 (kaggriculture.py sha256 `bc8a5487…`, equal to v2's pinned engine), Python 3.11.15, torch 2.6.0 (CPU), macOS 26.4 arm64. Games ran through `run_game.py`, a copy of the ship's `scripts/kaggle_local_episode.py` with `debug=False`, `KAGGRICULTURE_AGENT_STRICT=1` and every call recorded. I added only `--label`, a file-path opponent and a gzip replay. `owl` and `owl.rs` were checked to load from the package directory.
- **Seeds.** 93001-93008. Each seed was played in both seats against each anchor: 4 anchors × 8 seeds × 2 seats = 64 games per policy, identical for candidate and BC.
- **Load.** 2 game processes at a time (4 performance cores − 2), `nice -n 10`, OMP/MKL threads 1. About 58 s per game. Candidate games ran 08:46-09:18Z and BC games 09:20-09:51Z.

## Why these anchors

The v2 roster is 20 native Rust ports; v2 keeps no Python opponents. Of the anchors, only v43, smaller_market_shock, farm2945, tetsu65, tetsu342, v47 and v48 still have their original Python `main.py` (in v2 git history at `35fa47c5^`). The strongest anchors (cha22, pipe16, metav4) exist only as ports. In the v2 anchor-split reference, pipe16 won 17/18, metav4 15/18, smaller 13/18, farm2945 11/18 and v43 0/26. The four I picked span that range:

| anchor | how it runs | why |
|---|---|---|
| cha22 | v2's native `cha22_agent` binary, run through the documented JSONL protocol (`opponents/cha22/README.md`) by `anchors/cha22/main.py` | the newest and strongest import (it wraps Metav4), and the fixed opponent in v3 training |
| smaller_market_shock | original Python `main.py` (sha256 d5460fc2…, matches the v2 registry's `python_source_sha256`) | the strongest anchor still available in Python (13/18) |
| v43 | original Python `main.py` (sha256 919fc1d6…, matches the registry) | v2's named anchor, weak among the anchors (0/26) but still banks about 89k |
| starter | built-in `starter` of kaggle-environments | the weakest reference point |

## Limits

- **The cha22 binary is not freshly pinned.** v2's registry currently refuses it: the engine-crate digest changed and the binary (built 2026-09-28 10:34 +08) is older than `engine_rs/src/{lib,ffi,econ_attrib,myolie_features}.rs`. The cha22 entry and controller still match their pins, and `native_agents/`, `bin/` and `data/` are unchanged since the pin, so I used the binary as-is. Fully clearing this would need a clean rebuild in v2, which I did not do because v2 is read-only here.
- **The two packages come from different source trees** (merged source for the candidate, ship source for BC), because their configs need different reward schemas. The model code and the observation and action paths are the same across both.
- **This is a local strength check, not qualification.** Nothing shows how the agents behave under Kaggle's latency. 16 games per anchor cannot separate a win rate between 0.00 and about 0.19.
- **Not investigated:** why the candidate's bank drops to 44k against starter, and which decisions give the anchors their market share (sale timing, hires or investment). The replays needed for that are saved.

## Files

- `results.md` (this file), `tables.md`, `aggregate.json`
- `games-candidate.jsonl`, `games-bc.jsonl`: one row per game with seed, seat, banks, margin, statuses, call and fallback counts, turn quantiles, replay sha and runtime
- `games/{candidate,bc}/{receipts,logs,replays}/`: per-game receipts, agent logs and gzip replays
- `pkg-candidate/`, `pkg-bc/`: the local Mac packages, each with a `manifest.json`
- `pkg_local_mac.py`, `run_game.py`, `run_all.sh`, `aggregate.py`, `anchors/`
