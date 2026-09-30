# X1c: classifying ineffective commands (BC best and M@20M, mirror self-play)

Status: a diagnostic run by a Claude subagent on 2026-09-30. It is not Codex-verified. There was no training and no repository code change. The working artifacts are in `ops/earn-money-2026-09-30/x1/x1c/` (Mac) and `/root/x1/x1c/` (pod abl4mvr5w1mmn4).

## Question and stopping condition

- **Question.** Which classes of ineffective unit command cost material yield, meaning several thousand in bank per game? Plan section 2, X1c: only such a class earns F5 capability work.
- **Discriminating observation.** Harvests, PLANTs and sales that would have happened without the ineffective command, counted per class.
- **Stopping condition.** Each ineffective command classified for 8 games per checkpoint, or the ~60 min budget spent.

## Method (exact)

- **Code.** `/root/kg-v3-anchor` at 0f70773, with its built venv and the pinned `kaggriculture-engine` crate (engine_rs, `ATTRIB_VERSION` 1).
- **Play** (`x1c/drive.py`).
  - Mirror self-play: the same checkpoint plays both seats, with no opponent mix.
  - Config: `configs/kaggriculture_4rank_margin.yaml` with `env.n_envs=8`, `opponent_mix=None`, `native_threads=1` and `rl.dtype=float32`. Run on CPU with `CUDA_VISIBLE_DEVICES=""`, `nice -n 19` and 4 torch threads per process.
  - **Sampling mode.** Actions are sampled exactly as run_ppo's evaluation does it (`model(obs, deterministic=False)`, `run_ppo.py:2301/2307`). This is not greedy decoding. Seed: `torch.manual_seed(1000)`.
  - **Env.** Built with `create_env(base_seed=1000, rank=0, world_size=1)`, which gives **game seeds 1008-1015** (read from `env.seed_state()` after reset). Both checkpoints play the same 8 seeds. That is 16 seats per checkpoint, and each game runs 719 learner steps.
  - For each step, the joint action JSON is dumped (`owl.kaggriculture.codec.decode_actions`).
  - Commands:
    - `drive.py /root/bc-best/checkpoint_bc_best.pt bc 1000 8 bc.jsonl`
    - `drive.py /root/runs/M-margin-J2-4rank-20260930/20260930-033319/checkpoint_00_020_004_864.pt m20 1000 8 m20.jsonl`
  - Wall time: 651 s and 635 s, run concurrently.
- **Classify** (`x1c/replay/src/main.rs`, built as a standalone cargo binary against the engine crate).
  - Each game is replayed through `Game::new_with_seed_decimal` and `step_with_market_metrics`.
  - **Replay parity.** The replay reproduced the driver's terminal banks exactly in 16/16 games for each checkpoint.
  - **Identifying ineffective commands.** Per step and seat, the rise in `A_SLOT_INEFFECTIVE` (attrib offset 117) marks exactly which actor's command was ineffective. There is one command per actor per step, and slots 0-14 are exact. For M@20M, 1 of 15,109 commands fell in the lumped slot 15 and is left unclassified.
  - **Classifying.** Each command is classified by counterfactual clones of the start-of-step `Game`. Units act in farmer-then-hand order, before the market (`lib.rs` `step_in_place`), and the other seat's program is kept unchanged.
    1. **PLANT blocked, no seeds.** A PLANT whose crop had demand above seeds (`lib.rs:1540-1565`) and 0 seeds of that crop at the start of the step. Seeds bought this turn arrive only after the unit phase.
    2. **PLANT over-demand.** The same block, but with 0 < seeds < demand. **PLANTs lost** = the effective PLANTs if the program keeps the first `s` PLANTs of that crop in unit order (the rest PASS), minus the effective PLANTs in the actual step.
    3. **Predictable alone.** The command is still ineffective when the unit acts alone from the start-of-step state, with every other own unit on PASS.
    4. **Same-target collision.** The command is effective alone, but ineffective when paired with an earlier own unit j whose own command succeeded, and both act on the same tile. For moves, the tile is the destination.
    5. **Other-resource collision.** As in 4, but on a different tile. In practice this is PICKUP against shared shed stock.
    6. **Multi-unit interaction.** No single earlier unit explains the failure.
  - Commands: `x1c-replay bc.jsonl > classify.jsonl; x1c-replay m20.jsonl >> classify.jsonl` (3.3 s). Aggregation: `agg.py classify.jsonl > agg.json`.
- **Live learner.** scratch-bank-lr2e3 `perf/steps_per_second` was 4715-4888 before this work, 4480-5040 during it, and 5008 at 07:16. There was no sustained drop above 10%. No GPU was used.

## Results (per seat, over 16 seats = 8 games x 2 seats per checkpoint)

| | BC best | M@20M |
|---|---:|---:|
| final bank per seat (mean) | 71,181 | 36,648 |
| effective HARVEST per seat | 280.8 | 120.0 |
| sale cash per seat | 87,348 | 43,252 |
| **ineffective commands per seat** | **628.7** (10,059 total) | **944.3** (15,109 total) |
| 1. PLANT, no seeds at step start | 272.8 (43.4%) | 222.8 (23.6%) |
| 2. PLANT over-demand PASS | 203.4 (32.3%) | 55.1 (5.8%) |
| 3. Predictable alone (other verbs) | 118.8 (18.9%) | 622.4 (65.9%) |
| 4. Same-target collision | 32.7 (5.2%) | 42.3 (4.5%) |
| 5. Other-resource collision (PICKUP) | 0.4 | 1.5 |
| 6. Multi-unit interaction | 0.6 | 0.25 |
| predictable from start-of-step state (1+3) | 391.6 (62%) | 845.1 (89%) |

Per-game banks, seeds 1008-1015, as (seat 0, seat 1):
- **BC:** (64763, 90201), (99479, 80915), (92569, 94980), (75633, 66762), (72928, 58741), (76114, 94250), (27842, 59444), (39343, 44931).
- **M@20M:** (43482, 34081), (28313, 41405), (45991, 14654), (36281, 35234), (41245, 23227), (44053, 37245), (39332, 33670), (45504, 42644).

### The requested classes

- **Blocked move.** All blocked moves are predictable alone. None came from a collision.
  - BC: 1.4 per seat.
  - M@20M: 132.7 per seat (NORTH 103.8, WEST 28.9). This is a drift artifact: units walk into walls or locked cells.
- **PICKUP/PLACE.**
  - BC: 57.8 per seat, of which 55.9 are predictable alone (PLACE 34.1, PICKUP 21.8), 0.8 are same-target collisions and 1.0 are shed-stock interactions.
  - M@20M: 66.3 per seat, of which 63.5 are predictable, 1.1 are collisions and 1.7 are shed-stock interactions.
- **Same-target collisions** are almost all a second unit repeating an upkeep verb on a tile an earlier unit already served. Pair counts over 16 seats:

  | Pair | BC | M@20M |
  |---|---:|---:|
  | WATER->WATER | 242 | 299 |
  | HARVEST->HARVEST | 110 | 181 |
  | COLLECT_FERTILIZER x2 | 66 | 50 |
  | CARE x2 | 39 | 21 |
  | FEED x2 | 26 | 31 |
  | PLANT->PLANT | 20 | 50 |

- **Other predictable-alone verbs.**
  - BC: HARVEST 26.4, FEED 10.9, WATER 8.1, CARE 3.7, COLLECT_FERTILIZER 3.7, PLANT 2.6 and FERTILIZE 2.4.
  - M@20M: HARVEST 171.2, WATER 152.4, FEED 28.8, FERTILIZE 22.4, CARE 17.8 and PLANT 14.4.
- **What the drift added.** M@20M has +315.6 ineffective commands per seat over BC. The change by class is:

  | Class | Change per seat |
  |---|---:|
  | Predictable alone | +503.6 |
  | Same-target collisions | +9.6 |
  | PLANT, no seeds | -50.0 |
  | PLANT over-demand | -148.3 |

  So the rise is single-unit commands that are wrong from the start state (HARVEST on unready tiles, WATER on ineligible tiles, walking into walls). It is not parallel-frame collisions.

### Harvests and sales lost next to each class

- **Direct yield loss is 0 by construction for classes 1 and 3-6.**
  - A collided HARVEST or upkeep command hit a tile the earlier unit had already served. The 340 (BC) and 555 (M@20M) harvest units that the colliding HARVESTs would have collected alone were in fact collected by the first unit.
  - A predictable-alone HARVEST, WATER or FEED would produce nothing even with the unit acting alone.
  - Their cost is the wasted unit-turn (opportunity), not lost yield.
- **Crude labor-value bound (average product, not marginal).** The bank per effective unit command is 14.74 (BC: 71,181 / 4,827.6) and 6.93 (M@20M: 36,648 / 5,285.9). The numbers below are wasted turns x that ratio, per seat. They are an upper-leaning estimate, because units often have nothing better to do and PASS turns are common.

  | Class | BC | M@20M |
  |---|---:|---:|
  | No-seed PLANT | 4.0k | 1.5k |
  | Over-demand PLANT | 3.0k | 0.4k |
  | Predictable alone | 1.75k | 4.3k |
  | Collisions | 0.48k | 0.29k |

- **PLANT over-demand: the one class with a real yield loss.**
  - **BC:** there were 1,011 seat-step-crop block events with seeds on hand. The trimmed counterfactual would have planted **95.9 more PLANTs per seat** (wheat 85.9, strawberry 9.4, carrot 0.6).
  - **BC value, upper bound.** The bound assumes every lost PLANT is never replanted. It is priced at this run's own yield per effective PLANT, sold fraction and mean sale price:
    - wheat: 85.9 x 3.3 units x 0.80 sold x 25.6 = 5.66k;
    - strawberry: 9.4 x 4.4 x 1.0 x 239.5 = 9.77k;
    - carrot: 0.07k.
    - The total is **up to about 15.5k per seat**, or about 4.4k harvest units at stake in total.
  - **M@20M:** 334 block events and 23.8 PLANTs lost per seat (wheat 23.1, melon 0.6, strawberry 0.1). The upper bound is about 1.6k per seat.
  - The bound overstates the loss when the unit plants the same tile a turn later. The paired fix arm below measures the actual effect.

### Paired fix arm: the value of the PLANT over-demand class (BC)

- **What the arm did.** `X1C_FIX=1 drive.py /root/bc-best/checkpoint_bc_best.pt bcfix 1000 8 bcfix.jsonl` played the same seeds (1008-1015), the same torch seed and the same sampling mode as the BC run. The one change: before each step, any seat that submitted more PLANT X commands than it held X seeds (with seeds above 0) kept the first `s` PLANTs in unit order, and the rest became PASS. These were re-encoded with `codec.encode_actions`.
- **Scale and parity.** 1,004 surplus PLANTs were trimmed in total (62.75 per seat). Wall time was 597 s. The replay again matched the driver banks in 8/8 games.
- **Outcome.** The mean bank per seat was **68,173, against 71,181 for BC**.
  - The paired per-game difference in seat-mean bank (fix minus BC), for seeds 1008-1015: -9,100, -9,751, +4,818, -29,778, -19,716, -25,289, +16,033 and +48,721.
  - The mean is **-3.0k, with SE 9.1k over n = 8 games**, so the 95% CI (t7) is about [-24.6k, +18.6k].
- **Other counts in the fix arm.**
  - Ineffective commands: 513.9 per seat (over-demand is gone; no-seed PLANT 323.0, predictable-alone 156.9, collisions 32.9).
  - Effective PLANTs: 168.6 per seat, against 206.1 in the BC run.
  - Effective HARVESTs: 246.7 per seat, against 280.8.
- **Reading.** Trajectories diverge after the first trimmed step, so this arm cannot resolve an effect below about 20k. It gives **no evidence that repairing over-demand earns money**. Effective PLANTs even fell, which suggests the blocked PLANTs are largely made up in later turns. So the 15.5k-per-seat upper bound above is not realized in this measurement. Resolving an effect of a few thousand needs many more seeds (roughly 16x more) or a counterfactual that stops diverging, such as a per-event continuation with common random numbers.

## Reading

- **Collisions do not cost material yield.** Same-target collisions (the parallel-frame architecture class) are 33-42 commands per seat, about 5% of all waste. They cost at most about 0.5k per seat even at average product, and they lose no harvest.
- **PLANT over-demand is the only class with a direct yield loss.** It is an engine rule: one surplus PLANT blocks every PLANT of that crop. BC hits it often, with 96 lost PLANTs per seat, including strawberry. Its static upper bound is about 15.5k per seat. The paired repair arm measured -3.0k ± 9.1k (n = 8), which is not a detectable gain. So no class has a demonstrated yield cost of several thousand per game yet, and **none earns F5 on this evidence**.
- **Seedless PLANTs are BC's largest single class.** A PLANT with no seeds at the start of the step is 43% of BC's waste. It may be the teacher's same-turn buy-and-plant habit, which fails because units act before the market; that is inferred from the ordering, not verified.
- **The drift to M@20M adds predictable single-unit waste.** It adds about 500 commands per seat of HARVEST, WATER and moves that are wrong from the start state. That points to decision drift (M1/M3), not architecture (M6 collisions).

## Limits

- The results come from one sampled rollout per seed: 8 seeds, one sampling seed, and the CPU float32 path.
- "Predictable" means predictable from the engine start state. That the observation encodes each relevant field is assumed from the plan's claim (`observe.rs:365-379`) and was not checked per command.
- The labor-value numbers are an average-product heuristic, not a counterfactual.
- One M@20M command (slot 15) is unclassified.
- J/2 and hz4 were not run.
