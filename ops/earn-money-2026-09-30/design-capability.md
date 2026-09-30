# Design (capability-first): making higher earnings learnable in self-play

Date: 2026-09-30. Tree: kg-v3-int `main` 3e89425. Author: a Claude design subagent. This is not Codex-verified. It is a plan: nothing was launched, and no repository code was changed.

The owner asked for self-play in all cases ("let's switch back to self play no matter what"). Every run and diagnostic below is self-play. Fixed opponents appear only as offline evaluation panels, never as a training mix.

The plan keeps the owner's term M reward for the first run. Reward changes are the owner's decision and are marked as options.

## 0. A new supported finding (from existing logs, no compute)

**Every Kaggriculture rollout is synchronized by game phase.** The game has a fixed length of 720 steps, and all 256 envs reset together. The Kaggriculture configs set no `rl.truncation_prob`. So every env is always at the same game step.

Each iteration collects 64 steps, which is about 2.1 of the 30 days. Its 16 optimizer steps therefore see only one game phase. Games end in only 1 iteration in about 11. I decoded the rank-0 `[nt-probe]` records with `raw_decode`:

| Run | Iterations | Iterations with any game end | Gap between them |
|---|---|---|---|
| M (r350xr3w) | 1,469 | 130 | 11 (99 times), 12 (30 times) |
| J/2 (nw3klj2s) | 150 | 13 | 11 (9 times), 12 (3 times) |

Medians for M by phase. The phase is ((iteration − 1)·64 mod 720) / 64, with n ≈ 130 iterations per phase:

| Phase (≈ days) | Raw adv std | Explained variance | approx_kl | Teacher unit_kind KL |
|---|---|---|---|---|
| 0 (days 0–2, investment) | 0.018 | 0.966 | 0.0097 | 5.7 |
| 4 (days 8–10) | 0.018 | 0.980 | 0.0101 | 7.2 |
| 8 (days 21–23) | 0.020 | 0.980 | 0.0107 | 7.8 |
| 9 (game end) | **0.034** | **0.791** | 0.0110 | 6.4 |
| 10 | 0.027 | 0.894 | 0.0097 | 5.2 |

J/2 shows the same pattern. Its end-phase adv std is 0.24–0.28 against 0.09 elsewhere, and its end-phase explained variance is 0.27–0.45 against 0.7–0.9.

**What follows from this.**
- The terminal win/loss/draw term enters only the end-phase iterations. That term is half of the owner's term M return.
- The investment decisions of days 0–2 (land, strawberry seed, sheep) are trained in about 1 iteration in 11. Those iterations use returns that are only bootstrapped, from a winner critic, with no finished game in the batch.
- No update averages gradients across game phases. Each fixed-size step, whose size is set by the LR under Muon (evidence-ppo C1), moves the shared trunk and the unit_kind head toward one phase's noise. Nothing in the batch holds the other phases in place, apart from teacher KL at 0.005.

v2 found and fixed both of these mechanisms. Synchronized phases caused KL spikes and forgetting, and staggering cut KL volatility from 1.98 to 0.45. Updates with no finished game drifted, which was fixed with `--require-outcomes` (evidence-v2 (d) and (e)).

I infer, without having checked the Orbit Wars code, that Isaiah's Orbit Wars games end at varying times through elimination. If so, his recipe never met this mechanism, which would make it a Kaggriculture-specific break from the recipe that worked.

**Evidence against or limiting this finding.**
- approx_kl is flat across phases (0.0097–0.011), so there are no v2-style KL spikes here. The LR caps each step.
- Teacher KL per phase is measured on that phase's states, so it mixes state distribution with policy change.
- Nothing causal has been tested yet.

## 1. Diagnosis

The question is which mechanisms stop self-play from earning. They are ranked by support, and grouped by error class as CLAUDE.md requires.

### D-A: Training error, a supported mechanism with an untested causal role

Phase-synchronized, outcome-sparse, fixed-size updates make the policy drift phase by phase.

- **For:**
  - The synchronization is supported (section 0).
  - The step size is independent of the signal (evidence-ppo C1 and C2: Muon orthogonalization, per-minibatch advantage normalization).
  - The advantage under term M in mirror play is mostly noise (raw std about 0.02, EV above 0.9).
  - The vs-cha22 control drifted at the same per-update KL.
  - v2 saw both effects and fixed them.
- **Against:** approx_kl is flat by phase. No run has isolated synchronization, so the slide could come entirely from LR-sized noise steps even with desynchronized phases.

### D-B: Execution and action-space error, supported as a gap, unattributed as a cause of the slide

The action space contains a large, legal region of choices with no effect, and drift moves probability into it.

- **For:**
  - BC wastes 565 unit commands a game in self-play (844 against cha22), against cha22's 65.
  - PPO at M20 raised this to 996, and its harvests halved from 248 to 115.
  - Grammar masks do not check effect. `grammar.rs` has no checks on watered state, seeds or money.
  - `unit_kind` carries 60–80% of teacher KL. That is the head where no-effect verbs live.
  - 14 drought and 13 starvation deaths per BC self-play seat show that wasted commands displace upkeep. Each death also destroys an investment.
- **Against:**
  - Labor is nearly free (fib hire cost, about 13–15 hands a day), so a wasted command costs something only when it displaced a needed one.
  - The share of the bank lost to ineffective commands is not measured.

### D-C: Architecture error, a candidate

Unit frames and market orders are decided in parallel, with no conditioning across units within a turn (`kaggriculture_actor.py:6-8`).

- Two units can target the same tile.
- If total PLANT demand for a crop exceeds the seeds held, the engine turns every PLANT of that crop into PASS (`lib.rs:1550`).
- BC fits per-unit marginals of a coordinated teacher, so it inherits these collisions. This is one plausible reason BC earns 62–80k while its teacher earns more.
- **Against:** the teacher's own bank and waste in the BC corpus have not been measured (D-bc below). Collisions are not counted.

### D-D: Decision and credit error, open

Long-payback investments get negative or noisy credit.

- Under the margin potential, a purchase is an immediate negative reward. Payback comes 100–300 steps later, beyond the roughly 10-step window of λ = 0.9.
- Section 0 adds a second problem: investment steps are updated only in iterations that contain no outcomes.
- **For:** M20 dropped strawberry (200 → 5 tile-days), sheep (133 → 23) and land (7 of 8 seats never buy). v2 measured purchase credit inverted by a late critic plus truncated GAE (0 of 16 correct).
- **Against:** the drift in D-A alone would also remove the rarest and most precise chains first. The check is D-adv.

### Information: not a blocker

Every economic lever is in the current observation tokens (evidence-econ section 5). The hidden parts are the RNG shop types and the rival's shed, and they are hidden from every policy. Nothing needs memory between turns.

### Unresolved attribution

The data cannot yet split the slide among D-A, D-B, D-C and D-D. The design therefore runs one discriminating check per mechanism before a long run, and changes one mechanism per training arm.

## 2. Changes

All changes go through the shared path: `scripts/run_ppo.py`, `python/owl/train/ppo.py`, and the Rust/PyO3 adapter in `src/kaggriculture/`. They add no second trainer and no memory. No opponent identity enters inputs, losses, rewards, normalization or selection.

### C0: Economic telemetry (prerequisite; no behavior change)

- Carry the 32-field terminal econ vector that already exists (`terminal_metrics` `econ_0` and `econ_1`, `lib.rs:1118-1175`) through the training rollout's terminal path.
  - Seam: `PendingBatch.metrics` in `src/kaggriculture/env.rs` feeds the `train/bank_games` aggregation in `ppo.py`, in the Kaggriculture branch near `ppo.py:848`.
  - Log these per update as game means: `econ/sell_cash`, `sell_units`, `ineffective`, `harvest`, `water`, `feed`, `drought`, `starvation`, `unsold_end`, `seeds_unused_end`, `unused_land_cash`, `idle_hand_steps`, `floor_sale_units`.
  - Also log `train/joint_bank_mean` (bank_0 + bank_1), which shows whether both seats are sliding into poverty together.
- Add rollout action histograms from the sampled action tensors already on device: the shares of each `market_kind` (BUY_LAND, BUY_ANIMAL, BUY_SEED by item, SELL, HIRE) and of each `unit_kind` verb.
- Add the raw advantage by action family: the mean and std of the advantage before normalization, grouped by sampled `market_kind`/item and by `unit_kind`. The groups are BUY_LAND, BUY_ANIMAL, BUY_SEED STRAWBERRY, PLANT and HARVEST. This makes D-adv an online measurement.
- Add `train/game_phase_hist`, the distribution of env game steps in the batch, so synchronization is visible.

### C1: Stagger game phases once at startup (fixes D-A's synchronization)

- **New key:** `rl.initial_stagger: bool`, default false.
- **Behavior:** when true, env i's first game is truncated at a step u_i drawn uniformly from {1..719}, with a critic bootstrap. This reuses the existing truncation path (`ppo.py:688-710`, `_apply_truncation`, which covers stateless models; the Kaggriculture model is stateless). Only a per-env truncation step for game 1 is new.
- Every later game runs the full 720 steps. Because the length is fixed, the stagger then lasts for the whole run.
- **Result:** every iteration holds all phases, and about 23 game ends globally (256·64/720), so every update carries terminal outcomes.
- **Rejected alternative:** a constant `truncation_prob` with a fixed `truncation_step`. It would cut paybacks in every game, and it spreads phases only slowly.

### C2: Effect-aware unit masks (addresses D-B; capability, no retraining needed)

- **New key:** `env.action_spec.effect_masks: bool`.
- **Seam:** `src/kaggriculture/grammar.rs`, mirrored in `owl/kaggriculture/gpu_grammar.py`'s tables.
- **Rule:** mask a unit verb or target only when the current observation fully determines that it has no effect, using the engine's own commit predicate on the start-of-step state:
  - WATER on a tile already watered today, or on a tile with no crop;
  - HARVEST on a tile with no ready units;
  - FEED on an animal already fed today, or with no WHEAT held;
  - CARE on an animal already cared for;
  - COLLECT_FERTILIZER with none available;
  - PLANT of a crop with 0 seeds held, or on an occupied tile;
  - SELL of more units than are held.
- Masks never check other units' same-turn choices; that is C3.
- **Properties:**
  - BC logits are unchanged; only the probability over allowed choices is renormalized. The BC checkpoint therefore runs as-is, and the change can be judged offline before any RL.
  - It removes the no-effect region that drift fills.
  - It ships inside `owl.rs` with the agent. Masks are a function of the current observation only, so the policy stays stateless.
- **Required test:** for each verb, over oracle and BC traces, the mask must never remove an action the engine commits.

### C3 (conditional on D-cap): Resolve over-planting within a turn (addresses D-C)

- **New key:** `env.action_spec.plant_resolver: bool`.
- When the joint sampled PLANT demand for a crop exceeds the seeds held, only the excess PLANTs become PASS, taken from the highest unit index first. The engine would otherwise turn every PLANT of that crop into PASS.
- It is deterministic, uses only the current observation and the sampled joint action, and is applied in the adapter before `engine.step`. It is shipped identically in the agent.
- PPO log-probs stay on the sampled action, and the resolver becomes part of the environment.
- Build it only if D-cap shows that PLANT blocks cost material yield. Autoregressive unit frames within a turn are allowed (prefix state resets each observation) but cost throughput. They are the fallback only if collisions other than PLANT dominate.

### Reward and optimizer minimum

- **Reward:** the owner's term M, unchanged (`configs/kaggriculture_4rank_margin.yaml` reward block).
- **Option R1, owner's decision, only if D-adv shows consistently negative purchase advantages:** a book-value margin potential.
  - Before the last step, Φ_t = margin_score(W_self − W_opp), where W = `seat_wealth` (`lib.rs:1417`, which marks productive assets at book). At the terminal step, Φ_T = margin_score(bank_self − bank_opp).
  - Resets are symmetric, so the whole-game return equals term M exactly. Only the timing changes: a purchase is no longer an immediate loss.
  - It is still zero-sum, and the critic range limit of [−1.5, 1.5] is unchanged.
- **Optimizer:** J/2's LRs (muon 1e-4, adamw 5e-6) and `target_kl: null` in run 1. That isolates C1. If C1 holds the bank but does not raise it, the next lever is evidence-ppo's D3: a larger batch per optimizer step (`rl.gradient_accumulation_steps` 1 → 4, so 4 optimizer steps per iteration) plus `rl.target_kl` 0.005.

## 3. Cheap offline diagnostics, run first

The CPU runs are niced, use no pod, and use the eval seeds of `configs/kaggriculture_4rank_vs_cha22.yaml`.

**D-sync (done here, from logs).** Confirmed; see section 0.

**D-cap (CPU, about 10 minutes, no model change).**
- Play BC best in self-play on 8 seeds (16 seats) and log the per-verb breakdown. The engine already asserts `A_VERB_INEFFECTIVE` / `A_SLOT_INEFFECTIVE` (`lib.rs:1928-1934`), available from `step_with_market_metrics`.
- Classify each ineffective command:
  - (a) predictable from the start-of-step observation (C2 can fix it);
  - (b) a collision on the same target within the turn;
  - (c) a PLANT-block PASS, counting the PLANTs lost.
- Repeat with the M20 checkpoint (`checkpoint_00_020_004_864`).
- **Discriminating observation:**
  - If (a) is at least 60% of 565 and involves upkeep verbs, C2 is the capability lever.
  - If (b)+(c) dominate, the problem is the architecture (C3 or sequential frames).
  - If the total yield lost is small (a few harvests a game), D-B is not the earnings lever. In that case drop C2/C3 from the path and keep only C0 and C1.

**D-mask (CPU, about 10 minutes, after C2 is implemented and tested).**
- Play BC best self-play on the same 8 seeds with `effect_masks` off and on, with sampled actions.
- Measure bank, deaths, harvests and ineffective commands.
- Two same-recipe seeds differed by 16.6k, so use the same seeds and a paired difference per seat.
- **Pass:** paired mean bank gain of at least 5k with fewer deaths. The masked BC then becomes the start policy; it is the same weights with masks on.
- **Fail:** no gain, or a lower bank, which would mean the masks remove something useful and must be audited.

**D-bc (CPU, a few minutes).**
- In the BC corpus `bc-best/shards-top1` (team "M & M & P & Q"), measure the teacher's own final bank and its ineffective command count by replaying its actions through the engine.
- If the teacher earns well above BC's 62–80k and wastes little, the gap is imitation fidelity under parallel frames (D-C). Starting-state and data changes then matter more than RL.
- If the teacher also earns about 70k, BC is already faithful. More money must then come from RL or from cha22-like behavior.

**D-adv (CPU, about 20 minutes).**
- Roll out BC best and M20 in self-play for 8 games with term M.
- Compute the critic values, GAE (γ 1, λ 0.9, horizon 64, bootstrapped as in training), and raw advantages on BUY_LAND, BUY_ANIMAL, BUY_SEED STRAWBERRY and PLANT frames, split by game phase.
- **Discriminating observation:**
  - A consistently negative mean at purchase frames (for example a t-stat below −2) supports D-D, which makes R1 worth taking to the owner.
  - A mean near zero with high variance supports pure drift (D-A).

**D-ceiling (CPU, about 5 minutes).**
- Play cha22 against cha22 for 4 games and log the joint bank.
- This gives the ceiling for two competent sellers, so no bank target is set against cha22's 143–179k, which it earned against weak sellers.

## 4. First run (pod, short, paired)

- **Mechanism tested:** D-A's synchronization. Does desynchronizing game phases (C1) stop the BC economy sliding under term M self-play at J/2's LR?
- **Inputs and code path:** `scripts/run_ppo.py` with a new preset `configs/kaggriculture_4rank_margin_stagger.yaml`. It equals `kaggriculture_4rank_margin.yaml` with `rl.initial_stagger: true` and C0 telemetry.
  - Control arm: the same preset with `rl.initial_stagger: false`.
  - Both arms use the same seed and start.
  - If D-mask passed, both arms run with `effect_masks: true`. The masks are then part of the start policy and do not confound the pair.
- **Start:** `/Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt` with `--load-model-weights-mode model_only`, keeping the BC critic head. The BC-critic arms were the only ones that rose. The teacher and last_best are the BC best, not J/2 final.
- **Length:** 300 iterations per arm, about 4.9M env steps and about 20 minutes each at M's median of 4.1 s per iteration. The two arms run one after the other on 4 ranks.
  - For comparison, M slid 63.9k → 40k by iteration 150, and J/2 was past its peak by iteration 60.
  - `checkpoint_freq` stays at the owner's 10M. The probe's evaluation is offline.
- **Expected discriminating observation:**
  - Compare the own bank averaged over windows of 11 iterations, taken from the games that ended in each window (both arms have about 256 games per window).
  - If synchronization drives the slide, the stagger arm stays at or above start − 5k through iteration 300 while the control repeats the J/2 slide. The expected control slide is 17k or more below start, based on J/2, A2 and M.
  - Secondary signals: the stagger arm's `teacher/unit_kind_kl` grows more slowly, and `econ/ineffective`, `econ/drought` and `econ/starvation` do not rise.
- **Metrics to watch:**
  - bank and `train/joint_bank_mean` per window;
  - `econ/*` (sell cash, units sold, ineffective, deaths, harvest);
  - the market_kind shares (BUY_LAND, BUY_ANIMAL, BUY_SEED STRAWBERRY);
  - raw advantage by family;
  - teacher KL by head;
  - `train/game_phase_hist` (it must be flat in the stagger arm by iteration 12);
  - explained variance, approx_kl and clipfrac.
- **Offline evaluation after the run:** each arm's final checkpoint against BC best on CPU, 8 seeds with both seats swapped (16 games), reporting win rate, bank margin, own bank, legality and completion. This is selection evidence only, not held-out qualification.
- **Loss condition:** the stagger arm's window bank falls 17k or more below its start by iteration 300, or falls as far as the control arm (within 17k). Either way, synchronization is not the lever.
- **Stopping condition:**
  - Stop at 300 iterations.
  - Stop early if either arm's window bank falls below 30k (the bank M reached; nothing further to learn).
  - Stop on any legality or completion failure.
  - No automatic extension. A win is followed by one 1,500-iteration run of the stagger preset, judged the same way plus the 10M evaluation. If the bank holds but does not rise, add evidence-ppo's D3 (batch per step plus target_kl) as the next single-mechanism arm.

## 5. When to abandon this design

- **D-sync is refuted as a cause** if the pair shows no difference within 17k and both slide. Then the drift is pure LR-sized noise stepping (D-A without synchronization). Move to the optimizer levers: a larger batch per step, target_kl, or a fixed BC-KL anchor at 0.05–0.1. The anchor needs the owner to confirm it is not "adaptive".
- **Drop C2** if D-mask gives no paired gain (below 5k) or D-cap shows that no-effect commands cost little yield.
- **Keep C3 and sequential frames off the path** if D-bc shows the teacher earns about the same as BC and D-cap shows few collisions or PLANT blocks.
- **The design cannot reach "earning for real" under term M** if C1 holds the BC bank but no arm ever beats BC best in the offline evaluation, and `train/joint_bank_mean` stays flat while the margin telemetry moves. Mirror self-play would then reward only relative play. Bring the owner the measured joint-bank and flooding (`floor_sale_units`) evidence, the D-ceiling number and option R1. Do not choose a reward on the owner's behalf.
- **Stop spending on training** if D-adv shows the critic ranks purchases wrongly even with synchronization fixed (negative purchase advantage in the stagger arm's online telemetry). Inspect GAE and the critic first: v2 needed segments of 256 or more with λ = 1 to fix exactly this.

## Limits

- All run evidence is one seed. The only replicate pair differs by 16.6k.
- The phase table is a correlation from logs.
- Orbit Wars ending games at varying times is inferred, not checked.
- C1, C2 and C3 are designs. Their tests and `just py-prepare` / `just rs-prepare` are still to run.
- The CPU diagnostics use small samples (8 games) with sampled actions.
- Each implemented adaptation needs its cookbook record.
