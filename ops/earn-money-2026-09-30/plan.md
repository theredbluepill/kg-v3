# Plan: getting the self-play agent to earn money (2026-09-30)

Owner request (verbatim, 2026-09-30): "let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?"

Status: a synthesis by a Claude subagent of the three designs and three critiques in this folder (`design-signal.md`, `design-optimizer.md`, `design-capability.md`, `critique-*.md`) and the four evidence reports (`evidence-runs.md`, `evidence-econ.md`, `evidence-ppo.md`, `evidence-v2.md`). Tree kg-v3-int `main` 3e89425. It is not Codex-verified. Nothing in it has been run, and no repository code has been changed. Every earlier run is a single seed. The only replicate pair (ablations D and J) ended 16.6k apart, so no single-seed difference under about 17k can be ranked.

Constraints kept throughout:
- Self-play only. Fixed opponents (BC best, cha22) appear only as evaluation panels and labels, never as a training mix.
- `scripts/run_ppo.py` stays the one trainer. Every change goes through its shared path.
- The policy stays stateless: current-observation tokens only, with no opponent identity anywhere.
- No adaptive anchor.
- The reward is the owner's decision. The current reward is term M: 0.5 x cash difference + 0.5 x terminal sign. All runs below keep term M unless the owner decides otherwise (section 5).

Plain summary: none of the three designs is likely to earn more than BC on its own; each critique scored that chance at 2/10. The evidence says two things. First, the optimizer moves the policy by a fixed amount each update, whatever the signal says, and every run has drifted downhill from BC. Second, term M in mirror self-play does not pay for both seats getting richer, and may score long-payback investment as a loss. So the plan (a) runs cheap diagnostics that tell these apart, then (b) runs one short self-play attribution run with a noise-only twin, then (c) picks a step-size fix, a credit fix or a reward question for the owner from what those show. More money for real most likely needs the drift controlled first, then either an objective that pays for absolute earnings or a better starting economy. Both of those are owner decisions.

## 0. What "earning for real" can mean in self-play

- **Score.** The final bank is the score. SELL from the shed is the only inflow, and unsold goods are worth 0.
- **Demand cap.** Demand is the town drain, and both seats share it. Over a game it is worth about 195-211k at base prices for both players combined. Wheat and eggs are effectively uncapped, and they glut.
- **Where BC stands.** BC self-play already makes about 152k joint (76k per seat), which is 72-78% of that bound.
- **cha22's number is inflated.** cha22's 129-179k comes against weak sellers, and part of it is demand BC leaves unclaimed.
- **Mirror ceiling (inferred).** A mirror twin's plausible ceiling is about 90-100k per seat, plus the wheat and egg glut. cha22 against cha22 has not been measured; diagnostic X1b measures it.
- **What in-run numbers can show.** In mirror play, own_bank_mean is exactly joint bank / 2 (`telemetry.py:55`), so in-run telemetry can only show whether both seats get richer together.

How success is judged:
- **Primary:** a paired, seat-swapped evaluation against the fixed BC best on at least 32 seeds (64 games). Report:
  - own bank against BC's own bank on the same seeds (about 76k);
  - the margin;
  - the win rate with its 95% CI, where the lower bound must be above 0.5;
  - legality, completion and runtime.
- **Secondary:** self-play joint bank, and the econ counters (units sold, deaths, ineffective commands, tile-days, animal-days).
- **Held-out label only:** a cha22 panel.

## 1. Mechanisms

### 1.1 Ranked, with confidence

| # | Mechanism | Error class | Confidence | Evidence |
|---|---|---|---|---|
| M1 | **The step size does not depend on the signal.** Muon orthogonalizes the gradient and AdamW is scale-free. Per-minibatch advantage normalization rescales the advantage to unit size. So every update moves the policy by an amount set by the LR, whether the advantage is signal or noise. Under term M in mirror play the advantage is mostly noise. | Decision (optimizer) | The step-size fact is **supported, high**. Its causal share of the slide is **untested, medium**. | approx_kl follows the LR (about 0.012 at 1e-4) across advantage std 0.029-0.39. J/2's advantage std fell 0.34 -> 0.06 at constant KL. vs-cha22 had the same KL as M. Lower LR slows the slide but does not stop it (5/5 full-LR arms collapsed; 4/4 runs at 1e-4 slid after iteration 150). Teacher KL grows without bound (M 4.9 -> 8.3) while EV is 0.92-0.98. v2 gained only under a tight trust region (LR 1e-6 to 3e-6, fixed KL, KL stop at 0.05). |
| M2 | **Game phases are synchronized and most updates contain no outcomes.** Games have a fixed length of 720, all 256 envs reset together, and there is no `truncation_prob`. So each iteration trains on one 64-step phase, and only about 1 iteration in 11 contains any game end. The day 0-2 investment phase is updated only from bootstrapped returns. | Decision (training) | Synchronization is **supported, high**: run M's log has 256-game blocks every 11-12 iterations. Its causal role is **untested, low-medium**. | End phase: raw advantage std 0.034 and EV 0.79, against 0.018 and 0.97 in other phases. v2 found both effects and fixed them (staggering cut KL volatility 1.98 -> 0.45; `--require-outcomes`). **Against:** approx_kl is flat across phases, and Muon momentum 0.95 already mixes about 20 steps. Staggering does not extend how far λ = 0.9 carries credit. |
| M3 | **Long-payback investments are lost in a directed way.** | Decision (credit or objective) | The pattern is **supported**. Its cause is **open** among three readings: (a) credit inversion, because λ 0.9 over a 64-step horizon credits about 10 turns while payback takes 100-300 turns; (b) term M correctly scores investment as a margin loss in a shared market; (c) drift removes the rarest, most precise chains first. | M@20M against BC on the same seeds: strawberry tile-days 200 -> 5, sheep 133 -> 23, and 7 of 8 seats never bought land. Cows and hands were kept. v2 measured (a): the credit sign was wrong in 16 of 16 buy-9 forks, and was fixed by λ = 1 over 256 turns. v2 also measured (b): cha22's expansion raised its own bank by 4.6k but cut the margin by 6.8k. |
| M4 | **Term M pays nothing for a shared rise in money,** and the zero-sum critic cannot represent a shared return. | Architecture and objective | **Supported structurally, high.** It limits "for real". It is **not the cause of the slide under M**, which has no common mode. | `_values` gives 2p−1 per seat, so V0 + V1 = 0 (`kaggriculture.py:1077-1084`, `ppo.py:848-856`). The one v2 mirror recipe that raised own bank was a non-zero-sum own-bank target with a per-seat critic (m17: 82.0k -> 86.8k). |
| M5 | **Value and teacher gradients change the policy through the shared trunk** (H-V). | Architecture | **Candidate, untested.** | `critic_value_hidden` is a slice of the trunk output. vf_coef is 2.0, and gradient norms of 13-15 are clipped to 10 in 66-100% of iterations; which loss dominates is unmeasured. Early rises came in 5 of 6 BC-critic runs and 0 of 8 fresh-critic arms. |
| M6 | **Execution losses.** Parallel unit frames, ineffective commands and upkeep deaths. | Execution and architecture | The gap is **supported**. Its role in the slide is **unattributed**. | BC wastes 565 commands per game (cha22: 65) and has 27 deaths per seat. M@20M: 996 ineffective commands, and harvests fell 248 -> 115. Over-demanded PLANT turns every PLANT of that crop into PASS (`lib.rs:1540-1565`). Self-play entropy **rose** (J/2 6.1 -> 8.0, hz4 7.4 -> 9.8). A pure random walk would not raise entropy, so this is a directed pattern that has not been explained. "Ineffective" also counts blocked moves and PICKUP/PLACE (`lib.rs:3069`), not just upkeep verbs. |

**Information is not implicated.** The current observation holds every economic lever, including `watered_today`, `fed_today`, `cared_today` and `fertilizer_available` (`observe.rs:365-379`). Nothing needs memory between turns.

**Working reading.** M1 is the engine of the drift. M2, M3 or M5 may decide what the drift removes first. M4 caps how far "earning" can be rewarded under term M. M6 is a baseline inefficiency that drift makes worse.

### 1.2 Refuted

- **Gradient clipping as the cause of fixed-size steps.** Under Muon the clip cannot change the step size.
- **Critic quality or critic range as the main bottleneck under M.** EV is 0.91-0.985. Critic warm-up is not a fix either: ablation arm C collapsed fastest.
- **Keeping the policy close to the teacher is enough by itself.** Ablation F held teacher KL at 0.22 and still fell.
- **An LR of 1e-4 alone keeps the economy over a long horizon.**
- **The vs-cha22 control had a near-zero per-step signal.** Its reward_margin_abs_mean was 0.0014-0.0024, above M's.
- **cha22 wins by hiring more labor, or BC loses by overspending.**
  - cha22's hands per day (9-10) are close to BC's (7-9).
  - v2's clone spent 5-6k less than cha22 and sold 32k less.
  - The gap is production and command effectiveness.
- **Net-worth potential shaping as a new learning signal.** At γ = 1 it telescopes to an offset the critic absorbs.
- **Death shaping or ineffective-command shaping alone** (v2 m6, m7).
- **"AdamW moves in proportion to evidence" (argued, not measured).** At low SNR both optimizers move about c·lr·(√T·noise + T·signal/noise) over T steps. O1's AdamW swap is mostly a hidden LR cut of about 15x.

### 1.3 Unattributed

- **The main causal question:** is the slide noise stepping (M1), a direction that points downhill (M3, M5), or phase synchronization (M2)? No noise-only control exists.
- **Why vs-cha22 slid faster than self-play at the same KL.** The candidates are the always-lost terminal, a margin penalty for cha22's own earning, or its higher gradient norm.
- **Which link of the chain breaks first.** Per-family action statistics were never logged.
- **How the ineffective-command rise (565 -> 996) divides** between drift and architecture collisions.
- **Whether the leaderboard #1 teacher earns much more than BC.** If it does, the gap is imitation fidelity, not RL.
- **Whether in-run bank telemetry understates a checkpoint's bank.** J/2's last window read 53k, but J/2 final earned 64-77k in evaluations.
- **Seed variance for every run.**

## 2. Diagnostics to run first, in order

Where they run: GPU items run on a pod GPU that no learner is using. CPU game runs go on the pod CPU by default. If run on the Mac, they are niced and bounded to about 15 minutes each, as the evidence-econ runs were. No training runs on the Mac. Scripts are working artifacts under `ops/earn-money-2026-09-30/`.

### X1. Economic triage on CPU (no repo code, about 45-60 min in total)

- **X1a. How much does the BC teacher earn?**
  - **Method.** Replay the leaderboard #1 team's actions from the BC corpus (`bc-best/shards-top1`) through the engine. Record the teacher's final bank, ineffective commands and deaths, and the opponent in each game.
  - **If the teacher earns well above BC's 62-80k with little waste:** the missing money is imitation fidelity under parallel frames (M6 architecture, and owner option O3). RL is not the first lever.
  - **If the teacher earns about 70-80k:** BC is already faithful, and more money must come from RL or from a different teacher.
- **X1b. The ceiling with two competent sellers.**
  - **Method.** cha22 against cha22, 4-8 games. Record the joint bank and per-seat banks.
  - **Output.** It sets the realistic self-play target. It is the denominator for "for real".
- **X1c. Classifying the ineffective commands.**
  - **Method.** Run BC best and M@20M (`checkpoint_00_020_004_864`) in self-play on 8 seeds each, using `step_with_market_metrics` with `A_VERB_INEFFECTIVE`/`A_SLOT_INEFFECTIVE` (`lib.rs:1928-1934`). Classify each ineffective command as one of:
    - predictable from the start-of-step observation;
    - a collision on the same target within the turn;
    - a PLANT-block PASS, counting the PLANTs lost;
    - a blocked move or PICKUP/PLACE.
  - **Also count** the harvests and sales lost next to each class.
  - **Readout.** Only a class that costs a material yield (for example several thousand in bank per game) earns capability work (F5).
- **X1d. Which link of the chain breaks first.**
  - **Method.** Play the checkpoints in training order on the same seeds: BC best, J/2 final, hz4@10M, M@10M and M@20M. Record the 32 econ counters, strawberry tile-days, animal-days, land purchases, deaths, harvests and units sold.
  - **Readout.** The first counter to move is where the chain breaks:
    - investment first (land, strawberry, sheep) points to M3;
    - upkeep deaths or command waste first points to M6 or undirected drift.

### X2. Instrumentation and diagnostic code (about 0.5-1 day, required before any run)

All of this code sits on the shared path. It changes no behavior by default. It needs tests, `just py-prepare` (plus `just rs-prepare` if a Rust counter is added), the mapped docs, and one cookbook adaptation record that inventories it.

- **C0 telemetry, per completed-game window, to nt-probe records and W&B.** The econ counters already in `terminal_metrics` `econ_0`/`econ_1` (`lib.rs:1118-1175`) cover:
  - `econ/sell_cash`, `sell_units`, `ineffective`, `harvest`, `water`, `feed`, `drought`, `starvation`, `unsold_end`, `seeds_unused_end`, `idle_hand_steps`, `floor_sale_units`.

  Also log:
  - `market_kind` shares (BUY_LAND, BUY_ANIMAL, BUY_SEED by item, SELL, HIRE) and `unit_kind` verb shares;
  - entropy per head;
  - the raw advantage mean, SE and count by chosen family, **stratified by game phase**;
  - per-loss gradient norms on the trunk (policy, value, teacher KL, teacher value);
  - `train/game_phase_hist`;
  - approx_kl by phase.
- **Null seam.** Add `rl.debug_advantage_null: none | game_sign`, default `none`.
  - With `game_sign`, each env-game's raw advantages (both seats) are multiplied by one random ±1 before normalization.
  - Returns, values and teacher targets are unchanged.
  - This keeps the within-game correlation and the opposite-sign structure between seats, but destroys action credit.
  - It is logged loudly as `diag/advantage_null`.
  - It is a diagnostic knob only: no identity, no change to the reward.
- **Stagger seam.** Add `rl.initial_stagger: bool`, default false.
  - Env i's **first** game is truncated at a step u_i drawn uniformly from {1..719}, bootstrapped by the critic. This reuses the stateless truncation path at `ppo.py:688-710`.
  - A truncated game publishes no metrics (`env.rs:491-506`), so the bank telemetry stays clean.
  - Later games run the full 720 steps. Each iteration then carries all phases and about 23 game ends.

### X3. Rollout set R and the gradient probe (pod GPU, about 30-60 min, no parameter updates)

- **Collect.** 13 consecutive rollout iterations (256 games) through the trainer's own collection path, from BC best with term M and no update. Repeat from M@20M. Iterations 1-12 cover one full game cycle. Hold out **whole games**, not an iteration: the last iteration is only early-game states.
- **Read out:**
  - (i) Per-loss gradient norms and the cosine between policy and value gradients on the trunk. If value or teacher terms dominate the clipped norm and the cosine is materially negative, **M5** is supported.
  - (ii) The mean pairwise cosine between minibatch policy gradients, per phase, at a 16-game step (today) and a 64-game step. Compare it with 30-50 per-game sign-flip nulls, and estimate B_simple (McCandlish).
    - If the 16-game cosines fall inside the null band in most phases, the step is noise-driven, which supports **M1**.
    - B_simple sets the batch for F1.
    - If even the full 256-game gradient is inside the null band in every phase, there is no signal to extract. Stop the optimizer route and take the objective question to the owner (O1).
  - (iii) Cross-phase interference. Take one optimizer step on phase-A rows and measure the KL change on held-out phase-B games. If it is large, **M2** is supported.
  - (iv) Compare unit_kind and market_kind cosines against the null, **per frame** (teacher KL is summed over entities, so raw shares mostly count frames).

### X4. Optimizer-response replay (pod GPU, about 1 h)

- **Method.** Replay R's iterations 1-12 through `_update_minibatch`/`_step_optimizer` from BC weights with a fresh optimizer. Run each arm with real advantages and with the per-game sign-flip null:
  - Muon 1e-4, 16 steps per iteration (today);
  - Muon 3e-5 and 1e-5;
  - Muon 1e-4 with `gradient_accumulation_steps` 4 (4 steps per iteration);
  - AdamW 5e-6 with 4 steps;
  - a policy-loss-zero arm.
- **Measure** KL(BC || updated) per head on the held-out games, and R_move = KL_real / KL_null.
- **Readout.**
  - The setting with the highest R_move at an acceptable per-update KL becomes F1's step rule.
  - An AdamW trunk is adopted only if its R_move clearly exceeds Muon's **at matched KL**. Otherwise the plan stays on Isaiah's Muon, by the recorded decision to resolve recipe differences toward Isaiah.
  - If the policy-loss-zero arm moves the policy about as much as the real arm, **M5** is supported.
  - This also calibrates `target_kl`. Banks rose at an approx_kl of 0.008-0.013, not 0.002-0.004, so O1's 0.005 would have capped the only good phase.
- **Limit.** After the first iteration the replay is slightly off-policy.

### X5. Paired investment forks (pod CPU, about 1-2 CPU-hours, parallel)

- **Determinism first.** Replaying one game from its seed and recorded actions must give identical banks at every turn.
- **Setup.** Pick 24 BC self-play states where BC made a long-payback purchase (BUY_LAND, BUY_ANIMAL sheep, BUY_SEED strawberry).
  - Fork each one: **buy now** against **the same purchase one day later**, and where it applies, buy-k against buy-k′. Buy-against-no-order is not used, because BC re-issues the order.
  - Continue both branches with BC on 4 sampling seeds each.
- **Measure.** Δ final own bank, Δ margin and Δ joint bank, each with a CI. Compare their signs with the λ = 0.9 GAE advantage and the λ = 1 Monte Carlo advantage from the BC critic on the same rows.
- **Outcomes:**

| Outcome | Reading | Next step |
|---|---|---|
| (i) Buying now raises **both** own bank and margin, but the λ 0.9 advantage is negative | Credit inversion, M3(a) | F3 |
| (ii) Buying now raises own bank but **not** the margin | Term M is anti-investment by construction, M3(b) | Owner question O1 |
| (iii) Buying now does not raise own bank under BC play | Not a credit problem | Execution or labor (F5), or BC's own play |

### X6. Return-variance decomposition (minutes, from R)

- **Method.** Split per-transition return variance into own-bank, opponent-bank and terminal-sign parts under term M and under each reward option in O1.
- **Use.** It predicts each option's action-attributable signal share before any GPU run. An option must beat M by at least 1.5x to be worth a run.

Order and cost: X1 (about 1 h CPU) and X2 (code) in parallel, then X3, then X4 (together about 2 h of pod GPU), then X5 and X6 (CPU). Wall clock is about 1.5-2 days, dominated by X2's code, tests and records.

## 3. The first self-play run: S1, an attribution triple

- **Question (from M1 and M2).** Under the owner's term M, is the slide from BC best driven by signal-blind noise steps? Does de-synchronizing game phases change it?
- **Why not a run meant to beat BC first.** No design survived critique above 2/10 on that axis. Every earlier run would count as the "control" of a blind experiment, and none of them had a noise twin.

### Inputs

- **Config:** `configs/kaggriculture_4rank_margin.yaml` with only these changes:
  - the 2-rank batch split from `configs/kaggriculture_2rank.yaml`: `env.n_envs` 128 and `rl.segments_per_minibatch` 8 per rank. Two arms then share the 4-GPU pod at the **same global batch**: 256 envs, 16 optimizer steps per iteration and 16,384 env steps per iteration;
  - it is committed as `configs/kaggriculture_2rank_margin.yaml`, with a header carrying provenance and the owner's quotes;
  - everything else is unchanged: reward term M, muon_lr 1e-4, adamw_lr 5e-6, the 1,000-step warm-up (about 62 iterations), target_kl null, teacher last_best at 0.005/0.005, and checkpoint_freq 10M.
- **Start:** `--load-model-weights /Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only`. This keeps the BC critic head and starts a fresh optimizer. The teacher is the **BC best**, not J/2 final.
  - This control has never been run: M started from J/2 final with its optimizer state.
- **Length:** `--max-env-steps 3276800` (200 iterations), ending in `checkpoint_final.pt`.
- **Arms**, 2 seeds each (env seed and torch seed), 6 runs as 3 concurrent pairs:
  - **A0, control.**
  - **A1, noise-null:** `-o rl.debug_advantage_null=game_sign`. The value and teacher losses stay live, so if it slides, the slide comes from step size plus trunk and teacher terms, not from reward content.
  - **A2, stagger:** `-o rl.initial_stagger=true`.
- **Cost.** 2-rank iteration time is unmeasured. The estimate is 6-9 s per iteration, so 20-30 min per run and about 1.5-2 h of pod time. Every run logs to W&B spoon/kg-v3 (netrc on the pod first) under one group with v3 identifiers.
- **Code path:** `run_ppo.py`, then the shared PPO update, then the null or stagger seam.

### Primary measurement

- **Offline paired evaluation** of each `checkpoint_final.pt` against BC best on pod CPU: 32 fixed seeds, seat-swapped, 64 games per checkpoint.
- **Report:** win rate with a Wilson CI, own bank, rival bank, margin, joint bank, the X1 counters, legality, completion and runtime.
- **In-run:** own and joint bank per completed-game window (about 11 iterations; the stagger arm reads every iteration), C0 counters, and teacher KL per head.
- **Limit:** this is selection evidence, not held-out qualification.

### Expected discriminating observations

| A0 control | A1 null | A2 stagger | Reading | Next |
|---|---|---|---|---|
| Slides | Slides by the same amount (within the two-seed spread) | any | **M1 dominates.** Update direction does not matter at this LR. | F1 |
| Slides | Holds near BC, with slower teacher-KL growth | any | **The signal points downhill** (M3 or M4 direction; M5 if X3 flagged it). | Use X5 to choose F3 or O1 |
| Slides | any | Holds, and both seeds beat both A0 seeds | **M2** is material. | F2, combined with whichever of the first two rows applies |
| Holds | Holds | Holds | Inconclusive at 200 iterations. Earlier runs slid mostly after iteration 100-150. | Extend A0 and A1 once to 400 iterations, then stop |
| Rises and beats BC in the paired eval | Flat | any | Real signal is working, a first. | F6 |

- **Loss conditions** (stop an arm): a legality or completion failure; window own bank below 30k (M's floor, nothing more to learn); teacher/kl above 8 nats.
- **Stopping condition.** 200 iterations and then the offline evaluation. There is no automatic extension beyond the one 400-iteration rule above.
- **Receipt.** `ops/earn-money-2026-09-30/s1/`: config hash, tree SHA, W&B ids, checkpoint paths and evaluation JSON. Weights stay out of git.

## 4. Follow-up runs, each conditional on an outcome

- **F1: step-size control (S1 row 1, or X3 shows a noise-driven step).**
  - **Regime.** v2's regime is the only precedent for self-play gains in this game. Apply the X4 winner under Muon: a lower LR and/or `gradient_accumulation_steps` sized to X3's B_simple, plus `target_kl` set from X4's measured KL.
  - **If it still drifts:** first add a scale floor for advantage normalization (v2's global lagged RMS with no centering, floor from X3's null distribution). After that, and only if the owner allows it (O2), a fixed BC teacher-KL coefficient.
  - **Design.** Each arm runs with a null twin, 2 seeds and 400 iterations.
  - **Pass:** the paired evaluation against BC shows no bank loss beyond the CI, and teacher KL stays bounded.
  - **Next:** a 10M-step run, F6.
- **F2: stagger (S1 row 3).** Make `initial_stagger: true` the default in the Kaggriculture presets (cookbook record), and combine it with F1 or F3.
- **F3: credit horizon (X5 outcome i).**
  - **Change.** `gae_lambda` 0.97-1.0 with `horizon` 128-256. Scale `n_envs` down so env steps per iteration stay equal, after a memory check.
  - **Pairing.** Combine it with F1's larger batch per step, because variance rises.
  - **Arms.** A null twin, 2 seeds.
  - **Check.** Purchase-row advantages (C0) must turn non-negative, and strawberry, sheep and land days must hold at 80% or more of BC.
- **F4: objective (X5 outcome ii, or S1 row 2 without credit inversion).** Term M cannot reward "for real" earnings, so take O1 to the owner. If the owner picks an own-bank term:
  - build the per-seat critic offset head. It is zero-initialized so it matches BC exactly at step 0, and it is behind `model.critic_offset`;
  - loader rule: only the `critic_offset_head.*` keys may be missing;
  - the first arm stops the head's gradient from reaching the trunk, to keep M5 out of the comparison;
  - measure the mean δ on purchase rows against the control, not just `ev_common`.
- **F5: capability (X1a teacher much richer than BC, or X1c shows a class that costs yield).**
  - **Cheap first:** a deterministic PLANT over-demand resolver in the adapter (only the excess PLANTs become PASS).
  - **Costly, only if X1c justifies it:** effect-aware unit masks. They need a new per-row mask input carried through Rust buffers, the rollout, the PPO ratio, teacher precompute, BC evaluation and the Kaggle agent, with a parity test. That is days of work.
  - **Two masks are probably unsound:** SELL capped at the start-of-step stock, and FEED with no WHEAT held, because market orders resolve after unit actions.
  - **Evaluate** masked checkpoints against masked BC.
- **F6: the "earn more" run.**
  - **When:** only after a configuration holds BC's economy against its null twin.
  - **Run:** 10M env steps on 4 ranks from BC best, in self-play.
  - **Evaluations at 10M:** paired against BC best, plus a held-out cha22 panel (label only).
  - **Promotion:** at 0.7 or more, as today.
  - **Target:** own bank and joint bank toward X1b's ceiling.

When to stop and record a limit instead of spending more:
- **Signal limit.** X3 finds the full-batch gradient inside the null band in every phase. Under term M at this batch there is no signal; that is the objective question (O1).
- **Signal repaired but not enough.** F1 holds BC but no arm beats BC in two consecutive 10M evaluations while joint bank stays flat. Mirror term M does not pay for money; take O1 and O3 to the owner with the measured numbers.
- **Execution, not credit.** X5 outcome iii plus X1c showing small losses. The money is in BC's play or its data (O3), not in RL.

## 5. Decisions for the owner

- **O1. What the reward pays for.** The owner owns the reward; decide after X5 and X6.
  - (a) Keep term M. Earnings are measured but only relative play is rewarded. This fits if X5 shows investment raises the margin.
  - (b) Add a non-zero-sum own-bank term with the per-seat critic offset head. For example: own-bank weight 0.5, scale 250k, cap 0.5; margin weight 0.25, cap 0.25; terminal 0.25. These weights are agent-proposed and not measured. It is v2 m17's idea, which raised mirror own bank 82.0k -> 86.8k. Term A's earlier coefficients were never owner-confirmed; the owner approved the term, not the numbers.
  - (c) A book-value margin potential: `seat_wealth` (`lib.rs:1417`) during the game and the bank at the end. The whole-game return equals term M exactly; only the timing moves, so a purchase stops being an immediate loss. It is still zero-sum.
  - **Recommendation:** (c) if X5 shows credit inversion under M; (b) if X5 shows investment raises own bank but not the margin; (a) otherwise.
- **O2. A fixed BC teacher-KL coefficient** (`rl.teacher_mode: fixed`, `rl.teacher_init` set to the BC best, `teacher_kl_coef` 0.05-0.1, never adapted).
  - **Question:** is this the "adaptive anchor" you ruled out? It is only needed if F1's step-size fixes alone still drift.
  - **Evidence:** ablation arm A kept 41k against 0.1k for the control.
  - **Risk:** it may also cap how far the policy can move toward a cha22-level economy.
  - **Recommendation:** allow it as fixed, and only as F1's last rung.
- **O3. The starting economy.**
  - **Option:** BC on cha22 trajectories, or cha22 as the teacher. It is still self-play, not an opponent mix.
  - **Why:** cha22 earns 2-3x BC with 65 ineffective commands.
  - **Conflicts:** it conflicts with "BC imitates one player (leaderboard #1)".
  - **Caveats:** v2's cha22 clone still lost 26-27k against cha22. cha22's number is inflated by weak rivals, and X1b measures by how much.
  - **Recommendation:** decide after X1a and X1b. If leaderboard #1 itself earns about BC's 70-80k and cha22 against cha22 is well above that, this is the most direct lever for money for real.

Not owner decisions, noted here for transparency:
- The optimizer stays Isaiah's Muon unless X4 measures a clear advantage. That is recorded, not offered as a menu.
- CPU diagnostics run on the pod CPU, or bounded and niced locally.
- Every seam and config above gets a cookbook adaptation record before it is used.
