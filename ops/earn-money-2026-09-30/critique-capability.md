# Critique: design-capability.md (adversarial)

Date: 2026-09-30. Tree: kg-v3-int `main` 3e89425. Reviewer: a Claude critique subagent, not Codex-verified. It checks the design against the code and one run log. Nothing was launched and no repository code was changed.

## Verdict

- **Scores:**
  - (a) Probability that self-play PPO earns more than BC: **2/10**.
  - (b) Information value of the first run as written: **4/10**. With the fixes in section 5 it would be about 7/10.
  - (c) Cost: **4/10**, where 10 is the most expensive. C0+C1 plus about 40 minutes of pod time is cheap. C2 costs far more than the design says (section 2, point 4).
- **Synchronization is real.** The phase-synchronization finding holds and is the design's best contribution.
- **But its role is overstated.** The design claims desynchronization fixes things it does not touch: investment credit, and whether outcomes reach investment steps.
- **The first run can mislead in either direction.** It tests synchronization and "there is any usable signal" together. It has one seed per arm, and its pass threshold equals the seed noise.
- **Even a pass stops short of the owner's question.** The best outcome is "the BC bank no longer slides". The design has no step that earns more than BC.

## 1. Claims I checked

| Claim | Check | Result |
|---|---|---|
| Fixed-length games, so no natural desynchronization | `engine_rs/src/lib.rs:704` sets `episode_steps` to 720. `lib.rs:1487` ends the game at `previous_step >= episode_steps - 2`, which is 719 action steps. Nothing in the game ends it early. | Confirmed. The design's 720 is really 719, which does not matter. |
| All envs end together | M log (`kg-v3-runs/M-margin-J2-4rank-20260930/*.log`): 520 records show `train/bank_games: 256.0` and 5,356 show `0.0`. Games only ever end in blocks of 256, about 1 record in 11.3. | Confirmed. |
| Kaggriculture configs set no `truncation_prob` | `configs/kaggriculture_4rank_margin.yaml` | Confirmed. |
| The truncation path already exists and is stateless-only | `ppo.py:688-710` and `1175-1233`. `_truncation_keeps_transition_reward` is True for Kaggriculture (`ppo.py:3044`). | Confirmed. The truncate branch in `env.rs:491-506` publishes no transition and no metrics, so the cut first game does not pollute bank telemetry. |
| The critic is zero-sum and MSE on a 2p−1 value | `ppo.py:637-645` rejects `winner_ce` for Kaggriculture. `ppo.py:848-856` shows the two seats' values sum to zero. | Confirmed. |
| PLANT over-demand blocks the whole crop | `lib.rs:1540-1565`: a crop whose PLANT demand exceeds the seeds held has every one of its PLANTs turned into PASS. | Confirmed. |
| Unit actions resolve before market orders | `lib.rs:1455-1463`: `apply_player_actions`, then `process_market`, then `town_consume`. | Confirmed. |
| The observation holds the upkeep state | `src/kaggriculture/observe.rs:365-379, 523-578`: `watered_today`, `fed_today`, `cared_today`, `fertilizer_available`, `consecutive_un*`. | Confirmed. The masks in C2 can be computed from the current observation. |
| "Ineffective" means upkeep verbs | `lib.rs:3069-3077`: `ineffective_unit_commands` is submitted minus committed over **move, production and logistics** actions. | **Wrong framing.** The 565 per game includes blocked moves and PICKUP/PLACE. The C2 upkeep masks address only part of it, and nobody has measured which part. |
| C2 is "mirrored in gpu_grammar's tables" | `grammar.rs:598-660` (`grammar_tables`) derives masks from fixed prefixes. `gpu_grammar.py` holds static `GrammarTables` with a digest. No mask depends on the observation. | **Wrong.** C2 is a new mask input that varies per row (section 2, point 4). |
| The phase table (adv std, EV, KL by phase) | Not re-derived. Only the game-end cadence was spot-checked. | Unverified by me. |

## 2. Wrong or unsupported mechanism claims

1. **"Every update carries terminal outcomes" is not an investment-credit fix.**
   - GAE with λ = 0.9 and horizon 64 puts the terminal sign directly into advantages only within about 10–30 steps of game end. Stagger does not change that.
   - Day 0–2 investment frames still get their credit through the critic bootstrap, with or without stagger.
   - Phase-0 states are about 1/11 of all rows either way. Stagger changes which rows share a minibatch. It does not change how often each phase is trained or how its credit is computed.
   - So "investment decisions are updated only in iterations that contain no outcomes" is true but irrelevant to D-D. C1 targets cross-phase interference only.

2. **"Nothing in the batch holds the other phases in place" is overstated.**
   - Muon momentum is 0.95, which is about 20 steps of memory, against 16 optimizer steps per iteration. Each step therefore already mixes about 1.25 iterations, which is two adjacent phases.
   - Teacher KL (0.005) acts on every row.
   - Neither fact refutes interference. They do weaken the "one phase per step" picture.

3. **Under the design's own drift hypothesis, stagger should do little.**
   - If the advantage is mostly noise (raw std about 0.02, section 0), a minibatch that mixes phases is still noise. Muon's orthogonalized step is still full-size.
   - Stagger helps only if each phase carries real signal that another phase's step erodes. That is a "signal plus interference" hypothesis, and the design never states it.
   - A null result is therefore ambiguous. It could mean "synchronization is irrelevant" or "there was no signal to protect".
   - Section 5 reads a null result as "pure LR-noise stepping". That conclusion needs a noise-null arm, and the design has none.

4. **C2 costs much more than the design says, and "no retraining needed" hides consistency work.**
   - Each env and actor needs a boolean mask per step. It must be computed in Rust, carried through a new caller-owned buffer, and stored in the rollout, because the PPO ratio must recompute log-probs under the same mask.
   - It must also reach the teacher precompute, so teacher KL is taken on the masked distribution; BC eval; and the Kaggle `owl.rs` agent, with a JSON-observation parity test.
   - This is days of work. It touches the data pipeline that CLAUDE.md requires to stay canonical, and it needs its own throughput measurement.

5. **One C2 rule is probably unsound as written.**
   - SELL is a market order, and market orders resolve **after** unit actions (`lib.rs:1457-1461`).
   - A PLACE into the shed in the same step may raise the stock that SELL draws on. A mask of "SELL of more units than are held" at the start of the step would then remove sells the engine would commit.
   - The required mask-never-removes-a-committed-action test would catch this only if the traces contain PLACE and SELL in the same step. Check the order first.
   - FEED "with no WHEAT held" has the same risk if another unit acts earlier in the same turn.

6. **"Wasted commands displace upkeep" (D-B) is unsupported.**
   - The design's own evidence says labor is nearly free, at about 13–15 hands a day.
   - A plant dies because no unit went to that tile, not because a unit watered an already-watered tile.
   - Masking WATER on a watered tile does not route a hand to a dry one. A mask improves how probability is spent on the unit's current tile only. It does not improve coverage.
   - The 27 deaths per seat may be a routing and coverage failure, which parallel unit frames cannot coordinate (D-C). They may not be an effect-mask failure at all.

7. **The claim that the control will slide 17k or more is extrapolated.**
   - Term M from BC best, at this learning rate, has never been run. M started from J/2 final.
   - The runs cited slid 10–33k under different rewards and starts. J/2 itself slid about 10–15k in-run.
   - A control slide of about 10–15k is well within what the evidence predicts.

## 3. Confounds that make the first run non-discriminating

- **Low power.**
  - One seed per arm, against a replicate gap of 16.6k.
  - The loss rule is "falls as far as the control (within 17k)". If the control slides 12k and stagger holds at 0, the run is scored as a failure. That is the most likely false negative.
  - If stagger holds and the control also holds, as A2 held through iteration 124, the run is also uninformative.
- **Joint test.** Synchronization and signal existence are tested together, and there is no noise-null arm (section 2, point 3).
- **Warm-up overlap.**
  - LR warm-up is 1000 optimizer steps, which is about 62 iterations. Every earlier "rise" happened inside that window.
  - During the stagger arm's first 11–23 iterations, game 1 is cut with a critic bootstrap. That bootstrap lies in (−1, 1), while the term-M return-to-go spans ±1.5 (config header), so it is biased.
  - The stagger arm also reports no bank until its first full games end.
  - Early windows are therefore not comparable between arms. Compare from about iteration 60 on.
- **Masked start.**
  - If D-mask passes, both arms start from masked BC. That is fine within the pair.
  - The offline evaluation "against BC best" must then use **masked** BC best. Otherwise the gain from the masks is credited to PPO.
- **In-run bank as the endpoint.**
  - evidence-runs "Open" already reports that the in-run bank understates a checkpoint's steady bank: J/2's last interval read 53k, yet it earned 64–77k as last_best.
  - The primary endpoint should be a paired offline evaluation on fixed seeds.
- **D-mask sample size.**
  - 8 games give 8 independent pairs, not 16. The two seats share a market, so their banks are correlated.
  - The paired SD is unknown, so the 5k gain threshold may sit inside the noise.
  - CPU time is cheap. Use 32 or more seeds and report the paired SD and a CI.

## 4. Constraint check

| Constraint | Status |
|---|---|
| Stateless policy | OK. The masks in C2 and the resolver in C3 use only the current observation and the within-turn sampled action. C1's per-env cut step is trainer state, not policy state. |
| No opponent identity | OK. Everything is self-play, and fixed bots appear only in offline panels. |
| No adaptive anchor | OK in run 1. Section 5 mentions a "fixed BC-KL anchor 0.05–0.1" and correctly defers it to the owner. It is a change to the existing `teacher_kl_coef`, so present it to the owner as that, not as a new "anchor". |
| One trainer | OK. All seams are in `run_ppo.py`, `ppo.py` and the Rust adapter. |
| Owner's reward authority | OK. Term M is unchanged, and R1 is marked as the owner's decision. The R1 algebra checks out: a symmetric reset gives Φ_0 = 0, and at the terminal step Φ switches to bank. |
| Train and Kaggle parity | Risk. C2 and C3 change what gets executed. They must ship in `owl.rs` and need a parity test against the Kaggle observation JSON. The design names the shipping but not the parity test. |
| A cookbook record per adaptation | Stated in Limits; not planned per change. |

## 5. Cheaper or better ways to get the same answer

1. **Run the CPU diagnostics first. They need no code and gate everything else.**
   - D-bc: does the teacher earn well above BC?
   - D-ceiling: cha22 against cha22.
   - D-cap: classify the ineffective commands, **including moves and logistics**.
   - Together these take about 30 minutes on CPU. They decide whether the money is behind imitation fidelity or coordination (D-C) or behind RL. C1 and C2 cannot settle that.
2. **Add a noise-null twin** (advantages shuffled within the batch) to the first pod run.
   - It is a flag of a few lines, and both sibling designs already propose it.
   - It is the cheapest way to tell "the reward carries usable signal" from "the slide is the step size alone".
   - If the noise-null slides like the control, neither stagger nor any reward change matters until the size of the step depends on the signal.
3. **Test the interference claim offline before building C1.**
   - From one checkpoint, take one optimizer step on phase-A rows and measure the change in the policy loss or KL on phase-B rows.
   - Or measure the gradient cosine across phases. That takes minutes of GPU time.
   - If cross-phase interference is negligible, drop C1.
4. **Change the first-run design.**
   - Run three arms: control, stagger and noise-null, with 2 seeds each and 150–200 iterations. That is about 6 × 12 minutes.
   - Primary endpoint: a paired offline evaluation of the final checkpoint against its start, on at least 32 fixed seeds with seats swapped (64 games). Report own bank, margin and win rate.
   - Keep the in-run window bank as secondary.
   - This costs about as much as the design's 2 × 300 iterations and gives far more.
5. **Defer C2** until D-cap shows that no-effect commands predictable from the observation make up a large share, **and** that removing them in a replay counterfactual would have raised harvests or sales.
   - If deaths are a coverage failure, C2 will not help. The lever is then coordinated unit frames (D-C), whose cost to throughput has to be measured.
6. **Name an owner-level option the design leaves out.**
   - cha22 earns 2–3× BC, with near-zero waste.
   - Changing the BC warm-start data or the teacher to cha22 trajectories, in self-play with no opponent identity, is a data change and not an opponent mix.
   - It contradicts the owner's recorded "BC imitates one player (leaderboard #1)", so it goes to the owner as an option. It is not the agent's choice. It is the most direct route to "earn money for real" that the evidence supports.

## 6. The most likely way it fails

- **What happens:** C0 and C1 are built, and both 300-iteration arms slide by similar amounts, or by amounts that differ by less than 17k.
- **What gets concluded:** section 5 then says "synchronization refuted, pure noise stepping", with no noise-null to support it.
- **What it costs:** a day of code and 40 minutes of pod time, spent learning what a noise-null twin would have shown directly.
- **Second most likely failure:** C2 is built at a cost of days. The D-mask gain comes out under 5k, because deaths are about coverage and not wasted verbs. BC stays at about 70k.
- **Even the best case falls short:** stagger holds the bank at start. That still does not make the agent earn more than BC. The design's path beyond holding is a 1,500-iteration extension with no new mechanism for rising.

## 7. What to keep

- The phase-synchronization finding, which I confirmed from code and the M log. Keep C1 as a cheap stage-2 lever, backed by the existing `truncate_envs` path.
- C0 telemetry, which every design needs: econ counters, `joint_bank_mean`, action-family shares and raw advantage by family.
- D-bc, D-cap, D-ceiling and D-adv as diagnostics that run first.
- The PLANT resolver in C3 as a cheap, deterministic fix to use if D-cap shows PLANT blocks cost yield. The engine behavior it fixes is confirmed at `lib.rs:1540-1565`.
