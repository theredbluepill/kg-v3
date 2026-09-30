# Critique: design-signal.md (signal-first self-play), 2026-09-30

Adversarial review by a Claude subagent. It is not Codex-verified and nothing was run. Tree: `/Users/poonszesen/kg-v3-int` @ 3e89425. I re-read `src/kaggriculture/reward.rs` (all 209 lines), `python/owl/model/kaggriculture.py:329-366, 1077-1120`, `python/owl/train/ppo.py:152-210, 800-870, 1460-1560`, `configs/kaggriculture_4rank_bc_finetune_bank.yaml`, `engine_rs/src/lib.rs` (episode_steps 720, turns_per_day 24, `seat_wealth` at 1417), and `evidence-v2.md:50-62, 80-110`.

## Verdict

H-A is arithmetically correct, and C2 (a per-seat critic offset) is the one change a non-zero-sum "earn money" reward needs in self-play. The design is wrong to make C2 plus A2's reward the first run. H-A cannot explain the slide in J/2, hz4, M or vs-cha22, because their rewards have no common mode at all. The design also misses a better-supported mechanism: value-loss interference through the shared trunk. The first run is single-seed against single-seed, with a pass bar set at the noise floor, and it adds a trunk confound. Run the gates first: Dx4 extended to three arms, and Dx2. Keep C1 and C2 as code, but do not launch Run 1 until the step-size and trunk questions are answered.

## 1. Mechanism claims checked against the code

**Confirmed.**
- `_values` returns `2·exp(logp_self) − 1` over a two-token softmax, so V_0 + V_1 = 0 (`kaggriculture.py:1077-1084`).
- A2's arithmetic holds. The config has w_b 0.25, S 100k, cap 0.25, econ_shaping 0.2 with econ_cap 0.25, and a terminal scale of 0.5. So −0.0025 per 1k is right, and 0.9^100 = 2.7e-5 is right.
- E1 passes `validate`: the active caps sum to 0.75, below 1, and its terminal scale is 0.25.
- `seat_wealth` exists (`lib.rs:1417`), so E2 is feasible.
- The engine is deterministic from config, seed and actions (`lib.rs:6`), so Dx2's prefix replay is feasible.
- `truncation_prob` already exists in `ppo.py:195`. De-synchronizing phases needs only config.

**H-A is overstated for purchases.** The purchase penalty is −w_b·c/S to the buyer only. Split into modes, half is common and half is antisymmetric. The zero-sum critic can baseline the antisymmetric half. So only about half the immediate penalty is structurally unbaselined, not all of it.

**The magnitudes are tiny.**
- A2's common-mode return is 0.011-0.015 per 64-turn segment. That puts the per-turn common-mode advantage at about 1e-3 to 3e-3.
- The zero-sum advantage std is 0.03-0.26.
- H-A is therefore a small directional bias on purchase turns, about −0.01 for a 4k purchase. It is not the dominant signal.
- It is plausible that H-A bent A2 toward selling. It cannot explain why every reward slid.

**H-A covers no failed run except A and A2.**
- Under J/2 and hz4 (terminal sign plus relative death penalties), under M (margin plus sign plus relative penalties) and under vs-cha22, every reward term is odd in the seats (`reward.rs`: `delta[1-s] - delta[s]`, an odd `margin_score`, and the sign). The common mode is exactly zero.
- M lost the most investment (strawberry tile-days 200 to 5, sheep 133 to 23). That pattern cannot come from H-A.
- The design says so in section 1, but Run 1 still targets H-A. So Run 1 tests a repair for a secondary failure on the one reward family that already did best (A2 held 64-67k to iteration 124).

**Misquote of the v2 evidence.** The design says buy-9's "advantage was negative in 0 of 16 pairs". The source (`evidence-v2.md:52-54`) says A(9) − A(5) was −0.035 at λ 0.9, with the correct sign in 0 of 16 pairs. The advantage was *wrongly negative* in 16 of 16 pairs. The design's wording states the opposite of the evidence it relies on.

**Dx1(b) is misspecified.**
- A "per-seat Monte-Carlo mean" baseline is a constant. GAE with a constant baseline gives δ = r, exactly as with no baseline, so the myopic bias would *not* vanish. That test cannot tell H-A apart from anything.
- The correct oracle is either the λ = 1 (Monte-Carlo) common-mode return from each state, or an offline regression of future own-bank score on current-observation features.
- Also, Dx1(a) under A2 will show BUY_* below 0 almost by construction: an immediate −w_b·c/S inside a 10-turn window. That confirms arithmetic, not behavior. The informative quantity is the sign agreement between each family's advantage and the Dx2 fork outcome.

**Advantage-by-family attribution is confounded.**
- The advantage is one scalar per seat-turn (`ppo_clip_mode: per_player` sums entity log-probs).
- Purchases, plantings and sales co-occur in the same turn and cluster by hour of day and game phase.
- A raw per-family mean conflates family with phase. Dx1 needs stratification by day and hour, or a regression on family indicators with phase controls.

**Missing mechanism: value gradients through the shared trunk (H-V).**
- `critic_value_hidden` is a slice of the shared trunk output `x` (`kaggriculture.py:366`).
- `vf_coef` is 2.0, grad norms of 13-15 sit above the clip of 10 in 66-100% of iterations, and `evidence-v2.md:50` notes that which loss dominates the norm is unmeasured.
- The strongest factor in `evidence-runs.md` is that 0 of 8 fresh-critic arms rose, against 5 of 6 BC-critic arms. That is exactly what large value-loss gradients damaging actor features would produce.
- The design does not consider H-V at all. C2 adds a new, initially wrong value target (the common mode) that sends fresh value gradients into the same trunk. It could worsen the drift it is meant to cure.

**Synchronized phases get only a footnote.**
- With 256 envs starting together, 720-turn games and a 64-turn horizon, all 16 optimizer steps of an iteration see the same 64-turn slice of the game, with Muon momentum 0.95.
- That is a directional, phase-cycling push, not isotropic noise. It is the most plausible reason why "drift" removes specific behaviors, such as the day-0 to day-3 investment phase.
- v2 marked it S (`evidence-v2.md` section 2, item 4). The fix is config-only (`rl.truncation_prob` with `truncation_step`), yet the design puts it under "also check".

**Calibration understates how little headroom there is.**
- BC self-play is already at about 152k joint, against a joint demand bound of 195-211k at base prices. That is 72-78% of the bound before the wheat and egg glut.
- Under E1, going from 76k to 100k own bank is worth 0.5 × 24k / 250k = 0.048 of return, against ±0.25 terminal and ±0.25 margin.
- Even E1 leaves the "earn more" signal at about a tenth of the coin-flip terms. Dx3 is the right check, and it should run before C2 is built, not after.

## 2. Constraint check

- **Stateless and no opponent identity:** met. C2 reads only the seat's own current-observation critic token, and holds no state or identity.
- **One trainer:** met (C1, C2 and Dx4 all sit on the `run_ppo.py` and `ppo.py` shared path).
- **No adaptive anchor:** met. The existing fixed `teacher_kl_coef` of 0.005 is the unchanged recipe.
- **Owner's reward authority:** partly met.
  - Run 1 uses A2's reward, not the latest owner reward M, and the design flags that.
  - The design misstates the provenance, though. The A2 config header says "the owner gave none" for the values, and "the owner has not confirmed the numbers" (w_b 0.25 was the implementer's correction).
  - "Owner-approved term A" is true of the term, not of E0's coefficients. The owner question must say so.
  - E1 is correctly marked [owner].
- **Checkpoint loading:** the new loader rule (allow exactly the missing `critic_offset_head.*` keys) is a new mode beside the existing `model_only` and `model_fresh_critic_head` (`run_ppo.py:145`). It needs its own test and a cookbook adaptation record. C1's tile-day counters are a Rust engine change, so they also need `just rs-prepare`, the docs mapping and a record. The design lists only the Python side.

## 3. Why Run 1 would not discriminate

1. **Seed noise equals the pass margin.**
   - The pass bar (at least 68k over iterations 150-250, against A2's 51-52k) is 16-17k above A2. The seed-noise floor from ablations D and J is 16.6k, and the comparison is one seed against one seed.
   - A pass is weak evidence and a fail is uninformative.
   - Open question 2 adds that in-run bank telemetry may understate a checkpoint's bank.
2. **C2 also changes the trunk.**
   - C2 changes the gradient mix into the shared trunk (H-V) and the share of the clipped norm (clip 10, norms 13-15) left for the policy.
   - A difference against A2 could therefore come from the altered effective policy step, not from common-mode credit.
   - Log per-loss gradient norms, or run with the offset head detached from the trunk (stop-grad on `critic_value_hidden` into `critic_offset_head`), so the credit effect is isolated.
3. **The engagement gate is too easy.** `ev_common > 0.3` can be met by fitting the phase-of-game trend of the common mode (bank grows with time) without valuing assets. It does not show that o(s') − o(s) credits purchases. Measure the purchase-transition δ directly: the mean δ on BUY_* turns under C2, against A2's.
4. **H-N can mask everything.** If the drift is step-size-driven (the optimizer angle's A), a correct signal still slides at iteration 150+. The design's abandonment rule 4 covers this, but only after about 300 iterations of 4-rank time.

## 4. Cheaper ways to get the same answers

- **A per-loss gradient-norm probe (minutes, no training).** Take the BC best, run one rollout on the pod and take one backward per loss term: policy, value, teacher KL, teacher value. Record ‖g‖ and the cosine between the value and policy gradients on the trunk. This settles "which term dominates the clipped norm" and tests H-V directly. Add it to C1.
- **A three-arm Dx4 instead of one arm.** Stop each arm at iteration 150:
  - (i) shuffled advantages, as the design proposes;
  - (ii) the policy loss off (value plus teacher only), which tests H-V;
  - (iii) the real J/2 recipe with `truncation_prob` de-synchronizing phases.

  Arms (i) and (ii) are two debug keys, and arm (iii) is config only. Together they separate noise steps, trunk interference and phase-coherent drift before any reward or critic change. Arm (i) alone cannot tell arm (ii)'s effect from its own, because shuffled advantages keep the value loss.
- **Dx2 before Dx1.** The fork test is the ground truth, and the engine is deterministic.
  - Use v2's paired contrast (buy-k against buy-k′, or buy now against buy next day), not "purchase against no order". BC will often re-issue the order next turn and wash the effect out.
  - Budget: about 192 full-game continuations at 720 turns each. That is likely well above 15 minutes of CPU time, so run it niced on the pod CPU, not on the Mac.
- **Dx3 before building C2.** If E1's action-attributable variance share is not at least 1.5 times M's, C2 plus E1 is not worth building for a run yet.
- **H-C under the owner's current reward M needs no new head.** C4 (λ 0.97 to 1, horizon 128-256) plus phase de-synchronization is config only, and it is v2's measured repair. If Dx2 shows purchase-credit inversion under M, that run respects the owner's latest reward without asking for a reward change.

## 5. Most likely way it fails

C2 is built, engages (`ev_common` above 0.3), and the own bank still follows A2's path down after about iteration 124. The reason is that the slide is driven by step size and phase-coherent updates, possibly amplified by value gradients in the shared trunk. The common-mode credit is 1-10% of the advantage, far too small to steer against that. The run ends at about 300 iterations, single seed, with an ambiguous 5-15k difference against A2 that the noise floor cannot rank. Secondary risk: the new offset target's value gradients speed up the drift early, as the fresh-critic arms did.

## 6. Scores (1-10)

| | Score | Reason |
|---|---|---|
| (a) Probability that the design makes self-play PPO earn more than BC | **2** | The design fixes a secondary bias that exists only under own-bank rewards. It leaves the dominant slide (seen under every reward) to other angles. Self-play headroom above 76k is small (about 72-78% of the joint demand bound is already used), and E1's earning signal is about a tenth of the coin-flip terms. |
| (b) Information value of Run 1 | **3** | Single seed against single seed with a pass bar at the noise floor, a trunk-gradient confound and an easy engagement gate. The diagnostics are worth much more: Dx2 about 8, Dx4 extended to three arms about 8, Dx3 about 6, and Dx1 about 4 until it is fixed (non-constant oracle, phase stratification). |
| (c) Cost (10 = most expensive) | **6** | C1 includes new Rust counters, docs, tests and records. C2 needs model, loader, test and docs work. Four CPU diagnostics, one pod probe and a 4-rank run of about 300 iterations, plus an owner round-trip for the reward. |

## 7. Recommended re-ordering

1. Add C1 telemetry, plus per-loss gradient norms and the purchase-transition δ.
2. Run the gradient-norm probe (H-V).
3. Run Dx4 as three arms (shuffle, policy-loss-off, de-synchronized phases), 150 iterations each.
4. Run Dx2 (paired forks) and Dx3.
5. Only then choose:
   - C4 under M, if Dx2 shows inversion and the step size is handled;
   - C2 plus own-bank, if the owner wants absolute earning and Dx3 shows a usable signal share. In that case, detach the offset head from the trunk in the first arm.

Take the owner's decision on E0 or E1 coefficients with the correct provenance: the term was approved, the numbers were not.
