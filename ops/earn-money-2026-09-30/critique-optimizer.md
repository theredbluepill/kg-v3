# Adversarial critique: design-optimizer.md ("movement proportional to evidence")

Status: Claude-authored critique. It is not Codex-verified. It is read-only: nothing was launched and no code was edited.
Tree: `/Users/poonszesen/kg-v3-int` @ 3e89425.
Code re-read for this critique:
- `python/owl/train/optimizer.py:207-262`
- `python/owl/train/ppo.py:140-205, 1313-1378, 1466-1485, 1636-1661, 3231-3277, 3444-3455, 3616-3644`
- `python/owl/kaggriculture/telemetry.py:36-70`
- `scripts/run_ppo.py:1822-1835`
- `configs/kaggriculture_4rank_margin.yaml`
- `cookbook/decisions/recipe-choices-align-to-isaiah-without-owner-escalation.md`

Evidence tables were read from `evidence-runs.md`, `evidence-ppo.md` and `evidence-v2.md`.

## Verdict in one paragraph

The code facts in the design are correct. Muon runs on every 2-D non-I/O matrix. Advantages are normalized per minibatch and all-reduced across ranks. `target_kl` is checked per minibatch against a rank-reduced approx_kl and breaks only at a window boundary. Teacher KL is summed over entities.

The central mechanism claim does not hold in the regime the design itself describes. In that regime the per-step gradient is mostly noise. There, AdamW is not more signal-sensitive than Muon: under idealized assumptions both accumulate displacement ≈ c·lr·(√T·noise + T·μ/σ) and differ only in the constant c. So O1 is mainly a disguised learning-rate cut of about 15x, plus a batch reshuffle that does not change SNR per sample. The prior evidence already says a lower LR slows the slide. The O1 pair will most likely end flat (outcome 2), or "holding" in a way that cannot separate fixed from slowed.

Three further problems:
- Its success metric (own/joint bank in self-play) is exactly the common mode that term M assigns zero reward to.
- Its target_kl value is miscalibrated against the logs.
- Its 64-game eval threshold of 0.55 does not discriminate.

The null-twin idea, S2 telemetry and D0 are worth keeping. Of these, D0 gives most of the information at a small fraction of the cost.

## 1. Mechanism claims checked against code and logs

### 1.1 The claim that AdamW makes the step depend on the signal is wrong in the low-SNR regime (major)

The design's argument: |m|/√v ≈ 1 for a consistent signal and ≈ 0.23 for pure noise, while Muon's orthogonalization restores noise to full size. This holds only when the per-step signal dominates the per-step noise (μ ≳ σ per coordinate). In that regime there is no drift problem to solve.

The design's own premise is the opposite regime: advantage std about 0.02 and a "mostly noise" gradient. Write the signal as μ ≪ σ. Then:

- **AdamW.** √v ≈ σ, so the signal part of each step is lr·μ/σ, not lr. The noise part is lr·0.23 per step, but consecutive steps share momentum with a correlation time of (1+β1)/(1−β1) = 19. Over T steps, Σm ≈ Σg, so displacement ≈ lr·(√T·1 + T·μ/σ).
- **Muon** (β 0.95, nesterov). The momentum norm is about 0.16σ, which is rescaled to 1. The signal part is lr·μ/(0.16σ). The noise correlation time is 39. Displacement ≈ 6.2·lr·(√T + T·μ/σ).
- **Same form.** Both have the same signal-to-noise ratio of accumulated displacement, √T·μ/σ. Only the scale constant differs. Neither can "stand still": bias-corrected Adam keeps unit-scale steps under pure noise, because v tracks the noise.
- **The 4x batch per step.** It halves σ per step but takes 4x fewer steps, so over a fixed sample budget the ratio is unchanged: (T/4)·2μ/σ against √(T/4). A larger batch reduces total movement. It does not raise per-sample SNR.
- **Net scale.** Per iteration: Muon 1e-4 × 16 steps against AdamW 5e-6 × 4 steps. Both the signal and the noise displacement shrink about 15x. O1 ≈ today's recipe at a Muon LR of about 7e-6, with the same data-limited SNR.

What survives:
- AdamW does weight each coordinate by its own SNR. That helps only if the signal is concentrated in a few coordinates (for example head weights) while the trunk is noise. That is testable in D1's per-group readout, but the design does not state it as the mechanism.
- The real rationale for small steps is the curvature/diffusion picture. Noise costs loss in proportion to lr² times curvature, while signal gains in proportion to lr. So a smaller LR can turn net progress positive at a narrow BC optimum. That is an **LR** argument, not an optimizer-family argument.
- The design's "signal-to-noise contrast rises about 4x" and its movement-budget table are therefore unsupported. The table counts noise per iteration as 4 × 1.2e-6 and ignores momentum correlation, which multiplies AdamW's noise displacement by √19 ≈ 4.4 over multi-iteration horizons.

Consequence: a cheaper equivalent that stays aligned with Isaiah is `-o optimizer.muon_lr=7e-6` (or 1e-5): a one-line override, with no optimizer swap and no Decision deviation. D0 should include a "Muon at matched per-iteration KL" arm. If AdamW's R_move is about equal to matched-Muon's R_move, the optimizer swap buys nothing.

### 1.2 target_kl 0.005 contradicts the logs it cites (major)

The design says: "Banks rose only while approx_kl was 0.002-0.004 (warm-up)." `evidence-runs.md` table 1b says otherwise:

| Run | Window | Mean approx_kl | Bank |
|---|---|---|---|
| J/2 | iterations 21-63 | 0.0084 | peak 81.6k at iteration 45 |
| Ablations D and J (/10, BC critic) | iterations 21-46 | 0.0128 | D peaked at 83.8k at iteration 45 |
| A2 | iterations 21-63 | 0.0081 | rose to 79.1k at iteration 57 |

- Every observed rise happened at approx_kl of about 0.008-0.013. At 0.005 the cap sits below the only regime that ever produced a rise.
- If the cap binds, "outcome 2 → raise LR x2 once" does nothing: a higher LR only hits the same per-iteration KL cap sooner.
- **Break-path detail** (`ppo.py:1361-1365`, `ppo_epochs: 1`). Window 1's approx_kl is computed before any step, so it is about 0 and can never trip the check. At most steps 2-4 are cut. The cap therefore bounds the drift of 1-3 steps, measured on other games' data, not a per-iteration budget.
- **Units** (`ppo.py:159`, per_player). approx_kl is the joint KL summed over all K unit frames of a player-turn. A 0.005 budget therefore means different per-frame freedom in early game (few units) and late game (many units). This couples the cap to game phase, which is synchronized.

### 1.3 "unit_kind carries 60-80% of teacher KL, hence per_player credit dilution" is confounded by counting (moderate)

`_output_action_kl_components` sums each head's KL over entities (`ppo.py:3270-3277`, via `_sum_entropy_component`). unit_kind has one frame per unit, and market_kind has far fewer frames. Its share of a per-player-turn sum is dominated by frame count, whatever the credit. Normalize by active frames before drawing any credit-dilution inference. D1's per-head cosine is the right test, but the premise that motivates it is weak.

### 1.4 An unexplained directed signature: entropy rises in self-play (moderate)

- Self-play entropy rose: J/2 6.1 → 8.0 and hz4 7.4 → 9.8, with ent_coef 1e-6.
- vs-cha22 entropy fell: 6.1 → 4.9.
- A pure random walk in logit space tends to lower entropy, not raise it.
- So something directed is diffusing the unit heads in self-play. Candidates, all unmeasured:
  - the push-down/push-up asymmetry of PPO on K-frame joint ratios;
  - advantage correlating with K after per-minibatch centering;
  - the teacher term.
- This matters for money because unit frames are sampled in parallel with no cross-unit conditioning (evidence-econ). Higher unit entropy directly produces more conflicting commands: M's ineffective commands went 565 → 996, and a PLANT is voided when total seed demand exceeds seeds held.
- The design files execution as a "baseline inefficiency, not the cause". The data fit **decision drift amplified by an execution architecture that cannot absorb diffuse unit policies**.
- Per-head entropy is not on the design's watch list. It should be, in both arms.

### 1.5 Mechanism B and D4 miss the case that v2 actually found (major)

`evidence-v2.md` §3: "relative reward can actively penalize a profitable investment". cha22's expansion raised the learner's own bank by +4.6k but the rival's by more, so the margin changed by −6.8k. "The policy's avoidance agrees with the objective."

- In a shared-demand market, term M may *correctly* score investment as negative. That is an objective property, not a credit inversion.
- D4 has readouts for "λ 0.9 negative with λ 1 ≥ 0" (B) and "both ≥ 0" (not B). It has none for **both significantly negative**, which is this objective case.
- In that case O1 would faithfully optimize toward less investment, which means less money. M@20M dropping strawberry tile-days from 200 to 5 and sheep from 133 to 23 is what this predicts.
- D4 also compares investment turns selected by the policy's own actions with same-day non-investment turns. That is observational and confounded by the state that led to the purchase.
- The decisive, cheap version is v2's paired fork protocol: from stored states, force invest versus not-invest, then run N mirror continuations and read the term-M return difference. That is ground truth for what the reward pays, independent of the critic. Add this readout; it can also end the optimizer-first programme before any pod time.

### 1.6 The success metric is the quantity term M does not pay for (major, framing)

- `self_play_bank_metrics` pools both learner seats (`telemetry.py:55`). In mirror self-play, `own_bank_mean` is exactly joint bank / 2, so the proposed `econ/joint_bank_mean` is redundant in training.
- More importantly, term M is zero-sum. Its gradient for the common mode (both seats richer) is exactly zero. So "own and joint bank rising" in training is not what O1 optimizes. It can only happen as a by-product of best-response asymmetry.
- The only reward-aligned own-bank readout is against a fixed opponent (the eval against BC). Outcome 1 should be judged on eval margin and eval own bank against fixed BC, with a cha22 panel added as a held-out label (allowed: a label, not a model input). It should not be judged on the training-time bank.

## 2. Constraint check

| Constraint | Status |
|---|---|
| Self-play only | Kept. Both arms are mirror self-play. The eval against BC is evaluation, not collection. |
| Reward term M unchanged, owner's authority | Kept in config. The outcome table expects term M to deliver a common-mode gain it does not reward (1.6). That is not a violation, but the expectation is mis-specified. |
| One trainer | Kept. S1/S2/S3 are in the shared `run_ppo.py`/`ppo.py` path. `probe_signal.py` is an ops diagnostic that must call the trainer's own functions, not reimplement the loss. |
| No adaptive anchor | Kept. L3 (a fixed coefficient) is correctly flagged for owner confirmation. target_kl is a per-update trust region against the rollout policy, not an anchor. |
| Stateless, no opponent identity | Kept. The permutation null is rank-local over player-turns and exposes nothing. S2 counters are telemetry only (`telemetry.py` docstring contract). |
| Isaiah alignment Decision (adopted) | **Tension.** The Decision says recipe differences are implementation choices, to be "resolve[d] toward Isaiah" with the residual recorded, and "Do not present alignment alternatives to the owner as a menu." The design switches the trunk from Muon to AdamW and adds target_kl, both departures. It then offers "If the owner prefers Isaiah alignment, L1 plus S3 under Muon is the fallback", which is exactly a menu. Given 1.1, the Muon-LR-cut arm is the aligned default. AdamW needs D0 evidence that it beats matched Muon. |
| Run-before-reasoning rule | Met for O1. D0/D1/D4 each state a question, code path and readout. D1's cost is understated (see 4). |

## 3. Confounds that make the first run non-discriminating

1. **Four simultaneous changes.** O1 changes optimizer family, batch per step (4x), target_kl, and Muon weight decay (0.01 → 0), all against the M recipe. If O1-real holds, the cause cannot be localized. Given 1.1, the most likely cause would be "effective LR / 15", which a one-flag Muon arm would have shown.
2. **Holding versus slowing.** At about 15x less movement, prior slides of 10-33k over 100-1000 iterations would stretch past the 610-iteration window. "Real flat and null flat" (outcome 2) is then the expected result under *both* hypotheses, fixed and merely slowed. The design treats outcome 2 as "steps too small" and raises LR, but with target_kl binding that changes nothing (1.2).
3. **The null is not matched to real noise.**
   - A global permutation of advantages across player-turns makes the noise iid. It destroys two structures of real advantage noise: temporal correlation within a game (λ 0.9 over about 10 turns) and seat antisymmetry (seat 1 ≈ −seat 0 in the same game), which makes real gradients partly cancel.
   - The null's gradient noise is therefore spread differently from real noise, so its drift is not "what the real noise does".
   - Better null: **one random sign per game**, applied to both seats and all turns. It preserves within-game temporal correlation and the ± seat structure, while E[A·∇logπ] = 0. Add this as the default `advantage_null` mode.
4. **Eval power.**
   - Eval games = `cfg.env.n_envs` (`run_ppo.py:1666`). That is 128 per rank in the 2-rank config, not the 64 the design states.
   - Even at 128 games the SE at 0.5 is 0.044. A threshold of 0.55 is about 1.1 SE, so "eval ≥ 0.55" cannot separate "better than BC" from "equal to BC".
   - Use seat-swapped paired seeds, report the win-rate CI and the paired bank-margin CI, and require the lower CI bound > 0.5. Alternatively, report it only as an estimate.
5. **D0's held-out readout covers only early game.** Iteration 13 is game steps about 48-112 of the second game, because all 256 envs are phase-locked. KL(BC‖updated) measured there ignores the mid- and late-game harvest/sell decisions where M lost money. Hold out whole envs across all 12 iterations instead of a time slice.
6. **Baseline B0 sits in warm-up.** B0 is the first 2 intervals (about iterations 1-22), which is exactly where BC-critic runs rose. A warm-up bump inflates B0 and makes the loss condition trip early. Use the BC checkpoint's own bank on the same seeds (evidence-econ: 76k in self-play, CPU) as the reference.

## 4. Cheaper ways to get the same answers

| Question | Design's route | Cheaper route |
|---|---|---|
| Is the policy's movement signal-blind, and does AdamW fix it? | D0 plus the O1 pod pair | **D0 alone**, adding a Muon arm at matched per-iteration KL and the per-game sign-flip null. It runs offline on one GPU in hours. If R_move(AdamW) ≈ R_move(Muon at matched KL), drop the optimizer swap: O1 reduces to a Muon LR cut. |
| Was the past slide noise? | D2 is optional | A Muon real/null pair of the M recipe (per-game sign flip) for about 100-150 iterations, from BC. J/2 slid 30k between iterations 45 and 79, so 150 iterations is enough to see the slide or not. That is about 1/4 of O1's pod time, and it attributes the historical failure, which O1 cannot. |
| Does term M pay for investment? (mechanism B, or the objective) | D4 on observational λ-contrasts | v2's forced-fork mirror continuations from about 20 stored states × 16 continuations, on CPU or one GPU. That is ground truth for the reward's preference and removes the critic from the question. |
| Which chain link breaks? | S2 in a new run | The existing checkpoints (J/2 final, hz4@10M, M@10M/20M) already exist, and evidence-econ already ran M@20M. Running the same CPU econ runner on the other three gives the chain order now, with no training. |
| D1 null band | 500 permutations × 12 phases × 20 minibatches, about 120k backward passes | 30-50 sign-flip nulls are enough for a 95% band on a pairwise-cosine statistic. Alternatively, compare split-half cosine of real against null with bootstrap over games. This is about 10x cheaper. |

## 5. The most likely way it fails

1. **D0 is skipped or diluted, O1 launches, and both arms hold near BC for 610 iterations (outcome 2).** O1 is, to first order, an LR cut of about 15x. The per-sample signal is unchanged, and the only in-run rises in the record were within seed noise at approx_kl about 0.01, which target_kl now forbids.
2. **The eval against BC reads 0.45-0.55 with 128 games,** which is not interpretable.
3. **The LR ×2 escalation hits the same target_kl cap.** The design exhausts its abandonment rule 5 ("outcome 2 twice") after about 2 pod runs.
4. **The owner's question stays unanswered.** Even a perfect optimizer only finds what term M rewards in mirror play, which is margin over a copy. v2 evidence says that can point *away* from investment in a shared market (1.5). The cha22-style economy (all-in to about 3k by day 10, 3-4x strawberry tile-days) is also far outside BC's sampled support at ent_coef 1e-6: v2 sampled BUY_LAND/tomato 0/2000 times, at log-prob −13. It is not reachable by small local steps.

The second most likely failure is outcome 3 in both arms: drift survives, driven by the value/teacher gradients through the shared trunk. The null arm keeps both at full strength, so it cannot separate PG noise from value interference. Add a D0 arm with policy loss coefficient 0: value and teacher terms only, no PG.

## 6. What to keep, and a tighter sequence

Keep:
- S1, changed to a per-game sign flip, with permute as a secondary mode.
- S2 econ counters, including per-head entropy and ineffective commands by kind. The per-step `transition_econ_before/after` arrays already exist in `python/owl/kaggriculture/env.py:155-156`.
- D0, as a hard gate, with the added arms.
- The forked-continuation version of D4.

Proposed order:
1. S1 and S2 land (`just py-prepare`, cookbook adaptation record).
2. Offline D0 and fork-D4 on one free pod GPU.
3. If D0 shows matched-Muon ≈ AdamW, the first pod run is the Isaiah-aligned **Muon LR cut, real versus sign-flip null**, with a 150-iteration first readout, then extension to 10M. Otherwise it is O1.
4. If fork-D4 shows term M pays against investment in mirror play, stop optimizer work and bring that measured fact to the owner. It is the owner's reward call, presented as evidence, not a menu.

## 7. Scores (1-10)

| Axis | Score | Reason |
|---|---|---|
| (a) Probability that it makes self-play PPO earn more than BC | **2** | At best it stops the slide. Its mechanism is mostly an LR cut. target_kl is set below the only regime that produced rises. Term M does not reward the common mode and may penalize investment. Exploration of cha22-like plans is not addressed. |
| (b) Information value of its first run (the O1 pair as written) | **4** | The null twin is a real control, and S2 adds lasting telemetry. But four changes are confounded, "held" cannot be told from "slowed" at this length, the null is mismatched, and the eval is underpowered. With D0 as a hard gate plus the fixes in 3 and 4, this rises to about 7. |
| (c) Cost (10 = most expensive) | **5** | S1 is small. S2 is moderate (the counters exist). `probe_signal.py` is moderate. D1 as specified is heavy (about 120k backward passes). The O1 pair is 4 GPUs for about 610 iterations plus one or two escalation reruns. The cheaper routes in 4 cut this to about 3. |
