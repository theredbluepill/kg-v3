# Design: self-play that earns money, signal-first (2026-09-30)

Angle: fix what the reward and the advantage tell the policy about earning,
keeping the optimizer as it is unless evidence forces a change. Tree 3e89425.
Inputs: evidence-runs.md, evidence-econ.md, evidence-ppo.md and evidence-v2.md
in this folder, plus code read for this note: `src/kaggriculture/reward.rs`,
`python/owl/model/kaggriculture.py:1077-1120` and
`python/owl/train/ppo.py:820-870`. This is an agent proposal. It is not
Codex-verified and nothing in it has been run. All evidence is single-seed.

## 0. Constraints this plan keeps

- Self-play only, and `scripts/run_ppo.py` stays the one trainer.
- The policy is stateless: it reads current-observation tokens only, and
  opponent identity is not an input anywhere.
- No adaptive anchor. This plan proposes no anchor of any kind (a fixed BC-KL
  anchor belongs to the optimizer angle and needs the owner).
- The reward is the owner's call. Every reward change below is marked
  **[owner]** and is offered as an option, not adopted. The owner's current
  reward is term M: 0.5 × cash difference + 0.5 × terminal sign.

## 1. Diagnosis

### H-A (supported by code; its effect is untested): the critic structurally cannot value an own-bank reward

`_values` returns `2·p_self − 1` from a softmax over the two seats' critic
tokens, so V_0 + V_1 = 0 always (`kaggriculture.py:1077-1084`; the same point
is noted at `ppo.py:848-856`). Any non-zero-sum part of the return is therefore
never baselined. This includes term A's own-bank reward, and it includes the
very thing "earn money for real" needs: a shared rise or fall in money.

For that common-mode part, δ_t = r_t exactly. With γ = 1 and λ = 0.9 its
advantage is Σ 0.9^k · r_{t+k}: roughly a 10-turn myopic cash reward.

- **Purchases look bad.** A purchase costing c gets −w_b·c/S at once, which
  under A2 is −0.0025 per 1k spent. The payback 100-300 turns later arrives
  with weight 0.9^100 ≈ 3e-5.
- **Sales look good.** Any SELL gets immediate credit.
- **Conclusion.** Term A, as tested (A at w_b 1.0 and A2 at w_b 0.25), was
  structurally anti-investment and pro-immediate-sale. It was not a clean test
  of "reward absolute earning".
- **Consistent with this:** A2 delayed the slide by about 60 iterations but
  did not stop it. A2's `return_common_mean` was 0.011-0.015 per segment, a
  real but unbaselined signal.
- **Not verified:** A2 logged no investment counters and left no checkpoint,
  so the predicted loss of investments under A2 has not been observed.

### H-C (plausible, unmeasured): cash-keyed credit on investments under every reward tried

Under term M the purchase transition pays −0.5·c/50k at once, so the critic
must raise V(s') by exactly that amount plus the payback. Under terminal-only
rewards (J/2, hz4) the purchase δ is V(s') − V(s). If the winner critic reads
the current cash lead as its win predictor, both rewards give a negative δ at
purchases.

- **Evidence for:**
  - M@20M against BC on the same seeds dropped exactly the long-payback
    assets:
    - strawberry tile-days fell from 200 to 5;
    - sheep fell from 133 to 23;
    - 7 of 8 seats never bought land.
  - Cows and hands, which pay back quickly, were kept.
  - v2 measured this inversion directly: buy-9 was better in 16 of 16 forks,
    yet its advantage was negative in 0 of 16 pairs. It was fixed only by
    λ = 1 over 256-turn segments.
- **Evidence against:**
  - A perfect critic makes the reward's timing irrelevant at γ = 1: the
    potential-shaping identity gives δ' = δ.
  - EV is high (0.91-0.985), so the critic is good on average. It is
    unmeasured on purchase transitions in particular.

### H-N (supported, and not a signal issue): fixed-size steps on a low-SNR signal

The audit shows approx_kl is set by the learning rate under Muon, not by the
signal. So the share of each step that is useful is the gradient's
signal-to-noise ratio. Under term M in mirror play, three things make that
ratio small:

- **The terminal sign is nearly a coin flip,** yet it is half the return.
- **The opponent's sampled actions are noise.** The margin term pays
  Δbank_self − Δbank_opp, and the opponent's half is noise for this seat's
  credit, roughly doubling the per-step variance.
- **A shared slide is invisible.**

vs-cha22 slid with a dense per-step margin, so the signal alone cannot be the
whole cause. This is the strongest evidence against a signal-only fix, and it
sets the abandonment gate in section 5.

### Reading

- **Most likely: H-N drift plus a directional anti-investment bias (H-A or H-C).**
  The bias decides which parts of the economy the drift removes first. The
  M@20M pattern fits this reading:
  - long-payback assets are lost;
  - ineffective commands rise from 565 to 996, which looks like undirected
    loss of precision.
- **Signal-first can fix:**
  - the direction of the updates (H-A, H-C);
  - part of the signal-to-noise ratio, by removing opponent noise and paying
    for shared money.
- **Signal-first cannot fix:**
  - step size (H-N);
  - the architecture limit on ineffective commands. Parallel unit frames with
    no cross-unit conditioning cause PLANT-block and same-tile collisions
    (`lib.rs:1550`). That is an execution/architecture error to diagnose
    separately.

### Calibration for "for real" in self-play

Town drain is shared: about 195-211k at base prices for both seats combined,
and wheat and eggs are uncapped.

- A mirror twin's cash ceiling is therefore about 100k plus the wheat and egg
  glut, not cha22's 143-179k, which it earned against weak sellers.
- BC self-play earns 76k.
- Realistic self-play success: own bank and joint bank rise toward the demand
  bound.

## 2. Changes (on the `scripts/run_ppo.py` shared path only)

### C1. Telemetry (required before any run; no behavior change)

In `scripts/run_ppo.py`, where `terminal_metrics` are gathered (about lines
1822-1846), log per-iteration means to the nt-probe records and to W&B:

- the 32 engine econ fields, under `env/econ/<name>`;
- `env/joint_bank_mean`;
- tile-day counters by crop and animal-days by animal (new; the engine state
  has them, but no counter exists yet).

In `python/owl/train/ppo.py`, after GAE, log the raw and normalized advantage
mean, SE and count per chosen action family: `train/adv_by_kind/<kind>` for
BUY_LAND, BUY_ANIMAL, BUY_SEED (by crop), HIRE, SELL, PLANT, WATER, HARVEST,
FEED and PASS. This is the in-run discriminator for H-A and H-C.

With C2 enabled, also log `train/ev_common`: the explained variance of the
common-mode return by V_0 + V_1.

### C2. A per-seat critic offset head (model seam; bit-identical at step 0)

In `python/owl/model/kaggriculture.py`:

- **Value.** V_s = (2·p_s − 1) + o_s, where o_s =
  `critic_offset_head(critic_value_hidden[s])`.
- **Head.** A new `OutputProjectionMLP(trunk, 1)` whose output layer (weights
  and bias) is zero-initialized. Exclude it from the generic
  output-layer init in `get_output_layers`, so the zero init survives.
- **Loss.** The winner softmax and its teacher value distillation are
  unchanged; the MSE value loss (`vf_coef` 2.0) trains both parts.

The head is gated by a model config flag, `critic_offset: true`, which
defaults to false, so old configs stay byte-identical.

When loading BC best with `--load-model-weights-mode model_only`, allow
exactly the missing `critic_offset_head.*` keys and fail on any other missing
or extra key.

- **Why this is legal.** It is a new head on current-observation critic
  tokens, which the owner allows ("new game stems/heads"). It holds no state
  and reads no identity.
- **Effect under each reward:**
  - under term M or terminal-only (zero-sum) it should learn o ≈ 0, so it is
    inert;
  - under any own-bank term it baselines the common mode;
  - it also lifts the (−1, 1) range limit.

Required checks and records:

- Update `docs/model-architecture.md`.
- Add a unit test: at init, values equal the BC critic's exactly, and a
  state_dict round-trip works.
- Run `just py-prepare`.
- Add a cookbook adaptation record.

### C3. Reward options [owner] (none adopted here)

**E0, for Run 1 (a mechanism test).** Reuse A2's reward exactly
(`configs/kaggriculture_4rank_bc_finetune_bank.yaml`, which is owner-approved
term A: "A is good"). Only C2 changes, so the run tests H-A. It needs the
owner's OK because the latest owner reward is M.

**E1, the "earn for real" follow-up, only if Run 1 passes.** Own-bank is the
dominant term and does not saturate within reach:

| Key | Value |
|---|---|
| `econ_bank_weight` | 0.5 |
| `econ_bank_scale` | 250000.0 |
| `econ_bank_cap` | 0.5 (saturates at 250k; 75k scores .15, 179k scores .36) |
| `econ_margin_weight` | 0.25 |
| `econ_margin_scale` | 50000.0 |
| `econ_margin_cap` | 0.25 |
| terminal scale | 0.25 |
| `econ_shaping` | 0 |

The weights are agent-proposed and not measured optima. This is v2 m17's idea
(a non-zero-sum own-bank target with a per-seat critic, which raised mirror
own bank from 82.0k to 86.8k), not its model.

**E2, conditional on Dx1/Dx2 showing H-C.** Re-time the potential on
book-value wealth: Φ = `seat_wealth` (`lib.rs:1417, 2824`) during the game and
Φ = bank at the terminal transition. The telescoped objective is exactly the
owner's cash terms, because unsold goods are zeroed at the end. It moves
purchase credit to purchase time and charges plant and animal deaths at death
time. The limit is v2's: with a perfect critic it changes nothing, since
δ' = δ. It helps only if the critic misvalues assets, which is what Dx1/Dx2
measure. It needs a read-only wealth input in `src/kaggriculture/env.rs` and
an `econ_*_potential: cash|wealth` key in `RewardConfig`.

### C4. Credit window (config; conditional on H-C, not in Run 1)

Set `rl.gae_lambda` to 0.97 and `rl.horizon` to 128, with `env.n_envs` 32 per
rank so env steps per iteration stay equal. This is v2's working direction
(λ = 1 over 256 turns), not measured here. It raises variance, which feeds
H-N, so it goes in only after Dx1/Dx2 show purchase-credit inversion under the
chosen critic.

Also check the phase synchronization that v2 found: Kaggriculture configs set
no `truncation_prob`, and the fixed 720-turn games all start together.

## 3. Cheap diagnostics first (no training)

Each takes the BC best (`/Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt`),
and M@20M where noted. All run niced and time-bounded, on the local CPU (at
most about 15 min each) or on the pod's CPU. Scripts live in this `ops/`
folder as working artifacts.

- **Dx1: advantage by action family**, the main discriminator.
  - **Run.** 16 BC self-play games; the econ runner already does this. Record
    per seat and turn: the chosen actions, banks, the critic value, the 32
    counters and, if exposed, `seat_wealth`. Recompute the rewards offline for
    terminal-only, M, A2 and E1 by mirroring `RewardConfig::transition`, then
    run GAE with γ = 1, λ = 0.9 and 64-turn bootstrapped segments,
    (a) with the BC critic as is (zero-sum) and (b) with an oracle common-mode
    baseline (per-seat Monte-Carlo mean).
  - **Output.** Mean ± SE of the raw advantage per family, per reward.
  - **Supports H-A:** under A2 with the zero-sum critic, BUY_* is below 0 by
    more than 3 SE and SELL is above 0, and the bias vanishes with the oracle
    baseline.
  - **Supports H-C:** under M or terminal-only, BUY_LAND, BUY_ANIMAL and
    BUY_SEED STRAWBERRY are below 0 by more than 2 SE, and Dx2 shows those
    purchases raise the final margin.
  - **Refutes both:** purchase advantages are within 1 SE of 0 and match the
    other families.
  - **Repeat on M@20M** (`kg-v3-runs/M-margin-J2-4rank-20260930/checkpoints/`)
    to see whether the bias grew during training.
- **Dx2: paired forks (v2's buy-9 protocol).**
  - **Setup.** First check determinism: replaying one game from its seed and
    recorded actions must give identical per-turn banks. Then pick 24 BC
    states where BC chose a long-payback purchase. Fork each by prefix replay:
    keep the purchase, or replace it with no order. Continue both with BC on 4
    sampling seeds each, 192 continuations in total.
  - **Measure.** Δ final own bank, Δ margin and Δ joint bank with a CI, and
    the agreement of their sign with Dx1's advantage.
  - **Reading.** This is the ground truth for whether investing pays under BC
    play, and whether the training signal says so.
- **Dx3: return-variance decomposition,** from Dx1's data.
  - Split per-transition advantage variance into own-bank, opponent-bank and
    terminal-sign parts. Report the terminal sign's share of per-game return
    variance under M.
  - This predicts the signal-to-noise gain of E1 over M before any GPU time.
    If E1's action-attributable variance share is not at least 1.5× M's, E1
    is not worth a run.
- **Dx4: noise-null twin, the gate** (a short pod probe, shared with the
  optimizer angle's D2).
  - **Setup.** The J/2 recipe from BC best with raw advantages shuffled
    across rows within each minibatch. This keeps the advantage distribution
    and destroys action credit. It needs a debug key, `rl.debug_advantage_mode:
    shuffle`, in `ppo.py`, logged loudly.
  - **Stop** at iteration 150.
  - **If it slides like A2 and M** (bank at iterations 100-150 within 17k of
    theirs), the reward content is not steering at this learning rate. See
    section 5.

## 4. First run (Run 1): H-A, "own-bank with a critic that can hold it"

- **Mechanism.** Term A failed because the zero-sum critic left its
  common-mode return unbaselined, which made it a 10-turn cash reward that
  penalizes investment (H-A).
- **Precondition.** Dx1 shows the H-A bias, and Dx4 does not trigger
  abandonment.
- **Inputs.**
  - Config: `configs/kaggriculture_4rank_bc_finetune_bank.yaml` (A2's; E0,
    owner OK needed) plus `-o model.critic_offset=true`.
  - Code: C1 and C2.
  - Start: `--load-model-weights /Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only`,
    with the BC critic kept and the BC best as teacher (not J/2 final).
  - Optimizer, learning rate, schedule and the PPO keys: exactly A2's, so the
    critic is the only change against A2 (bqtke7iq).
  - Four ranks, W&B spoon/kg-v3 under a v3 run identifier, with a netrc on
    the pod first.
- **Code path.** `run_ppo.py`, then the shared PPO update, then
  `KaggricultureModel._values` with the offset.
- **Expected discriminating observation.** Against A2 at the same iteration:
  - `train/ev_common` above 0.3 by iteration 60;
  - `train/adv_by_kind/BUY_*` mean at least −1 SE;
  - strawberry tile-days, animal-days and land purchases at 80% or more of
    BC's;
  - `own_bank_mean` at 68k or more over iterations 150-250, where A2 was at
    51-52k. The 17k margin is the seed-noise floor from ablations D and J.
- **Loss conditions** (any of these stops the run):
  - own bank below 55k for 2 consecutive intervals after iteration 124, which
    is A2's path;
  - investments below 50% of BC while `ev_common` is above 0.3, which means
    H-A is fixed but H-C or H-N dominates (take C4 or hand over to the
    optimizer angle);
  - `ev_common` below 0.1 at iteration 60, which means the head did not
    engage (an implementation problem: one repair, then stop).
- **Stopping condition.** Iteration 300 (about 4.9M env steps) or a loss
  condition. Then run a seat-swapped evaluation against the BC best: 64 games,
  reporting win rate, own and rival bank, joint bank, legality, completion and
  runtime.
- **Metrics to watch:**
  - `own_bank_mean` and joint bank;
  - `env/econ/*`: units sold, ineffective commands, deaths, and tile-days and
    animal-days;
  - `adv_by_kind`, `ev_common`, `return_common_mean`,
    `return_zero_sum_abs_mean`;
  - teacher KL by head (`unit_kind`, `market_kind`), approx_kl, clipfrac.

If Run 1 passes, Run 2 is E1 (the owner's weights) from BC best with C2, to
measure how far above 76k the self-play economy goes, judged against the joint
demand bound.

## 5. When to abandon this design

1. **Dx4's noise-null twin slides like A2 and M** (within 17k at iterations
   100-150). Update direction is then irrelevant at this learning rate, and
   step size (H-N) must be fixed first; the optimizer angle's larger batch or
   target_kl comes first. Signal changes resume only after that.
2. **Dx1 shows no directional bias.** If purchase and sale advantages are
   within 1 SE of 0 under every reward and critic, there is no signal error
   to repair.
3. **Dx2 shows the purchases do not pay under BC play** (Δ final own bank
   ≤ 0). Then the loss of investment is not a credit error; it is an
   execution or architecture question (ineffective commands, parallel-frame
   collisions).
4. **Run 1 engages but does not help.** If `ev_common` is above 0.3 and
   purchase advantages are at least 0 while the bank still follows A2 (below
   55k at iterations 150-250), the signal was repaired and did not matter.
   Record it as a supported limit and reopen only with a step-size change.

## 6. Unresolved attribution

- Why vs-cha22 slid faster than self-play: the always-lost terminal, a
  penalty for cha22's own earning, or a larger gradient norm.
- How much of the ineffective-command rise, from 565 to 996, is drift and how
  much is an architecture collision.
- Seed variance: every run is single-seed.
- Whether in-run bank telemetry understates a checkpoint's steady-policy bank.
