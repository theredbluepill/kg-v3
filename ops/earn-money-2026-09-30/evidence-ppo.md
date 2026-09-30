# PPO mechanics audit: why BC-started PPO drifts down in self-play

Working evidence for the owner's question (2026-09-30): "let's switch back to
self play no matter what, and think about how do we get the agent to earn
moneny for real?" This covers the PPO-mechanics angle only. It was read-only:
nothing was launched and no worktree was edited except this file. It is not
Codex-verified. Every run cited is **one seed**.

Code: `/Users/poonszesen/kg-v3-int` at `3e89425`. Isaiah recipe:
`git -C /Users/poonszesen/kg-v3 show upstream/main:configs/scaling_6m.yaml`
(upstream `32b3ec9`). Run metrics are rank-0 `[nt-probe]` iteration records in
`/Users/poonszesen/kg-v3-runs/<run>/<run>.log`, extracted with a scratch
parser (medians over iteration bins, or single bank-bearing iterations as
listed).

Labels: **S** = supported, **R** = refuted, **O** = open.

## 1. What the update path actually does (current tree, margin preset)

| Setting | Current (`configs/kaggriculture_4rank_margin.yaml`) | Isaiah `scaling_6m` | Source |
|---|---|---|---|
| Init | BC best (or J/2 final) warm start | from scratch | preset header l.22-23, l.66-68 |
| Game length | 720 turns | Orbit Wars 500 steps | `upstream/main:src/rules_engine/state.rs:287` |
| Global envs / horizon | 4 x 64 = 256 / 64 | 256 / 64 | l.96, l.144 |
| Optimizer steps per iteration | 16 (64 envs / 4 segments per rank) | 16 | l.151-153; `ppo.py:3444-3454` |
| Samples per optimizer step | 16 segments x 64 turns x 2 seats = 2,048 player-turns, from only 16 games | same count, shorter games | `ppo.py:3444-3454` |
| PPO epochs | 1 | 1 | l.151 |
| gamma / lambda | 1.0 / 0.9 | 1.0 / 0.9 | l.155-156 |
| Critic | per-seat 2p-1 winner head, MSE loss (winner_ce refused for Kaggriculture), no value clip | same head family; `value_loss` default mse | `ppo.py:636-646`, `ppo.py:2205-2218`, l.158 |
| vf_coef / ent_coef | 2.0 / 1e-6 | 2.0 / 1e-6 | l.159-160 |
| clip_coef, clip mode | 0.2 on the **joint per-player** ratio (all heads of one player-turn summed) | same | `ppo.py:2240-2258`, l.163 |
| normalize_advantages | **true, per minibatch** (mean/std over the 16 segments of that step, all-reduced across ranks) | same | `ppo.py:1476-1485`, `ppo.py:3616-3643` |
| max_grad_norm | 10 over all parameters, before the optimizer step | 10 | `ppo.py:1654-1657`, l.161 |
| target_kl | null (the check at `ppo.py:1641-1644` never fires) | null | l.162 |
| Optimizer | Muon for every 2-D non-input/output matrix (lr 1e-4, wd 0.01, momentum 0.95, nesterov, torch 2.9 `adjust_lr_fn` "original" = lr x sqrt(max(1, A/B))); AdamW for input/output layers + 1-D params (lr 5e-6) | Muon 2e-3 / AdamW 1e-4 (20x higher) | `python/owl/train/optimizer.py:231-270`; torch 2.9 `torch/optim/_muon.py::_adjust_lr` |
| LR schedule | 1,000-step linear warm-up, cosine over 400k steps to 1% (effectively constant after warm-up in every run: 1,436 iterations x 16 = 23k steps) | same | l.137-141; M log `optimizer/learning_rate` 9.93e-5 at it 1436 |
| Teacher | `last_best`, KL 0.005, value 0.005; promotion needs win rate >= 0.7 vs last_best | same | l.176-178; `scripts/run_ppo.py:126` |
| Reward | term M: per-step clamp(.5 x margin / 50k, +-.5) difference + 0.5 x terminal sign; zero-sum | Orbit win/loss | `src/kaggriculture/reward.rs:135-140`, `:186-199` |

The only recipe-level differences from Isaiah are the 20x lower learning
rates, the warm start and the game/reward. Every PPO mechanic is Isaiah's
unchanged (**S**, diff of the two configs above).

## 2. Claims

### C1. The per-update policy change is set by the learning rate, not by the advantage signal. **S**

- Muon orthogonalizes the (momentum) gradient by Newton-Schulz, so each
  matrix update has singular values near 1 times lr x sqrt(max(1, A/B)); the
  gradient's magnitude is discarded (torch 2.9 `_muon.py`). Grad-norm clipping
  multiplies the whole gradient by one scalar, so it cannot change a Muon step
  size. AdamW divides by its own RMS, so clipping is also nearly irrelevant
  for the heads.
- Measured: `policy/approx_kl` per update tracks LR through warm-up and is
  the same whatever the raw advantage scale:
  - J/2 control (`nw3klj2s`): LR 2.2e-5 -> kl 0.0040 (it 10-17); LR 1e-4 ->
    0.0128-0.0133 (it 64-124), while `train/advantage_std` fell 0.34 -> 0.06.
  - main-J (`gq94cyyp`): LR 1.28e-5 -> 0.0025; 1.65e-4 -> 0.0175 (it 50-53).
  - M (`r350xr3w`): kl 0.009-0.012 at adv std 0.017-0.023 (it 4-1436).
  - vs-cha22 (`kifqbyx5`, margin saturated): kl 0.009-0.013 at LR 1e-4,
    identical to self-play M.
  - Clip fraction 13-20% at LR 1e-4 in every run: a large share of
    player-turns hit the trust-region edge every update, signal or not.
- Independent evidence: the 10-arm ablation found trunk and actor movement
  "scale almost exactly with LR: 0.09-0.11x at 0.1x LR, whatever the loss mix,
  critic head or value path"
  (`/Users/poonszesen/kg-v3-pod6000/ops/rebuild-2026-09-29/pod-6000-2026-09-29/ablation/attribution.md`,
  "Which factors helped" item 1).

### C2. Per-minibatch advantage normalization inflates a near-zero signal to unit scale. **S**

- `_normalize_masked_advantages` divides by the std of the current 16-segment
  minibatch (`ppo.py:3616-3643`), so the policy term always sees unit-variance
  advantages.
- Raw `train/advantage_std` falls 10-25x once the BC critic adapts: J/2 0.55
  -> 0.15 (it 23-147); A2 0.57 -> 0.14 (it 12-248); J2-resume 0.037 -> 0.016;
  M 0.017-0.023; vs-cha22 0.51 -> 0.024-0.035. `optimizer/grad_norm` stays
  13-15 through this fall (J/2 15.4 at it 10-17, 14.8 at it 117-124): a 6x
  weaker raw signal gives the same gradient norm.
- Consequence: the policy gradient dominates the shared trunk's gradient
  direction (value loss ~1.7e-4 x vf 2.0 in M; teacher term 0.005 x KL) and
  Muon turns that direction into a full-size step (C1).

### C3. Under term M in mirror self-play, the advantage is mostly noise. **S** (magnitude), **O** (exact SNR)

- Magnitudes in M: `train/reward_margin_abs_mean` 0.0009-0.0012 per turn;
  `train/return_zero_sum_abs_mean` 0.014-0.019 per 64-turn segment (about
  1.4-1.9k bank margin); raw advantage std 0.017-0.023; explained variance
  0.91-0.985. The critic is not the bottleneck: it predicts 98% of return
  variance, and what remains is the advantage.
- By construction, 0.5 of each game's return is a terminal sign that is a
  coin flip when both seats run the same policy, and the margin term is
  zero-sum (`reward.rs:186-199`). A change that makes both seats richer
  scores zero. Self-play relative reward cannot directly pay for "earn more
  money", only for "earn more than a copy of yourself".
- Each optimizer step averages 2,048 player-turns from only 16 games
  (C-table), and turns within a game share its outcome. With lambda 0.9 the
  credit window is about 10 turns, so there are roughly 16-100 independent
  credit units per step. The gradient signal-to-noise ratio was **not
  measured**. The discriminating check is in section 4, D1.
- The zero-signal control supports "noise-driven": vs-cha22 lost every game
  by more than 50k, which saturates the margin clamp and makes the terminal
  sign constant. Its bank still fell 62.8k (it 12) -> 39.6k (it 102) -> 9.6k
  (it 147) -> 3-5k (it 180-281), with approx_kl 0.010-0.013 per update and
  unit_kind entropy 4.06 -> 2.76. One seed. Opponent-play credit is not
  separated from drift.

### C4. The policy random-walks away from its anchor, and bank falls as teacher KL grows. **S** (correlation), **O** (causation)

- `teacher/kl` (nats per player-turn vs the teacher) grows without bound:
  J2-resume 0.8 (it 11-155) -> 3.5 -> 5.9 -> 6.5 (it 724-979); M 4.9 -> 8.3;
  vs-cha22 0.02 -> 4.9 (it 248). unit_kind carries most of it (M 4.9 -> 8.3
  of total 3-10), then market_kind and market_item (0.4-0.65 each).
- Bank peaks early at low LR and low KL, then falls as KL passes about 1 nat:
  - J/2: 74.6k (it 23, KL 0.11) -> 81.6k (it 45, KL 0.78) -> 51.4k (it 79, KL 2.0).
  - A2: 71.8k (it 12) -> 79.1k (it 57, KL 1.05) -> 50.9k (it 248, KL 4.3).
  - The ablation had the same shape: LR/10 plus BC critic (arm D) rose
    72.2k -> 83.8k inside warm-up (<= 736 steps, LR still below peak).
  - Some real signal exists at small steps. It is lost once cumulative drift
    dominates.
- The anchor is weak by construction. The teacher-KL gradient is 0.005 x
  (p_student - p_teacher) per logit, against a unit-scale normalized
  advantage per sample. `last_best` is never promoted (M evals 0.14 / 0.11 <
  0.7; `.../M-margin-J2-4rank/stop.md`). In M and J2-resume the teacher was
  J/2 final, not the BC best (`.../M-margin-J2-4rank/launch.md`,
  "Warm start").

### C5. Grad clipping at 10 vs norms 13-15 is not the cause. **S** (reasoning plus C1), **O** (no ablation)

The clip scale factor is about 0.7. It changes neither the Muon step (scale
invariant) nor, materially, the AdamW step. Raising or lowering the clip alone
should not change the drift. No run isolated it.

### C6. The critic range (+-1 vs term M's +-1.5 mid-game return-to-go) is not binding in practice. **S** (typical), **O** (tails)

EV 0.91-0.985 in M, and value loss 1.3-2.0e-4. Typical |margin| of 10-18k
keeps returns far inside +-1. Tails (|margin| >= 50k) were not measured.

### C7. Critic warm-up would not fix it. **S** (one arm)

Ablation arm C (value-only warm-up, full LR) collapsed fastest: 70.8k -> 0 by
game 2. The BC critic head (arms D and H) already gives EV 0.79-0.84
(`attribution.md`, table and critic-quality bullet). In M, EV is 0.98.

### C8. Why Isaiah's identical mechanics worked there. **O** (inference)

- From a random init, a fixed-size noisy step costs nothing: there is no
  skill to lose. Orbit Wars' win/loss signal is also decisive and its games
  are shorter (500 steps).
- `last_best` promotion kept moving the anchor with real improvement.
- From a BC policy whose value lies in a precise plant -> water -> harvest ->
  sell chain, the same fixed-size steps on a mostly-noise direction destroy
  information faster than the weak relative signal adds it.
- This is consistent with C1-C4, but not tested: no from-scratch
  Kaggriculture control exists.

## 3. Levers (self-play only; all untested unless noted)

| Lever | Mechanism it targets | Expected effect | Risk / limit |
|---|---|---|---|
| **L1. Larger batch per optimizer step** (for example 64 segments per step and 4 steps per iteration: `segments_per_minibatch` 16 per rank, same envs) | SNR per step (C3); fewer fixed-size steps (C1) | 4x fewer steps per env step, about 2x better direction SNR, so drift per sample drops about 4x (about 2x if pure random walk) | Slower learning per sample; memory headroom unmeasured |
| **L2. Lower Muon LR further, or LR decay tied to KL** (for example 2e-5, or cosine over about 20k steps, not 400k) | C1: drift is proportional to LR x steps | Directly reduces drift. Ablation: trunk movement 0.1x at LR/10 | Only slows the walk unless the signal improves too; J/2 at 1e-4 still slid |
| **L3. target_kl early stop** (for example 0.003-0.005 on approx_kl; code exists at `ppo.py:1641-1644, 1361-1365`) | Per-update budget (C1) | Caps per-iteration movement at a fixed KL, whatever the LR | Isaiah never ran it; with 1 epoch it truncates minibatches, so it acts like a lower LR. Needs a unit test on the break path |
| **L4. Fixed BC-KL anchor at a larger coefficient** (`teacher_mode: fixed`, BC best, 0.05-0.1) | C4 unbounded drift | Bounds KL. Ablation arm A (0.1, full LR) kept 41.4k vs control 92; arm F (0.1, LR/10) +4.6k, inside the 5k spread | **Owner said "we dont want adaptive anchor".** A fixed one is not adaptive, but it needs owner confirmation. It also caps how far the policy can go (cha22's 143-179k may need a large KL) |
| **L5. Replace per-minibatch normalization with a running scale** (EMA std across iterations), or no normalization | C2 | A weak iteration then gives a small policy gradient relative to the value and teacher terms, so the anchor and critic dominate the direction when there is no signal | **Under Muon the step size is still unchanged (C1).** Only the loss mix changes. Pair it with AdamW-only or L1/L3 to shrink steps |
| **L6. Optimizer whose step scales with gradient magnitude for the actor** (AdamW everywhere at a low LR, or Muon only for the trunk with policy heads frozen early) | C1 | Small or noisy gradients give smaller effective moves (Adam's v remembers prior scale) | Loses Muon's throughput benefit. Isaiah never used it at this scale |
| **L7. gamma/lambda** (lambda 0.95-0.98, keep gamma 1) | Delayed payback beyond the about 10-turn credit window | Better credit for investments that pay back 10-50 turns later | More variance: worse SNR unless L1 is also used. Payback lengths not measured from the engine |
| **L8. Reward SNR (owner's call)** | C3: the terminal coin flip is 50% of the return; zero-sum pays nothing for a shared rise | Lowering the terminal share, or adding own-bank term A (exists: `econ_bank_weight`), gives a non-relative signal. Per-minibatch normalization then makes it a cross-game batch baseline: richer games get positive advantage | The critic cannot represent the common mode (2p-1 per seat sums to zero). A2 (0.25 own-bank) still slid 79.1k -> 50.9k at LR 1e-4, so reward alone did not stop drift |
| L9. Critic warm-up | — | None expected (C7) | Arm C collapsed fastest |
| L10. Grad clip change | — | None expected (C5) | — |

Ranking by evidence: L2/L3 (step budget) and L1 (SNR) address the two
measured mechanisms, C1 and C3. L4 is the only one with an in-repo positive
datum (arm A), but it needs owner approval. L5 alone is not expected to help
under Muon. L8 is the owner's decision.

## 4. Discriminating checks before any run (stated, not run)

- **D1 (gradient SNR, no training):** on BC best and on M's 20M checkpoint,
  collect one self-play rollout, compute the 16 per-minibatch policy
  gradients without stepping, and report the mean pairwise cosine and the
  gradient noise scale tr(Sigma)/|G|^2.
  - Expected: cosine ~0 means noise-dominated steps (supports C3/L1); a
    clearly positive cosine means a coherent direction (the problem is then
    the direction, not the noise).
  - Stop: one checkpoint pair.
- **D2 (drift without signal):** the vs-cha22 saturated run already is one.
  A self-play twin with the reward zeroed would test whether bank falls at
  the same rate as in M.
  - If yes: drift is optimizer-driven (C1).
  - If much slower: the M signal is anti-aligned with money (reward
    problem).
- **D3 (step budget):** M preset with L1 (4 steps per iteration) plus L3
  (target_kl 0.005), from BC best, 100 iterations.
  - Watch: own_bank_mean vs the J/2 bins at matched teacher KL.
  - Loss condition: bank below 60k by the time teacher KL exceeds 2.

## 5. Open / not attributed

- The exact gradient SNR (D1) has not been measured.
- The engine's plant-to-sale payback length has not been measured (L7).
- Whether the entropy collapse in vs-cha22 (not seen in M) is
  noise-driven logit spreading or a real vs-bot signal is unknown.
- Seed variance is unknown for every run cited.
- M and J2-resume confound the reward change with an LR re-warm and a J/2
  teacher.
- The M eval figures (0.14 / 0.11, own bank 32.6k) come from receipts and the
  task statement, not re-read from W&B in this audit.
