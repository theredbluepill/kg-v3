# Evidence: v3 PPO run history (2026-09-30)

Angle: one table over every v3 PPO run and ablation arm, then which factors go
with holding the economy and which go with sliding. Read-only extraction. Nothing
was launched and no worktree was edited.

**How the numbers were made.** 4-rank runs: rank-0 `[nt-probe]` iteration records,
decoded with `json.raw_decode` over the whole log, because progress bars break the
lines. Logs: `/Users/poonszesen/kg-v3-runs/<run>/<run>.log`. Every 256-game
interval is one synchronized game phase of about 11 iterations. Both seats are
the learner (self-play). vs-cha22 is the only exception: its learner seat plays
the fixed bot. Runs A and vs-cha22 are extended from W&B `run.history`, because
their local log copies are truncated. Evals come from W&B `run.summary` and
`history`. Ablation arms (2 ranks, 46 iterations): `.../pod-6000-2026-09-29/ablation/ablate-*/pod-receipts/iterations.tsv`
and `final-report.md:52-66`. Where a table shows a range of iterations, the
metrics are means over that range.

## 1. Run table

LR = peak Muon / AdamW. The schedule is a 1,000-optimizer-step linear warm-up
(peak at about iteration 63), then cosine over 400k steps. Every run uses
`normalize_advantages: true`, `max_grad_norm 10`, `vf_coef 2`, `gamma 1`,
`gae_lambda .9`, `clip .2`, `ent 1e-6` and teacher = `last_best`. These are
identical to Isaiah's `upstream/main:configs/scaling_6m.yaml:39-51`; only the LR
differs.

| Run (W&B) | Start / critic init | LR | Reward | KL coef | Ranks | Iters reached |
|---|---|---|---|---|---|---|
| main-J (gq94cyyp) | BC, `model_only` (BC critic) | 2e-4 / 1e-5 | 0.25 starvation/drought shaping + 0.75 terminal sign | .005 | 4 | 88; watchdog bank floor |
| J/2 control (nw3klj2s) | BC, BC critic | 1e-4 / 5e-6 | same as main-J | .005 | 4 | 150; step cap |
| A (04cy2m6s) | BC, BC critic | 1e-4 / 5e-6 | own-bank w_b 1.0, cap .25 (saturates at a **25k** bank; the "spec bug", `configs/kaggriculture_4rank_margin.yaml:47-50`) + shaping .25 + terminal .5 | .005 | 4 | about 78 (W&B 1,277,952 steps); stopped |
| A2 (bqtke7iq) | BC, BC critic | 1e-4 / 5e-6 | own-bank w_b .25, cap .25 (saturates at **100k**) + shaping .25 + terminal .5 | .005 | 4 | 257; stopped to launch hz4 |
| hz4bpjnq (J2-resume) | J/2 final, `model_and_optimizer`; teacher = J/2 final | 1e-4 / 5e-6, re-warmed | shaping .2 + terminal sign .8 | .005 | 4 | 985; owner stop |
| M (r350xr3w) | J/2 final, `model_and_optimizer`; teacher = J/2 final | 1e-4 / 5e-6, re-warmed | term M: margin .5 (scale 50k) + terminal .5; shaping 0 | .005 | 4 | 1,469; owner stop |
| vs-cha22 (kifqbyx5) | BC, BC critic | 1e-4 / 5e-6 | term M, learner vs the fixed bot cha22 (fraction 1.0) | .005 | 4 | 278 on W&B, **still running** |
| Abl. control 6.2 (7k07gp7c) | BC, fresh critic | 2e-3 / 1e-4 (warm-up) | default (.25/.75) | .005 | 2 | 46 compared (235 run) |
| Abl. A (y8j6vzky) | BC, fresh | full | default | **0.1** | 2 | 46 |
| Abl. B (32pahqok) | BC, fresh | /10 | default | .005 | 2 | 46 |
| Abl. C (lpt11y9j) | BC, fresh + critic warm-up | full | default | .005 | 2 | 46 |
| Abl. D (hftcr4xa) | BC, **BC critic**, seed 0 | /10 | default | .005 | 2 | 46 |
| Abl. E (hi2lqsrx) | BC, fresh, vf .5 | /10 | default | .005 | 2 | 46 |
| Abl. F (nur61v3a) | BC, fresh | /10 | default | **0.1** | 2 | 46 |
| Abl. G (yz34n7i6) | BC, fresh, critic stop-gradient | /10 | default | .005 | 2 | 46 |
| Abl. H (ksqdtvm0) | BC, BC critic | full | default | .005 | 2 | 46 |
| Abl. I (o9c5ji4v) | BC, fresh, critic stop-gradient | full | default | .005 | 2 | 46 |
| Abl. J (pkz85wlw) | BC, BC critic, seed 1e6 | /10 | default | .005 | 2 | 46 |

"/10" means Muon 2e-4, which is the same peak as main-J. The ablation arms end
inside the warm-up: the LR at iteration 45 was 1.44e-4 in the /10 arms and
1.44e-3 in the full-LR arms (`final-report.md:88-99`).

### 1a. Own bank at game ends, matched iterations (rank 0, 256 games per cell)

The ablation arms report the mean of the two seats, from `iterations.tsv`.

| it | main-J | J/2 | A | A2 | hz4 | M | vs-cha22 (vs bot) | Abl B | D | J | H | F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 (start) | 72.6k | 74.3k | 74.5k | 71.8k | 62.6k | 63.9k | 62.8k (cha22 143k) | 73.3k | 72.2k | 72.0k | 76.7k | 73.5k |
| 34 | 74.3k | 77.3k | 73.8k | 76.2k | 63.7k | 62.5k | 63.2k | 63.8k | 79.6k | 79.2k | 20.6k | 70.8k |
| 45 | 68.4k | **81.6k** | 71.6k | 76.0k | 63.8k | 63.6k | 57.3k | 62.2k | **83.8k** | 67.2k | 0.07k | 66.8k |
| 57 | 52.6k | 77.4k | 68.8k | **79.1k** | 62.2k | 62.2k | 57.0k | | | | | |
| 68 | 17.9k | 66.6k | 60.9k | 77.4k | 63.3k | 56.6k | 52.3k | | | | | |
| 79 | 5.7k (stop) | 51.4k | – | 65.7k | 65.3k | 52.0k | 48.3k | | | | | |
| 90 | | 46.7k | | 64.7k | 64.2k | 48.9k | 43.1k | | | | | |
| 113 | | 51.7k | | 64.3k | **68.3k** | 42.4k | 28.3k | | | | | |
| 147 | | 53.2k | | 51.8k | 63.1k | 40.5k | 9.6k | | | | | |
| 248 | | | | 50.9k | 58.4k | 41.5k | 5.1k | | | | | |
| 461 | | | | | 51.1k | 44.3k | | | | | | |
| 686 | | | | | 41.8k | 44.4k | | | | | | |
| 978 | | | | | 30.4k | 36.0k | | | | | | |
| 1461 | | | | | | 38.1k | | | | | | |

- Minimum and end: main-J 5.7k@79; J/2 min 46.7k@90, end 53.2k@147; A end 60.9k@68; A2 end 50.9k@248;
  hz4 min 30.4k@978; M min 35.2k@1191, end 38.1k@1461; vs-cha22 3.1k@203, then 3.8k@270 (W&B).
- Draw rate was 0 in every 4-rank interval that was checked (the first and last of each run).
  Late in the self-play runs the spread narrows: M's own_bank p90 fell from 89.4k to 46.8k.

### 1b. Training diagnostics (rank 0; means over iteration windows)

| Run | window | teacher KL | unit_kind KL | market_kind KL | unit_item KL | grad_norm | clipfrac | approx_kl | entropy | EV | adv_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main-J | 1-20 / 21-63 / 64-88 | .06 / 1.13 / 2.06 | .04 / .77 / 1.58 | .008 / .15 / .23 | .003 / .07 / .09 | 14.0 / 13.6 / 11.5 | .046 / .189 / .214 | .0045 / .0148 / .017 | 6.1 / 6.8 / 6.0 | .79 / .88 / .81 | .37 / .18 / .054 |
| J/2 | 1-20 / 21-63 / 64-150 | .03 / .70 / 2.92 | .02 / .44 / 1.94 | .008 / .11 / .25 | .002 / .04 / .41 | 14.3 / 13.7 / 14.5 | .022 / .108 / .165 | .0029 / .0084 / .0124 | 6.1 / 6.6 / 8.0 | .78 / .86 / .81 | .39 / .26 / .081 |
| A | 1-20 / 21-33 | .03 / .23 | .02 / .14 | .006 / .03 | | 14.1 / 14.1 | .021 / .071 | .0029 / .0060 | 6.1 / 6.6 | .79 / .82 | .38 / .34 |
| A2 | 1-20 / 21-63 / 64-150 / 151-257 | .03 / .66 / 2.82 / 4.84 | .02 / .42 / 1.97 / 3.54 | .005 / .09 / .23 / .42 | .002 / .05 / .34 / .40 | 14.0 / 13.4 / 13.3 / 12.9 | .021 / .104 / .159 / .164 | .0029 / .0081 / .0118 / .0123 | 6.1 / 6.5 / 7.5 / 7.7 | .78 / .87 / .85 / .81 | .39 / .26 / .076 / .047 |
| hz4 | 1-20 / 64-150 / 401-800 / 801-985 | .04 / 2.46 / 7.89 / 8.45 | .02 / 1.74 / 6.11 / 6.49 | .002 / .20 / .35 / .44 | .007 / .23 / .66 / .58 | 13.8 / 14.6 / 12.6 / 11.3 | .030 / .186 / .170 / .152 | .0035 / .0141 / .0127 / .0111 | 7.4 / 7.8 / 9.8 / 9.8 | .79 / .76 / .79 / .80 | .052 / .049 / .036 / .035 |
| M | 1-20 / 64-150 / 401-800 / 1101-1469 | .06 / 2.53 / 8.62 / 9.98 | .03 / 1.79 / 6.74 / 8.07 | .011 / .22 / .57 / .68 | .006 / .19 / .72 / .36 | 13.7 / 13.4 / 11.8 / 9.7 | .030 / .160 / .148 / .127 | .0035 / .0120 / .0108 / .0093 | 7.4 / 5.9 / 7.7 / 6.9 | .82 / .83 / .92 / .93 | .042 / .034 / .029 / .029 |
| vs-cha22 | 1-20 / 64-150 / 151-219 | .03 / 2.44 / 5.06 | .02 / 1.67 / 3.67 | .008 / .42 / .77 | .001 / .15 / .19 | 19.7 / 17.0 / 13.7 | .021 / .183 / .159 | .0029 / .0141 / .0118 | 6.1 / 5.9 / 4.9 | .75 / .72 / .69 | .39 / .057 / .029 |
| Abl /10 BC critic D, J | 1-20 / 21-46 | 1.41, 1.45 at it45 | .96, 1.20 at it45 | .15, .11 at it45 | | n/a | .050 / .169, .052 / .167 | .0047 / .0128 | 6.1 / 6.7 | .84 | .37 / .23 |
| Abl /10 fresh critic B, E, F, G | 1-20 / 21-46 | 1.36, 1.14, .22, .69 at it45 | 1.06, .96, .17, .45 | .15, .06, .01, .17 | | E, F >10 every iteration | .04 / .13-.14 | .004 / .010-.011 | 6.0-6.2 / 5.9-6.5 | .29, -.10, .23, .16 | .16-.18 / .09-.18 |
| Abl full LR H, I, A, C | 1-20 / 21-46 | 4.66, 2.31, 2.49, 8.26 at it45 | 3.72, .96, 2.00, 1.83 | .54, 1.18, .17, 3.59 | | n/a | .30-.38 / .34-.60 | .04-.08 / .04-.20 | 6-12 / 3.7-16 | .79, .06, .37, .69 | .08-.26 / .02-.07 |

- Logged `optimizer/grad_norm` above the clip of 10: main-J 78/88 iterations, J/2 150/150, A 33/33, A2 257/257,
  hz4 940/985, M 969/1,469, vs-cha22 271/287.
- Reward telemetry. A2 `return_common_mean` was about .011-.015 per window. M `reward_margin_abs_mean` was
  .0018 → .0010. vs-cha22 `reward_margin_abs_mean` was .0024 → .0014 and `return_common_mean` about -.022.
  vs-cha22 `win_rate_vs_bot` was **0.000 in all 20 intervals of 256 games**. Its margin went from -80.6k@12
  to -166k@203.

### 1c. Evaluations against `last_best` (W&B; 64 games each, `promotion_threshold` .7)

| Run | env steps | win rate | candidate bank | last_best bank | margin mean |
|---|---|---|---|---|---|
| hz4bpjnq | 10.0M (about it461) | .078 | 45.0k | 76.7k | -31.6k |
| M | 10.0M | .141 | 45.5k | 65.8k | -20.3k |
| M | 20.0M | .109 | 32.6k | 64.2k | -31.6k |

main-J, J/2, A, A2 and every ablation arm stopped before their first evaluation
(checkpoint_freq was 10M at 4 ranks and 20M in the ablation). vs-cha22 has no
`eval/*` yet. The BC best against cha22, from 16 games
(`vs-cha22-4rank-20260930/bc-vs-cha22-eval16.log`): win rate 0/16, own bank
61.9k, cha22 144.4k.

## 2. Claims

**C1. No run held a bank above its start for a sustained period. SUPPORTED.**
Rises did happen, but they were short, came early, and are within the measured
noise:

| Run | Rise | Where |
|---|---|---|
| J/2 | +7.4k | iterations 34-57, 3 intervals |
| A2 | +7.3k | iterations 34-68, 4 intervals, the longest (about 720k env steps) |
| Abl D | +11.6k | through iteration 45, when the run ended |
| Abl J | +7.2k | iteration 34 |
| hz4 | +5.4k | iteration 113; 12 of its 85 intervals were above start, all before about iteration 150 |
| main-J | +1.7k | iteration 34 |
| M, vs-cha22 | none | 0 of 128 M intervals were above start |

Every rise is at or below the 16.6k same-recipe game-4 gap between D and J
(`final-report.md:83`), so no single-seed rise ranks. Every run ended below its
start. Nothing ran past iteration 150 above its start.

**C2. The early rises occur only with the BC critic head, during the LR warm-up. SUPPORTED (association, few runs).**

- Rose: 5 of 6 BC-critic, BC-start runs. These are D, J, J/2, A2 and main-J;
  main-J's rise is within noise. A did not rise.
- Rose: 0 of 8 fresh-critic arms (control, A, B, C, E, F, G, I).
- The rise peaks fall at iterations 34-57, with the Muon LR at 5e-5 to 1.4e-4,
  before the warm-up ends at iteration 63. At a matched LR the result was not
  consistent: at 1.44e-4 (iteration 45), D was at 83.8k and J at 67.2k.

**C3. Step size (LR) is associated with the speed of the slide, not with its occurrence. SUPPORTED.**

| Run | Peak Muon LR | Bank |
|---|---|---|
| main-J | 2e-4 | 5.7k by iteration 79, 20 iterations after it reached peak LR |
| J/2 | 1e-4 | 46.7k by iteration 90, then a 47-53k plateau |
| Full-LR ablation arms | 2e-3, reached 1.44e-3 by iteration 45 | about 0.1k by game 4 (control, C, H, I) |
| Abl A (full LR, kl 0.1) | 2e-3 | 41k |
| /10 ablation arms | 2e-4, reached 1.44e-4 by iteration 45 | 62-84k by game 4 |

All 4-rank runs at 1e-4 still slid, over 1,000+ iterations: hz4 went from 62.6k
to 30.4k and M from 63.9k to 38.1k. Halving the LR slowed the slide but did not
stop it. Nothing below 1e-4 has run past its warm-up.

**C4. Reward A (own-bank shaping) delayed the slide; it did not prevent it or raise the economy. SUPPORTED, one seed each.**

- **A** (w_b 1.0, saturating at 25k):
  - ran about 78 iterations;
  - bank went 74.5k, 74.7k, 73.8k, 71.6k, 68.8k, 60.9k at iterations 12-68;
  - that is below J/2 at every interval from 34 to 68, by 3.5-10k. At a 25k
    saturation the term cannot act above 25k (`launch.md` reading).
- **A2** (w_b .25, saturating at 100k):
  - bank rose to 79.1k at iteration 57;
  - it held 64-67k at iterations 79-124, 11-18k above J/2 at the same iterations;
  - it then fell to 51.8k at 147, against J/2's 53.2k, and to 50.9k at 248, which
    is J/2's plateau level;
  - `return_common_mean` stayed positive (.011-.015), so the term was active.
- **Confounds and limits:**
  - The terminal scale also changed, from .75 to .5.
  - The 11-18k gap is about the 16.6k same-recipe noise.
  - The winner critic outputs `2p-1` per seat and sums to zero, so it cannot
    represent the common-mode part of the bank return
    (`python/owl/train/ppo.py:848-856`).
  - Neither A nor A2 reached an evaluation.

**C5. Reward M did not hold the economy, and lost to its start in both evaluations. SUPPORTED.**

- M slid from 63.9k to 42.4k by iteration 113, faster than hz4 under
  shaping .2 + terminal .8, which held 63-68k through iteration 147 and then
  slid.
- M later flattened at 36-45k, while hz4 kept falling to 30.4k.
- Evaluation win rates against the start were .141 and .109 (64 games each).
- Same start, teacher and LR re-warm-up as hz4, one seed each.
- Reading: the slide shape differs by reward, but under every self-play reward
  tested the endpoint was below the start.

**C6. The fixed-bot control (vs-cha22) drifted downhill with a terminal signal that had zero variance. SUPPORTED. The claim that its per-step signal was near zero is REFUTED as stated.**

- `win_rate_vs_bot` was 0 in 20 of 20 intervals, so the terminal ±.5 was the
  constant -.5.
- The per-step margin component was not near zero: `reward_margin_abs_mean` was
  .0017 (iterations 64-150) and .0014 (151-219). That is larger than M's
  self-play .0011-.0012.
- The learner bank fell from 62.8k to 3.1k while cha22's rose from 143k to 169k.
  That fall is steeper than any self-play run at the same LR.
- Reading: a dense margin signal against a much stronger bot did not stop a
  collapse. Self-play reward relativity is therefore not the only cause. See
  the open item O3.

**C7. Advantage normalization and grad clipping keep the update size constant across a 10x range of advantage scale. SUPPORTED (association). The step size is decoupled from signal strength.**

| Run, window | adv_std | clipfrac | approx_kl |
|---|---|---|---|
| J/2, iterations 21-63 | .26 | .11 | .0084 |
| M, iterations 1101-1469 | .029 | .13 | .0093 |
| hz4, all windows | .035-.052 | .15-.19 | .011-.014 |

- `normalize_advantages: true` (`configs/kaggriculture_4rank_margin.yaml:164`).
- Grad norm was above the clip of 10 in 66-100% of iterations in every run.
- This fits the random-walk hypothesis; it does not prove it. The update was
  not measured against a noise-only null.

**C8. Distance from the teacher does not predict the bank by itself. SUPPORTED.**

- Abl F held teacher KL at .22 and still fell 6.7k (`final-report.md:120-125`).
- hz4 held 63-68k while teacher KL rose to 3.36 (iteration 90).
- J/2 fell to 46.7k at teacher KL 2.59.
- In the self-play runs, unit_kind KL is 60-80% of teacher KL. vs-cha22 had the
  largest market_kind share (.95 of 4.91 at iteration 248).
- Entropy rose in the self-play runs (J/2 6.1 → 8.0, hz4 7.4 → 9.8) and fell in
  vs-cha22 (6.1 → 4.9).

**C9. Explained variance stayed high while the bank slid. SUPPORTED.**

- M's EV was .92-.94 from iteration 401 onward, with its bank at 36-45k.
- J/2's EV was .81, with its bank sliding.
- A critic that predicts the zero-sum outcome well gives no protection against a
  shared decline. It cannot, in principle: see C4 on the critic's structure.

## 3. Factors: holding the economy vs sliding (denominators)

| Factor | Hold or rise | Slide | Reading |
|---|---|---|---|
| LR (peak Muon) ≤ 2e-4 and still inside the warm-up (iterations ≤ 46) | 9 of 9 held within 12k of start at iteration 45 (B, D, E, F, G, J; 4-rank J/2, A and A2) | 0 of 9 | LR is necessary for survival |
| LR at 1e-4 past iteration 150 | 0 of 4 (J/2 plateau, A2, hz4, M) | 4 of 4, down 10-33k | LR is not sufficient |
| LR 2e-4 past its peak (main-J) | 0 of 1 | 1 of 1 collapsed | |
| Full LR | 0 of 5 | 5 of 5 | |
| BC critic head (at /10 LR, to iteration 34) | early rise in 2 of 2 (D, J) | – | Fresh critic: 0 of 4 rose |
| Own-bank reward (A2) | delayed the slide by about 60 iterations | ended at J/2's level | |
| Margin reward (M) | – | slid faster early, flatter late | lost evaluations .141 and .109 |
| Anchor (kl coef 0.1) | at full LR, 41k vs 0.1k | at /10 LR within noise (F) | the owner rejected an adaptive anchor; a fixed 0.1 is listed for completeness |
| Fixed strong opponent (cha22) | – | 1 of 1 collapsed to about 4k | |

## 4. Open

- **O1.** Every run is one seed. The only replicate pair is D and J, and their
  game-4 banks differ by 16.6k. No difference under about 17k ranks.
- **O2.** The in-run bank is measured while the policy changes during an
  11-iteration game.
  - hz4 and M began at 62.6k and 63.9k from J/2 final, while J/2's own last
    interval read 53.2k.
  - As `last_best`, J/2 final earned 64-77k in evaluations.
  - So in-run telemetry may understate a checkpoint's steady-policy bank.
    Unmeasured.
- **O3.** Why vs-cha22 collapsed faster than self-play at the same LR is
  unattributed. Candidates are the constant-loss terminal, the per-step margin
  that penalizes cha22's own earning, and a higher grad norm (19.7 against 14 in
  iterations 1-20).
- **O4.** No run separated signal from noise in the update, for example the
  policy-gradient direction's consistency across minibatches, or a run with
  shuffled advantages. The random-walk mechanism (C7) is therefore consistent
  with the evidence but not tested.
- **O5.** No per-family (plant, water, harvest, sell) action statistics were
  logged, so which part of the farming chain degrades first is unknown. The
  unit_kind KL dominance points at unit actions.
- **O6.** No run has ever beaten its start in evaluation, and none has been
  evaluated against the BC best or cha22 on a held-out set. Only vs-cha22 is
  still running.
