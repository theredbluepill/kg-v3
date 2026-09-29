# PPO collapse attribution: control, A–I, and the combined recipe J

**Pre-landing diagnostic; one seed per arm, and the same seed in every arm.**
All ten runs are 2-rank runs on `kg/pod-ppo-prelanding` `e74d67e` (0
porcelain lines on the pod) with the default `env.seed` 0, so rank seeds
are 0/1 in every arm. **Seed variance is unknown.** The arm-to-arm spread
below is not a seed-variance estimate: it mixes small policy differences
with GPU nondeterminism under identical seeds. Nothing here is
Codex-verified. It supports no selection, ranking or submission. The
measurements come first. The interpretation is in its own section and is
marked as inferred.

## Arms

| Arm | LR | Critic head | Value path to trunk | `vf_coef` | `teacher_kl_coef` | Hook | W&B |
|---|---|---|---|---|---|---|---|
| Control 6.2 | full (Muon 0.002 / AdamW 1e-4) | fresh | on | 2.0 | 0.005 | none | `7k07gp7c` |
| A | full | fresh | on | 2.0 | **0.1** | none | `y8j6vzky` |
| B | **/ 10** | fresh | on | 2.0 | 0.005 | none | `32pahqok` |
| C | full | fresh | on | 2.0 | 0.005 (0 in warm-up) | critic warm-up, **changes training** | `lpt11y9j` |
| D | / 10 | **BC** (`model_only`) | on | 2.0 | 0.005 | trunk audit, telemetry only | `hftcr4xa` |
| E | / 10 | fresh | on | **0.5** | 0.005 | trunk audit | `hi2lqsrx` |
| F | / 10 | fresh | on | 2.0 | **0.1** | trunk audit | `nur61v3a` |
| G | / 10 | fresh | **stop-grad** | 2.0 | 0.005 | stop-grad, **changes training**, plus trunk audit | `yz34n7i6` |
| H | full | **BC** | on | 2.0 | 0.005 | trunk audit | `ksqdtvm0` |
| I | full | fresh | **stop-grad** | 2.0 | 0.005 | stop-grad plus trunk audit | `o9c5ji4v` |

All W&B runs are in `spoon/kg-v3`. Every arm shares the nine input hashes
(config, model config, shared launcher, `run_ppo.py`, `ppo.py`,
`logging.py`, `env.py`, `rs.abi3.so`, BC best `fd854587…6f51`); each
result.md records them. Every arm also uses `rl.eval_replay_games=0`,
`--max-env-steps 753664` and 16 optimizer steps per iteration.

## Measurement

### Reproduction check (made for this note)

`extract_iterations.py RUN_LOG` was rerun on every `pod-receipts/run.log`,
and on `../6.2/pod-receipts/run.log` with `MAX_ITER` 46. Each output equals
the committed table byte for byte:

| Arm | Table | sha256 (first 16 hex) |
|---|---|---|
| Control | `ablate-A-anchor/control-6.2-iterations-1-46.tsv` | `f0aed514ab9c28b0` |
| A | `ablate-A-anchor/pod-receipts/iterations.tsv` | `4962064c3be56696` |
| B | `ablate-B-lr10/pod-receipts/iterations.tsv` | `79a508a1f68c9d74` |
| C | `ablate-C-critic-warmup/pod-receipts/iterations.tsv` | `dd9cca2fef56ff67` |
| D | `ablate-D-lr10-bccritic/pod-receipts/iterations.tsv` | `735723f5e4ae6a58` |
| E | `ablate-E-lr10-vf05/pod-receipts/iterations.tsv` | `f8eb0cc2877b6f41` |
| F | `ablate-F-lr10-anchor01/pod-receipts/iterations.tsv` | `e72b99983e972028` |
| G | `ablate-G-lr10-critic-stopgrad/pod-receipts/iterations.tsv` | `d57ee7c5c37650e9` |
| H | `ablate-H-full-bccritic/pod-receipts/iterations.tsv` | `313c7c5fdc8f8cb5` |
| I | `ablate-I-full-critic-stopgrad/pod-receipts/iterations.tsv` | `475d73fba3ac2bba` |

Every table has iterations 1–46, 0 nonfinite metrics, and
`optimizer/steps` 736 at iteration 46.

### Per-arm values

- **Banks** are the completed-game final banks at iterations 12, 23, 34 and
  45, as the mean of the two seats' rank-reduced means.
- **Teacher KL** is the per-iteration `teacher/kl` at iteration 45 (game-4
  end, matched phase) and at iteration 46 (first iteration of game 5).
- **EV** is the arithmetic mean of `train/explained_variance` over
  iterations 1–46.
- **Trunk** is ||θ − θ0|| / ||θ0|| at iteration 46, given as actor-only /
  shared:
  - D–I: in-run `trunk_audit.tsv`. Both ranks are equal to every printed
    digit.
  - A, B, C: each arm's `checkpoint_final.pt` compared with the BC best
    (`ablate-D-lr10-bccritic/pod-receipts/final_checkpoint_change_vs_bc.tsv`).
    The method reproduces D's in-run value exactly.
  - Control: **not measured.** Its only kept checkpoint is from iteration
    235.

| Arm | Game 1 | Game 2 | Game 3 | Game 4 | Game 4 − game 1 | Teacher KL it 45 / 46 | EV mean | Trunk actor / shared |
|---|---|---|---|---|---|---|---|---|
| Control | 73,182 | 54,960 | 7,484 | 92 | −73,090 | 3.25 / 5.28 | 0.54 | n/a |
| A | 71,438 | 60,140 | 50,198 | 41,432 | −30,006 | 2.49 / 2.14 | 0.37 | 4.98e-2 / 7.41e-2 |
| B | 73,284 | 65,373 | 63,801 | 62,212 | −11,072 | 1.36 / 0.64 | 0.29 | 5.12e-3 / 8.02e-3 |
| C | 70,750 | 4 | 0 | 0 | −70,750 | 8.26 / 35.2 | 0.69 | 2.89e-2 / 7.26e-2 |
| D | 72,239 | 76,413 | 79,571 | **83,790** | **+11,550** | 1.41 / 1.45 | **0.84** | 5.07e-3 / 8.43e-3 |
| E | 75,198 | 73,961 | 72,933 | 66,989 | −8,209 | 1.14 / 0.63 | −0.10 | 5.10e-3 / 7.86e-3 |
| F | 73,460 | 71,944 | 70,798 | 66,810 | −6,650 | 0.22 / 0.21 | 0.23 | 5.14e-3 / 7.78e-3 |
| G | 70,118 | 64,582 | 66,544 | 63,414 | −6,705 | 0.69 / 0.58 | 0.16 | 5.17e-3 / 7.81e-3 |
| H | 76,738 | 60,066 | 20,616 | 74 | −76,665 | 4.66 / 4.92 | 0.79 | 4.88e-2 / 8.14e-2 |
| I | 70,870 | 49,994 | 4,801 | 103 | −70,766 | 2.31 / 3.21 | 0.06 | 4.70e-2 / 8.59e-2 |

**Unsized spread.** Game-1 banks, which come before any arm has moved far
from BC, range 70,118–76,738 across all ten arms. Across the five LR / 10
arms (B, D, E, F, G), where teacher KL was 0.04 or less at game-1 end, they
range 70,118–75,198, a spread of 5.1k. Differences of about 5k or less are
treated below as indistinguishable from this spread. That threshold is an
observed range, not a confidence interval.

### Factor effects from matched pairs

Each pair differs in exactly one factor. Δ is the second arm minus the
first; ratios are the second arm over the first.

| Factor (from → to) | Pair | Game-4 bank Δ | Game 4 − game 1 | Teacher KL it 45 ratio | Teacher KL it 46 ratio | Shared / actor trunk ratio |
|---|---|---|---|---|---|---|
| **LR full → / 10** | control → B | 92 → 62,212 (+62.1k) | −73.1k → −11.1k | 0.42x | 0.12x | n/a (control unmeasured) |
| | H → D | 74 → 83,790 (+83.7k) | −76.7k → +11.6k | 0.30x | 0.29x | 0.104x / 0.104x |
| | I → G | 103 → 63,414 (+63.3k) | −70.8k → −6.7k | 0.30x | 0.18x | 0.091x / 0.110x |
| **Critic head fresh → BC** | B → D | 62,212 → 83,790 (**+21.6k**) | −11.1k → **+11.6k** | 1.04x | 2.27x | 1.05x / 0.99x |
| | control → H | 92 → 74 (0) | −73.1k → −76.7k | 1.43x | 0.93x | n/a |
| **Value path on → stop-grad** | B → G | 62,212 → 63,414 (+1.2k) | −11.1k → −6.7k | 0.51x | 0.91x | 0.97x / 1.01x |
| | control → I | 92 → 103 (0) | −73.1k → −70.8k | 0.71x | 0.61x | n/a |
| **`vf_coef` 2.0 → 0.5** | B → E | 62,212 → 66,989 (+4.8k) | −11.1k → −8.2k | 0.83x | 0.99x | 0.98x / 1.00x |
| **Anchor 0.005 → 0.1** | B → F | 62,212 → 66,810 (+4.6k) | −11.1k → −6.7k | 0.16x | 0.33x | 0.97x / 1.00x |
| | control → A | 92 → 41,432 (**+41.3k**) | −73.1k → −30.0k | 0.77x | 0.41x | n/a |

Supporting measurements:

- **Critic quality (EV mean, 1–46):** D 0.84 and H 0.79, both with the BC
  head. The fresh-head arms at LR / 10 were B 0.29, F 0.23, G 0.16 and
  E −0.10. The control had 0.54 and I 0.06.
- **Per-update step:** the approx-KL mean over iterations 1–46 was
  0.0077–0.0093 in every LR / 10 arm, against 0.050–0.130 at full LR. The
  clip-fraction mean was 0.094–0.117 against 0.33–0.47.
- **Gradient clipping:** `optimizer/grad_norm` exceeded `max_grad_norm` 10
  in every iteration in E and F, per their results.

Arm C (value-only critic warm-up at full LR) has no matched partner. It
collapsed by game 2, the fastest of any arm, and is excluded from the
recipe.

## Interpretation (inferred from the measurements above)

### Which factors helped

1. **Learning rate is the dominant factor for survival.** In all three
   pairs, LR / 10 turns a game-4 collapse to about 0.1k into 62–84k. Trunk
   and actor movement scale almost exactly with LR: 0.09–0.11x at 0.1x LR,
   whatever the loss mix, critic head or value path. Every full-LR arm
   except A (anchor) collapsed below 1k by game 4. This includes H (BC
   critic) and I (stop-grad).
2. **The BC critic head is the only factor that turned the bank trend
   upward, and only at LR / 10.** B → D is +21.6k at game 4, four times the
   unsized spread, and flips game 4 − game 1 from −11.1k to +11.6k. At the
   same step size, trunk movement was equal (1.05x) and matched-phase
   teacher KL was equal (1.04x). So the head changed the *direction* of the
   update, not its size or its distance from BC. At full LR the same head
   did nothing for game 4 (control → H). It raised game 3 from 7.5k to
   20.6k, which is within an unsized single-seed difference. The head
   changes three things at once: the critic's initial function, the teacher
   value target and the initial value-gradient size. D cannot say which of
   the three mattered.
3. **The anchor (0.1) helps at full LR and is not shown to help at
   LR / 10.** At full LR, control → A gave +41.3k, a large partial delay:
   A still fell 30k over four games. At LR / 10, B → F cut game-4 teacher
   KL 6x (0.16x) but gave +4.6k in banks. That is within the spread. Its
   effect on banks at LR / 10 is not established.
4. **Stop-grad and `vf_coef` 0.5 show no bank effect beyond the spread.**
   - Stop-grad halved matched-phase teacher KL at LR / 10 (0.51x) and cut
     it less at full LR (0.71x). It did not move banks (+1.2k, 0). It
     weakened the critic: EV 0.16 against 0.29, and 0.06 against 0.54.
   - `vf_coef` 0.5 gave +4.8k in banks, which is within the spread. It
     made the critic worse (EV −0.10). The trunk-gradient norm was
     unchanged (E result).
5. **Distance from BC does not rank the arms.**
   - At LR / 10, game-4 teacher KL spans 0.22 (F) to 1.41 (D).
   - The arm with the most drift, D, is the only one whose banks rose.
   - F and G stayed closest to BC and still fell 6.7k.
   - What separated D from the rest is critic quality (EV 0.84 against
     0.29 or less), not closeness to BC.
6. **Additivity.** The anchor's LR / 10 effect is within the spread, and
   no arm combines the anchor with the BC head. So additivity with D is
   untested and cannot be claimed. D's rise came with teacher KL rising to
   1.41. A coefficient that held F at 0.22 could suppress that movement as
   easily as protect it. Which one happens is unmeasured.

### Does any arm show PPO improving on BC?

**Only D, and only on self-play banks.**

- **The BC level.** In the 6.1 forced evaluation, the BC policy's games
  ended near 70k per seat (last-best 72,377 over 128 games). Game-1 banks
  here are 70.1–76.7k.
- **D's banks.** D's game-3 and game-4 banks are 79.6k and 83.8k (seat 0
  85,232 / seat 1 82,347). Both are above every arm's game-1 bank and above
  the 6.1 BC evaluation, and they were still rising at game 4.
- **No other arm** ended any game after game 1 above its own game-1 bank.

**Limits:**

- **Self-play banks are not strength.** Both seats run the same policy, so
  a joint rise may reflect a less contested mirror economy, not a stronger
  player.
- **No held-out evaluation ran** (`rl.eval_replay_games=0`). There is no
  win rate or bank margin against BC or any other opponent.
- **One seed.** The seed is shared with every other arm.

### Unresolved attribution

- **Seed variance.** It is unknown in every arm, and all ten arms used the
  same env seed.
- **The BC head's three coupled changes:** initial critic function,
  teacher value target and initial value-gradient size.
- **Whether D's rise persists past 46 iterations**, or whether it is a
  slower version of the same drift. Teacher KL was still rising
  monotonically at game ends: 0.046 / 0.29 / 0.81 / 1.41.
- **Anchor × BC head, stop-grad × BC head, and any LR between LR / 10 and
  full.** None of these was run.
- **Which action families produce D's gain or the others' decline.** This
  was not measured per family.
- **Teacher value distillation.** Its trunk gradient was blocked together
  with the value loss in G and I. It was not separated.

## Combined recipe J

**J = LR / 10 plus the BC critic head, with nothing else changed.** This is
D's training configuration. The factors that helped beyond the spread are
exactly these two:

- LR / 10: three of three pairs.
- The BC head at LR / 10: +21.6k and the only upward trend.

The other factors are left out for these reasons:

- **Anchor 0.1:** its bank effect at LR / 10 is within the spread, and its
  interaction with the BC head is unmeasured. It could suppress the drift
  that came with D's gain (point 6 above).
- **Critic stop-grad:** no bank effect at either LR, and a weaker critic,
  which is the opposite of the lever that worked.
- **`vf_coef` 0.5:** within the spread, and a worse critic.
- **Critic warm-up (C):** harmful.

**The best critic-path fix is a load mode, not code.** The BC critic head
comes from `--load-model-weights-mode model_only`, a mode the canonical
`scripts/run_ppo.py` already has. J therefore needs **no training-changing
hook**. The only hook is D's telemetry-only trunk audit, kept for
comparability.

**Exact specification:**

- **Code:** `kg/pod-ppo-prelanding` `e74d67e`, config
  `configs/kaggriculture_2rank.yaml`, 2 ranks.
- **Load:** `--load-model-weights /root/bc-best/checkpoint_bc_best.pt
  --load-model-weights-mode model_only`. The BC critic head was confirmed
  loaded in D: `critic_head` ||θ0|| 22.0106.
- **Overrides:** `-o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`.
  Everything else is config default: `vf_coef` 2.0, `teacher_kl_coef`
  0.005, `teacher_value_coef` 0.005, `ent_coef` 1e-6, and the schedule with
  1,000 warm-up steps and cosine decay to 400k at `lr_min_ratio` 0.01.
- **Hooks:** `ablate-D-lr10-bccritic/trunk_audit_launcher.py`
  (`f98720bd…78f9`, DIAGNOSTIC-ONLY, telemetry-only, reads parameters and
  gradients). No `critic_stopgrad_launcher.py` or `critic_warmup_launcher.py`.
- **Launcher:** `ablate-D-lr10-bccritic/run_d.sh` (`ed87c412…1b22`) runs
  exactly this. It already passes `-o rl.eval_replay_games=0`, so that key
  must not be repeated (H attempt 1 failed on the duplicate).

**What a J run must add to be informative** (for its run statement, not
part of the recipe):

- **A different env seed**, `-o env.seed=<N>` with N ≠ 0, so the run is a
  replicate of D rather than a re-draw of the same seed. This override has
  not been dry-run.
- **A longer horizon** than 46 iterations, to test whether the rise
  persists. `run_d.sh` hard-codes `timeout 600` on torchrun and `timeout
  700` on the GPU sampler, which kills anything much past about 46
  iterations. A longer J therefore needs a copy of the launcher with only
  those limits raised.
  - For scale: 235 iterations (3,850,240 env steps, 6.2's length) at D's
    measured 1,839 game SPS is about 35 minutes, or about $2.4 at
    $4.18/h. This is a projection, not a measurement.
- **A held-out evaluation** of the final checkpoint against the BC best and
  at least one other opponent, reporting win rate and bank margin by seat
  with denominators.
  - `rl.eval_replay_games` does not provide this: it is replay export.
  - The in-trainer last-best evaluation runs only at `checkpoint_freq` 20M
    env steps.

**Loss conditions** (proposed, anchored to observed values, not validated):

- Game-end banks fall below the J run's own game-1 bank for two
  consecutive games.
- A replicate at a new seed does not reproduce a positive game 4 − game 1,
  which would make D's +11.6k within seed noise.
- Any nonfinite metric, or an iteration with other than 16 optimizer steps.
- The held-out evaluation shows no win-rate or bank-margin gain over the
  BC best. In that case the self-play rise is not strength.
