# Evidence from v2 history: getting an RL agent to earn money (read-only)

Scope: `~/kaggriculture-v2` @ 23f75800 (READ-ONLY) and `kg-v3` branch `kg/reference-2026-09-29` (65f0eac).
Only mechanisms and evidence are reported here. No v2 score, gate or ranking transfers to v3. v2 bank
figures are given only to show a mechanism's size inside v2.
Paths abbreviated: `V2=~/kaggriculture-v2`, `SP=V2/ops/myolie-dagger-2026-09-22/selfplay.py`,
`LOG=V2/cookbook/log.md`, `REG=V2/cookbook/references/myolie-league-pool-registry.md`,
`GAP=V2/ops/myolie-league-b1-2026-09-26/codex-assist/cha22-gap/README.md`.
Status: **S** = supported in v2 by the cited check, **R** = refuted, **O** = open. "(v3: O)" means the v2
result is supported but whether it applies to v3 has not been checked.

## 1. Did v2 PPO ever improve on its BC/start in self-play? Yes, slowly, under a tight trust region

- **S**: v2 learners were cloned from cha22 (a1-it048, "cha22 behaviour cloning", REG:16). Mirror or league
  PPO was then promoted repeatedly on fresh paired gates: m8 over m3 at gate 15, with own bank 78.0k→81.6k
  and margin +5,350 (CI +490..+10,210) (REG:96). Later promotions: gates 17, 21, 27, 29, 35 and 41 (REG:97-110).
  These are small, cumulative gains. None of the steps was a jump.
- **S**: every v2 phase ran with sampling temperature 0.5, **a fixed KL to a teacher (beta 0.5, later 0.05)**,
  `--require-outcomes` and `--stop-ref-kl 0.05` (REG:109). The LR was about 1e-6 to 3e-6 on Muon/AdamW (SP:4022-4024).
  The documented reason: in sp-p1, LR 2e-5 "reached approx KL 0.02 after 3-5 steps; 3e-6 used 0.004-0.01 per update"
  (SP:4023). The gradient clip norm was 1.0 (SP:3995).
- **S**: larger steps hurt. p2 restarted the cosine LR at about 2.9e-6, 5x b4's end of 5.8e-7, and lost at gate 1:
  85 vs 118 wins of 880, with a v43 decrease (LOG:1502-1507). The same LR jump was aborted in p3, and p3r
  restarted at a constant 1e-6. In the one-epoch T1 sweep, 1e-4 lost 1,392 greedy service choices net while
  1e-5 gained 326 (V2/cookbook/references/myolie-service-update-magnitude-needs-direction-checks.md, table).
- **S**: a loss term that is too large destroys the economy within a few updates. The m19 a002 investment floor
  (about 600 per learner turn) moved KL to the teacher from .05 to .42, entropy from .39 to 1.75, and own bank from
  89.1k to 61.2k within 4 updates (LOG:325-333).
- **S**, in the v3 reference branch: PPO from the BC best at AdamW 1e-4 had KL 1.89/1.05/0.37 over its first 3
  iterations. Its win rate against the BC incumbent fell 12.5%→3.5% (`git show kg/reference-2026-09-29:cookbook/references/bc-bootstrap-uses-native-replay-features-and-current-heads.md`,
  "Evidence and limits"). The run then crashed, so this is only 3 iterations of evidence.
- **S**: many PPO phases were flat or worse, not better. m4: margin −5,441. m5: margin −5,231, significantly
  worse. m6 and m7: flat. m12: plateau. m13 (cha22-only opponent) and m14 (share 0.25): own bank flat at about
  85k (REG:92-102).
- **O (v3)**: none of the six v3 runs had v2's teacher-KL/stop-KL guard, and their LR was far above 1e-6.
  Whether a fixed (non-adaptive) KL to BC is allowed is the owner's call ("we dont want adaptive anchor").
  v2 evidence only says that v2's gains happened inside such a guard.

## 2. Why v2 PPO drifted or collapsed: diagnosed mechanisms

1. **Advantage normalization amplified noise. S.** With per-rival lagged scales, three rivals were each scaled
   52-64x and produced about 99.9% of the squared update (LOG:1875-1880). v2 then switched to one **lagged**
   global RMS with **no centering and a floor of 0.01**, which caps amplification at 100x (SP:1429-1454, 4224-4228).
   The code's stated reason: "A same-batch divisor can reverse the expected update" (SP:1432).
   (v3: **O**. v3 normalizes each batch without a floor. On a near-zero, saturated reward like vs-cha22 this is
   the same risk, but it has not been measured in v3.)
2. **Value gradient and actor gradient share one clip group. S (v2 v5 it011).** Value gradients through the
   shared trunk shrink the actor-head step by 58-160x under one shared clip
   (V2/cookbook/references/myolie-margin-value-loss-limits-policy-updates.md:45). An unnormalized margin value
   target makes this worse. (v3: **O**. v3 gradient norms of 13-15 against clip 10 mean the clip is active. Which
   term dominates the norm is unmeasured.)
3. **A late critic plus truncated GAE inverts credit for purchases. S.** Seed 950004, step 215: buy-9 beats buy-5
   in 16/16 mirror continuations (+0.59 win_share). But A(9)−A(5) is −0.035 at λ 0.9 and −0.023 at λ 0.98, in
   0/16 pairs. The critic carries no outcome information before about turn 300 (LOG:1131-1140). Against anchors
   the inversion held in 32/32 pairs. **Training moved the logit gap toward the wrong action**, from 0.02 (a1) to
   0.17 (m3) (LOG:1182-1192). With a zero critic, λ 0.9 passes about 0 of the effect back 500 turns
   (V2/ops/myolie-dagger-2026-09-22/NETWORTH-SHAPING-DESIGN.md:61).
   Repair: 256-turn segments and λ = 1. This was m8, the first clear own-bank and margin gain (REG:96). m7, with
   128-turn segments and λ 0.98, stayed flat at 68-82k (REG:95).
   (v3: **O**, and highly relevant. The v3 horizon is 64 with λ 0.95, so a bootstrap-driven drift *direction*,
   not only a random walk, is plausible for purchase and investment heads.)
4. **Synchronized game phases. S.** When all games start together, each update trains on one phase. KL spiked up
   to 19.7 after resets, and the policy forgot the other phases. Staggering the start turns cut training-KL
   volatility from 1.98 to 0.45 and raised share
   (V2/cookbook/references/synchronized-collector-games-starve-updates-of-game-phases.md:29-38).
   (v3: **O**. The v3 Kaggriculture configs set no `truncation_prob` (python/owl/train/ppo.py:189-195), and games
   have a fixed length of 720. Check whether the 64-step batches are phase-locked.)
5. **Updates with no terminal outcome drift from the start. S (as stated in code).** "After a staggered
   (re)start no game ends for ~720 turns; league-b1/b2 drifted from INIT in exactly those updates, whose
   advantages were critic bootstrap residuals only". The repair was `--require-outcomes` (SP:4160-4163). The
   underlying run logs were not re-read here. (v3: **O**. This is the same shape as vs-cha22 drifting downhill
   on about 0.0006 reward per step.)
6. **Pure self-play from scratch never learned to earn. S.** Before the BC start, every v6 checkpoint ended at
   zero bank (V2/cookbook/references/synchronized-collector-games-starve-updates-of-game-phases.md). In the v3
   reference branch, random or early policies drained 3000 to 0 by turn 192 through market orders, while PASS or
   no-market controls kept 3000 (bc-bootstrap Reference, "Mechanism diagnosis").
7. **Cash blindness. S (v2 rare4).** Rare4's greedy action ignored its own cash in 0 of 36 states in a cha22 loss,
   and it ordered HIRE below 50 cash. Frozen-trunk critic heads learned EV of about 0 from cash (LOG:1926-1931).
   (v3: **O**. Whether v3 actor and critic actually use the bank token is not verified.)

## 3. Reward designs tried in v2

| Design | Result | Status |
|---|---|---|
| win_loss, then win_share (0.9 win + 0.1·(2·share−1)) (SP:404-414) | the baseline for all gains in §1 | S |
| share 0.5 (m4) / 0.25 (m14) | m4 margin −5,441; m14 own bank flat at about 85k | S (no gain) |
| margin/scale (`--reward margin`) | coupled with value clipping, see §2.2; no promoted run found | O |
| deaths shaping, capped zero-sum (m6) | no reduction in deaths over 45 updates (REG:94) | R, as a standalone repair |
| + ineffective-command term (m7) | flat bank; it cannot reach the step-215 credit (REG:95) | R, as a standalone repair |
| net-worth potential shaping | **NO-GO**: shifts held-out EV by −0.005..+0.002, because with γ = 1 it telescopes to a critic offset. Net worth helps as a **critic input** (late EV +0.075) (NETWORTH-SHAPING-DESIGN.md:17-27) | R as reward; S as critic feature |
| **own-bank target**, 0.7·win_share + 0.3·clip((own−200k)/200k) (m17, mirror, beta .05) | own bank 82.0k→86.8k in training; deaths and ineffective commands down 22-29%; promoted at gate 27 (LOG:435-445, 447-456) | S (one run) |
| all-econ-failure capped penalty + market entropy (m18) | self-play banks flat at 86-91k; promoted on wins (+0.083), own bank only 88.6k→89.4k (REG:106) | S (small) |
| investment floor loss (m19 a002) | own bank collapsed 89.1k→61.2k in 4 updates (LOG:325-333) | R at that scale |

- **S**: "In mirror, win_share only rewards beating the twin, so absolute earnings go unrewarded" (LOG:451). The
  one v2 recipe that directly raised own bank in mirror added a non-zero-sum own-bank term. That term needs a
  per-seat critic, not a joint zero-sum critic (LOG:452-455).
- **S**: relative reward can actively penalize a profitable investment. cha22's expansion (land, tomato, 2 hires)
  raises Myolie's own bank by +4.6k (12/16) but the rival's by more, so the margin changes by −6.8k. The policy's
  avoidance agrees with the objective (LOG:977-980). BUY_LAND/tomato was sampled 0 of 2000 times at T 0.5
  (logp −13) across m8-m11. So investment is also an **exploration** gap.

## 4. What made cha22 strong (v2 mechanism evidence)

- **S**: the gap between the cha22 clone and cha22 is **production, not overspending**. On the same seats, the clone
  loses 26-27k bank. Its sales are 31.8-32.3k lower, and it actually spends 5-6k less (GAP:38).
- **S**: cha22 invests: land, sheep, HIRE (hires 6,351 vs 4,527; land 5,000 vs 3,000) (GAP:34-36). On seed
  950004, cha22 harvests 414 wool and earns 99,490 from wool; the clone harvests 86 wool and earns 20,760. HARVEST
  committed: 488/498 for cha22 vs 148/434 for the clone. Starvation deaths: 0 vs 16 (GAP:53-60).
- **S**: the failures are chained. A 4-wheat under-purchase at step 215 leads to empty-shed FEEDs and sheep
  starvation. Correcting only that quantity restores the exact teacher state at step 266 and adds 12-69k bank
  (GAP:70-76). But adding the teacher's expansion alone can *lower* bank by 22k when labor and feed do not
  follow (GAP:85-87). Investment only pays together with hiring, feed transport and harvest.
- **S**: day-level timing. The first error is on day 9, the upkeep loss on days 10-12, and the bank gap opens on
  days 16-30. Early cash can make the failing policy *look ahead* (GAP:49). Tomato needs about 8 days to first
  yield. Late planting (after day 25) cannot pay back
  (V2/ops/myolie-economics-2026-09-27/README.md §5).

## 5. How much money is actually available in a shared market

- **S (v2 anchors)**: season income is demand-bound. Joint bank per seed was 145.7k-229.1k. Within one seed it
  varies 0.7-15% across pairings. An uncontested agent against an idle one banked 154k, while contested anchors
  banked 67-116k
  (V2/cookbook/references/anchors-split-a-world-sized-market-with-margins-near-one-percent.md:46,58).
- **O (v3)**: consequence to check. cha22's 143-179k *against BC* may partly be demand left over by the weaker
  opponent. In mirror self-play, each twin's ceiling may be about half the joint bank, not cha22's figure. v3
  should log the **joint** bank per seed (own + opponent) to separate "the economy shrinks" from "the split
  shifts".

## 6. Implications for v3 (hypotheses, not results)

Each item still needs a stated question, discriminating observation and stopping condition before any run.

1. **Step size and trust region first.** v2 gains came at LR about 1e-6 with KL about 0.004-0.01 per update and a
   hard stop at KL 0.05 to the start. Every v3 run slid from BC. Discriminating check: the per-iteration
   KL-to-BC trajectory in the `[nt-probe]` logs against the bank slide.
2. **Credit horizon.** A horizon of 64 with λ 0.95 cannot carry purchase and investment credit, which lands
   100-500 turns later (§2.3). v2's fix was longer segments or λ = 1. Check the critic's EV by game phase and the
   advantage sign on a paired purchase fork (the v2 buy-9 protocol).
3. **Normalization and clip coupling.** Use a lagged, floored advantage scale (§2.1). Measure the value vs actor
   share of the clipped gradient norm (§2.2). Skip or down-weight batches with no terminal outcomes (§2.5).
4. **Reward.** Relative reward (margin or W/L) leaves absolute earnings unrewarded in mirror play and can
   penalize investment (§3). The only v2 mirror recipe that raised own bank added an own-bank term. The reward
   choice remains the owner's.
5. **Target the production chain.** Invest, hire, feed and harvest, then sell (§4). Telemetry should track
   HARVEST committed/submitted, starvation and drought, and sales revenue, not bank alone.

## Not verified here

- v2 run logs behind the `--require-outcomes` claim (§2.5) were not re-read.
- W&B for v2 runs was not queried.
- v3 code applicability was checked only for `truncation_prob` (§2.4).
- All v2 figures come from v2's engine and panels. The v3 engine reuses v2 semantics, but that equivalence was
  not re-checked for these claims.
