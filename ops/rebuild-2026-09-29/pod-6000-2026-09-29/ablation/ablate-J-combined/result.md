# Result: PPO collapse ablation J, "ablate-J-combined" (recipe J: LR / 10, BC critic head, env.seed 1000000)

**Execution: PASS. D's game-4 rise did not reproduce at the new seed.**

- **Banks** (mean of the two seats): 72.0k, 74.6k, 79.2k, then **67.2k**. Through game 3 J tracked D (D 72.2k, 76.4k, 79.6k). At game 4 D rose to 83.8k, but J fell 12.0k from its game-3 bank.
  - Game 4 − game 1 is **−4.8k**. D had +11.6k and B −11.1k.
  - −4.8k lies inside the 5.1k unsized spread from `../attribution.md`. This is discriminating outcome 2 from the run statement: D's game-4 rise cannot be told apart from seed noise.
  - The game-3 rise, +7.2k over game 1, did reproduce.
- **Trunk movement:** shared 8.45e-3 and actor-only 5.13e-3 at iteration 46. D had 8.43e-3 and 5.07e-3. As predicted from D, movement is set by the LR.
- **Teacher KL:** 1.45 at the game-4 end (iteration 45; D 1.41, B 1.36) and 1.28 at iteration 46 (D 1.45, B 0.64). As predicted from D, it is not below 0.64.
- **Critic:** the mean explained variance over iterations 1–46 was **0.84**, equal to D's 0.84 (B 0.29). The critic advantage reproduced, but it did not stop the game-4 fall.
- **Orchestrator's brief prediction** (flat near 70k, teacher KL at 46 well below 0.64, lowest trunk movement of all arms): the banks were not flat, the KL was 1.28, and the trunk was equal to D's and B's, not lowest. Not met.

This is a pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod). It is one replicate at one new seed and has not been Codex-verified. Receipts are in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/pkz85wlw`, state `finished`, online. It was renamed through the API after the run (`wb_rename_j.py`, exit 0) to `ablate-J-combined` in group `kg-v3-ppo-collapse-ablation`. The original name was `ppo-20260929-233547` in group `ppo`; job_type and tags are unchanged. Summary `_step` is 753,664 (`pod-receipts/wandb_pkz85wlw_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`.
  - 46 complete iterations on both ranks, 736 optimizer steps (16 per iteration).
  - 0 nonfinite metrics (the stop was armed and never fired), 0 tracebacks.
  - Launched at 23:35:43Z with the GPUs idle; ended 23:43:39Z, 476 s wall.
  - Mean record wall over iterations 2–46 was 9.76 s: 1,678 game SPS, with rollout 6.06 s, teacher 0.91 s and update 2.79 s. D ran at 1,839 SPS. The rollout difference was not investigated.
  - After the run: both GPUs 0 MiB / 0 %, no compute apps (`idle_after.csv`).
- **Code identity:**
  - All eleven hashes in `pod-receipts/hashes.sha256` equal D's main run. They cover the nine shared inputs, `run_d.sh` `ed87c412…1b22` and the hook `f98720bd…78f9`.
  - Overrides reached the trainer as `['rl.eval_replay_games=0', 'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001', 'env.seed=1000000']`.
  - The BC critic head was loaded: ||θ0|| is 22.0106 on both ranks.
- **Checkpoint:** `checkpoint_final.pt` `b9aab352…a2a0`, kept on the pod at `/root/runs/ablate-J-combined/20260929-233547/`.
- **Spend:** 517 s of GPU runs (the 476 s run plus the 41 s dry run), about **$0.60** at $4.18/h.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | J (seed 1e6) | D (seed 0, same recipe) | B (LR / 10, fresh critic) |
|---|---|---|---|---|
| 1 | 12 | 71,007 / 73,059 | 73,073 / 71,405 | 73,360 / 73,208 |
| 2 | 23 | 74,935 / 74,353 | 76,902 / 75,924 | 65,621 / 65,125 |
| 3 | 34 | 77,958 / 80,480 | 80,549 / 78,593 | 63,689 / 63,913 |
| 4 | 45 | 67,789 / 66,620 | 85,232 / 82,347 | 62,104 / 62,321 |

## Per-iteration metrics at the key iterations (J)

The full table is `pod-receipts/iterations.tsv`, from `../extract_iterations.py`. `value_loss` is `loss/value_loss` from the rank-0 iteration records.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0014 | 0.0007 | 0.0003 | 0.0013 | 0.004 | 0.551 | 0.53 | 0.152 |
| 13 | 0.049 | 0.030 | 0.0088 | 0.0048 | 0.053 | 0.415 | 0.75 | 0.084 |
| 23 | 0.27 | 0.15 | 0.056 | 0.0090 | 0.109 | 0.501 | 0.53 | 0.123 |
| 34 | 0.63 | 0.44 | 0.083 | 0.0132 | 0.174 | 0.404 | 0.59 | 0.079 |
| 46 | 1.28 | 0.68 | 0.30 | 0.0126 | 0.166 | 0.214 | 0.75 | 0.022 |

- At the game-4 end (iteration 45), teacher KL was 1.45 and unit-kind KL 1.20. D had 1.41 and 0.96. Market-kind KL was 0.11 in both.
- Explained variance fell below 0.6 only at iteration 1 and at the game boundaries (iterations 12, 23, 34). The mean over iterations 1–46 was 0.84.
- Policy entropy at game ends (iterations 12, 23, 34, 45) was 4.94, 5.67, 6.32 and 6.87. D had 5.00, 5.70, 6.75 and 8.29, and B 4.90, 5.31, 6.16 and 8.02. It rose in all three arms, and J's rose the least by game 4. Why it rose was not investigated. `ent_coef` is 1e-6.
- `optimizer/grad_norm` was 12.4–15.2. It exceeded `max_grad_norm` 10 in every iteration, as in D and B.

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

The DIAGNOSTIC-ONLY hook recorded these in-run values after each listed iteration's last optimizer step (`pod-receipts/trunk_audit.tsv`). Both ranks are equal to every printed digit.

| After iteration | Actor-only | Critic-only | Shared trunk | D shared |
|---|---|---|---|---|
| 12 | 6.21e-4 | 8.31e-4 | 1.04e-3 | 1.04e-3 |
| 23 | 1.74e-3 | 2.54e-3 | 2.92e-3 | 2.94e-3 |
| 34 | 3.21e-3 | 4.65e-3 | 5.36e-3 | 5.38e-3 |
| 46 | 5.13e-3 | 7.52e-3 | 8.45e-3 | 8.43e-3 |

- **Gradient audit** (rank 0, before each iteration's first optimizer step):
  - Critic-only norm: 0.33–2.68 (1.21 at iteration 1).
  - Shared norm: 9.6–17.1.
- **Group sizes and ||θ0||** equal D's: actor-only 873,406 / 88.45, critic-only 66,561 / 22.06, shared 5,312,256 / 156.81.

## Reading

**Supported:**

1. **Trunk movement is fixed by the step size.** Two seeds of the same recipe gave the same trajectory to within 1 % at every audited iteration. B, with a fresh head, is within 5 %. So the three coupled BC-head changes do not change how far the trunk moves.
2. **The BC head's critic quality is reproducible.** The mean EV was 0.84 at both seeds, against 0.29 for B.
3. **The bank *trend* through game 3 reproduces:**
   - J +7.2k and D +7.3k over game 1, where B lost 9.5k.
   - Game 3 − game 1 is therefore consistent across two seeds for the BC head. It is one more observation, not a variance estimate.
4. **D's game-4 result does not reproduce:**
   - J's game-4 bank fell 12.0k below its game-3 bank, to 4.8k under game 1.
   - The attribution's proposed loss condition was "a replicate at a new seed does not reproduce a positive game 4 − game 1". It is met.
   - B → D's +21.6k at game 4 is therefore not a stable effect of the head. Against J, the recipe's game-4 bank is 5.0k above B (67.2k against 62.2k), which is inside the spread.

**Not supported / unresolved:**

- **Whether J's game-4 drop begins a collapse or is game-to-game noise.** The attribution's loss condition, "below game 1 for two consecutive games", needs game 5, and 46 iterations cannot show it.
  - Teacher KL was still rising at game ends: 0.036, 0.27, 0.63, 1.45.
  - Unit-kind KL at the game-4 end was higher than D's (1.20 against 0.96).
  - This is consistent with slow drift that eventually costs banks, but that has not been shown.
- **Seed variance.** It is now two draws for one recipe, not an estimate. Game-4 banks for the same recipe differ by 16.6k across the two seeds. That is more than three times the 5.1k spread, which the attribution took from game-1 banks. The spread therefore understates later-game variance, and the attribution's single-seed game-4 deltas (B → D +21.6k, B → E +4.8k, B → F +4.6k) need replicates before any ranking.
- **Which action families drove J's game-4 drop.** Not measured per family. Unit-kind KL grew faster than market-kind KL.
- **Self-play banks are not strength.** No held-out evaluation ran (`rl.eval_replay_games=0`), so there is no win rate or bank margin against BC or any other opponent.
- **The cause of the rollout-time difference from D** (6.06 s against 5.10 s) was not investigated.
