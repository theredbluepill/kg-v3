# Launch: vs-cha22-4rank, PPO from BC against the fixed bot Cha22 (pod abl4mvr5w1mmn4, 2026-09-30)

**The run is live, with no step or time cap.** Only the owner stops it, or the watchdog on a nonfinite loss. This record covers the launch through iteration 94, including the first game ends. Run statement: `run-statement.md`, commit `bf713cf`, committed before the launch.

**This is a pre-landing launch** (owner: "Sure you can just launch the same reward on CHA22 run now?"). `kg/rebuild-opponent-mix` at `0f70773` is still under independent review, running in parallel. The previous run on this pod, M-margin-J2-4rank (`r350xr3w`), was stopped for this launch (`../M-margin-J2-4rank/stop.md`).

## Identity

- **Code:** `kg/rebuild-opponent-mix` at `0f7077319ed731894594e9a65f0671f81df37527`, delivered to the pod as a git bundle (`0f70773 ^87beaf0`) and checked out in `/root/kg-v3-anchor`. It had 0 porcelain lines at launch (`receipts/git_state.txt`).
- **Environment build:**
  - The venv is fresh, not copied from `/root/kg-v3-M`. A copied venv keeps `#!/root/kg-v3-M/.venv/bin/python` shebangs and an editable `owl.pth` pointing at M's source. `uv sync --frozen --group dev --extra flash-attn` with `UV_LINK_MODE=copy` took 68 s.
  - `uv run --frozen maturin develop --release` took 49 s. It is mandatory here, because `env.rs` and `opponents.rs` changed.
  - `owl.rs.assert_release_build()` passed, and `owl` and `owl.rs` import from `/root/kg-v3-anchor/python/owl/`. The `.so` sha256 is `c6a09b8e…`.
  - The build versions are torch 2.9.0+cu128 and flash_attn 2.8.3. Log: `/root/prep_anchor.log`.
- **Config:** `configs/kaggriculture_4rank_vs_cha22.yaml` (sha256 `1d0d61b1…4400`), with no `-o` overrides. `FullConfig.from_file` on the pod (`receipts/config_check.txt`) gives:
  - `opponent_mix` bot `cha22`, fraction 1.0;
  - `reward_mode` `win_loss`, with `econ_shaping` 0.0, `econ_bank_weight` 0.0 and `econ_bank_cap` 0.0;
  - margin 0.5/50000.0/0.5 and **`terminal_scale` 0.5**;
  - muon_lr 1e-4 and adamw_lr 5e-6;
  - `checkpoint_freq` 10,000,000, `native_threads` 4, `n_envs` 64 per rank;
  - `teacher_mode` `last_best`, `teacher_kl_coef` 0.005, `teacher_value_coef` 0.005, and `teacher_init` None.
- **Config against M:** `diff` of M's run-dir `config.yaml` against this run's shows one difference, the added `opponent_mix: {bot: cha22, fraction: 1.0}` block. The reward, LRs, schedule, PPO coefficients, teacher and compile settings are all identical. Run-dir config sha256: `8041c60a…e2a3`.
- **Warm start:** `/root/bc-best/checkpoint_bc_best.pt`. Its sha256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51` was verified before launch and is recorded in `warm_start.json`, with mode `model_only`. `attempts.jsonl` has `start_env_steps` 0, `source_commit` `0f70773…` and `telemetry_mode` `wandb-online`.
- **Teacher = BC best:** on a fresh launch with `--load-model-weights`, `run_ppo` builds the `last_best` model from the loaded student (`_create_eval_model_from_weights`) and sets it as the active teacher. `rl.teacher_init` is only read for `teacher_mode: fixed`, so no teacher argument is needed; M launched the same way. At the first iteration the LR was 1.6e-6, so the warm-up restarts fresh.
- **Command:** `run_cha22.sh` (this directory; sha256 `9a79e4bb…9bc8`, byte-identical to `/root/vs-cha22/run_cha22.sh`) was started as a session leader with `setsid nohup … & disown` at **2026-09-30T05:23:51Z**. It runs:
  `torchrun --nproc-per-node 4 /root/vs-cha22/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank_vs_cha22.yaml /root/runs/vs-cha22-4rank-20260930 --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only --log-mode wandb --wandb-mode online --experiment-id vs-cha22-4rank-20260930`
  - No `--max-env-steps`, no `--max-runtime-hours`.
  - Environment: `OMP_NUM_THREADS=1`, `KG_NT_NUMA=cpu`, `WANDB_ENTITY=spoon`, with the inductor and triton caches in `/root/sweep-cache`.
  - GPUs had no compute apps before launch (`receipts/idle_prelaunch.csv`).
- **Wrapper and watchdog:** `main_probe.py` (sha256 `26ba5b0c…`) and `watchdog.py` (sha256 `7a8c0e98…`) are byte-identical to M's. The watchdog's observe line reads `train/own_bank_mean` at `train/bank_games` > 0. In fixed-bot mode those are self-play fields and stay 0, so it prints only heartbeats. That is observe-only; the nonfinite `loss/*` stop is unaffected.

## Checks (all ranks, through iteration 94)

- **W&B:** **https://wandb.ai/spoon/kg-v3/runs/kifqbyx5**, online. Run name `ppo-20260930-052356`, experiment id `vs-cha22-4rank-20260930`.
- **16 optimizer steps per iteration:** every rank's `optimizer/steps` read 16, 32 and 48 at iterations 1-3, and `optimizer/minibatches_per_update` was 16.
- **All ranks:** ranks 0-3 each wrote iteration records 1-94.
- **No alarm:** 0 `Traceback`, 0 `log-ratio`, and every `loss/*` finite on every rank.
- **Learner mask:** `train/policy_active_ratio` reads 0.5, so only the learned seat trains; the bot's seat is masked out.
- **Memory:** the 2 s samples (608 of them) peaked at 38,981 / 38,981 / 46,247 / 39,001 MiB on GPUs 0-3. The pid-to-GPU map has rank pid 47887 on GPU 2. Its extra 7.3 GB over the other ranks is not attributed. M peaked at about 38,960 MiB on every GPU. Host RAM was 164 of 1,511 GiB in use.
- **SPS:** 
  - **1,672 env SPS**, from wall time over iterations 2-94. The mean `perf/steps_per_second` was about 1,750.
  - That is **about 0.46x M's 3,648**. Iteration wall time climbs within each game, from about 6 s just after a game reset to about 12 s late in the game. `native_step_seconds` is 72% of wall time. The update takes 1.48 s and the teacher 0.46 s, about what M took.
  - During the rollout, each rank's 4 native threads run at 60-80% CPU while the 256-core host sits about 98% idle. In the 2 s samples (450 per GPU, the first 15 min), mean GPU utilisation was 23-25% on every GPU.
  - **Reading:** the Cha22 bot stepping on the native threads bounds the run. `native_threads` above 4 might raise SPS. That is unmeasured and no profiler was run, so it is not a performance claim.
  - The BC CPU evaluation below ran at the same time, niced and on 4 threads. Over the iterations that overlapped it, SPS was 1,668, against 1,651 over iterations 2-32. No disturbance was visible.

## First game ends (global, 256 games against Cha22 per interval)

The table reads rank-0 records, whose `_vs_bot` metrics cover all ranks (`train/bank_games_vs_bot` 256). `teacher/kl` is the KL to the `last_best` teacher, which is the BC best.

| Iteration | LR (muon) | `win_rate_vs_bot` | `own_bank_mean_vs_bot` | `opponent_bank_mean_vs_bot` | `margin_mean_vs_bot` | `teacher/kl` |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1.9e-5 | 0.0 | 62,786 | 143,340 | −80,554 | 0.02 |
| 23 | 3.7e-5 | 0.0 | 59,781 | 143,468 | −83,688 | 0.09 |
| 34 | 5.4e-5 | 0.0 | 63,179 | 141,642 | −78,463 | 0.23 |
| 45 | 7.2e-5 | 0.0 | 57,341 | 143,213 | −85,872 | 0.69 |
| 57 | 9.1e-5 | 0.0 | 56,975 | 142,698 | −85,723 | 0.95 |
| 68 | 1.0e-4 | 0.0 | 52,262 | 147,771 | −95,509 | 1.21 |
| 79 | 1.0e-4 | 0.0 | 48,312 | 147,865 | −99,552 | 1.11 |
| 90 | 1.0e-4 | 0.0 | 43,087 | 151,933 | −108,846 | 1.04 |

- **The first game end (iteration 12) is the near-BC sampled start.** The learner won none of 256 games, with its own bank at 62.8k against Cha22's 143.3k, a margin of −80.6k.
- **An early warning, not a verdict.** From iteration 45 on, as the LR passes about 7e-5 on its way to the 1e-4 peak at iteration 63, the own bank slides from 63k to 43k. The margin worsens from −80k to −109k, and the teacher KL rises from 0.02 to 1.2. That is below the BC start on the training telemetry. The run statement's loss condition reads the 10M fixed-bot evaluation, which is not yet available. main-J, hz4bpjnq and M each slid the same way once the LR warmed up. The cause (the LR, the objective or the teacher weight) is still not attributed.
- **Mechanism risk, from the config and the observed margins:**
  - Term M pays `m = clamp(0.5 × margin / 50000, −0.5, 0.5)` as a potential, plus `0.5 × sign(margin)` at game end.
  - Against Cha22 every game is lost, so the terminal part is a constant −0.5. With mean margins of −80k to −109k, most games end with `m` saturated at −0.5.
  - A saturated game returns about −1 whether it is lost by 60k or by 110k. So beyond −50k the reward cannot tell a smaller loss from a larger one.
  - The per-game margin distribution is not logged, so the saturated fraction is unmeasured. `train/return_max` at the game ends is 0.018-0.025.

## BC baseline

- **Trainer-path baseline:** the trainer runs no fixed-bot evaluation at iteration 0, so the first 10M checkpoint gives the first `eval/*_vs_bot`. The first game end above (iteration 12, LR 1.9e-5) is the sampled near-BC baseline: 0/256 wins, own 62.8k, opponent 143.3k, margin −80.6k.
- **Separate CPU evaluation of the BC best against Cha22:**
  - Script: `/root/vs-cha22/bc_vs_cha22_eval.py` (copy in this directory). It calls `run_ppo._evaluate_against_bot`, the trainer's own fixed-bot evaluation: sampled actions, with the learned seat alternating by env parity.
  - Settings: `env.n_envs` 16, `env_steps` 0, CPU, `rl.dtype` float32 and 4 torch threads, with no compile, under `nice -n 19` and no GPU.
  - It ran for 656 s (05:28-05:39Z). Log: `/root/runs/bc-vs-cha22-eval16.log`.
  - **Result:** `eval/win_rate_vs_bot` **0.0** (0/16; 0/8 in seat 0 and 0/8 in seat 1). `eval/own_bank_mean_vs_bot` **61,882**, `eval/opponent_bank_mean_vs_bot` **144,422**, and `eval/margin_mean_vs_bot` **−82,540**.
  - Limits: 16 games, one world seed (`env_steps` 0), and float32 on CPU against the trainer's bf16 GPU path. It agrees with the iteration-12 interval.

## Processes (pod) and how to stop

- `run_cha22.sh` is pid **47845**, the session and process-group leader.
- torchrun is pid 47880, and ranks 0-3 are pids 47885-47888, each in its own session.
- The watchdog is pid **48195**, run as `watchdog.py 47845 /root/runs/vs-cha22-4rank-20260930.log`. It stops on a nonfinite `loss/*` only. Its log is `/root/runs/vs-cha22-watchdog.log`.
- **To stop the run** (owner's decision only; by PID, no `pkill -f` pattern):
  1. `kill -TERM -47845; kill -TERM 47885 47886 47887 47888`.
  2. Confirm with `nvidia-smi --query-compute-apps=pid --format=csv`.
  3. The watchdog exits by itself once the run is gone. If it does not, `kill 48195`.
  4. On the Mac, `kill 80077` stops the copy-off loop. Kill its `sleep` child too, with `pkill -P 80077`.
- A SIGTERM leaves no `checkpoint_final.pt`. Only the 10M-cadence checkpoints are certain.

## Custody

- **Pod files:**
  - log `/root/runs/vs-cha22-4rank-20260930.log`;
  - run dir `/root/runs/vs-cha22-4rank-20260930/20260930-052356/`;
  - receipts `/root/receipts/vs-cha22-4rank-20260930/`: git state, hashes, idle-prelaunch, config check, and 2 s and 60 s nvidia-smi samples.
- **Mac copy-off:** `copyoff.sh` (this directory) (sha256 `8f02cd9a…efc0`) is M's loop with the name, destination and log list changed. It runs under nohup as pid **80077**, every 10 min, to `/Users/poonszesen/kg-v3-runs/vs-cha22-4rank-20260930/`, with `SHA256SUMS`. It also copies the watchdog log and the BC evaluation log. Its first pass was at 05:28:20Z.
- **First checkpoint and evaluations:** at 10M env steps, about iteration 611. At the measured SPS that is roughly 100 min after launch, around 07:05Z. The trainer then runs the last-best evaluation, which alone drives promotion, and the fixed-bot evaluation in both seats (`eval/*_vs_bot`, `eval/*_vs_bot_seat_{0,1}`).

## Spend

- The pod costs $8.36/h (4x RTX PRO 6000), and the run keeps billing until the owner stops it.

## Unresolved

- One seed.
- Pre-landing: the independent review of `kg/rebuild-opponent-mix` is still open.
- **Win rate starts at the floor (0/256, 0/16).** `win_rate_vs_bot` can only show improvement, never a loss. The discriminating metric for now is `margin_mean_vs_bot` and `own_bank_mean_vs_bot`.
- The train-telemetry slide after the LR warm-up (iterations 45-90) is unattributed among the LR, the objective (term M saturating at −50k) and the teacher weight.
- The throughput limit is the Cha22 stepping on 4 native threads. No profiler was run.
- Rank 2's extra 7.3 GB of GPU memory is unexplained.
- The margin scale of 50,000 is agent-proposed. It is saturated for typical losses against Cha22.
