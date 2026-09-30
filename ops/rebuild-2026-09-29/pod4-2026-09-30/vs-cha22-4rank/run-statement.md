# Run statement: vs-cha22-4rank-20260930 (the cha22 anchor), pre-landing launch

**Owner, verbatim (2026-09-30, latest):** "Sure you can just launch the same reward on CHA22 run now?"

This came right after the main agent reported that the self-play run M-margin-J2-4rank (W&B `r350xr3w`) lost both of its evaluations to its start point:

- at 10M env steps, a 14.1% win rate against `last_best`;
- at 20M env steps, a 10.9% win rate.

The agent recommended keeping teacher KL on for cha22. The owner's earlier messages on this setup were:

- "OK, for fixed bot, we can use cha22 (check ~/kaggriculture-v2)."
- "can we acceleerate this setup?"
- "implement the new rewrad first before we revisit the cha22 anchor setup."
- "is anchor thing ready?"

The readings below are the agents' interpretation, not owner adoption. The run takes the only 4-GPU pod, so M was stopped first (`../M-margin-J2-4rank/stop.md`).

**This is a pre-landing launch.** `kg/rebuild-opponent-mix` at `0f70773` is still under independent review, running in parallel. The branch has not landed on the main line.

- **Question.** Can our PPO pipeline improve on BC against one FIXED bot, Cha22? The pipeline is the canonical `scripts/run_ppo.py`, the same reward as M (term M) and learner-masked fixed-opponent collection. Mirror self-play gives no fixed yardstick, and a fixed bot does. The owner's view, as relayed, is that "the agent can learn from BC, but not our RL pipeline".
- **Inputs and code path.**
  - **Code:** `kg/rebuild-opponent-mix` at `0f7077319ed731894594e9a65f0671f81df37527`, sent as a git bundle and checked out clean in `/root/kg-v3-anchor`. The environment is a fresh `uv sync --frozen --group dev --extra flash-attn`, followed by `uv run --frozen maturin develop --release`, which is mandatory because the Rust env and `opponents.rs` changed. `owl.rs.assert_release_build()` must pass.
  - **Config:** `configs/kaggriculture_4rank_vs_cha22.yaml`, with no `-o` overrides:
    - `opponent_mix` bot cha22, fraction 1.0;
    - term M: `econ_shaping` 0, bank weight 0, margin 0.5/50,000/0.5, `terminal_scale` 0.5;
    - muon_lr 1e-4 and adamw_lr 5e-6;
    - `checkpoint_freq` 10M and `native_threads` 4;
    - teacher `last_best` with `teacher_kl_coef` 0.005.
  - **Warm start:** `/root/bc-best/checkpoint_bc_best.pt`, sha256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`, loaded with `--load-model-weights-mode model_only`. The optimizer and the LR warm-up start fresh.
  - **Teacher:** on a fresh launch, `run_ppo` builds the `last_best` teacher from the `--load-model-weights` student (`_create_eval_model_from_weights`), so the teacher is the BC best. `rl.teacher_init` is read only for `teacher_mode: fixed`, so no extra argument is passed. M launched the same way.
- **Command** (pod). `/root/vs-cha22/main_probe.py` and `watchdog.py` are byte copies of `/root/M-margin/`'s:

  ```bash
  cd /root/kg-v3-anchor
  OMP_NUM_THREADS=1 KG_NT_NUMA=cpu WANDB_ENTITY=spoon \
  TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton \
  .venv/bin/torchrun --nproc-per-node 4 /root/vs-cha22/main_probe.py scripts/run_ppo.py \
    configs/kaggriculture_4rank_vs_cha22.yaml /root/runs/vs-cha22-4rank-20260930 \
    --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only \
    --log-mode wandb --wandb-mode online --experiment-id vs-cha22-4rank-20260930
  ```

  There is no `--max-env-steps` and no `--max-runtime-hours`.
- **Baseline.**
  - The trainer runs no fixed-bot evaluation at iteration 0. Its first `eval/*_vs_bot` comes at the 10M checkpoint.
  - The BC start is therefore read in two ways:
    - the first game-end interval's `train/win_rate_vs_bot`, `train/own_bank_mean_vs_bot` and `train/margin_mean_vs_bot`, at about iteration 12. That is the sampled near-BC policy under a warming LR.
    - if cheap and safe, a small separate evaluation of BC against Cha22 (CPU, niced), recorded in `launch.md`.
- **Discriminating observation.**
  - Improvement means `eval/win_rate_vs_bot` and `eval/margin_mean_vs_bot` rise above the BC start over successive 10M checkpoints. Both seats count (`eval/*_vs_bot_seat_{0,1}`).
  - **Loss condition:** they fall below the BC start.
  - A flat or falling curve while `loss/*` stays finite and the learner mask holds points at the objective or the optimizer, not at collection.
- **Stopping condition.** The owner decides. A watchdog on a nonfinite `loss/*` is the only automatic stop.
- **Limits.**
  - One seed.
  - Pre-landing: the review of the cha22 branch is still open.
  - Cha22 stepping throughput at 64 envs per rank and the wall time of the fixed-bot evaluation are unmeasured.
  - Cha22 is parity-qualified by a light three-game corpus, at the default configuration only.
  - The margin scale of 50,000 is agent-proposed.
  - The M run's loss against its start is not attributed among the reward, the LR re-warm-up and the teacher.
