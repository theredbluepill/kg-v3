# Run statement: vs-cha22-4rank-20260930 (the cha22 anchor), not launched

Owner, verbatim, 2026-09-30, in order: "OK, for fixed bot, we can use cha22
(check ~/kaggriculture-v2).", "can we acceleerate this setup?", "implement the
new rewrad first before we revisit the cha22 anchor setup." and "is anchor
thing ready?". The readings below are the agents' interpretation, not owner
adoption.

- **Question.** Can this RL pipeline (the canonical `scripts/run_ppo.py`,
  term M, learner-masked fixed-opponent collection) improve on the BC policy
  against one fixed bot, Cha22? The orchestrator relays the owner's view that
  "the agent can learn from BC, but not our RL pipeline". Mirror self-play
  gives no fixed yardstick; a fixed bot does.
- **Inputs and code path.** A clean checkout of the merge commit on
  `kg/rebuild-opponent-mix`, with `uv run maturin develop --release` and
  `owl.rs.assert_release_build()`. Config: `configs/kaggriculture_4rank_vs_cha22.yaml`,
  with no `-o` overrides. Warm start: `/root/bc-best/checkpoint_bc_best.pt`
  (sha256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`),
  `--load-model-weights-mode model_only`. The optimizer and LR warm-up are fresh,
  and the last_best teacher starts at BC.
- **Command** (pod; the wrapper is a byte copy of `/root/M-margin/main_probe.py`
  under its own path, so the M-margin stop line
  `pkill -f M-margin/main_probe.py` cannot reach this run):

  ```bash
  mkdir -p /root/vs-cha22 && cp /root/M-margin/main_probe.py /root/vs-cha22/main_probe.py
  cd /root/kg-v3-anchor
  OMP_NUM_THREADS=1 KG_NT_NUMA=cpu WANDB_ENTITY=spoon \
  TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton \
  .venv/bin/torchrun --nproc-per-node 4 /root/vs-cha22/main_probe.py scripts/run_ppo.py \
    configs/kaggriculture_4rank_vs_cha22.yaml /root/runs/vs-cha22-4rank-20260930 \
    --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only \
    --log-mode wandb --wandb-mode online --experiment-id vs-cha22-4rank-20260930
  ```

  There is no `--max-env-steps` and no `--max-runtime-hours`. The 2-rank twin
  is `configs/kaggriculture_2rank_vs_cha22.yaml` with `--nproc-per-node 2`.
- **Discriminating observation.**
  - The baseline is the near-BC policy. Games end about every 12 iterations
    (720 steps / horizon 64), and the first game-end interval's
    `train/win_rate_vs_bot` and `train/margin_mean_vs_bot` are the sampled BC
    policy under a warming LR.
  - Improvement means a sustained rise of `eval/win_rate_vs_bot` and
    `eval/margin_mean_vs_bot` (both seats, `eval/*_vs_bot_seat_{0,1}`) over
    successive 10M-step checkpoints, compared with that baseline.
  - A flat or falling curve while `loss/*` stays finite and the learner mask
    holds points at the objective or the optimizer, not at collection.
- **Stopping condition.** The owner stops the run. A watchdog on a nonfinite
  `loss/*` is the only automatic stop.
- **Limits.**
  - There is no pre-training BC-vs-Cha22 evaluation; the first interval is a
    sampled-policy proxy.
  - One seed.
  - Cha22 stepping throughput at 64 envs per rank and the fixed-bot evaluation
    wall time are unmeasured.
  - Cha22 is parity-qualified by a light three-game corpus at the default
    configuration.
  - At merge time the pod's four GPUs held the live run
    `M-margin-J2-4rank-20260930`. This run needs them free, and the owner
    decides when.
