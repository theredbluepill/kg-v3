# Run statement: early 2-rank Kaggriculture PPO smoke (brief 1.5 draft, pre-landing)

Written before launch. **Pre-landing diagnostic:** the code is the unverified
merge branch `kg/pod-ppo-prelanding` (Task 3.1 remainder `7a55e81`, Phase 4.4
`4a662ad`, BC handoff `a814544`, merged onto `kg/rebuild-pod-6000-evidence`
`ce44850`); Codex is at its usage limit until Oct 6, so nothing here is
Codex-verified. Results: pending Codex review.

- **Question (integration, not performance or policy quality):** does the
  native env -> pinned caller-owned buffers (entry fence) -> stateless policy
  -> rollout storage -> PPO update path execute correctly on two ranks with
  matched collection/replay and disjoint rank seeds? This is brief 1.5's
  "Draft pod run statement for Claude".
- **Inputs and code path:** pod `aki4vy8kpfldpa` (2x RTX PRO 6000 Blackwell,
  driver 595.91.07), `/root/kg-v3` checked out at the `kg/pod-ppo-prelanding`
  commit that adds this statement; `.venv` from the env receipt (torch
  2.9.0+cu128, flash-attn 2.8.3, wandb 0.26.1). No Rust source differs from
  the pod's built `994818b` extension (`git diff 994818b` over src, engine_rs,
  Cargo, pyproject, uv.lock is empty), so maturin is not rebuilt.
  Config `early-smoke/config.yaml`: the 2-rank recipe with the brief's
  overrides (128 envs/rank, spm 8, accum 1, horizon 64, 1 epoch, target_kl
  null, random-init embed 32 / depth 1 / 1 head / mlp 2 / scratch 0, eager,
  bf16, no compile, `episodeSteps: 33` = 32-transition games, seed 23000,
  rank offsets 0/1 stride 2, reward win_loss W 0.2 / S 4 / D 1 / cap 0.25 /
  I 0 / I-cap 0.1, no teacher, no BC weights, pin_memory true,
  native_threads 1, checkpoint_freq null, eval_replay_games 0). Launcher:
  `early-smoke/launch_kg_run_ppo.py`, a runpy-style wrapper that calls
  `run_ppo.main()` unchanged and only observes (W&B project must be `kg-v3`
  and online; per-phase time and CUDA peaks; env seed/stride; native step
  time; terminal-record samples).
  Command: `timeout --kill-after=30 360 .venv/bin/torchrun --nproc-per-node 2
  <launcher> scripts/run_ppo.py <config> <run> --log-mode wandb --wandb-mode
  online --max-env-steps 32768 --max-runtime-hours 0.1666667`, with
  `OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon`.
  32,768 global env steps = 2 iterations x (128 envs x 2 ranks x 64).
- **Expected discriminating observations:** exit 0 after exactly 2
  iterations; finite loss/policy/value/entropy/grad metrics; optimizer steps
  16 after iteration 1 and 32 after iteration 2 (128/8 minibatches, accum 1);
  rank 0 constructs seed 23000 stride 2 and rank 1 seed 23001 stride 2
  (disjoint residues mod 2); no first-minibatch log-ratio alarm (0.05 nats);
  terminal records present each 32 steps with raw bank fields; W&B run in
  `spoon/kg-v3` online and `finished`.
- **Failure-status probe (isolated, after the smoke):** same config, a second
  run dir, `KG_PROBE_FAIL_BEFORE_UPDATE=1` makes the launcher raise inside the
  first `_update` call before any minibatch, `--max-env-steps 16384`,
  `timeout --kill-after=30 150`. Expected: nonzero exit, a rank-tagged
  traceback, and the W&B run state `failed` or `crashed` (not `finished`). An
  unavailable W&B API makes this inconclusive, not a pass.
- **Stopping condition and budget:** success after 2 complete iterations
  with the checks above. Stop on a nonfinite value, the alarm, wrong counts,
  a native exception or the cap. Cap 10 min wall for both runs together
  (external `timeout` watchdogs 360 s + 150 s) and US$5 (at $4.18/h, 10 min
  is about $0.70). No retry beyond the remaining budget.
- **Records:** `early-smoke/` receipts: git state, file hashes, run logs,
  `[kg-probe]` lines, nvidia-smi samples, W&B history JSON and state for both
  runs, `result.md`. Bulk run dirs stay on the pod.
