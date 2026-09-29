# Run statement: PPO collapse ablation J, "ablate-J-combined" (recipe J: LR / 10 with the BC critic head, new env seed)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

- **Question:** the attribution (`../attribution.md`, commit `d7bc061`) chose
  recipe J = LR / 10 plus the BC critic head, which is D's training
  configuration. D was the only arm whose self-play banks rose (72.2k at game
  1 to 83.8k at game 4), but every arm ran with `env.seed` 0, so seed
  variance is unknown. **Does recipe J reproduce D's rising banks when the
  games are drawn from a different seed?** A plain rerun of D at seed 0 would
  only re-draw GPU nondeterminism over the same games.
- **Change versus B (`ablate-B-lr10`, W&B `spoon/kg-v3/32pahqok`):** the
  recipe, which is exactly one factor versus B:
  `--load-model-weights-mode model_only` (BC critic head kept; the initial
  last-best teacher's value target follows it) instead of
  `model_fresh_critic_head`. LR is already / 10 in B.
- **Change versus D (`ablate-D-lr10-bccritic`, W&B `spoon/kg-v3/hftcr4xa`),
  the reference this run replicates:** only `-o env.seed=1000000`. This is
  the "different env seed" the attribution says a J run must add to be
  informative. Training streams draw `base_seed + rank + k * world_size`
  (`scripts/run_ppo.py`), so seed 1 would reuse D's rank-1 games on rank 0;
  1,000,000 keeps every game disjoint from D's (global 256 envs, about 6
  resets). Everything else is D: pod checkout `e74d67e` (0 porcelain lines),
  `configs/kaggriculture_2rank.yaml`, `-o rl.eval_replay_games=0`
  (passed by the launcher, not repeated) `optimizer.muon_lr=0.0002
  optimizer.adamw_lr=0.00001`, BC best `fd854587…6f51`, 2 ranks,
  `KG_PROBE_STOP_ON_NONFINITE=1`, `--max-env-steps 753664` (46 x 16,384).
  `vf_coef` 2.0, `teacher_kl_coef` 0.005 and the schedule are config
  defaults. No anchor, stop-gradient, `vf_coef` or warm-up change.
- **Launcher and hook, unchanged from D:** `../ablate-D-lr10-bccritic/run_d.sh`
  (`ed87c412…1b22`) and its DIAGNOSTIC-ONLY, telemetry-only
  `trunk_audit_launcher.py` (`f98720bd…78f9`), which loads the shared
  launcher `b393ad31…92df` unchanged and only reads parameters and
  gradients. It records ||θ − θ0|| / ||θ0|| from the loaded weights for the
  actor-only, critic-only and shared-trunk groups after iterations 12, 23, 34
  and 46, plus per-group gradient norms before each iteration's first
  optimizer step. No training-changing hook.
- **Hook and override verification (done before this statement):** dry run
  `run_d.sh ablate-J-dryrun 32768 1,2 debug -o optimizer.muon_lr=0.0002
  optimizer.adamw_lr=0.00001 env.seed=1000000`, W&B off, exit 0, 41 s
  (`dryrun-receipts/`).
  - Overrides reached the trainer as `['rl.eval_replay_games=0',
    'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001',
    'env.seed=1000000']`; the run's `config.yaml` has `env.seed: 1000000`,
    `muon_lr 0.0002`, `adamw_lr 1e-05`, `eval_replay_games 0`.
  - All eleven input hashes equal D's main run (config, model config, shared
    launcher, `run_ppo.py`, `ppo.py`, `logging.py`, `env.py`, `rs.abi3.so`,
    BC best, `run_d.sh`, hook).
  - BC critic head loaded: ||θ0|| 22.0106 on both ranks, as in D.
  - Trunk audit after iterations 1 and 2: shared 1.18e-5 / 4.58e-5 (D dry
    run 1.20e-5 / 4.63e-5); both ranks equal.
  - Iteration-1 metrics differ from D's dry run (explained variance 0.536
    against 0.562, advantage std 0.546 against 0.537), consistent with
    different games.
- **Command:** `nohup run_d.sh ablate-J-combined 753664 12,23,34,46 wandb -o
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001 env.seed=1000000`.
- **Measurements, iterations 1–46:** `../extract_iterations.py` (banks at
  iterations 12, 23, 34, 45; teacher, unit-kind and market-kind KL; approx
  KL; clip fraction; advantage std; explained variance) plus
  `loss/value_loss`; `trunk_audit_summary.py` for the audit. Compared with D
  and B.
- **Prediction (from D, the arm with this exact configuration):** banks rise
  from game 1 to game 4 (D +11.6k); teacher KL at the game-4 end near D's
  1.41, not below B's 0.64; shared-trunk change near D's 8.43e-3, since
  movement tracked LR in every pair and not the head; explained-variance
  mean well above the fresh-head arms' 0.29. The orchestrator's brief
  predicted flat banks near 70k, teacher KL at iteration 46 well below 0.64
  and the lowest trunk movement of all arms; D's measurements contradict the
  last two, and I record that prediction here only as given.
  **Discriminating observations:**
  1. Game 4 − game 1 positive and beyond the 5.1k unsized spread: D's rise
     reproduces at a new seed.
  2. Game 4 − game 1 within ±5.1k: D's rise is not distinguishable from seed
     noise; B → D's +21.6k is weakened.
  3. Banks fall like B's (−11.1k) or worse: the BC-head effect in D was seed
     luck.
- **Confounds, named in advance:** one new seed is a single replicate, not a
  variance estimate; the BC head still changes three things at once (initial
  critic function, teacher value target, initial value-gradient size);
  self-play banks are not strength and no held-out evaluation runs here.
- **Stopping condition and budget:** 46 complete iterations, the first
  nonfinite metric or trainer fault, or the watchdog `timeout --kill-after=30
  600`. D took 439 s (about $0.51 at $4.18/h); the dry run took 41 s (about
  $0.05). W&B online; afterwards the run is renamed through the API to
  `ablate-J-combined` in group `kg-v3-ppo-collapse-ablation`. GPUs left idle.
