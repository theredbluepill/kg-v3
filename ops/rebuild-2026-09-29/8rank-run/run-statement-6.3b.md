# Run statement (DRAFT) — plan 6.3b 8-rank qualification, then the recipe-J main run

**Status: draft, not approved, nothing launched.** No 8-GPU pod exists. Creating one is a new billable resource: the owner must see the live hourly price (`cost.md`, re-read immediately before creation) and approve it explicitly. The main run needs every qualification check below to pass first. Fill every `<…>` before launch and freeze the statement (record its SHA-256) before the first pod-changing command.

## Part A — 6.3b qualification

- **Question.** Does `configs/kaggriculture_8rank_bc_finetune.yaml` (recipe J: the 8-rank config with both LRs / 10, `checkpoint_freq` 10M) run correctly on an 8x RTX PRO 6000 pod from the BC best with `--load-model-weights-mode model_only`? At what complete-work throughput, with which `native_threads`, and with what all-reduce share? It is a qualification and throughput diagnostic, not a learning result. **Substitution:** plan 6.3b names `configs/kaggriculture_8rank.yaml`; this package qualifies the recipe-J preset instead, because the main run uses it. The two files have identical shapes and differ only in `muon_lr` and `adamw_lr` (`test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs`), so the qualification's memory, throughput and all-reduce figures apply to both.
- **Inputs and code path.**
  - Source: the integration commit `<commit>`, staged with `stage_from_mac.sh` as a git bundle (nothing pushed); `git status --porcelain` must be empty on the pod.
  - Environment: `setup.sh` runs `uv sync --frozen --group dev --extra flash-attn` (torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3 from `uv.lock`) and `maturin develop --release`. The driver must be 595.91.07 on all 8 GPUs, or setup stops and points to the ATEN A/B re-probe.
  - BC best: SHA-256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51` (`copy_bc_best.sh` from `~/kg-v3-runs/bc-best/`, or volume `4llk4uaf20` in EU-RO-1).
  - Trainer: the canonical `scripts/run_ppo.py` under `torchrun --nproc-per-node 8`; no probe wrapper and no hook.
  - Telemetry: W&B online in `kg-v3`, credential via the pod credential Workflow, with experiment ids `kg-v3-6.3b-*`.
  - Hardware: pod `<id>`, `<Secure|Community>`, `<data center>`, `<vCPUs>`, driver `<…>`, at `$<live price>/h`.
- **Steps** (`qualify.sh <step>`; receipts under `/workspace/kg-v3-receipts/6.3b/<step>/`, copied to `ops/rebuild-2026-09-29/8rank-run/receipts/` afterwards):
  1. `seeds` (CPU). `seed_probe.py` builds each of the 8 ranks' native envs the way `run_ppo`'s fresh launch does and reads `seed_state`. Then the native seed-partition tests at world size 8 run. **Expected:** 256 distinct construction seeds, rank r in residue `base + r (mod 8)`, `ok: true`; the tests pass. The same probe on the Mac at this commit gave exactly that (`local-checks.md`). **Limit:** `run_ppo` does not log per-rank seeds, so this checks the launch arithmetic and native streams, not the run's own record.
  2. `memory-smoke`. One full iteration (`--max-env-steps 16384`) with the teacher on (the BC best seeds the last-best teacher), and a forced last-best evaluation (`-o rl.checkpoint_freq=16384`). **Expected:** exit 0; `teacher/cache_bytes` 418,643,968 per rank; the evaluation logged; the nvidia-smi peak per GPU at or below 83,204 MiB (85 % of 97,887). The 2-rank 6.1/6.2 peak allocated was 38,457 MiB in teacher precompute at 128 envs per rank. At 32 envs per rank the teacher cache and chunk are a quarter of that, so the peak is expected well below target. **Limit:** integration has no per-phase peak telemetry (the pre-landing probe wrapper is not on this branch), so this is the whole-GPU nvidia-smi peak, allocator reserve included. That is an upper bound on the allocated peak. **Limit (dense states):** plan 6.3b asks for a forced evaluation "at dense BC positions". No code path starts an evaluation from BC positions, so this step forces a normal evaluation from game start, and its one iteration's rollout, teacher precompute and update see only turns 0–63 of each game. The dense late-game rollout, teacher and update memory is covered instead by step 5's nvidia-smi peak (`peak_memory.txt`), since 30 minutes is expected to span several 720-turn games (one game is about 11.25 iterations; the 8-rank iteration time is unmeasured); that peak is a whole-run upper bound, not per phase.
  3. `threads-sweep`. `native_threads` 2, 4 and 8, 12 iterations each (about one 720-step game), same seed. The native step was the largest per-iteration cost at 2 ranks: 38.0 / 36.8 ms per step for 128 envs at `native_threads` 2, about 2.4 s of a 3.3 s rollout (6.2). **Decision rule:** the vCPU budget is `nproc / 8` per rank. Take the smallest `native_threads` whose mean `time/rollout_seconds` over iterations 2–12 is within 5 % of the best, with `8 × native_threads` + 8 trainer processes ≤ vCPUs. `qualify.sh threads-sweep` enforces the vCPU skip: when `nproc` is below 72 (`8 × 8 + 8`) it runs 2 and 4 only and writes the reason to `times.txt`. It does not read the cgroup quota; `kg_preflight` records `cpu.max` in `vcpus.txt`, and the operator checks it. Record the choice as `KG_NATIVE_THREADS` for the main run. Committing it as a config value is a change with its own record, and it must change `configs/kaggriculture_8rank.yaml` and the preset together: changing the preset alone fails `test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs`.
  4. `allreduce` (after `kg_preflight`'s idle-GPU check). `nvidia-smi topo -m` (RTX PRO 6000 has no NVLink; expect PCIe, but read it). `allreduce_bench.py` times one FP32 all-reduce of the model's trainable-parameter bytes, per call and × 16 optimizer steps per iteration. **Output:** the component cost and its share of `time/update_seconds` from step 5. That share is an upper bound on the exposed cost, because DDP overlaps its buckets with the backward pass.
  5. `complete-work 30 <threads> [nsys]`. 30 minutes of recipe J from the BC best. With `nsys` present, a 60 s capture window starts after 5 minutes (the canonical timeline profiler; the capture's overhead is reported). The capture uses `--kill=none`, so the learner keeps running when the window closes, and `--wait=all` where the pod's `nsys profile --help` (saved as `nsys_profile_help.txt`) offers it; the step stops before launch if that nsys has no `--kill`. **Report**, from `summarize_run.py` over complete iterations after iteration 1: global env-step SPS and seat rows per second (2 per env step, both self-play seats), the per-phase means (rollout, teacher, update, iteration), `optimizer/steps` +16 and `train/env_steps` +16,384 every iteration, `teacher/cache_bytes`, the nonfinite count (0), W&B state `running` then `finished` online (`attempts.jsonl` `telemetry_mode: wandb-online`), and the NCCL all-reduce kernel time from the capture against the update time. The first-minibatch log-ratio alarm (`rl.first_minibatch_logratio_limit` 0.05) raises inside the trainer when it fires, so "quiet" means exit 0 with no `first_minibatch_logratio` error in `run.log`. **Limit (learner turns):** plan 6.3b asks for SPS "in learner turns (valid learner-seat actions)". `run_ppo` logs no valid-action count, so seat rows are reported and labelled as rows, not valid actions. The model-only component estimate (0.90–0.95 scaling efficiency against 2 ranks) excludes the engine, copies, GAE, logging and all-reduce, so it predicts nothing end to end.
- **Pass (all needed before Part B):** steps 1–5 as expected; the driver probed; flash-attn on sm_120 on all 8 GPUs (`setup.sh` step 7); W&B online; no nonfinite value; no alarm; every iteration with 16 optimizer steps.
- **Stop at the first failure** and report it. Do not retry blindly: attribute it as an information, decision, execution or architecture error first.
- **Budget.** About 2 hours of pod time including setup (`cost.md`), and a 20–40 minute hard timeout per launch inside `qualify.sh`.
- **Safety.** Idle GPUs before each launch (`kg_preflight`). No driver, CUDA or security change. The credential is never printed. The pod is not stopped or deleted without the owner's instruction; stopping a pod without a network volume loses its disk, so pull receipts first (`pull_from_pod.sh`).

## Part B — MAIN RUN statement (recipe J, 8 ranks)

- **Question.** From the BC best, does recipe J at 8 ranks keep the self-play economy healthy past the LR warm-up (the 2-rank ablation never reached peak LR)? Does the last-best evaluation promote at the 10M cadence? The run supports only those diagnostics. Strength against other opponents needs a separate held-out evaluation.
- **Recipe.** `configs/kaggriculture_8rank_bc_finetune.yaml`: `muon_lr` 0.0002, `adamw_lr` 1e-5, the same schedule (warm-up 1,000 optimizer steps = 62.5 iterations; cosine to 400k steps), `vf_coef` 2.0, `teacher_kl_coef` 0.005, `teacher_value_coef` 0.005, `ent_coef` 1e-6, and `checkpoint_freq` 10,000,000. A checkpoint and a last-best evaluation (promotion at win rate ≥ 0.7, which refreshes the teacher) come every 610–611 iterations. The first comes at iteration 611 (10,010,624 env steps). The global workload is the 2-rank ablation's (256 envs, 16 optimizer steps and 16,384 env steps per iteration), so the LR carries over by construction. That is reasoning, not an 8-rank measurement.
- **Launch** (on the pod, after Part A):

  ```bash
  KG_NATIVE_THREADS=<from step 3> [KG_DURABLE_DIR=<network volume mount>] \
    bash ops/rebuild-2026-09-29/8rank-run/launch.sh /workspace/kg-v3-runs/main-<date> <RUNTIME_HOURS>
  ```

  which runs

  ```bash
  torchrun --nproc-per-node 8 scripts/run_ppo.py configs/kaggriculture_8rank_bc_finetune.yaml <out> \
    --load-model-weights /workspace/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only \
    --log-mode wandb --wandb-mode online --experiment-id kg-v3-8rank-recipe-j-<date> \
    --max-runtime-hours <RUNTIME_HOURS> -o env.native_threads=<N>
  ```

  `RUNTIME_HOURS` is the owner's runtime cap: `run_ppo` stops after the iteration that crosses it and writes `checkpoint_final.pt`. An outer `timeout` of the cap plus 45 minutes kills a hung run. On the Mac, keep `pull_from_pod.sh <pod> /workspace/kg-v3-runs/main-<date> 600` running for the whole run.
- **Expected records.**
  - `warm_start.json` with SHA-256 `fd854587…6f51` and mode `model_only`;
  - `attempts.jsonl` with `telemetry_mode: wandb-online` and the W&B URL;
  - a checkpoint and an `eval/*` row (win rate, `eval/promoted`, banks and margin) at each 10M interval;
  - `train/own_bank_mean` at every game end (about every 11.25 iterations);
  - `watchdog.log`, the watchdog's append-only `checkpoints.sha256` (every `checkpoint_last_best.pt` version), `launch.sh`'s end-of-run `checkpoints_final.sha256`, and the pulled checkpoints on the Mac (`~/kg-v3-runs/main-<date>/`);
  - the final pull's lines `checkpoints.sha256: verified N / mismatched 0 / missing 0` and `checkpoints_final.sha256: verified N / mismatched 0 / missing 0`, with exit 0.
- **Stop conditions** (the watchdog enforces 1 and 3; the trainer enforces 2):
  1. any nonfinite logged metric;
  2. the first-minibatch log-ratio alarm (the trainer raises, so torchrun exits nonzero);
  3. `train/own_bank_mean` below 20,000 at two consecutive game intervals (iterations with `train/bank_games > 0`);
  4. the runtime cap (a normal stop).

  **Failure to promote is not a stop:** `eval/promoted` 0 at an interval only keeps the current last-best teacher. A W&B read outage does not stop the run. The watchdog logs it, and a telemetry-dependent stop cannot fire during it. The operator reports any outage.
- **Checkpoint custody.** The pod disk is lost when a pod without a network volume stops. The watchdog hashes every checkpoint into the append-only `checkpoints.sha256` and, with `KG_DURABLE_DIR`, copies and re-hashes it. After the run, `kg_post` writes `checkpoints_final.sha256` beside it; both use paths relative to the run's `OUT_DIR`. `pull_from_pod.sh` copies every checkpoint to `~/kg-v3-runs/<run>/` on the Mac and checks both lists. It keeps every version of `checkpoint_last_best.pt` by hash, because promotion rewrites that file in place, and it checks each recorded version against its hash-named copy. A mismatch fails the pull. In the final pull (no interval) a missing file fails it too; while the run is going, missing files are only reported, since the pod may have written them after the pull's copy. Checkpoints and corpora stay out of git.
- **Resume after an interruption:** `torchrun --nproc-per-node 8 scripts/run_ppo.py <run dir> --log-mode wandb --wandb-mode online --max-runtime-hours <remaining>`. `run_ppo` resumes from the run directory's latest checkpoint, and W&B needs `resume="must"`, online only. A resume replays at most the steps since the last 10M checkpoint. Its rollout seeds start past every seed the checkpoint trained on.
- **What the run cannot show.**
  - Self-play banks are not strength.
  - The last-best evaluation plays only the previous best.
  - Win rate and bank margin by seat against the BC best and at least one other opponent, with denominators, need a separate evaluation of the pulled checkpoints.
  - One seed. The ablation's same-recipe game-4 gap was 16.6k.
- **Owner decisions before launch:** approve the pod and its live price; set `RUNTIME_HOURS`; choose whether to create the pod in EU-RO-1 with volume `4llk4uaf20` (in-place BC original and durable checkpoint copies) or elsewhere (the Mac copy and pulls).
