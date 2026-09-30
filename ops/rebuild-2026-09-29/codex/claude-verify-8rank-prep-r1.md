Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify `kg/rebuild-8rank-prep` r1 (8-rank prep: 10M cadence, recipe-J presets, run package)

- Worktree under review: `/Users/poonszesen/kg-v3-8rankprep`, HEAD `2eb0ee4`. Diff: `kg/isaiah-gap-closure...HEAD`, 3 commits (`669737a`, `ef515a6`, `2eb0ee4`), 41 files.
- Scratch worktree for the mutations: a detached worktree at `2eb0ee4` in this session's scratchpad (`verify-8rank`). All mutations were restored and checked with `git diff --quiet`.
- Owner request as relayed by the harness: "cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank". This review launched nothing and touched no pod.

## Summary

The core change is correct:
- The cadence is 10M on every Kaggriculture GPU config, and Orbit is untouched.
- The presets are byte copies of their base configs with only the two learning rates changed.
- The tests kill every config mutation I tried.
- The Decision quotes the owner verbatim.
- The evidence claims are scoped correctly.

The draft run package has two script defects that would bite on the pod, and one cookbook note contradicts itself. All three are small, local fixes. Apply them before any of the package runs on a pod.

## Checks

### 1. Cadence: 10M everywhere for Kaggriculture, Orbit untouched, no stale 20M. PASS, with one P3.

- **Configs.** `git grep checkpoint_freq configs/` shows `10_000_000` in `kaggriculture_{2,4,8}rank.yaml`, `kaggriculture_1gpu_eager.yaml` and both `*_bc_finetune.yaml`. `kaggriculture.yaml` (CPU) keeps `1_000`. Every Orbit config (`baseline*`, `scaling_*`, `stateless_*`, `trunc_*`, `winner_ce_*`, `extend_training`) keeps `20_000_000`. The diff touches no Orbit config.
- **Semantics.** In `scripts/run_ppo.py`, `env_steps_per_iteration = horizon * n_envs * world_size` (line 404) is global. `_next_periodic_checkpoint_step` gives `(env_steps // f + 1) * f`. One threshold drives the periodic checkpoint, `_evaluate_against_last_best`, and promotion at `LAST_BEST_WIN_RATE_THRESHOLD`, which refreshes the teacher (lines 617–693).
  - The first interval closes at iteration 611 (611 × 16,384 = 10,010,624). Later intervals close at 1,221 (+610), then 1,832 (+611). The "611, then 610–611" claims in the Decision, configs, README and run statement are right.
- **Stale 20M text.** `git grep -E "20M|20,000,000|20_000_000|1,221|checkpoint cadence"` over the md/yaml/py/sh files, excluding Orbit configs and codex transcripts:
  - `README.md:125` describes the Orbit `baseline.yaml`. Correct.
  - Config comments give "half of scaling_6m's 20M". Correct.
  - The plan and `preserve-the-ppo-recipe…` show 20M as history with the 10M update. Correct.
  - The old log entries are history.
  - **P3:** `cookbook/references/shared-ppo-adapts-game-batches-without-a-second-loop.md` still says, under "## Current state", "Periodic evaluation now follows upstream's 20M-step `checkpoint_freq`" (line 50), and lists "the 20M-step checkpoint cadence" as aligned (line 17). The Decision leaves this on purpose because the note describes the reference branch. But "Current state … now follows" reads as a claim about the current code. Suggest one clause, "(reference branch; Kaggriculture configs now use 10M, see the cadence Decision)".

### 2. Fine-tune presets differ from their base configs only in the two LRs, and the tests cover them. PASS.

- **Byte diff.** Stripping comments, `diff` of `kaggriculture_8rank.yaml` against `kaggriculture_8rank_bc_finetune.yaml`, and of the 2-rank pair, shows only `adamw_lr 0.0001 → 1.0e-05` and `muon_lr 0.002 → 0.0002`. Dropping the prepended header (`tail -n +29` and `+26`) leaves only the extra line "# The base config's header follows." and the two LR lines. So "byte copies with two values changed and a header prepended" holds.
- **Test coverage.** `test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs` compares the whole loaded `FullConfig` with the LRs restored, so it is a real diff test. The workload and per-rank-shape tests are parametrized over the presets. The teacher, env and model tests include them through `_ALL`.
- **Test run.** `tests/kaggriculture/test_configs.py` passed in the scratch worktree: 78 passed. Separately, `tests/kaggriculture/test_configs.py tests/scripts/test_run_ppo.py tests/kaggriculture/test_native_env.py` at HEAD: 590 passed in 5.62 s.

### 3. Mutations: 7 config mutations, 3 SHA and custody mutations

Every config mutation was run against `tests/kaggriculture/test_configs.py`. All were killed:

| # | Mutation | Result |
|---|---|---|
| V1 | 8-rank preset `checkpoint_freq` 10M → 20M | killed: cadence test and preset diff test (2 failed) |
| V2 | 2-rank preset `vf_coef` 2.0 → 1.0 | killed: preset diff test |
| V3 | 8-rank preset `teacher_value_coef` 0.005 → 0.05 | killed: diff test and teacher test |
| V4 | 8-rank base `checkpoint_freq` → 20M | killed: 3 tests (cadence, preset diff, ranked-differ) |
| V5 | 2-rank preset `muon_lr` 0.0002 → 0.00021 | killed: LR / 10 test |
| V6 | 8-rank preset `native_threads` 2 → 4 | killed: diff test |
| V7 | 8-rank preset `lr_min_ratio` 0.01 → 0.02 | killed: diff test |

The SHA and custody paths have no automated tests, so I exercised them directly:

- **S1: wrong BC file.** `copy_bc_best.sh` with `KG_BC_LOCAL` pointing at a random-bytes `checkpoint_bc_best.pt` printed `shasum: … did NOT match` and exited 1 before any ssh. The gate works. The Mac's durable copy `~/kg-v3-runs/bc-best/checkpoint_bc_best.pt` hashes to `fd854587…6f51`, the same constant as in `setup.sh`, `common.sh`, `copy_bc_best.sh` and the run statement.
- **S2: watchdog durable-copy check.** I ran `watchdog.watch` against a real process group (`sleep`) and a scratch run directory.
  - Clean run: "copied … (digest)".
  - Fault injected in `copyfile`: "COPY MISMATCH …; retrying next poll", and the corrupt copy was never promoted into place.
  - Fault injected with the verification also mutated away: the corrupt copy was accepted silently. So the check is load-bearing and it works.
  - `watchdog.py --self-test` passed.
- **S3: Mac-side verification in `pull_from_pod.sh`.** I used a fake `rsync` that maps `fakepod:` to local paths, with a wrong digest in the receipts file. This found **F2** below:
  - (A) With a watchdog-format line (`<run>/checkpoint_….pt`) it printed `HASH MISMATCH …`, but still exited 0.
  - (B) With the line format `kg_post` writes (absolute pod path) it printed nothing and exited 0, so verification was silently skipped.

### 4. Setup, launch and watchdog scripts

- **Quoting.** Commands are built as arrays (`cmd=(…)`, `"${cmd[@]}"`, `printf '%q '`). Variables are quoted. The ssh strings use single-quoted remote paths inside double quotes, and the credential install's `$HOME` and `$$` expand on the remote side. `bash -n` was already recorded; I found no quoting defect.
- **torchrun arguments.** They match run_ppo's parser: `--load-model-weights`, `--load-model-weights-mode model_only` (also run_ppo's default), `--log-mode wandb`, `--wandb-mode online`, `--experiment-id` (a W&B group; the regex accepts `kg-v3-8rank-recipe-j-YYYYMMDD`), `--max-runtime-hours`, `--max-env-steps`, and `-o env.native_threads=N` (the value goes through `yaml.safe_load`, so it arrives as an int and passes `strict=True`).
  - `run_ppo` writes into `output_dir/<timestamp>/` (line 1063), which matches the `*/attempts.jsonl` globs.
  - The metric keys the watchdog uses exist: `train/bank_games` and `train/own_bank_mean` from `self_play_bank_metrics(prefix="train/")`.
  - The `attempts.jsonl` fields `telemetry_mode` (`wandb-online`), `wandb_entity`, `wandb_project` and `wandb_run_id` match `logging.py`.
- **Driver gate.** `setup.sh` step 2 requires exactly 8 GPUs, each on driver 595.91.07, which matches `KAGGRICULTURE_PROBED_COMPILE_STACK`. On any other driver it stops and points to the ATEN A/B re-probe; it never changes the driver. Step 6 then calls the real `check_compile_stack(installed_compile_stack())`.
- **Credentials.**
  - `export_wandb_netrc_entry.py` refuses to write to a TTY.
  - The pod side uses `umask 077`, a temp file then `mv`, and mode 600. It refuses an existing `~/.netrc` and an empty entry.
  - `setup.sh` prints only `wandb.Api().default_entity`.
  - Nothing uses `set -x`. The secret is never printed.
- **Checkpoint copy-off.** The watchdog hashes every checkpoint, re-hashes the durable copy, and names `last_best` versions by digest. `pull_from_pod.sh` keeps `last_best` versions by digest too. That part is sound, apart from F2.

### 5. Run statements against plan 6.3b. MOSTLY PASS, with P3 gaps.

The following are covered:
- the memory smoke with the teacher and a forced evaluation;
- complete-work SPS over complete iterations with 16 optimizer steps each, and phase costs;
- the all-reduce with `nvidia-smi topo -m`;
- vCPU and `native_threads`;
- the log-ratio alarm;
- W&B `kg-v3` online;
- seeds at world size 8, with the limit stated that `run_ppo` does not log per-rank seeds;
- the live price and owner approval before creating the pod.

The P3 gaps:
- Plan 6.3b's memory smoke asks for a forced evaluation "at dense BC positions". The `memory-smoke` step forces a normal evaluation from game start (`-o rl.checkpoint_freq=16384`) and does not say it dropped the dense-position part. Either add it or record the omission as a limit.
- Plan 6.3b qualifies `configs/kaggriculture_8rank.yaml`. The package qualifies `kaggriculture_8rank_bc_finetune.yaml` instead. The shapes are identical and only the LRs differ, so this is defensible, but the run statement should say it substitutes the preset for the base config.
- The plan asks for "learner turns (valid learner-seat actions)". `summarize_run.py`'s `learner_seat_sps` is `2 × env steps / s`, which counts seat rows, not valid actions. Its column `perf/learner_seat_rows_per_second` is a BC-only metric (`bc.py:747`) that PPO never logs, so that column is always empty. Label the figure as rows, or drop the dead column.
- The threads-sweep decision rule says to skip 8 threads when the pod has fewer than 72 vCPUs. `qualify.sh threads-sweep` always runs 2, 4 and 8. That is an operator step; state it in the script or enforce it.
- Committing the chosen `native_threads` to the preset alone fails `test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs` (V6). The base config must change with it. Worth one line in the run statement.

### 6. The Decision quotes the owner verbatim. PASS

`decider` and the body both read exactly "cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank". The source is `user-directive:2026-09-30:cut-the-checkpoint-interval-in-half`. The Decision calls itself an owner deviation from Isaiah, not an implementer choice. The cookbook lint hook exits 0 on the Decision, the preset Reference and the handoff Reference. The Decisions index and log entries are present.

### 7. Evidence scoping. PASS, with one P2 inside the handoff Reference

- **Checked against the source receipts.** I compared the numbers with `ddcbf84:…/ablation/{final-report,attribution}.md` and the D/J `result.md` files. They match:
  - explained variance 0.84 for D and J, against B 0.29, G 0.16, and ≤ 0.29 for every fresh head;
  - game-4 banks: control 92 → B 62,212; H 74 → D 83,790; I 103 → G 63,414;
  - D +11.6k and J −4.8k;
  - a 16.6k gap between the seeds.
- **Bank rises.** "Banks rising above the BC level is NOT established" appears in both preset headers, `README.md`, the preset Reference, the handoff correction and the run statement.
- **Inferred versus measured.** The 8-rank LR transfer is labelled "reasoning, not an 8-rank measurement". The CUDA 13.2 filter is labelled "inferred, not checked". The ablation is labelled pre-landing and not Codex-verified, and "inside the warm-up" (0.736 of peak at iteration 46, from 736/1000). `cost.md` states that no 8-rank SPS has been measured.
- **P2 (F3):** the handoff Reference's launch paragraph contradicts itself. See Findings.

## Findings

### F1 (P2, required). `qualify.sh complete-work … nsys` omits `--kill=none`, so the profiled learner would likely be terminated after about 6 minutes

- `qualify.sh:94–96` wraps `torchrun` in `nsys profile --trace=cuda,nvtx,osrt --delay=300 --duration=60 …` with no `--kill`.
- The repo's canonical workflow, `cookbook/workflows/profile-cuda-bottlenecks-with-nsight-systems.md:19`, says: "Choose non-terminating capture behavior (`--kill=none`, and `--wait=all` where supported)… Never kill or restart a learner merely to satisfy profiling."
- As I recall the Nsight Systems CLI, `--kill` defaults to `sigterm` when collection ends because of `--duration`. On that default, torchrun and its 8 workers receive SIGTERM at about 360 s:
  - the 30-minute complete-work measurement is lost;
  - `run_ppo_8` returns nonzero, and the step ends in `kg_fail`.
- `nsys` is absent on this Mac, so I did not execute this. The runtime effect is **PLAUSIBLE**. The contradiction with the workflow is **CONFIRMED**.
- **Fix:** add `--kill=none` (and `--wait=all` if the pod's nsys supports it), and record `nsys profile --help` in the receipt.

### F2 (P2, required). `kg_post` overwrites the watchdog's `checkpoints.sha256` with absolute paths, which silently disables the Mac-side hash check after the run ends

- `launch.sh:54` gives the watchdog `--receipts "$R"`. The watchdog appends `<digest>  <run_dir.name>/<checkpoint>` lines to `$R/checkpoints.sha256` (`watchdog.py:133–134`).
- After the run, `launch.sh:67` calls `kg_post "$R" "$OUT"`. `common.sh:52` then *truncates and rewrites the same file* with `find "$out" … -exec sha256sum {}`, which writes absolute pod paths.
- `pull_from_pod.sh:33–38` tests `[ -f "$DEST/run/$path" ]` and `continue`s on a miss. So every line is skipped with no output.
- The skipped check covers `checkpoint_final.pt` on the "pull once after it ends" that the README prescribes. The rewrite also erases the watchdog's history of `checkpoint_last_best.pt` digests.
- **Reproduced** with a fake-rsync harness: scenario B printed nothing and exited 0 on a wrong digest.
- **Fix:** in `kg_post`, write a separate file (for example `checkpoints_final.sha256`) with paths relative to `$out`, for example `(cd "$out" && find . -name 'checkpoint_*.pt' -exec sha256sum {} +)`, and keep the watchdog's file append-only.
- Also consider having `pull_from_pod.sh` count mismatches and unmatched lines and exit nonzero, or at least print a closing "verified N / mismatched M / missing K" line. Today a mismatch is only a stderr warning with exit 0 (scenario A). That second part is P3.

### F3 (P2, required, doc). `bc-best-starts-ppo-with-a-fresh-critic-head.md` gives contradictory launch requirements in one paragraph

The rewritten launch paragraph first says the recommended launch is `model_only` and that its `warm_start.json` "must show … mode `model_only`". Two stale sentences are left right after it:
- "So its value distillation … starts against the fresh head until the first promotion." Under `model_only` the teacher is the whole BC model. The Correction section says so itself: "value distillation starts against the BC head".
- "The run's `warm_start.json` must show SHA-256 `fd854587…6f51` and mode `model_fresh_critic_head`; the Phase 6.2 run statement must cite both."

**Fix:** scope those two sentences to the historical launch, or delete them.

### P3 (optional)

- **Shared-PPO Reference.** Its "Current state" still says "now follows upstream's 20M" (see check 1).
- **Watchdog crash on an empty `attempts.jsonl`.** `watchdog.wandb_run_path` raises an uncaught `IndexError` on an empty `attempts.jsonl` (reproduced in my harness). `run_ppo` creates the file with `open("a")` and writes one line, so the window is tiny. But one hit kills the watchdog for the whole run, taking the stop rules and the copy-off with it. Catch `IndexError`, or skip empty files.
- **Plan 6.3b gaps.** The dense-BC-positions evaluation, substituting the preset for the base config, the learner-turn labelling, and the 8-thread vCPU skip (see check 5).
- **`setsid` in `launch.sh:50`.** `setsid` is used without `-w`. It works because `bash launch.sh` is non-interactive (no job control, so `setsid` does not fork, and `$!` is the session and group leader). `setsid -w` would make that independent of how the script is invoked.
- **`allreduce` step.** It skips `kg_preflight`, so there is no idle-GPU check before the bench.
- **Test receipts.**
  - `mutations.log` records "149 passed" per mutation but not the pytest command. My whole-file run gives 78, so the selection differs and cannot be reproduced from the log.
  - `prepare.log` ran on "`7e87f54` + uncommitted changes", not on a commit. The code, config and test files did not change between `ef515a6` and `2eb0ee4`, and my targeted runs at `2eb0ee4` pass. But the log is not bound to the committed tree.

## Not checked

- Nothing ran on a GPU:
  - `setup.sh`, `check_flash_all_gpus.py` and `allreduce_bench.py`;
  - `qualify.sh` and `launch.sh` under real torchrun or nsys.
- No live W&B read.
- I did not re-run full `just prepare`.
- The ablation branch's code equivalence to the integration trainer is unverified. The preset Reference states that limit.

VERDICT: APPROVE WITH EDITS
