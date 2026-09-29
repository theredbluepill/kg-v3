# Task 3.5 bounded local functional check

Branch `kg/rebuild-3-1`, base `79ae216` (Task 3.1 remainder with verify r1/r2
fixes). Mac (Apple M5, 24 GiB), CPU only, `OMP_NUM_THREADS=2`. Python 3.12.13,
torch 2.9.0, wandb 0.26.1. No GPU, no training run beyond the two updates below.

## Question and stopping condition

Plan text: "Tiny model, 2 envs, 2 updates, CPU: finite losses, a checkpoint
written, and an evaluation plus promotion branch exercised through
test_last_best-style tests. This is the only local run."

Question: does `scripts/run_ppo.py`'s shared path (`main` -> `create_env` ->
`PPOTrainer` -> `_run_training_loop` -> `_evaluate_against_last_best` ->
promotion -> checkpoints -> logger) complete two updates on the native
Kaggriculture env with the shipped CPU config? Expected discriminating
observation: finite logged losses for two updates, loadable checkpoints, one
evaluation per update with both promotion outcomes, no replay export. Stop after
one pytest pass and one CLI run; each bounded to 2 min and 1 GB.

## 1. Test through `run_ppo.main()` (`tests/scripts/test_run_ppo.py`)

`test_kaggriculture_two_update_functional_check_through_main[promoted|held]`
launches `main()` with argv
`configs/kaggriculture.yaml <tmp>/runs --max-env-steps 8 -o rl.horizon=2
env.config.episodeSteps=6 --log-mode debug`: model `kaggriculture_cpu`
(embed 16, depth 1, 33,615 parameters), 2 envs, horizon 2, so 4 env steps and 2
optimizer steps per update, 2 updates.

Patched launch plumbing only: `assert_release_build`, `configure_torch`, the
single-process CPU session, the probed compile stack, a recording logger, and
`FullConfig.from_file`, which after the real validated load sets
`rl.checkpoint_freq` to one update (4 env steps). Isaiah's `checkpoint_freq`
floor is 1,000 env steps; reaching it in 2 updates at 2 envs needs horizon 250,
and a horizon-500 CLI trial measured 5.09 GB max RSS (CPU attention materializes
the 709-token score matrix), so the cadence is patched below the floor in the
test only. `LAST_BEST_WIN_RATE_THRESHOLD` is 0.0 (always promote) or 1.5 (never).

Assertions: logged steps `[4, 4, 8, 8]` (training, then evaluation, per update);
every training and evaluation metric finite, `loss/total_loss` present; 4
optimizer steps; each evaluation plays 2 native games to step 5
(`episodeSteps - 1`) with `eval/promoted` equal to the branch and
`eval/promotion_threshold` equal to the patched threshold; no `eval_replays/`
directory (`eval_replay_games: 0`); exactly `checkpoint_00_000_000_004.pt`,
`checkpoint_00_000_000_008.pt`, `checkpoint_final.pt` and
`checkpoint_last_best.pt` written; the final checkpoint loads through
`_create_eval_model_for_config` + `_load_model_from_checkpoint` with
`env_steps=8`, equal weights and the trainer's game count; weights changed from
the start. Promoted: last_best checkpoint at step 8 equals the final weights, the
teacher is last_best and active, and update 2 carries a teacher cache
(`teacher/cache_bytes > 0`) where update 1 has none. Held: last_best checkpoint
stays at step 0 with the start weights, no teacher, no teacher cache.

`pytest-functional.log`: both cases and the three native smoke tests pass,
0.70 s pytest, 2.39 s wall with startup, 349.7 MB max RSS.

## 2. CLI run with W&B offline

Command (`cli-command.txt`):

```text
OMP_NUM_THREADS=2 /usr/bin/time -l uv run python scripts/run_ppo.py \
  configs/kaggriculture.yaml ops/rebuild-2026-09-29/3.5/runs \
  --log-mode wandb --wandb-mode offline --max-env-steps 64 -o rl.horizon=16
```

Horizon 16 x 2 envs = 32 env steps per update, 2 updates. No evaluation is
reached: the validated `checkpoint_freq` floor (1,000) is not crossed in 64 env
steps; the evaluation and promotion branch is covered by the test above. The
native extension had to be rebuilt in release mode first (`main` asserts a release
build): `CARGO_BUILD_JOBS=2 uv run maturin develop --release`, 28.6 s, 0.90 GB
max RSS.

Result: exit 0, 2.95 s wall, 502.3 MB max RSS (`cli-run.stderr.log`). History
read from the offline `.wandb` file (`cli-wandb-history.json`, via
`read_wandb_history.py`):

| env step | total | policy | value | entropy | teacher KL / value |
|---|---|---|---|---|---|
| 32 | 0.0017179 | 1.54e-07 | 0.00088168 | -4.565e-05 | 0 / 0 |
| 64 | 0.0013280 | 6.41e-07 | 0.00068570 | -4.408e-05 | 0 / 0 |

`check_cli_checkpoint.py` (`cli-checkpoint-check.log`) loads
`checkpoint_final.pt` with the run's saved `config.yaml` through run_ppo's own
loader: `env_steps=64`, `player_step_total=128`, `total_games_played=0` (720-step
episodes do not finish in 32 turns), W&B run id `a7ohw4cb`, 33,615 parameters,
all finite, `eval_replay_games=0`. The W&B project is `kg-v3` (the logger's
Kaggriculture branch). The run was not synced. `~/.netrc` holds an
`api.wandb.ai` entry, so the host is not key-less; offline mode sent nothing.

Custody: `custody-sha256.txt` hashes the checkpoint, the run config, the offline
`.wandb` file and the source files used. `runs/`, `wandb/` and `*.pt` are
gitignored; the run directory stays local and is not durable evidence.

Earlier trials in the scratchpad (not evidence): a debug-build launch stopped at
`assert_release_build`; `-o rl.checkpoint_freq=128` failed validation
(`>= 1000`); horizon 500 with 2,000 steps ran both evaluations in 36 s but at
5.09 GB max RSS; horizons 64 and 16 measured 0.97 GB and 0.50 GB.

## 3. Checks

`just py-prepare` (`py-prepare-1.log`): format, lint, 3.11 syntax, mypy, 2,391
passed and 6 hardware/backend skips, docs freshness passes; 55.8 s wall, but
2.10 GB max RSS for the whole recipe (above the 1 GB bound; the functional test
and CLI run stayed below it).

## Limits

- No learning claim: warmup is 1,000 optimizer steps, so both updates run at a
  tiny learning rate; the losses are finite, not informative.
- The test patches the evaluation cadence below the validated floor, the
  promotion threshold and the logger; the CLI run never evaluates.
- CPU FP32 only: no compile, BF16, CUDA transfer, pinned memory or multi-rank.
- The eval outcome uses the unseeded global torch RNG for seat assignment
  (inherited); the test seeds torch before launching.

## Verification pass

Independent Claude pass at `268b1b4` (report:
`ops/rebuild-2026-09-29/codex/claude-verify-3.5.md` in the main checkout).
`verify/mutate.py` applied six mutations to `scripts/run_ppo.py`, each followed by
the two functional-check cases and a hash-checked restore
(`verify/mutations.log`): promotion `>=` to `>`, skipped last_best refresh,
promoted checkpoint written elsewhere, skipped teacher activation and a final
checkpoint at `env_steps=0` were all killed; `replay_dir` passed unconditionally
survived as an equivalent mutant, because `_evaluate_games` writes replays only
when `replay_games > 0`.
