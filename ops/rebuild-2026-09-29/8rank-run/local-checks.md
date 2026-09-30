# Local checks of the 8-rank run package (Mac, CPU, 2026-09-30)

These checks exercise the scripts' logic on the Mac only. No GPU, pod, torchrun or live W&B call was involved. The GPU paths of `setup.sh`, `qualify.sh`, `launch.sh`, `check_flash_all_gpus.py` and `allreduce_bench.py` are unexecuted.

- **Seed probe.** `uv run --no-sync python ops/rebuild-2026-09-29/8rank-run/seed_probe.py configs/kaggriculture_8rank_bc_finetune.yaml --world-size 8` exited 0 with a 304 MB peak RSS. It found 256 distinct construction seeds and base seed 0: rank r held seeds r, r + 8, …, r + 248, with next seed r + 256, and no violation (`local-seed-probe.json`). The first attempt failed because `run_ppo`, loaded by path, was not registered in `sys.modules` before its dataclasses ran; the fix registers it first.
- **Watchdog.**
  - `watchdog.py --self-test` passed: two low banks separated by a normal one do not stop; two consecutive game intervals below 20,000 do; an evaluation row with `eval/promoted` 0 never stops; NaN, inf, `"NaN"` and `"-Infinity"` stop; nonfinite `_`-prefixed system fields are ignored.
  - Scenario runs with a fake `wandb.Api` against a real process group (`sleep`), a scratch run directory with `attempts.jsonl` and two 3 MB checkpoint files, and a scratch durable directory:
    - (a) a NaN metric row: the watchdog wrote `watchdog_stop.json`, sent SIGTERM (child status −15) and exited 3;
    - (b) healthy rows with a failed promotion: no stop, exit 0 once the group ended;
    - (c) two low-bank game intervals: a stop with the reason naming both.
  - In all three the watchdog copied both checkpoints, `checkpoint_last_best.pt` under a hash-suffixed name, and wrote `checkpoints.sha256`.
  - The first scenario run exposed a defect. Stop handling sat inside the W&B read's `try`, so a signalling error (EPERM from `killpg` on a macOS zombie group) was logged as a telemetry outage and the loop continued. Stop handling now runs outside that block, and a refused probe of our own group counts as ended.
- **Summarizer.** `summarize_run.py` against a fake 13-iteration history gave 12 complete iterations after the first, deltas of 16,384 env steps and 16 optimizer steps, SPS computed from `_timestamp`, one game-interval bank and one evaluation row.
- **Shell and lint.** `bash -n` passes for all seven shell scripts. `uvx ruff format` and `uvx ruff check --select E,F,W,I,B --ignore E501` pass on the five Python files. `ops/` is outside `just prepare`'s lint scope, so these were run by hand. No `shellcheck` is installed.
- **Durable BC copy.** `/tmp/kg-v3-bc-best/` was copied to `/Users/poonszesen/kg-v3-runs/bc-best/`. The SHA-256 lists of both trees (7 files) are identical; see `README.md`.
