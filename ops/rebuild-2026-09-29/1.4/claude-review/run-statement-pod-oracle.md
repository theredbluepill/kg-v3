# Run statement: Task 1.4 reference oracle on the pod (CPU only, diagnostic)

- **Question.** Does the native `owl.rs.KaggricultureEnv` at `9dc2d02` reproduce the pinned reference `TrainingBatch` (`kg/reference-2026-09-29` at `65f0eac5`) bit for bit over the brief's 16 complete seeded games (17000..17015, 719 transitions each, fixed observation-local policy)? And does that oracle reject a native `dones` mutation?
- **Why the pod.** On the Mac, Codex's only recording attempt stopped at the 960 MiB watchdog while compiling the exported reference engine, before any game ran (`reference-recording-attempt.json`). The brief (G, Q4) moves recording and replay to the pod when they exceed the Mac budget. This is a CPU oracle, not training. No GPU is used.
- **Inputs and code path.**
  - Source: `kg/rebuild-env` at `9dc2d0236233377cb7518410140319b8f9cff797`, plus the reference commit, sent as git bundle `v3-env-oracle-9dc2d02.bundle` (sha256 `2f6590feaafbbe26e98b9feef2f33ee6b3d4efc3833ff448587d0552cbd605d7`, prerequisite `8fde43cd`, which the pod already has).
  - Pod: `w7ia3zvxqsvs3g`. A new, separate clone at `/workspace/kg-v3-env-oracle`, made with `git clone` from `/workspace/kg-v3-rebuild`, then fetching the bundle. `/workspace/kg-v3-rebuild` (checkout and `.venv`) and `/workspace/kg-v3` (reference) are not modified.
  - Environment: the clone's own `.venv` from `uv sync --frozen` (offline first), then `maturin develop --locked` (debug profile, as on the Mac).
  - Recorder: `scripts/record_kaggriculture_env_reference.py --reference 65f0eac5… --games 16 --first-seed 17000 --max-live-envs 1 --max-seconds 1800 --max-rss-mib 8192`. The brief lets Linux use larger explicit budgets. The recipe, game count and coverage assertions are unchanged.
  - Replay: `pytest tests/kaggriculture/test_env_reference.py tests/tools/test_record_kaggriculture_env_reference.py -q`.
  - Mutation: the edit from `mutate_reference_done.py` (`fill(row.done)` becomes `fill(!row.done)`), applied by hand. That script's `bounded.py` hardcodes a 960 MiB cap that a pod debug build may exceed. Rebuild, run the replay (expect failure), restore byte-exact (sha256 check), rebuild, run the replay (expect pass).
- **Expected discriminating observation.** The recorder publishes a fixture of at most 8 MiB compressed / 256 MiB expanded, with all coverage counters positive. The replay compares 11,504 transitions with zero divergence. The mutant fails with `first divergence game=0 seed=17000 step=0 seat=0 … field=dones`. A native divergence on the unmutated source would be a real defect: report the first divergence and diagnose it before any fix. Never edit the oracle to hide it.
- **Stopping condition.** Fixture published, replay green and mutation red then restored green. Stop earlier at the first hard failure (a crate missing offline, a build error, a recorder assertion, a divergence), which becomes the recorded result. Hard stop at 60 minutes wall from the first pod command.
- **Budget and safety.** The pod is already running ($4.18/h); nothing new is billed. Pre-check: both GPUs at 0 MiB / 0 %, no training processes (11:33Z). No driver, CUDA, torch or security changes. The pod is left running and idle. The clone directory is left in place for audit.
- **Artifacts.** The fixture `tests/fixtures/kaggriculture_env_reference_v1.{npz,json}` is copied back and committed. Receipts go in `ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/`.

## Addendum before the second pod phase (release checks, same clone and source)

- **Question.** Does a release build with the root `[profile.release.package.kaggriculture-engine] overflow-checks = true` turn a kernel i64 overflow into a caught, fully rolled-back step error (`release_dependency_overflow_is_caught`)? And what do the lifecycle phases cost in the optimized build (`measure_lifecycle`, ignored test), with and without that override?
- **Commands.** These are the exact pending commands from `results.md` §5, run in `/workspace/kg-v3-env-oracle` at `9dc2d02` with `CARGO_BUILD_JOBS=16`:
  - `cargo test --release --offline --locked --lib release_dependency_overflow_is_caught`
  - `KG_OVERFLOW_CHECKS_LABEL=enabled … measure_lifecycle -- --ignored --nocapture --test-threads=1`
  - the same measurement with `--config 'profile.release.package.kaggriculture-engine.overflow-checks=false'`
- **Expected observation.** The overflow test passes: the step returns `EnvError::Panic` with no published bytes. The timing test prints phase costs and asserts no speed. It measures one pod host, idle GPUs and one live env, so it is component timing only, not complete-update throughput.
- **Stopping condition.** All three commands have run, or the first build failure. Hard stop at 30 minutes wall.
