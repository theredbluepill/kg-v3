---
type: "Reference"
title: "Kaggle packaging ships the BC agent as a native, validated tarball"
description: "Task 7.4 ship build on kg/rebuild-7-4-ship: a 24.7 MB submission.tar.gz with main.py, owl, a pod-built CPython 3.11 abi3 x86-64 rs.abi3.so (GLIBC_2.35 max) and the slim BC best (fd854587...6f51). It encodes one seat with the training write_seat, runs a greedy CPU fp32 forward at 1 thread, decodes natively, and returns PASS on a caught fault. Strict local episodes qualified with 719 calls per seat and zero faults, once in a fresh Kaggle-image container under emulation and twice on the pod. Three more episodes in Kaggle mode (non-strict, fallback live) in a fresh pod venv also passed with zero faults and zero fallbacks: BC against starter in each seat, and self-play through the kaggle-environments CLI. Not submitted. Deviates from the brief: no in-image Docker build, no replay-parity or latency-benchmark receipts, no W&B. Branch kg/submit-08bc merges the ship path onto main 07c8fc99 so PPO run configs with the bank/margin reward-shaping keys load; the manifest now records only the checkpoint file name; a strict 5-turn Mac load of PPO checkpoint 08bc19ae passed. scripts/package_checkpoint.sh now packages a checkpoint from the Mac in about 28 s: it reuses the cached 08bc Linux module only while native sources equal its source commit 9a743fad, verifies hashes and every model tensor, and runs a 40-turn strict Kaggle-image episode. The 08bc module had the fixed-opponent controllers compiled in (default cargo features) and 08bc shipped it; the cache now holds a --no-default-features rebuild (3e5e4e55, 3.5 MB, no opponent strings, glibc max 2.35), with which the 50M checkpoint packaged and passed the 40-turn Kaggle-image episode without --allow-fixed-opponents. In the final sprint the same packager, with --final-turn-liquidation, built the CPU-pod panel packages from 60M to 210M (clean module, verified, Kaggle-image episode qualified in each archived receipt) and the three owner-agreed submissions, 90M, 170M and 210M, whose panels played the identical archives."
tags: ["kaggriculture-v3", "adaptation", "packaging", "kaggle-runtime"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.4-packaging.md"
  - resource: "repository:python/kaggriculture_main.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_agent.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_view.py"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:scripts/build_kaggriculture_submission.py"
  - resource: "repository:scripts/kaggle_local_episode.py"
  - resource: "repository:tests/kaggriculture/test_kaggle_agent.py"
  - resource: "repository:tests/kaggriculture/test_kaggle_view.py"
  - resource: "repository:tests/scripts/test_build_kaggriculture_submission.py"
  - resource: "repository:tests/scripts/test_kaggle_local_episode.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/kaggle-runtime-receipt.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/native-module-receipt.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-native-build.sh"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/custody.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/kaggle-image-episode-self-seed7.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-episode-self-seed20260930.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/pod-episode-starter-seat1-seed20260931.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4/py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/validation.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/kaggle_mode_episode.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/summarize_cli_run.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/run_validation.sh"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/episode-starter-seat0-seed20261000.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/episode-starter-seat1-seed20261001.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/cli-run-self-seed20261002-summary.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.4-ship/validation/receipts/venv-kaggle-freeze.txt"
  - resource: "repository:tests/kaggriculture/kaggle_fixtures.py"
  - resource: "repository:ops/submit-08bc-2026-09-30/mac_load5.py"
  - resource: "repository:ops/submit-08bc-2026-09-30/mac_load5.out"
  - resource: "repository:scripts/package_checkpoint.sh"
  - resource: "repository:native-cache/manifest.json"
  - resource: "repository:README.md"
  - resource: "repository:Cargo.toml"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/agent.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/core.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/envs/kaggriculture/kaggriculture.json"
  - resource: "kaggle-mcp:competitions/kaggriculture/pages"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/60M-ft-on/package/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/80M-ft-on/package/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-80M.log"
  - resource: "repository:ops/package-60m-2026-10-01/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/90M-ft-on/package/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/170M-ft-on/package/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/210M-ft-on/package/PACKAGE.md"
  - resource: "repository:ops/submit-90m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-170m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-210m-2026-10-01/receipt.md"
---

# Kaggle packaging ships the BC agent as a native, validated tarball

## Claim

Branch `kg/rebuild-7-4-ship` builds a Kaggle submission for the BC best checkpoint
and validates it by local execution. It carries the approved Task 7.4 brief
(`ops/rebuild-2026-09-29/briefs/7.4-packaging.md`, merged at `eb40556`). Nothing
was uploaded or submitted; submission is the owner's decision.

The existing-concept search covered packaging, submission, Kaggle, runtime and
BC. This note was the only one, as the design Reference for the brief. It is
revised in place (file name kept for incoming links) because its claim moved
from design to a built and executed artifact.

## What ships

`artifacts/7.4/submission.tar.gz` (gitignored): 24,704,468 bytes, SHA-256
`00e67809f9d4d3e40d477dc309be1df241dfc7c71a92f36241401ba873be3839`, built from
clean commit `6c49863`. The limit is 100 MiB. Its `manifest.json` (SHA-256
`4872a2d3…4a94`) binds every file hash, the source commit and tree, and the
two receipts below. `ops/rebuild-2026-09-29/7.4/custody.json` records the
archive hash and size.

- `main.py` (from `python/kaggriculture_main.py`):
  - sets 1 torch thread and 1 interop thread;
  - imports every bundled `owl` module at top level, because the agent
    directory is on `sys.path` only during Kaggle's lazy `exec`;
  - asserts a release extension, loads the model once and runs one warm-up
    turn;
  - defines `agent` last, because the loader takes the last callable.
- Weights: the BC best, original SHA-256
  `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`, slimmed
  to `{"model"}` only (25 MB fp32, 6,252,223 parameters). It is loaded strictly
  as `model_only`, with `force_flash_attn` overridden to false at runtime.
  The loader rejects hidden, recurrent, memory and opponent state names.
- Encoding: `encode_kaggriculture_seat_into` writes one `[1,1]` row through the
  same Rust `write_seat` as training. An empty rival private stands in for
  state the agent cannot see. The configuration comes from an allowlist that
  drops Kaggle's `__raw_path__`.
- Action: greedy (`deterministic=True`) decode through the native grammar, then
  a native encode/decode round trip that must reproduce the action exactly.
- Fallback: in non-strict mode, any exception or failed validation returns
  `{"farmer":["PASS"],"hands":[],"market":[]}` and increments a counter. So does
  a remaining overage bank below 2 s. Strict mode (`KAGGRICULTURE_AGENT_STRICT=1`)
  re-raises instead. The three qualifying episodes used it; the three
  Kaggle-mode episodes below did not.
- Native module: `owl/rs.abi3.so`, 3,477,136 bytes, SHA-256 `3558c26f…5375530`.
  It was built on the pod from clean `f66acf8` with CPython 3.11.13 and
  `maturin build --release --compatibility linux`. It is an ELF64 x86-64
  module whose highest glibc symbol version is `GLIBC_2.35`. A
  `manylinux_2_28` attempt was refused for newer symbols
  (`native-module-receipt.json`, both build logs). The builder refuses a
  non-x86-64 module, a glibc above the limit, a receipt whose hash differs from
  the module, a dirty tree, a checkpoint hash mismatch and an archive over
  100 MiB.

## Execution evidence

All three episodes ran the extracted tarball's `main.py` through
`kaggle-environments` 1.32.7's real file-agent loader with `debug=False`, in
strict mode. Each confirmed kaggriculture.py hash `bc8a5487…653e`.
Qualification needs zero `ERROR`/`TIMEOUT`/`INVALID` statuses at every step,
719 agent calls per seat from the harness's own counter, and zero exceptions.
It also needs zero raw returned actions failing the validator before Kaggle
normalizes them.

| Receipt | Host | Seats | Result |
|---|---|---|---|
| `kaggle-image-episode-self-seed7.json` | fresh container of the local Kaggle image (Python 3.11.13, glibc 2.35, torch 2.6.0+cu124), amd64 emulated on the Mac, `--network none --cpus=1.6 --memory=6.5g` | self-play, seed 7 | qualified; `owl` and `owl.rs` loaded from `/kaggle_simulations/agent`; turn 0 0.97 s / 0.79 s; steady p99 0.42 s, max 0.58 s (emulated) |
| `pod-episode-self-seed20260930.json` | pod EPYC 9535, CPython 3.11.13 + torch 2.6.0+cpu, `nice -n 19`, no GPU visible | self-play, seed 20260930 | qualified; turn 0 0.47 s; steady p99 0.124 s, max 0.129 s; banks 40,844 / 48,504 |
| `pod-episode-starter-seat1-seed20260931.json` | same | BC in seat 1 vs built-in `starter` | qualified; steady p99 0.125 s, max 0.150 s; banks 3,756 (starter) / 80,110 (BC) |

The minimum remaining overage bank stayed 60.0 s in every episode. No call
returned the fallback PASS.

`just py-prepare` passed on the ship branch with 2,777 passed and 18 skipped
(`py-prepare.log`).

### Kaggle-mode validation

`ops/rebuild-2026-09-29/7.4-ship/validation/validation.json` summarizes three
more episodes. The archive (`00e67809…3839`) was unpacked fresh on the pod, and
all 65 manifest files were re-hashed with no mismatch. The venv was new: uv,
CPython 3.11.13, and only torch 2.6.0+cpu, numpy 2.4.6, pydantic 2.12.4,
pyyaml 6.0.3 and kaggle-environments 1.32.7 with their dependencies. Runs used
`nice -n 19` with no GPU visible. This is Kaggle mode: `debug=False`,
`KAGGRICULTURE_AGENT_STRICT` unset (the fallback is live), the agent directory
not on `sys.path`, and a working directory outside it. The env configuration
gives `actTimeout` 1 s, a 60 s `remainingOverageTime` bank, 720 steps and a
1,200 s `runTimeout`.

| Episode | Steps, bad statuses | Calls, exceptions, invalid raw | Turn 0 / steady mean / steady max (s) | Min overage (s) | Peak RSS (MiB) | Banks | Fallbacks |
|---|---|---|---|---|---|---|---|
| BC seat 0 vs `starter`, seed 20261000 | 720, 0 | 719, 0, 0 | 1.316 / 0.122 / 0.260 | 59.68 | 459 | BC 39,981, starter 3,636 | 0 |
| BC seat 1 vs `starter`, seed 20261001 | 720, 0 | 719, 0, 0 | 1.264 / 0.121 / 0.242 | 59.74 | 458 | BC 33,352, starter 3,596 | 0 |
| CLI `kaggle-environments run`, self-play, seed 20261002 | 720, 0 | 719 per seat | seat 0: 1.219 / 0.122 / 0.162; seat 1: 0.457 / 0.122 / 0.136 | 59.78 / 60.0 | 625 | 62,278 / 61,665 | 0 |

"Fallbacks" means the agent's own counters for caught errors and overage-budget
PASSes, its printed fallback lines, PASS returns and non-empty stderr; all were
zero. The kaggriculture interpreter has no invalid-action penalty or status:
illegal unit actions are silent no-ops. Legality is therefore the framework
statuses plus the agent's validator applied to every raw returned action.
kaggle-environments 1.32.7 ships no dedicated agent-validation action, so its
CLI `run` of the submission against itself, the shape of Kaggle's validation
episode, stands in. Peak RSS is for the whole process: the env, the opponent
and the agent (or both agent copies). Replays stay on the pod in
`/root/ship/validate/out/`, and their hashes are in the receipts.

## Reusable findings

- **Timing.** The budget is 1 s per turn plus a 60 s per-seat overage bank.
  Imports, weight load and warm-up bill to turn 0 because the file agent is
  `exec`'d lazily inside the first timed call. In a fresh process the first
  copy's turn 0 took 1.22 to 1.32 s on the pod. That exceeds `actTimeout` and
  draws about 0.3 s from the 60 s bank, which is legal. A second copy in the
  same process took 0.46 s, because torch was already imported.
- **Loader.**
  - It picks the last callable in `main.py`.
  - It injects `configuration["__raw_path__"]`, which the Rust config envelope
    rejects, so the agent builds the configuration from an allowlist.
- **Failure modes.**
  - A non-dict return is silently normalized to PASS.
  - At termination the interpreter overwrites every status and reward, so a
    final-call fault ends as `DONE`.
  - Final statuses therefore prove nothing. The harness captures every
    `Agent.act` result itself, and a unit test injects a final-call fault.
- **Native build without Docker.** A `--compatibility linux` build on a newer
  glibc host is acceptable only when the highest referenced glibc symbol
  version is at or below the runtime's. The fresh runtime container must then
  load and play the module. Both held here.
- **Harness defect found and fixed.** `kaggle_local_episode.py` parsed
  `--episode-steps` but never passed it to `run_episode`. An intended 4-step
  load check therefore ran a full episode. The flag is now forwarded, and
  `test_cli_forwards_the_episode_step_bound` fails on the old code.

## Main-based ship branch for PPO checkpoints (`kg/submit-08bc`)

Owner, verbatim (2026-09-30): "can you package the 08bc and submit to kaggle
for probing? Note we have 4 submissions left, only use 1 of it."

The ship branch alone cannot load a current PPO run's `config.yaml`: main made
the six `econ_bank_*` and `econ_margin_*` reward-shaping keys required, and the
agent validates the whole `env` section. Branch `kg/submit-08bc` therefore
starts at `origin/main` `07c8fc99` and merges `kg/rebuild-7-4-ship` (`619349f`)
with `--no-ff`. Changed paths and reasons:

- `src/kaggriculture/observe.rs`: the one code conflict. Main's
  `PreparedObservation::snapshot` impl block and the ship branch's
  `from_seat_view` impl block are both kept. The result is byte-identical to
  the scratch merge that built the local Mac packages.
- `python/owl/model/`, `python/owl/train/`: untouched by the ship branch, so
  they equal `07c8fc99`. The additions since `0f70773` (the PPO run's training
  commit), the optional critic-offset head and the optional-state loader, are
  default off; 08bc's config does not enable them, and the strict load below
  admits its state dict.
- `scripts/build_kaggriculture_submission.py`: review item E1. The manifest
  records `checkpoint.original_name` (the file name) instead of the builder's
  absolute `original_path`. The builder test asserts it.
- `tests/kaggriculture/kaggle_fixtures.py`: the fixture training config now
  carries main's full `reward_shaping` key set; without it the agent tests fail
  validation on main.
- `cookbook/references/index.md`: dropped main's stale design-only entry for
  this note, which the merge left beside the ship entry.

Checks on this tree (Mac, arm64, torch 2.9.0 venv): the four ship test files
48 passed; `tests/kaggriculture` plus the two ship script tests 1,477 passed
and 4 skipped (CUDA); `cargo test --lib kaggriculture` 146 passed and 3
ignored, the seat-binding tests included; ruff clean on the touched files;
`mypy python/ scripts/` clean. `ops/submit-08bc-2026-09-30/mac_load5.out`:
the agent loaded the slimmed 08bc (`08bc19ae…4600`) strictly with the run's
`config.yaml` and played 5 turns in both seats in strict mode, with 10 calls,
0 caught errors, 0 budget passes, all statuses `ACTIVE` and a maximum act time
of 0.03 s. The slim hash from this venv (`5937ffa0…9dbd`) differs from the
earlier torch 2.6.0 local package's; the shipped slim hash is whatever the
build host's manifest records. This is a load check, not a Linux build, a
Kaggle-image episode or a strength measurement.

## One-command checkpoint packaging (`scripts/package_checkpoint.sh`)

Owner, verbatim (2026-09-30): "the packaging is now more efficient or?" Before
this, each package was assembled by hand from the 08bc steps: a pod native
build, a scratch builder venv, the builder, and a hand-run Docker episode.

Changed paths and reasons:

- `scripts/package_checkpoint.sh CHECKPOINT CONFIG OUT_DIR [--full-episode]`:
  - It makes a builder venv once with uv (CPython 3.11, torch 2.6.0 and
    numpy 2.4.6, matching the 08bc builder and the image's torch).
  - It refuses a dirty tree. It also refuses when any native source differs
    from the cached module's source commit, and says the `.so` must be rebuilt.
    The native sources are `src`, `engine_rs`, `opponents_rs`, `Cargo.toml`,
    `Cargo.lock`, `build.rs`, `rust-toolchain.toml`, `.cargo` and
    `pyproject.toml [tool.maturin]`.
  - It seeds and checks the cached module against the committed manifest. It
    requires the local image id to equal the runtime receipt's.
  - It runs the builder with both receipts and the checkpoint's SHA-256.
  - It extracts the archive fresh, re-hashes every file and compares every
    model tensor with `checkpoint["model"]`.
  - It plays a strict self-play episode in the Kaggle image (amd64 emulated,
    `--network none --cpus=1.6 --memory=6.5g`): 40 turns by default, 720 with
    `--full-episode`.
  - It writes `OUT_DIR/PACKAGE.md` with per-stage wall times.
- `native-cache/manifest.json`: custody of the cached module, its source
  commit, glibc symbol max, seed path and full pod build receipt. The `.so`
  itself is gitignored (`/native-cache/*.so*`). It first recorded the 08bc
  module `2bbdd2f0…8e4`; it now records the Kaggle-feature rebuild below.
- `README.md`: a section on usage, the rebuild rule and the opponent flag.

**Finding: the 08bc module carries the fixed-opponent controllers.** The 08bc
pod build ran `maturin build --release --compatibility linux` with default
cargo features. `Cargo.toml` defaults to `fixed-opponents`, and says the Kaggle
build passes `--no-default-features` because EcoBot and E776 carry no
redistribution license. The module's strings contain 1,013 lines matching
`kaggriculture_opponents`, the `opponents_rs/src/native_agents/e776.rs` path
and EcoBot messages. This explains the 14.3 MB size, against 3.5 MB for the BC
module, which predates the feature. The 08bc submission (ref 56711278) shipped
this module. The packager refuses such a module unless
`--allow-fixed-opponents` is passed, and records the flag in `PACKAGE.md`.
Whether to rebuild without them before another submission is the owner's call.

Checks on commit `be79f135` (Mac, owner's Docker):

- The 50M checkpoint `checkpoint_00_050_031_104.pt` (`0cc80065…a7c2`, config
  `62e0b5c1…75f7`) was packaged with `--allow-fixed-opponents` into
  `artifacts/c50/` (gitignored). The archive is 27,166,101 bytes, SHA-256
  `dd0243663f65bb1a64b7e9d67e4b39c69d0a33b1f751920117617eb3f1b6e142`; its inner
  `manifest.json` is `2e7ed159…b935`. `artifacts/c50/PACKAGE.md` is the receipt.
- Stage wall times: venv 0.8 s, preflight 0.5 s, build 2.4 s, verify 0.6 s and
  image episode 23.5 s, 27.9 s in total.
- All 65 files re-hashed clean. All 210 of 210 model tensors (6,252,223
  parameters) equal the checkpoint's, with max abs diff 0. An independent
  re-extract and compare agreed.
- The 40-turn strict self-play episode, seed 7, qualified: 39 calls per seat,
  0 bad statuses, 0 exceptions, 0 invalid raw actions and 0 default passes.
  `owl.rs` loaded from `/kaggle_simulations/agent`.
- Refusals exercised in a throwaway worktree: a dirty tree, a probe edit to
  `src/lib.rs` ("native sources differ … must be rebuilt"), the opponent guard,
  and a missing seed.

Limits: the archive hash changes on every build, because the manifest records
`built_utc`. The short episode covers 40 of 720 turns; `--full-episode` was not
run on c50 here. The module was not rebuilt in that check; the next section
adds the `--no-default-features` module. There is no shellcheck in this environment. Nothing
was uploaded or submitted.

## Kaggle-feature native module (no fixed opponents)

The 08bc module shipped the fixed-opponent controllers, so the module was
rebuilt with the Kaggle feature set and made the packager's default. This fixes
future packages only; the 08bc submission (ref 56711278) already carries the
old module.

- Build: pod `abl4mvr5w1mmn4`, `/root/sub08bc`, the same clean clone of
  `9a743fad` (tree `aaf9d4d2`), CPU only, `nice -n 19`, `CARGO_BUILD_JOBS=8`,
  fresh `CARGO_TARGET_DIR`; the live training run and `/root/kg-v3-anchor`
  were not touched. The command was `maturin build --release --compatibility
  linux --no-default-features`. Maturin still adds `extension-module` from
  `pyproject.toml [tool.maturin]`, so the build is `Cargo.toml`'s documented
  Kaggle feature set. Toolchain: rustc 1.97.0-nightly, maturin 1.15.0 and
  CPython 3.11.13. The build took 38 s.
- Module `3e5e4e55…664e`, 3,485,792 bytes (the BC module was 3.5 MB). Its
  highest glibc symbol is 2.35 (`hypotf`), and it needs only libgcc_s, libm,
  libc and ld-linux. `objdump -d` finds no zmm or AVX-512 mask instructions.
- Opponent check: `strings -a` finds 0 matches for `kaggriculture_opponents`,
  `opponents_rs`, `native_agents`, `e776`/`E776`, `EcoBot`, `cha22`/`Cha22`,
  `farm2945`, `metav4` and `ig_agent`. The old module has 1,013, 46, 7 and 77
  matches for `kaggriculture_opponents`, `e776`, `EcoBot` and `cha22`. Two
  matches are false positives: one `v43` string is a mangled `Env43` method
  name, and one `nm` "e776" is the address `0x1e7760`. On the pod,
  `assert_release_build()` passed and `kaggriculture_opponent_bots() == ()`.
- `native-cache/manifest.json` now names this module under `module`, with
  `fixed_opponents_compiled_in: false` and seed
  `artifacts/native-clean/rs.abi3.so`. It keeps the 08bc module and its
  receipt under `superseded_modules`, with the status "contains opponents, do
  not ship". The README section says the same.
- Package check on commit `b43260fb` (Mac, owner's Docker, without
  `--allow-fixed-opponents`): the 50M checkpoint `0cc80065…a7c2` went into
  `artifacts/c50-clean/` (gitignored). The archive is 24,719,006 bytes, SHA-256
  `7ba6dbb2231f828c7fd611b49c1a82691e02a1360ad1711e5b250b0fe23b95db`; the
  inner manifest is `8729f242…7934`. All 65 files re-hashed clean, and 210 of
  210 tensors are equal. The shipped `owl/rs.abi3.so` is `3e5e4e55…`. The strict
  40-turn self-play episode (seed 7) qualified: 39 calls per seat, 0 bad
  statuses, exceptions, invalid raw actions or default passes, and a steady
  p99 of 0.25 s. The total was 27.6 s.

Limits: the module was not built inside Kaggle's image, and it is not
stripped. The ymm (AVX2) lines were not traced to crates. Only 40 of 720 turns
ran. The old module's `.so` stays in the Mac cache for custody. Nothing was
uploaded or submitted.

## Sprint use (2026-09-30/10-01)

During the final sprint the packager built the archive for each checkpoint
that the CPU-pod anchor panel played. Package receipts are archived for every
paneled checkpoint from 60M to 210M. The 80M copy-back from the pod failed with
a broken pipe (`cpu-pod/eval-80M.log`), so its receipt was added on 2026-10-01
from the Mac build folder: clean module `3e5e4e55…`, 210/210 tensors equal,
rule 1 baked on and rule 2 left to the environment (off). The panel harness `ops/sprint-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh`
called `scripts/package_checkpoint.sh --final-turn-liquidation`, and each
`package/PACKAGE.md` is under
`ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/<ckpt>-ft-on/`.

Every archived receipt shows the clean module `3e5e4e55…` (no
fixed-opponent controllers). Each was verified: all files were re-hashed and
210/210 tensors were equal. Each passed one strict Kaggle-image episode. For
every checkpoint except 60M it was the default 40-turn episode. The 60M package
was built with `--full-episode` and ran the 720-turn episode instead of the
40-turn one (`episode_steps None`, `ops/package-60m-2026-10-01/PACKAGE.md`).

Three sprint archives were submitted, each with the owner's agreement:

| Checkpoint | Ref | Archive sha256 | Panel on the same archive |
| --- | --- | --- | --- |
| 90M | 56716929 | `5e460d20…` | 48/48 qualified |
| 170M | 56720629 | `428a63d7…` | 48/48 qualified |
| 210M | 56722061 | `45efe071…` | 48/48 qualified |

Each panel's game receipts name the same package hash prefix as the archive
that was submitted. The receipts are `ops/submit-{90m,170m,210m}-2026-10-01/receipt.md`.
The submission rule and its outcome are in the
[[../decisions/spend-kaggle-submission-slots-only-on-owner-agreed-checkpoints|submission Decision]].
None of the three ran the 720-turn Kaggle-image game; the 48 full panel games
on the Linux CPU pod are their full-length evidence.

## Deviations from the approved brief

- **Build route.** There was no build inside Kaggle's image: the pod has no
  Docker, and the brief forbids building on the Mac. The brief's fallback
  route was used instead, and the in-image load held. `Dockerfile.kaggle` was
  neither repaired nor used.
- **Builder script.** The starter's `build_kaggle_submission.sh` was not
  reused. A Python builder (`scripts/build_kaggriculture_submission.py`)
  stages the prebuilt module instead. The build context is the clean working
  tree, not a `git archive` export.
- **Load-check host.** The fresh-container check ran on the Mac under amd64
  emulation, not on the pod. Because of the harness defect above, it was a
  full episode, which the brief reserves for owner approval on the Mac. It
  took about 6 minutes at 1.6 CPUs.
- **Not done:**
  - T6 latency benchmark: 1 versus 2 threads under an equal 1.6-vCPU quota on
    amd64 hardware, with an external watchdog;
  - T7 replay-parity test against the native env and the training path;
  - W&B telemetry for the episode and benchmark;
  - the dated Kaggle-limits receipt;
  - README and `docs/containerization.md` updates.

## Limits

- It is not established that Kaggle production runs this image. The image is a
  local build of Kaggle's Dockerfile on `python:v163`.
- Production JSON key order is unverified (brief F9); the inventory and shed
  rank channels depend on it.
- No timing here is Kaggle hardware. The pod numbers had no vCPU quota, and the
  container numbers are emulated.
- The episodes validate packaging and legality on six seeds, not strength.
  The three wins over `starter` are one seed per seat assignment; the
  evaluation panel is Task 7.2.
- The Kaggle-mode venv uses the torch 2.6.0 CPU wheel, not the image's
  `+cu124` build, on a glibc 2.39 host.
- Only the latest 2 submissions count for the final leaderboard, so a new
  submission displaces an older one. The owner decides.

Reopen when:

- the validation episode's agent logs from an owner-approved submission exist
  (production image, key order, hardware timing);
- a replay-parity or latency receipt lands;
- the checkpoint or the native module changes.
