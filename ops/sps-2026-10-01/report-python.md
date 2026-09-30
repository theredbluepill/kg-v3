# Python rollout optimization report

Implementation and local verification are complete. The requested commit in
this worktree is blocked by sandbox permissions on its external Git metadata.
A portable commit is prepared in a writable temporary repository on branch
`kg/sps-python`; the original worktree remains uncommitted. No push, remote
access, Rust edit, preset change or training run occurred.

## Switches and changes

| Switch | Default | Enabled behavior |
| --- | --- | --- |
| `rl.compile_actor_heads` | `false` | Compile `actor.policy_core` with eager RNG and ATEN-only stack/claim checks, including actor-only compilation. |
| `rl.rollout_packing` | `false` | Build exact metadata from current CPU observation masks; avoid device nonzero/truth/scalar readbacks. |
| `rl.pinned_action_d2h` | `false` | Reuse pinned int64 buffers, issue both nonblocking copies, wait on one CUDA event before native consumption. |
| `env.skip_reward_telemetry_validation` | `false` | Skip duplicate Python finite scans; preserve native admission, shape/dtype checks and metric arithmetic. |

False keys are omitted from saved configs. RL flags reject Orbit configs.
Actor compile accepts only `default` and `max-autotune-no-cudagraphs`.
Packing retains the exact live token count; the context clears its mask on
exit. CPU attention/action-transfer paths are unchanged. Public reward oracles
validate by default. No model parameter, state-dict key or tensor schema changes.

Changed implementation:

- `python/owl/model/{kaggriculture,kaggriculture_actor,stateless_transformer_v1}.py`
- `python/owl/train/{ppo,utils,config}.py`, `scripts/run_ppo.py`
- `python/owl/kaggriculture/{config,env,rewards}.py`, `python/owl/game.py`
- New `scripts/bench_rollout_step.py`
- Four new Kaggriculture test modules, one benchmark test module, and the
  existing compile-target census test
- README, model architecture, RL API, cookbook Reference/index/log and the
  cuBLAS-only Decision/index

Adaptation record:
`cookbook/references/rollout-optimizations-preserve-sampling-behind-default-off-switches.md`.

## Parity evidence

- **All switches off, against base `b2276bc5`:** native digest
  `257eae38864aa2aa26373c7751be97b3b0d3c6d9d81ee80782036df41159a590`;
  two-update trainer metrics/weights digest
  `3ffd53a026b8b3be079c64b111226a6cb313dbdad586c81fbd28c974bd322fa2`.
  Both match exactly at `OMP_NUM_THREADS=2`. All 45 existing preset hashes
  match (`baseline-*` and `post-*` receipts).
- **CPU Inductor:** three fixed seeds give exact tokens, lengths, RNG and
  values. Event logp/entropy match at `atol=2e-6, rtol=0`; replay backward
  matches populated parameter gradients at `rtol=2e-4, atol=2e-5`.
  Eager-core external-noise tests are bit-exact in FP32/FP64/BF16 and across
  head chunks. Replay/greedy consume no random draws.
- **Packing:** exact metadata/results and full-model CPU padded/mock-varlen
  action/logp/entropy/value/RNG parity, learner ordering, chunking and exception
  cleanup pass. The real CUDA upper-bound test (bound 257, actual length 3)
  is skipped. Flash-attn 2.8.3 sources use cumulative entries for actual lengths
  and reject excess query blocks; the padded maximum is a valid bound.
- **Transport/integration:** CPU contiguous/noncontiguous bytes match;
  doubles verify pinned allocation, both async copies, event wait before native
  read, reuse and generic CUDA-device normalization. Real nondefault-stream
  CUDA parity is skipped. Two canonical PPO updates with packing, D2H and
  telemetry enabled match actions, densities, entropy, values, rewards, weights,
  all metrics under a deterministic clock and RNG exactly.
- **Telemetry:** on/off bytes and metrics match through nine native steps,
  autoresets and truncation. Extreme/negative/zero-bank results match bitwise;
  public NaN/Inf rejection and fast shape/dtype checks remain. Four finite scans
  disappear when both reward terms are active.

## CPU benchmark

```sh
UV_CACHE_DIR=/tmp/kg-sps-uv PYTHONPATH=python OMP_NUM_THREADS=2 \
TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/tmp/kg-sps-inductor \
uv run --no-sync python scripts/bench_rollout_step.py \
  --device cpu --steps 5 --warmup 2 --with-env \
  --output ops/sps-2026-10-01/benchmark-cpu.json
```

macOS 26.4 arm64, Python 3.12.13, Torch 2.9.0, two Torch threads; 20 envs /
40 seat rows, 709 padded tokens, fresh seed-401 weights from the 6,318,272
parameter `configs/kaggriculture_4rank_bank_critic.yaml`, FP32, eager trunk.
Each arm includes forward, within-turn decode, completed action transfer,
native step/telemetry and next-observation copy. One first call and two warmup
calls precede five measured calls: 100 game transitions / 200 seat turns
measured per arm, 160 transitions including startup/warmup. No game ends.

| Arm | Mean ms/step | Median ms/step | First call s | Seat turns/s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1433.418 | 1454.412 | 1.372 | 27.905 |
| actor | 1306.928 | 1312.850 | 2.599 | 30.606 |
| packing | 1345.044 | 1319.370 | 1.281 | 29.739 |
| D2H | 1357.452 | 1299.308 | 1.291 | 29.467 |
| telemetry | 1317.664 | 1310.402 | 1.794 | 30.357 |
| all | 1628.331 | 1366.954 | 1.279 | 24.565 |

All arms hash identically over every action step, full final observation and
CPU RNG. CUDA RNG is absent. Packing and pinned D2H are not effective
accelerator optimizations on CPU; their timing differences show this short
sequential sweep cannot attribute speed gains. The all-switch mean has a large
outlier. First-call timings use an existing Inductor disk cache, so they do not
measure clean-cache compilation. No confidence interval or live SPS claim.

`benchmark-cpu.json` records samples, resolved configs, versions and source/
native hashes. All recorded source hashes still match. The isolated sweep ran
after this task's test/build jobs finished; other system activity was not
controlled. `benchmark-preliminary.*` overlapped tests/builds and is excluded
from conclusions. The reused h200gate native extension passes
`rs.assert_release_build()`; its SHA-256 is
`698eabb6d6fbc88c8e4fa1882e0bfffbd7d4637161f6bfdf8d0bd5ba0ca9f337`.

## Checks

Direct prepare commands avoid writing through the shared virtualenv:

- Full Python suite: **3,071 passed, 11 skipped**. Skips identify CUDA, real BC
  artifacts, x86 quantization, bounded Mac regeneration and unavailable
  original sources; see `full-python-tests.log`.
- Targeted model/actor/PPO/env/compile/config/run_ppo suite: **2,375 passed,
  7 skipped** before the final added cases; the full suite includes those.
- Ruff format/check: **159 files formatted, all checks pass**.
- `mypy python/ scripts/`: **78 files, no issues**.
- Python 3.11 syntax, docs freshness and both markdown lint commands pass.
- Offline native wheel build and all Cargo formatting checks pass without
  editing Rust. Root all-target/no-default-feature Clippy, engine/opponent
  Clippy with existing allowances and both custody manifests pass.
- Rust: **407 passed, 5 ignored** (root 298, engine 72, opponents 37).
  Initial root tests lacked ignored Orbit fixtures; copied local fixtures
  resolve all eight failures. Both logs remain.
- Default digest/config comparisons, cookbook source lint, temporary-repo
  staged cookbook pre-commit check and `git diff --check` pass.

## GPU verification and delivery limit

No H200 check ran: the brief prohibits remote access. The diagnostic pod still
needs fixed-seed BF16 eager/compiled action/density/value/RNG parity, real CUDA
packing/D2H tests, GEMM cache census, and representative complete PPO update A/B
with valid learner-turn denominators and phase costs. Use Nsight Systems for
CUDA timeline attribution. Fresh pinned metadata cost, specialization/teacher
compile variants, memory and multi-rank behavior remain unmeasured. CPU tests
cannot prove all seeds avoid compiler rounding changes near an argmax tie.
This task does not change any standing-board result.

`git add -- README.md` failed with exit 128: the worktree's
`/Users/poonszesen/kg-v3/.git/worktrees/kg-v3-sps-py/index.lock` is outside the
writable sandbox (`commit-blocker.json`), and escalation is disabled. The
portable commit is made in `/tmp/kg-sps-base` on `kg/sps-python`, ending with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, and exported as
`/tmp/kg-sps-python.bundle` and `/tmp/kg-sps-python.patch`. It preserves the
requested commit without changing the restricted original Git directory.
The original branch has not advanced; nothing was pushed.

VERDICT: BLOCKED original-worktree commit is prohibited by sandbox filesystem permissions; implementation, verification and portable commit artifact are complete.
