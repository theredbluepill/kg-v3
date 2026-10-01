# Python rollout optimization verification

Task source: `brief-python.md` and `code-map.md`. Base:
`b2276bc5b70073b58a27f9e5fbd52473dbd48569`, branch `kg/sps-python`.

Target: four independent, default-off Python switches; identical default
behavior and unchanged native/PPO recipe; opt-in parity and a portable local
component benchmark. Completion: checks, bounded benchmark, cookbook record,
report and requested local commit. No remote machine, W&B session, training
episode, source dependency update or Rust code change is authorized by this brief.

Checks answer separate questions:

- Default compatibility: run the existing two-update native/trainer digest and
  all-preset config digests with the same CPU thread count against exported
  base Python and new Python. Expect byte identity; stop on any difference.
- Actor arithmetic: real CPU Inductor versus eager at fixed seeds. Expect exact
  actions, unchanged RNG consumption and values; tight FP32 logp/entropy and
  backward tolerance. Compiler numerics can differ from eager, so no arbitrary
  seed or GPU equality guarantee follows from this finite sample.
- Execution: compare CPU masks/packed outputs exactly; verify action-copy
  lifetime and event-before-native ordering with doubles and CUDA-only tests;
  run canonical two-update PPO parity for the noncompiler switches.
- Benchmark: after correctness checks finish, run six arms at 40 seat rows,
  fresh seed-401 weights and games, the 6.318M bank-critic preset, FP32 CPU,
  eager trunk, two Torch threads, two warmup calls after a cold first call and
  five measured steps each. Include native execution and H2D (CPU copy here).
  Expect identical action/state/RNG hashes and completed work; report observed
  times without an invented speed threshold. Stop after the six arms. No
  optimizer, teacher, evaluation, win-rate or end-to-end SPS claim.

The first timing pass overlapped test/build work and used an earlier receipt
schema. Its outputs are named `benchmark-preliminary.*`; they are excluded
from speed conclusions. The isolated pass is `benchmark-cpu.*`.

GPU handoff: run the same bounded A/B on the diagnostic H200 with the pinned
stack, additionally compare BF16 actions/densities/values and RNG, execute the
CUDA packing and nondefault-stream D2H tests, inspect the GEMM backend claim and
cache, and measure equivalent full PPO updates with valid learner-turn counts.
Use Nsight Systems for any CUDA timeline attribution. This task performs none
of those remote checks. The standing training board is unchanged: this is
implementation verification, not a new trained result or ranked option.

Local environment: the existing h200gate virtualenv/native module is reused by
gitignored symlinks; `PYTHONPATH=python`, `UV_CACHE_DIR=/tmp/kg-sps-uv`,
`uv run --no-sync`, `OMP_NUM_THREADS=2`. A wheel is independently built from
this tree using `maturin build --offline`; no dependency is installed or changed.
Ignored Orbit fixtures are copied from the existing local `kg-v3` worktree
after an initial Rust run reported the missing fixture files. Direct commands
cover the prepare recipes without writing through the shared virtualenv.
