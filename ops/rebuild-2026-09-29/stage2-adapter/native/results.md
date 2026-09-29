# Stage 2 native adapter, reward and codec qualification

Target: qualify the existing Python adapter/cold codec against the merged native
binding, remove every assigned missing-binding skip, and discriminate the
independent reward oracle against native execution. Completion: assigned focused
CPU suites pass; CUDA-only DMA proof remains explicitly pending on the pod.
No training, GPU execution, native build, dependency edit, git write or vendored
engine edit was performed by this subtask. Root owns final preparation.

## Changed files

- `python/owl/kaggriculture/rewards.py`: remove overflow rescaling so the oracle
  preserves native literal binary64 operation order; remove its unused helper.
- `tests/kaggriculture/test_env.py`: add native construction/reset/step/terminal,
  all 35 stable buffer/view identities, selected/no-op truncate with terminal
  and nonterminal transitions, late-seat invalid-row byte rollback and diagnostic
  checks. Keep every fake-binding test. Native action helper encodes each seat's
  current legal actor count and order limit through the real codec.
- `tests/kaggriculture/test_env_cuda_fence.py`: remove binding skip, retain CUDA
  availability skip; native action encoding occurs before the queued DMA delay.
  Real native step and fence-removal control are retained.
- `tests/kaggriculture/test_game.py`: enable real factory/rank streams and check
  construction plus 64 reset seeds (66 total per rank), exact successor at every
  reset, rank disjointness; delete each rank's env before constructing the next.
- `tests/kaggriculture/test_rewards.py`: enable all 11 Python/native admission
  cases, validate the recorded reward fixture's custody through its real loader,
  use actual recorded array names, add three exact native extreme-value probes.
- `tests/kaggriculture/test_codec.py`: enable 321 accepted and 43 rejected corpus
  cases; add 13 native encode rejection cases preserving output bytes plus real
  batch order/row equality. Existing fake forwarding/failure tests remain.

`env.py` and `codec.py` required no runtime edits: the real binding passes their
existing forwarding and runtime-import paths. Stub-only diagnostic types stay in
TYPE_CHECKING annotations under postponed annotations and do not execute.

## Red then green evidence

1. Initial removal of native-binding skips: 102 passed, 1 failed, 1 skipped,
   exit 1. The actual recorded fixture uses `econ_before`, `econ_after`, and
   `banks_after`; the prewritten Stage 1 test guessed `transition_*` names and
   failed with KeyError. Lifecycle, real rank streams, private live actions,
   admission and grammar corpus already passed against the existing implementation;
   no invented missing-production failure is claimed.
2. Expanded tests first run: 128 passed, 1 failed, 1 skipped, exit 1. Every new
   lifecycle and codec test passed. The new live reward probe failed at step 71
   for W=1e-320, starvation=drought=1e308: native [-.25, .25], oracle approximately
   [-9.999888e-13, 9.999888e-13]. An overflowing inner death sum was rescaled only
   in Python, changing the agreed arithmetic. This is an actual oracle defect,
   not a native gameplay/reward defect.
3. Correction: preserve `W * (s*S + d*D)` binary64 order; inner infinity remains
   infinity then caps, even for tiny positive W. Disabled components still
   short-circuit. Update the synthetic overflowing-inner-sum expectation and
   rename the float64-cap regression to saturation. Native code is unchanged.
4. Final focused suites: env+CUDA 39 passed/1 skipped; rewards 42 passed; game
   27 passed; codec 21 passed; every command exit 0 (129 passed/1 skipped total).
5. Ruff formatting initially exit 0 (3 reformatted), lint exit 1 (five split-
   assertion findings and an import-order finding), corrected formatting and
   lint both exit 0. Root's canonical preparation supplies full static/suite checks.

The fixture comparison retains Stage 1's inherited maximum one-float32-ULP
per-value tolerance unchanged. The loader checks manifest/source identity,
compressed/expanded hashes, array inventory and complete-trajectory custody
before fixture data reaches the oracle. The fixture compares 16*719=11,504
recorded transitions; it does not instantiate 16 simultaneous envs.

The extreme live comparisons require exact array equality on every actual native
transition, including terminal reward. Each parameter uses one game for exactly
96 transitions (episodeSteps97), fixed native reference action recipe, seed17000,
one thread; asserts nonzero starvation, drought and ineffective counters.
Recipes cover overflow saturation with cap .7 and huge ineffective weight,
disabled extreme coefficients, and tiny W with an overflowing inner death sum.
No performance or learned-policy conclusion follows these unit checks.

## Run statement and limits

Mechanism: native buffer/seed/lifecycle transport and independent arithmetic.
Inputs: merged native extension, typed Python adapter/codec, frozen Task 1.4
reference fixture/recipe, G grammar corpus; native tests use at most two live
games. Expected discriminator: exact buffer identity/rollback, defined seed
stride, completed-vs-autoreset counters/banks and native reward equality.
Stop: each subprocess has a 115-second cap; focused commands completed below
three seconds each. No native rebuild or large workload was run here.

All commands below inherited OMP_NUM_THREADS=2, CARGO_BUILD_JOBS=2,
CARGO_NET_OFFLINE=true, UV_OFFLINE=true. Each nested return code is recorded
without a pipeline in `commands.jsonl` and named log. Read-only inspections and
source edit commands returned 0; they are not test results.

## Commands

- `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py -q` — exit 1; log `unskip-red.log`; 1.942s.
- `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py -q` — exit 1; log `native-expanded-red.log`; 1.923s.
- `uvx --offline ruff format python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` — exit 0; log `format-lint-0.log`; 0.123s.
- `uvx --offline ruff check python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` — exit 1; log `format-lint-1.log`; 0.040s.
- `uvx --offline ruff format python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` — exit 0; log `format-green.log`; 0.184s.
- `uvx --offline ruff check python/owl/kaggriculture/rewards.py tests/kaggriculture/test_env.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_game.py tests/kaggriculture/test_codec.py tests/kaggriculture/test_env_cuda_fence.py` — exit 0; log `lint-green.log`; 0.073s.
- `uv run --offline pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_env_cuda_fence.py -q` — exit 0; log `env-green.log`; 2.046s.
- `uv run --offline pytest tests/kaggriculture/test_rewards.py -q` — exit 0; log `rewards-green.log`; 1.362s.
- `uv run --offline pytest tests/kaggriculture/test_game.py -q` — exit 0; log `game-green.log`; 1.358s.
- `uv run --offline pytest tests/kaggriculture/test_codec.py -q` — exit 0; log `codec-green.log`; 1.373s.

## Remaining skip and handoff

`tests/kaggriculture/test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma`
remains skipped only for `CUDA is unavailable`. Its body constructs the real
binding, queues all 35 copies after a CUDA delay, steps the native env and
compares to host snapshots. Patching `_fence` to no-op must expose at least one
mismatch. This hardware experiment has not run on the Mac.

Task 3.1 retains rollout storage/action transport/trainer adoption obligations;
this subtask proves no trainer collection or distributed execution. Root reports
those blockers, native-table/model checks, Rust mutation and final preparation.
No binding skip remains in the five assigned test modules.

The only implementation deviation in this subtask is correcting the Stage 1
independent reward oracle to match native overflow semantics. The source/stage
history should retain the previously mistaken rescaling claim as history while
current API prose states native order. Cookbook edits are deliberately reserved
for Claude's review per the owner instruction.
