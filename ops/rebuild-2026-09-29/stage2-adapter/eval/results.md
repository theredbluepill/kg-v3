# Stage 2 evaluation seam receipt

Target: construct independent native evaluation environments from the canonical
runner with the required evaluation seed/rank/world/device arguments, and retain
the launch stop until the existing PPO storage and mapping support Kaggriculture.
Discriminating observations: exact factory arguments, native seed custody, daily
RNG snapshots, reproducible terminal banks, and fail-before-allocation messages.
Stopping condition: focused tests and source checks pass. No training, model
forward, GPU execution, dependency edit, Git mutation, or vendored engine edit.

## Changed source

- `scripts/run_ppo.py`: Kaggriculture evaluation factory dispatch returns the env
  union; independent native construction receives the mixed evaluation seed,
  rank 0/world 1 and transfer device. Orbit's constructor branch is unchanged.
  Startup keeps its stop with the current Task 3.1 blocker. Policy evaluation
  explicitly stops before its still-Orbit observation/action mapper. The seed
  band comment correctly assigns training-only separation to Task 3.1.
- `tests/scripts/test_run_ppo.py`: replace the mixed test's stale Kaggriculture
  rejection (its Orbit setup/assertions are byte-unchanged); implement the
  skipped native evaluation seed/replay test; add CPU/CUDA factory-argument
  spies, real independent-instance test and policy-evaluation blocker test;
  strengthen the Kaggriculture fresh/resume startup assertions to the precise
  remaining blocker. All non-Kaggriculture tests are untouched.

## Commands and outcomes

Every entry ran through `python3 ops/rebuild-2026-09-29/stage2-adapter/eval/run_check.py
<receipt-name> <command>`; the runner records the actual subprocess exit code,
wall duration and full stdout/stderr in the matching `.json` and `.log`. It
sets `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true`
and enforces a 115-second timeout. Python checks were serialized with the other
agents' torch tests. Only one or two real CPU environments existed at a time.

The focused command is:

```sh
uv run --offline pytest tests/scripts/test_run_ppo.py -k 'create_eval_env or kaggriculture_eval or kaggriculture_native_eval or policy_evaluation or main_loads_kaggriculture or resume_startup_checks' -q
```

- `red`: exit 1; 9 failed, 1 passed, 97 deselected, 0 skipped. Recorded before
  production edits. The factory still raised the old native-pending error or
  lacked `create_env`; startup still named the absent native binding.
- `green`: exit 1; 1 failed, 9 passed, 97 deselected, 0 skipped. Production edits
  work, but the test's `farmer: []` was not a legal native unit command. Corrected
  the test to `PASS` with an empty market (native STOP).
- `green2`: exit 1; 1 failed, 9 passed, 97 deselected, 0 skipped. The test exposed
  a mistaken assumption that reset snapshots differ by seed. Kernel construction
  initializes the same board for every seed; the first end-of-day RNG makes
  differences visible. Read the engine before correcting this expectation.
- `green3`: exit 0; 10 passed, 97 deselected, 0 skipped; pytest 0.18 seconds,
  process 1.87 seconds. No remaining skipped test in this focused selection.
- `format`: `uvx --offline ruff format scripts/run_ppo.py tests/scripts/test_run_ppo.py`;
  exit 0, one file reformatted/one unchanged.
- `lint`: `uvx --offline ruff check scripts/run_ppo.py tests/scripts/test_run_ppo.py`;
  exit 1, two E501 long strings; split strings without changing their values.
- `lint2`: same lint command; exit 0, all checks passed.
- `mypy`: `uv run --offline mypy scripts/run_ppo.py`; exit 0, one source file clean.
- `diff_check`: `git diff --check`; exit 0, no whitespace errors.

## Seed/terminal proof and limits

The bounded native test uses two environments, `episodeSteps=3`,
`turnsPerDay=1`, `weedSpawnChance=.5`, one native thread and legal PASS/STOP rows.
Each of three independent invocations consumes construction seeds `base,base+1`,
then the caller's explicit reset consumes `base+2,base+3`; no constructor reset
is hidden. Different evaluation steps yield different game-seed tuples and
first-day snapshots. Repeating one step reproduces exact seed tuples, both
initial and first-day snapshots and both seats' final banks for each game.

The initial snapshots intentionally compare equal across distinct seeds:
`engine_rs/src/lib.rs:1236` constructs deterministic farms/market and stores the
seed; `engine_rs/src/lib.rs:4440` first uses that seed in `end_of_day`, including
weed sampling at line 4490. The test checks the first observable randomized
world after one legal transition, then finishes the game in the second. This is
a clarification of “different starting games,” not an engine change or a claim
that initial observations expose hidden seed identity. PASS yields equal banks
in these short games; the tested property is exact replay, not policy strength.

## Blocked: canonical trainer adoption (out of Stage 2 scope)

`PPOTrainer` cannot consume these batches yet:

- `python/owl/train/ppo.py:251`: `_PPORolloutBuffer` accepts Orbit spec types;
  line 287 accesses `action_spec.n_bins` on its non-pure/non-discrete branch;
  line 315 unconditionally constructs Orbit `ObsBatch`; line 402 allocates
  Orbit `ActionBundle` rather than Kaggriculture token rows/lengths.
- `python/owl/train/ppo.py:596`: the canonical trainer creates that buffer.
- `python/owl/train/ppo.py:2075`: `_map_action_bundle` handles only the three
  Orbit variants. Line 2115's observation mapper likewise lacks the
  Kaggriculture action-mask case.
- `scripts/run_ppo.py:1786` and line 1831: policy evaluation still uses Orbit
  actions and its concrete `ObsBatch` device mapper.

Consequently `main` retains `require_orbit_env`, with a preceding explicit
Kaggriculture stop (lines 181-188) that reports “Task 3.1 rollout storage and
action mapping are not implemented.” Both shipped rank configs and a resume
path assert this message and zero run-directory/env/model allocations. An
explicit `_evaluate_games` stop (lines 1424-1429) reports its observation/action
mapping blocker once the real factory returns a Kaggriculture adapter. The
existing fake seam evaluation test still proves raw-bank scoring independently.
No `ppo.py`, `train/config.py` or `distributed.py` edit was made.

## Deviations / handoff

- The owner explicitly brought `_create_eval_env` adoption into Stage 2 despite
  the brief body assigning runner changes to Tasks 3.1/3.3.
- The launch stop remains, as item 6 directs for the actual source blocker.
  Its message is local to `run_ppo.py` because editing `train/config.py` and its
  generic narrowing helper is explicitly forbidden in this stage.
- A narrow policy-evaluation guard is necessary now that the factory returns
  the actual env union: it prevents falling into unsupported Orbit transport
  and permits honest static typing without coercion or a second collector.
- Different native seeds do not change reset observations; the test observes
  seed custody plus the first daily RNG transition as explained above.
- Task 3.1 still owns rollout storage/action transport, observation-mask mapping,
  complete native-policy evaluation, training-only seed-band policy and the
  canonical trainer integration smoke. The pod still owns DMA and the early
  two-rank smoke. These tests qualify neither of those paths.

Parent integration owns full `just prepare`, remaining-skips inventory,
README/API/model documentation, the brief deviation appendix and final receipt.
