# Task 7.1: byte-exact evaluation controllers

## Inputs and supported mechanism

Integration input: `b8747b6e8acece5f561d09a75bb914364a60ac05`; run-1
stop committed as `21d0f45`; reference controllers/data:
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Original Python submissions
come read-only from sibling commit
`e8884aae82eddeb7a1aeae99ecceeca7c830d67e`; Starter comes from the hash-pinned
`kaggle-environments==1.32.7` engine.

Claude's revised placement resolves the former external-crate API boundary with
a v3-owned `Game` snapshot view in `opponents_rs`. It owns the frozen engine,
refreshes its full `StepSnapshot` after construction and successful steps,
exposes the snapshot accessors the imported code calls, and derives only the
hire-cost multiplier through serde. An authored `fib` port supplies Starter's
historical helper. Both private seats occur in the snapshot; tests must establish
that controllers ignore the rival's private state. Engine seed/RNG/hidden
counters are unreachable through the view. The view is for evaluation and
clones the full snapshot; there is no training-path or performance claim.

Scope: four byte-exact controllers plus E776 policy data; authored view,
registry/lifecycle wrapper, default-config-only runner, source/trace custody,
original-Python differential checks, docs/cookbook and binding-dependent skips.
No engine bytes, `policy_rows`, all-controller dispatcher, root-crate dependency,
model, PPO, reward, grammar, panel, replay export or packaging changes.

## Planned checks and discriminating observations

1. Compile the imported controllers and Starter's five inline tests through the
   view. Test snapshot refresh, `from_engine`, serde multiplier and `fib` against
   explicit expected values. A compiler failure or stale/incorrect snapshot is
   an architecture/view defect, separate from controller behavior.
2. Test registry keys, independent per-seat ownership, reset, and explicit
   repeated/skipped/wrong-seat failures. Run sequential bounded default matches:
   the same seed must preserve action hashes and final banks, and a different
   seed must change the sequence. Non-default matches must fail explicitly.
3. At several live steps, clone controller and view, perturb rival-private state
   and compare actions. An own-private/public-state positive control must change
   an action. Record fields unavailable to perturbation separately.
4. Generate original-Python traces with one fresh module/agent per seat and game.
   The planned base cycle places all bots in both seats: Starter–R04,
   R04–EcoBot, EcoBot–E776, E776–Starter. A second seed is conditional on the
   4,000,000-byte trace budget. Run at most two live games per invocation, under
   two minutes per game and 1 GB RAM. If a bound is exceeded, stop and retain the
   exact continuation command; do not widen the sample.
5. Rebuild each native game from seed/config, compare canonical action JSON
   including numeric representation for each seat/step, then apply recorded
   Python actions and compare full expected state. On a parity mismatch, stop
   widening; report first step/seat/field and inspect both sources to localize
   the cause. Never edit imported bytes or weaken equality to pass.
6. Count actual openings, day resets, weeds, shortages/rejected orders, hires,
   final-day sales and mid-episode reset/replay. Assert the recorded counts and
   list zero-count categories as gaps. Mutate each comparison and retain its red.
7. Check exact source/trace inventory, reference hashes, authored hashes, trace
   budget and mutation rejection. The trim updater accepts only the pinned
   integration, committed run-1 manifest or exact final output; it changes only
   five excluded reasons and the task's non-engine inventory.

## Bounds and completion

Every shell exports the task's offline/thread/TMPDIR settings. No training,
GPU/model diagnostic, panel or network operation is permitted. The requested
repository preparation runs its existing regression tests; it is not a model
experiment. Git metadata is read-only. Run-1 compile/source/check receipts stay
byte-exact; final checks are written separately under `run2/`.

Completion means all implemented paths and actual check outcomes are reported,
source bytes remain in custody, mismatches and coverage gaps are explicit, and
the note/index/log/coverage records agree with the final results. Planned checks
above are expectations, not reported passes. Actual command statuses, counts,
trace hashes, observed coverage, mismatch localization and any exact handoff
command belong in `results.md` and the `run2/` receipts.
