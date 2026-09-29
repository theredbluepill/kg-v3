# Task 7.1 R04 parity mismatch: angular-sector float reduction

This receipt localizes a native-controller decision mismatch against the requested
original Python submission on Python 3.12.13. It does not authorize a repair to
byte-exact imported files or certify another Python runtime.

## Frozen evidence and first mismatch

- Trace: `opponents_rs/fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz`,
  SHA-256 `39b8bc35c223594d2d6740fc35d0b8242e130351e0d69b2e4bc34594976cf97e`,
  178,476 bytes; seed 20260929, default configuration, 719 Python transitions.
- Seat 0 is Kaggle 1.32.7's original Starter; seat 1 is original R04 at sibling
  commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e`, entry source SHA-256
  `22d074391822206872448a6114a37ba2a4a39eda2ddbbcdc0bc8aa8cc6a64188`.
- First action mismatch: from-step 12, seat 1, `hands[2][0]`: native `WEST`,
  Python `NORTH`. Thirteen actions compared per seat: Starter 13 matched,
  R04 12 matched. Twelve native transitions matched the Python public/private
  state and typed status/reward data before stopping.
- A prior comparator attempt stopped on native scalar reward `0.0` versus Python
  JSON `0`. This was a harness mismatch: the existing engine replay comparator
  deserializes rewards as `Vec<f64>`. The new comparator adopts that existing
  typed reward contract. Action numeric representation, public/private state and
  their ordering remain strict; the step-12 action mismatch remains a failure.

## Source cause

Original `agents/r04/main.py:1070–1109` computes angular territory anchors.
The public state has 25 unlocked empty tiles at step zero, each weight 0.35.
`total_w = sum(p[3] for p in pts)` (1091) establishes the threshold per hand;
`acc += p[3]` (1098) then chooses sector boundaries. The pinned native
`opponents_rs/src/native_agents/r04.rs:849–912` implements the same structure,
but its total and coordinate reductions use sequential Rust `sum::<f64>()`
(882, 890, 894–895 and 903–908).

The actual runtime is `3.12.13 (main, Mar 3 2026, 15:35:03) [Clang 21.1.4]`.
Observed, without relying on a Python-version assumption:

| Reduction or threshold | Actual value |
| --- | --- |
| Python built-in `sum([0.35] * 25)` | 8.75 |
| Sequential accumulation of the same inputs | 8.749999999999996 |
| Python threshold per five hands | 1.75 |
| Sequential threshold per five hands | 1.7499999999999993 |
| Sequential accumulated weight after 15 points | 5.249999999999999 |
| Python threshold for group 3 | 5.25 |
| Sequential threshold for group 3 | 5.249999999999998 |

Thus the fifteenth point closes native group 3 but does not close Python group 3.
Internal anchors already differ at step zero: Python unit 3 is
`(1.8333333333333335, 1.6666666666666667)` while native is
`(1.9999999999999998, 1.9999999999999998)`; units 4 and 5 also differ. Roles,
jobs and selected targets agree through step 11 despite those anchor differences.
At step 12 the anchor penalty changes assignment: unit 3 targets Python `(1,1)`
versus native `(0,2)`, and unit 1 makes the converse assignment. This changes
unit 3's first move to `NORTH` versus `WEST`. Score construction reads anchors at
Python 1372–1375 and native 1597–1603; Hungarian selection consumes those scores
at Python 1397 and native 1617.

The engine's state and v3 observation view are not implicated by this bounded
finding. The native port does not reproduce the requested original-source oracle
under the actual recorded runtime. Historical native comments are not evidence
of compatibility with this runtime; the original competition runtime has not
been established by this task.

## Counterfactual and controls

`localize_r04_python.py` reads the frozen original source from Git via the same
hash guard, creates a fresh module, and replays only the first 13 frozen legal
observations (no live game and no new trace). All 13 original actions exactly
reproduce the stored Python actions. It records original jobs, anchors, targets,
candidate score pairs and selected pairs in `r04-python-debug.json`.

A second fresh module changes **only the `sum` binding inside the original
`compute_anchors` function** to straightforward sequential accumulation. The
function's bytecode and every other original global/function are preserved;
this is an explicitly altered diagnostic, never an oracle fixture. All 13
counterfactual anchors match the native debug output exactly, and all 13 actions
match native actions, including `WEST` at step 12. This isolates the floating
reduction as the cause of this first mismatch, rather than guessing from the
visible action difference. Native debug came from the unchanged imported
controller through `r04-diagnostic.rs` and is in `r04-native-debug.json`.

Original R04 clears and sets `KG_*` variables during module initialization.
A static AST check of this pinned source finds the only function referencing
`os.environ` is `_knob` at line 102; all `_knob` calls are module-level.
`GLOBAL_ASSIGN=1`, `ASSIGN_LOOKAHEAD=0`, and `ASSIGN_JOINT_ROUTES=0` are also
recorded in the Python debug receipt. Restoring the caller's environment after
initialization therefore does not change this pinned controller's runtime knobs.

## Bounds, coverage and reopening conditions

Exactly one live game was generated: 719 transitions in 1.2 seconds of generator
wall time (1.78 seconds including launch). `/usr/bin/time -l` then failed to read
`kern.clockrate` under the sandbox, so this first invocation has **no measured
RSS receipt**. The generator now records `resource.getrusage` peak RSS in future
summaries and aborts at the 1 GB bound, while the 120-second alarm cannot be
swallowed by an original agent's broad `except Exception` handler.

Generation stopped at the first substantive parity mismatch. No additional
pairings, second-seed games or later-step native qualification were collected.
EcoBot, E776, Starter seat 1 and R04 seat 0 remain without oracle comparisons.
The full Python trace coverage is in `coverage.json`; it is distinct from the
12-transition native qualification prefix. `buy_quantity_above_inventory_index`
is only an input proxy, because the market inventory field is a signed price
index; actual market shortages and individual rejected orders were not measured.
Final-day sales count submitted SELL orders, not proceeds. Mid-episode replay
was not generated. License/notice gaps remain unchanged.

Reopen controller parity only with an explicit contract for the required Python
runtime and an authorized change to the byte-exact native-source requirement
(or a newly pinned corrected upstream controller). Do not hide this mismatch by
relaxing action comparison, modifying the trace, or silently changing runtimes.
