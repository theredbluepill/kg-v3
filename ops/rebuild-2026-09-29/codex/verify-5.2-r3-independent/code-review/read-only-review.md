# Independent source review — Task 5.2, r3

Reviewed `b626f24e203205dadaacf32018c84edc06030522` against three-dot base
`2390c8e239a4d57770bdafd621c05abbcb326739` in `/Users/poonszesen/kg-v3-bc`.
This subtask was source review only. No tracked writes, source mutations, or
production training were performed. Main verifier owns test/mutation execution
and final custody/report.

No new actionable correctness finding was established in this source review.

## Prior findings

- R2 P3 (off-cadence terminal evaluation alters patience/stopping): **RESOLVED**
  for the reviewed step-budget continuation semantics. `bc.py:427–432` gates
  both patience fields on `scheduled`, but lines 433–436 keep selection strict.
  The initial/scheduled calls pass true (829, 863), the terminal call false
  (867–870), and history records the distinction (759). Direct and real-loop
  regressions at `test_bc.py:502,648` cover worse and best off-cadence NLL,
  unchanged stopping step, and identical continued parameters. Same source,
  settings, deterministic device and equivalent work budget are required;
  wall-clock timing and per-attempt runtime budgets are not deterministic.
- R2 rank-seed coverage gap: **RESOLVED by new fixture**, subject to the main
  verifier's independent mutation result. `test_bc.py:462` holds both partition
  lengths at 24 and compares whole epoch permutations, whereas the prior test
  could differ solely because its lengths differed. `_epoch_permutation` keeps
  the rank input (`bc.py:212`).
- R2 attempt-record malformed-input gap: **RESOLVED by new fixture**, subject to
  the main verifier's mutation result. `test_bc.py:822` covers missing file,
  empty lineage, a preexisting fresh-attempt file, wrong index, non-object line,
  and absent/non-string source.
- Prior R1 P2 fixes remain **RESOLVED** by source inspection: strict lowest-NLL
  selection separate from tolerance/patience (`bc.py:425–436`); each launch's
  current source, parent-state hash and source lineage (`train_bc.py:142–154,
  233–271`); and checked dataset/rank/config continuation (`bc.py:497–505,
  554–592`).
- Loss-sign, length-normalization and actual optimizer-update gaps remain
  **RESOLVED** in test design (`test_bc.py:349,397`).
- Other disclosed wiring/negative-input coverage gaps: **UNRESOLVED**. The
  source is present; the Reference accurately says CPU tests do not establish
  CUDA BF16, Inductor, real multi-rank DDP/NCCL, pinned transfer, full-scale data,
  throughput or live W&B. Further malformed shard/config branches lack dedicated
  fixtures. Overall coverage finding remains **PARTIAL**, not a source defect.

## Supported source behavior

- `train_bc.py:157–171` uses `build_bc_model`, shared `configure_model_compile`,
  shared `create_optimizer`/`create_lr_scheduler`, restore, and shared DDP.
  `bc.py:183–189` constructs and resets the registered v3 model. This adds a
  supervised BC objective/control loop and does not duplicate PPO rollout or
  policy-gradient semantics. There is no v2 model/collector import.
- `bc.py:325–327` uses shared autocast; `_train_step` (922–944) includes shared
  DDP `no_sync` for early microbatches, loss/count accumulation scaling, finite
  gradient clipping, real optimizer and scheduler steps. The current model has
  all parameter groups on the BC graph; a gradient-coverage fixture exists.
- `train_bc.py:106–113,158` retains workload and compiler-stack checks plus the
  model's cuBLAS-only compiler claim. This is source wiring evidence only.
- Critic handling is explicit: `bc.py:272–286,299–315` derives a per-seat
  self/opponent winner target from raw terminal banks, draws 0.5/0.5, with
  normalized teacher-forced policy NLL plus value CE. The equal coefficient is
  documented as an unmeasured starting recipe. Episode labels never enter the
  actor or critic.
- Loader schema matches the refreshed 5.1 brief read directly at `bcefd02`:
  per-episode compressed named v4 tensors, compact exact integer residency,
  manifest SHA-256/byte/layout/contract checking, duplicate-episode rejection,
  and paired seats. `bc_data.py:287–309` keeps disjoint `[rank::world_size]`
  rows. `bc.py:250–266` yields deterministic epoch/rank permutations and equal
  complete per-rank update counts. Validation uses every held-out row and an
  all-reduced numerator/denominator (`bc.py:354–384`).
- Best checkpoint keys in `bc.py:459–470` match both PPO metadata readers.
  A sibling resolved `config.yaml` is saved at `train_bc.py:138`; the advertised
  `--load-model-weights` and teacher paths load model weights, not BC optimizer
  continuation. Tests directly exercise those loaders (`test_bc.py:863`).
  Full Kaggriculture PPO execution remains dependent on the disclosed native
  environment/trainer seams and was not claimed verified.

Read governing scope, stateless-policy, recipe and evaluation Decisions, BC
Reference, prior r2 short/detailed report, current/previous BC implementation,
shared optimizer/distributed/compile utilities, model replay, PPO checkpoint
readers, current tests/config and the refreshed 5.1 shard brief. No external web
facts or performance claims are used.
