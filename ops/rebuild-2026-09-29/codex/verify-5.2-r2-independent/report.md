# Independent verification — Task 5.2 BC trainer, r2

Date: 2026-09-29. Branch: `kg/rebuild-bc-trainer`.
Reviewed HEAD: `eacab5e4833ca36fd0417bca83a3599355af187c`.
Requested three-dot base and actual merge base:
`2390c8e239a4d57770bdafd621c05abbcb326739`.

The bounded question was whether this implementation preserves Isaiah's shared
optimizer/BF16/compile/DDP mechanisms, selects the strict held-out NLL minimum,
implements L9 patience, preserves deterministic row order and produces the
checkpoint schema consumed by run_ppo. This review also rechecks every finding
in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-5.2-trainer-r1.md`
and the linked detailed report. Inputs were the full three-dot implementation,
the r1-to-r2 correction, governing cookbook/plan, shared training/model code,
the requested tests, isolated mutations and a no-update restart probe.
Completion means commands and mutation outcomes reported and tracked bytes
unchanged. No production training, corpus, GPU or live W&B run was launched;
synthetic CPU unit tests exercise their existing miniature update loops.

## Disposition of every r1 finding

The old r1 report remains unchanged. These statuses describe this HEAD.

| R1 finding | Status | Evidence and remaining scope |
| --- | --- | --- |
| P2: `min_delta` discards a strictly lower-NLL checkpoint | **RESOLVED** | `bc.py:416–425` updates patience against `improvement_nll` separately from strict `best_nll`. The new direct and real checkpoint tests retain 1.97 from `[2.0, 1.97, 2.4]` with tolerance 0.05. |
| P2: restart outputs inherit the original source identity | **RESOLVED** | `train_bc.py:142–155,233–271` resolves the source on each launch/restart and appends the attempt, source lineage and parent-state SHA-256; checkpoint sidecars and results copy that attempt. The real script restart test checks current source `b` after launch source `a`. |
| P2 companion: changed seed/batch silently passes resume compatibility | **RESOLVED** | `bc.py:485–494,542–580` persists and checks BC settings except the budget/path and compares the complete PPO config by content. Tests reject changed seed, batch, accumulation, tolerance, optimizer and PPO dtype while accepting a larger step budget. This does not establish the unconditional exact-continuation claim; see the new P3 below. |
| Significant test gap: reversed loss signs pass | **RESOLVED** | The hand-computed objective test independently checks length-normalized policy NLL, winner CE and their weighted sum. |
| Significant test gap: removing the real `optimizer.step()` passes | **RESOLVED** | `test_the_real_training_loop_lowers_the_objective` runs the actual BC update path and checks a positive objective reduction and changed floating model tensors. |
| Detailed report: removing length normalization passes | **RESOLVED** | The new numerical objective oracle includes unequal program lengths. |
| Detailed report: missing shared wiring, accumulation, clipping, scheduler, preflight and runtime-cap assertions | **UNRESOLVED** | These configurations/entry points remain incompletely asserted by the CPU suite; source wiring is present and the Reference explicitly lists this limitation. No GPU execution is inferred from passing CPU tests. |
| Detailed report: many malformed-data/config guards lack negative fixtures | **UNRESOLVED** | No new malformed-input suite was added. A surviving guard deletion is a coverage gap, not proof that malformed input reaches training; downstream rejection may remain. |
| Detailed report: rank's contribution to the permutation seed is unproven by the test | **UNRESOLVED** | The test still uses unequal partition lengths, so dropping rank from the seed survives. Source inspection confirms rank is included and row partitions stay disjoint. |

The r1 coverage finding is therefore **PARTIAL** overall: its loss-sign,
normalization and real-update gaps are closed; its other disclosed gaps remain.
Historical test/mutation counts in r1 are not assertions about this revision.

## New finding

### P3 — Qualify the exact-restart guarantee for off-interval stops

`python/owl/train/bc.py:854–855` performs a final validation when a budget or
runtime stop falls between scheduled evaluations. That validation updates
patience at line 744 and persists it in `bc_state.pt` at lines 784–800. A
compatible restart retains that extra observation but resumes the original
modulo-based evaluation cadence at line 849.

The no-update probe in `restart-cadence/probe.py` uses the real control flow,
checkpoint writes and compatibility check, with prescribed NLL and no optimizer
updates. With evaluation interval 2, patience 2, NLL 3 at step 0 and 4 thereafter:

- An uninterrupted budget of 4 evaluates at `[0, 2, 4]` and stops at step 4.
- A budget of 1 followed by a compatible extension to 4 evaluates at
  `[0, 1, 2]` and stops at step 2.

Thus the row sequence remains deterministic, but a restart can change the
patience/stopping trajectory even with unchanged source and every checked
trajectory setting. `README.md:479–480`, `bc.py:36–42` and the BC Reference's
residency/restart descriptions overstate this as repeating the uninterrupted
run. Narrow the guarantee to deterministic row/update continuation and state
that off-interval terminal evaluation changes selection/patience, or preserve
the scheduled evaluation semantics across such interruptions and add a test.
This is a documentation/qualification edit, not a failure of the requested
strict-minimum selection or disjoint sharding. Evidence: `restart-cadence/probe.log`.

## Supported behavior and limits

- The BC loop is a supervised objective, not a second PPO rollout or
  policy-gradient implementation. It calls the shared model/reset, optimizer,
  scheduler, autocast, compile, DDP and no-sync seams.
- Gradient accumulation divides the loss by its count; clipping and finite
  gradient rejection remain present. Compiled Kaggriculture regions use the
  existing cuBLAS-only claim/workload/stack paths.
- Critic training is explicit: normalized teacher-forced NLL plus winner CE
  from raw terminal banks, including draws. The equal coefficient is an
  unmeasured starting recipe; identity labels do not condition the policy.
- Rank slices are disjoint, row ordering is deterministic, and all validation
  rows contribute to reduced metrics. The aligned CPU restart oracle matches
  uninterrupted weights; the off-interval caveat is above.
- Existing run_ppo weight/teacher loaders and PPO metadata consume the best
  checkpoint. Full Kaggriculture PPO execution remains dependent on the
  disclosed native environment/trainer seams.
- Actual CUDA BF16, Inductor, multi-process DDP/NCCL, pinned transfers, real
  dataset scale, throughput and live W&B remain unqualified. This is review of
  the Task 5.2 script, not completion of the planned pod BC run.

## Requested checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — exit 0; **1,698 passed, 11 skipped**, 47.63 seconds (`pytest.log`).
- `uv run mypy python/owl scripts` — exit 0; **no issues in 66 source files**
  (`mypy.log`).
- The skips are unavailable native grammar/environment integration, four
  existing teacher/trainer integration seams, CUDA pinned memory/FlashAttention
  and the unavailable quantized backend. None skips a BC test.
- Environment: Python 3.12.13, torch 2.9.0, NumPy 2.4.4, Pydantic 2.13.3;
  CUDA unavailable (`environment.json`).

## Mutations

**120 unique mutations: 47 killed, 73 survived, zero setup-error executions or
timeouts.** Each substitution runs all 25 current BC tests. The inventory
covers the original guards/tests plus the revised selection, seven added tests,
continuation fields and new attempt-record guards. It reuses the earlier
inventory where source is unchanged, replaces retired guards, and adds 21
cases. These are coverage probes, not a correctness score.

The numerical objective catches reversed total loss, reversed winner CE and
removed length normalization. The real-update regression catches a removed
`optimizer.step()`. Reintroducing `min_delta` checkpoint gating, sliding the
patience reference, omitting continuation fields/compatibility, reusing the
original source, losing source lineage/parent hash, incorrect attempt indices
or start steps, and skipping sub-tolerance best-checkpoint saves all fail.

Survivors include the already disclosed wiring/config/data-negative gaps and
rank-seed dependency, plus deletion of fresh-existing-attempt, empty-history,
missing-attempt-file and source-type admission guards. The production guards
are present; their malformed-input branches lack tests. Removing the combined
attempt-record type/index guard also survives. Several guards still have
fallback/downstream rejection, so survival does not by itself imply acceptance
of invalid input.

Three isolated scratch copies ran independent case partitions. Import-path
probes confirm the copied BC, data, script and tests were the executed modules.
Each mutation was restored in `finally` and checked byte-for-byte, with a final
comparison to repository source. All three restored baselines pass **25/25**.
A substitution-text mismatch was corrected before any mutant was executed.
Exact substitutions, source hashes, per-case logs, restoration records and
results are under `mutations/`; `mutations/test-mutation-map.json` maps each test
to observed failures. No implementation or test was repaired during review.

## Other checks and custody

`git diff --check BASE...HEAD` returns 2 for 26 trailing-whitespace diagnostics
in the previously committed raw pytest evidence logs. Excluding `ops/` passes;
there is no source/config/document whitespace finding (`diff-check.log`).

All **1,930 tracked-file SHA-256 hashes remain unchanged**. Both worktree and
index `git diff --quiet` checks return 0. The only untracked path is this review
evidence directory (`tracked-before.json`, `custody.json`). The prior r1 verdict
and cookbook are untouched; the requested findings dispositions are above.

VERDICT: APPROVE WITH EDITS
