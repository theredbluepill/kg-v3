# Independent verification — Task 5.2 BC trainer, r3

Date: 2026-09-29. Branch: `kg/rebuild-bc-trainer`.
Reviewed HEAD: `b626f24e203205dadaacf32018c84edc06030522`.
Requested three-dot base and actual merge base:
`2390c8e239a4d57770bdafd621c05abbcb326739`.

The bounded question was whether this branch preserves Isaiah's shared
optimizer/BF16/compile/DDP mechanisms for the supervised BC objective, selects
the held-out NLL minimum, implements L9 patience, preserves deterministic
sharding and emits checkpoints consumed by run_ppo. The review includes the
full three-dot implementation and every finding in the requested historical
r2 verdict, `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-5.2-trainer-r2.md`,
including its linked detailed report. Completion means reported check and
mutation outcomes, reconciled findings and byte-identical tracked files.

No production training, corpus preparation, GPU or live W&B run was launched.
The requested existing CPU tests exercise their miniature supervised updates;
the additional runtime-cadence probe replaces updates with a no-op. No
implementation fixes or tracked test changes were made during verification.
The historical r2 verdict is untouched; its current dispositions are below.

## Disposition of every r2 finding

| Finding carried by the r2 short/detailed report | Status | Evidence at this HEAD and remaining scope |
| --- | --- | --- |
| New r2 P3: off-interval terminal evaluation consumes patience and invalidates exact-restart claims | **RESOLVED** | `bc.py:427–432` updates both patience fields only for scheduled evaluations; `bc.py:867–870` marks the terminal one unscheduled while strict minimum selection remains active. Direct and real-loop tests cover worse and better off-cadence NLL, identical resumed weights and stopping at step 4. The additional runtime-cap probe confirms the same behavior and history flags. Docs distinguish selection from scheduled patience. |
| R1 P2: `min_delta` discards a strictly lower-NLL checkpoint | **RESOLVED** | Selection at `bc.py:433–436` follows every strict minimum, separately from the improvement reference. Direct and real checkpoint regressions retain 1.97 from `[2.0, 1.97, 2.4]` with tolerance 0.05. |
| R1 P2: restart inherits original source identity | **RESOLVED** | `train_bc.py:142–154,233–271` resolves this attempt's source and records source lineage, parent-state SHA-256 and start step. Script restart regression checks new source in best/result records. |
| R1 P2 companion: changed seed/batch accepted silently on resume | **RESOLVED** | `bc.py:497–505,554–592` persists and checks trajectory settings plus data/world identity; tests reject seed, batch, accumulation, tolerance, optimizer and PPO config changes and admit a budget extension. |
| Reversed policy/total or winner-CE loss signs survive | **RESOLVED** | Independent hand-computed objective oracle and real-update regression remain present and pass. |
| Removed length normalization survives | **RESOLVED** | The objective oracle uses unequal program lengths and checks the normalized result. |
| Removed real `optimizer.step()` survives | **RESOLVED** | The real BC loop regression checks model tensor changes and a material objective decrease. |
| Rank contribution to permutation seed unproven | **RESOLVED** | New test uses equal 24-row partitions at both ranks and compares complete epoch permutations; dropping rank from the seed now fails. |
| R2 mutation finding: missing/empty attempt file, preexisting fresh file, malformed type/index/source guards lack fixtures | **RESOLVED** | New malformed-lineage test covers every listed case; mutations independently remove the guards and separate type/index predicates. |
| Shared wiring, accumulation, clipping, finite-gradient, scheduler, preflight and runtime-cap regression assertions incomplete | **UNRESOLVED** | Production calls are present by inspection, but removal mutations still survive the committed BC tests. This review's runtime-cap probe adds scoped external evidence, not a committed regression. CUDA BF16/Inductor/DDP execution remains unqualified. |
| Many malformed shard/config guards lack negative fixtures | **UNRESOLVED** | Those fixtures were not added. Guard-deletion survival is a coverage limit, not evidence that invalid data reaches training: downstream checks can still reject it. The attempt-record subset above is now closed. |
| Aggregate coverage finding | **PARTIAL** | Objective, real update, rank seed, attempt guards and scheduled patience are covered; the disclosed wiring and remaining malformed-input gaps are open. |

Historical counts in r2 describe that revision, not this one. No new actionable
correctness finding was established. The exact continuation claim is scoped to
unchanged source/settings, deterministic execution and an equivalent work
budget. It does not establish reproducible wall-clock timing or completion
under different per-attempt runtime budgets.

## Task acceptance and qualification boundary

- Shared model/reset, optimizer/scheduler, autocast, compile, DDP and no-sync
  paths are reused. BC adds its supervised objective, data order, held-out
  selection and stopping control; it does not implement another PPO rollout or
  policy-gradient trainer. No v2 model/collector import was introduced.
- The critic is explicitly trained with per-seat winner CE from raw terminal
  banks (draws 0.5/0.5), alongside length-normalized teacher-forced policy NLL.
  Its coefficient is documented as an unmeasured starting value. Episode and
  opponent labels do not condition actor/critic inputs or checkpoint selection.
- Rank-resident rows are disjoint; epoch/rank permutation and complete-update
  counts are deterministic. Evaluation includes every held-out row with reduced
  numerators and denominators. Off-cadence selection can retain an extra lower
  minimum without changing scheduled patience.
- The best checkpoint has run_ppo's exact keys beside its resolved config.
  Tests exercise both weight/teacher loaders and PPO metadata. Full
  Kaggriculture PPO remains dependent on the documented native environment and
  trainer seams; this review does not qualify those unfinished integrations.
- CUDA BF16, Inductor, multi-process DDP/NCCL, pinned transfers, real dataset
  scale, throughput, peak memory and live W&B remain unqualified. No performance
  or policy-quality claim follows from these CPU checks. This verdict covers
  the Task 5.2 script, not the planned pod BC run.

Independent source-review details: `code-review/read-only-review.md`.

## Requested checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — exit 0; **1,703 passed, 11 skipped**, 42.18 seconds (`pytest.log`).
- `uv run mypy python/owl scripts` — exit 0; **no issues in 66 source files**
  (`mypy.log`).
- All 30 BC tests pass. The 11 skips are existing native grammar/environment
  and teacher/trainer seams, CUDA pinned-memory/FlashAttention cases and the
  unavailable quantized backend; none skips a BC test.
- Python 3.12.13, torch 2.9.0, NumPy 2.4.4, Pydantic 2.13.3, macOS arm64;
  CUDA unavailable (`environment.json`).
- Doc-freshness check returns 0: no doc updates required. Source/config/doc
  three-dot `git diff --check` excluding `ops/` passes. Full diff check returns
  2 for 213 trailing-whitespace diagnostics in committed raw test evidence,
  not source/config/docs (`diff-check.log`); this is a historical evidence-log
  formatting issue.

## Additional no-update runtime probe

`runtime-cadence/probe.py` uses real control flow, checkpoint/state/history I/O
and compatibility checking with scripted NLL and a fake clock; `_train_step`
is a no-op. Each case has interval 2, patience 2, NLL 3 at step 0 and NLL 4 at
scheduled later evaluations. A cap after step 1 produces `max_runtime`, with
unchanged patience count and improvement reference. Restart reaches step 4
just like uninterrupted execution. A worse step-1 NLL 4 keeps the step-0 best;
a better step-1 NLL 2.5 keeps the step-1 best without altering the stopping step.
Both histories have steps `[0,1,2,4]` and scheduled flags `[1,0,1,1]`.
All assertions passed (`runtime-cadence/probe.log`).

## Mutation verification

**137 unique mutations: 68 killed, 69 survived, zero setup errors or timeouts.**
One duplicate execution was excluded from these unique counts.
All 120 r2 substitutions still matched this source and were rerun; 17 additional
probes target the new schedule flag, unscheduled selection/patience, the logged
flag and individual attempt-record predicates. Every mutation ran all 30 BC
tests. This covers each new test/guard with a corresponding behavior mutation;
`mutations/test-mutation-map.json` records tests that failed for each case.

| Inventory | Tried | Killed | Survived |
| --- | ---: | ---: | ---: |
| Original r2 inventory | 120 | 53 | 67 |
| Additional r3 probes | 17 | 15 | 2 |
| Total | 137 | 68 | 69 |

The prior rank-seed survivor and missing/empty/preexisting/malformed-source
attempt-guard survivors are now killed. Separate type/index/source predicate
mutants are killed too. Reintroducing unscheduled patience, treating terminal
validation as scheduled, failing to propagate the flag, marking initialization
or periodic evaluations unscheduled, skipping terminal evaluation, skipping
unscheduled selection or suppressing its best-checkpoint write all fail.
The prior loss-sign, normalization, real optimizer update, compatibility,
lineage and strict-minimum mutations remain killed.

The two additional survivors invert or delete `bc/scheduled_evaluation` from
history. **UNRESOLVED coverage gap:** the committed tests do not assert that
telemetry field. The source currently records it correctly, confirmed by the
independent runtime probe; this is not a demonstrated behavior defect. The
other 67 survivors are the already disclosed wiring/config/data-negative
coverage limits. Mutation survival is not a correctness score or proof that
malformed input is accepted.

Three isolated scratch copies ran independent case partitions. Import probes
confirm each copied BC/data/script/test module was executed; test bytes matched
HEAD. Each mutation was restored in `finally` and checked byte-for-byte. All
three initial and restored baselines pass **30/30**; all restoration records
say `all_byte_exact: true`. Source files in this worktree were never mutated.
Exact substitutions, per-case logs/results, category totals and restoration
hashes are under `mutations/`.

## Custody and verdict

All **2,089 tracked-file SHA-256 hashes are unchanged**. Worktree and index
`git diff --quiet` both return 0. The only untracked path is this review's
`ops/rebuild-2026-09-29/codex/verify-5.2-r3-independent/` evidence directory
(`tracked-before.json`, `custody.json`). No tracked implementation, test,
cookbook or prior verdict was edited, and no production training was launched.

The r2 P3 is resolved, prior P2 fixes remain intact, and the new tests close the
rank-seed and attempt-guard gaps. Remaining coverage and GPU/data qualification
limits are explicit and do not establish a new defect in this script review.

VERDICT: APPROVE
