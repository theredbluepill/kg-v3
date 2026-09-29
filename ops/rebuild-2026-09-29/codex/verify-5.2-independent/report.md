# Independent verification — Task 5.2 BC trainer

Date: 2026-09-29. Branch: `kg/rebuild-bc-trainer`.
Head: `50fea1ca3afec44b1551b1443c3a5ab20fc275c8`.
Requested three-dot base and actual merge base:
`2390c8e239a4d57770bdafd621c05abbcb326739`.

The bounded question was whether the BC implementation preserves the shared
Isaiah training mechanisms, selects the lowest held-out NLL, implements the L9
stop rule, preserves deterministic sharding/restarts and produces PPO-compatible
checkpoints. Inputs were the complete three-dot diff, current shared trainer and
model code, the governing cookbook, Task 5.2 plan, the refreshed 5.1 brief at
`bcefd02`, synthetic tests and isolated mutations. Completion requires the two
requested check commands, mutation outcomes and byte-exact tracked-file custody.
This review launches no production training, real-data, GPU or W&B run.

## Findings

### P2 — Separate minimum-NLL checkpoint selection from patience tolerance

`python/owl/train/bc.py:400` uses
`nll < self.best_nll - self.min_delta` to update `best_nll` and return the flag
that writes the best checkpoint. A valid nonzero `min_delta` therefore causes
the lowest-NLL checkpoint to be discarded. The no-training probe
`selection-probe.json` supplies NLLs `[2.0, 1.97, 2.4]`, `min_delta=0.05` and
patience 2. It stops with the retained best at step 0 / NLL 2.0 even though step
1 scored 1.97. `tests/kaggriculture/test_bc.py:393` explicitly expects this
behavior, so the passing test does not establish the requested minimum-NLL
selection. The supplied YAML's zero default avoids this case but the public
config accepts it. Save every strict absolute minimum; maintain significant
improvement/patience state separately, or disallow nonzero tolerance if that
feature is not intended.

### P2 — Bind resumed attempts to their actual source and continuation config

`scripts/train_bc.py:112–118` restores the original `launch.json` provenance;
only fresh launches call `_git_head()` at line 121. New checkpoint sidecars and
results reuse this old `source_commit` (`python/owl/train/bc.py:896` and `:812`).
A resumed attempt after a source change is consequently attributed to the old
implementation. This violates the source-bound evidence and separate-attempt
custody contract.

`probe_resume_identity.py` exercises the real `main` resume branch, real config
loader, real compatibility check and real launch-record reader, while stubbing
model/data loading and optimization. It records zero calls to `_git_head` and
passes the synthetic prior source `000…000` through to the training boundary
despite current HEAD `50fea1c…`. The same probe demonstrates that changing seed
20260929 → 20260930 and rows/rank 256 → 128 is admitted: compatibility checks
only dataset-manifest hash and world size (`python/owl/train/bc.py:509–521`).
Pure `TrainSharding` calls show those settings change row selection at the same
saved step. The checkpoint has no continuation-config fingerprint.

Validate the current source and trajectory-affecting settings before claiming
an exact continuation, or explicitly record a new attempt with actual source,
config and parent checkpoint identity. Keep allowed budget extensions distinct
from seed, batch, accumulation and scheduler changes. The source misattribution
is the finding; the config probe additionally bounds the advertised exact-restart
guarantee to unchanged settings. See `probe_resume_identity.log`.

## Supported behavior and scope limits

- The new loop is offline supervised BC, with no duplicated PPO rollout or
  policy-gradient semantics. It calls the shared model/reset, optimizer,
  scheduler, autocast, compile and distributed adapter paths. Gradient
  accumulation uses the shared no-sync context and divides the loss by the
  accumulation count. Actual GPU/BF16/Inductor/NCCL operation is unqualified.
- The critic treatment is explicit: length-normalized teacher-forced policy
  NLL plus a winner CE from raw terminal banks, including draws. The Reference
  justifies training the critic and discloses the unmeasured coefficient.
- The data layout matches the refreshed 5.1 shard brief. Episode identities
  are unique across splits; rank slices are disjoint; exact integer compaction
  is range-checked; all positive-admission shards undergo hash/size/schema and
  observation-contract checks. Zero-admitted records are skipped. Custody
  extras are preserved but not interpreted, as documented.
- Fixed settings produce deterministic rank-local order; the CPU restart test
  matches uninterrupted model weights. All validation rows are evaluated and
  metrics are summed across ranks. Real distributed execution remains open.
- The best-checkpoint test passes the existing `run_ppo` metadata and weight
  loaders with the neighboring PPO config. Integrated PPO remains blocked on
  the pre-existing environment/trainer seams, accurately disclosed by the
  Reference and skipped tests.

## Requested checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — exit 0; **1,691 passed, 11 skipped**, 46.16 seconds. See `pytest.log`.
- `uv run mypy python/owl scripts` — exit 0; **no issues in 66 source files**.
  See `mypy.log`.
- `git diff --check BASE...HEAD` — exit 0.

The skips cover unavailable native grammar/env integration, four pre-existing
teacher/trainer seams, CUDA pinned memory/FlashAttention and an unavailable
quantized backend. None skips a BC test. Environment: Python 3.12.13,
torch 2.9.0, NumPy 2.4.4, Pydantic 2.13.3; CUDA unavailable.

## Mutation checks

**100 unique mutants; 27 killed, 73 survived.** Two targeted survivors were
also rerun against all 18 BC tests and survived, for 102 total executions. No
mutation execution had a setup error or timeout. An initial runner syntax error
occurred before any mutation and was corrected. Each case changed one behavior
or guard on an isolated copy, using the current interpreter and verified source
resolution. Every change was restored byte-for-byte in `finally`, with hashes
compared to the original source. The post-restoration BC suite passes all 18
tests in 2.20 seconds.

The inventory covers each of the 18 new tests, explicit runtime/input/state
guards, BC config bounds, manifest schema/field constraints, exclusive-create
writes and the shared training entry points. Exact substitutions, selectors and
per-case logs are under `mutations/manifest.json`, `mutations/results.json` and
`mutations/extra/`. These counts are a coverage diagnostic, not a correctness
score: removing the `min_delta` behavior actually corrects minimum selection but
is killed by the current test's incorrect expectation. Some other kills arise
from explicit downstream dtype/shape/admission errors or expected error wording.

Meaningful coverage gaps include:

- Reversing both loss terms' signs survives **all 18** tests. The direct toy
  optimizer test only requires `final < 0.5 * initial`; increasingly negative
  losses satisfy it. Removing length normalization also survives.
- Removing the real trainer's `optimizer.step()` survives **all 18** tests.
  The toy optimization test uses its own update loop, while integrated loop
  tests exercise scripted selection, checkpoint equality and repeatability.
  They do not establish that the actual BC loop learns. Add an independent
  numerical objective oracle and a regression that proves the real update
  path changes parameters and improves the intended objective.
- Removing autocast, compile, DDP, startup workload/stack checking, accumulation
  normalization, clipping, scheduler stepping, preflight and runtime-cap
  handling survives. The CPU fixtures do not exercise these configurations or
  assert their wiring. GPU execution is already an explicit documentation gap;
  call-contract tests and bounded CPU accumulation/scheduler/runtime tests can
  close several wiring gaps without a GPU run.
- Many malformed-data/config guards have no corresponding negative fixture.
  A survivor is not automatically an admission bug: a missing-file or shape
  guard may still fail downstream. The per-case log identifies the precise
  removed guard; do not infer that every surviving mutation admits bad input.
- Removing rank from the permutation seed survives, including the full suite.
  Row partitions stay disjoint, but order no longer has the documented rank
  dependency. The test's unequal rank sizes can distinguish permutations even
  when rank is ignored, so this is not evidence of that dependency.

## Custody

The worktree started clean. An SHA-256 inventory captured all 1,798 tracked
files before verification. The post-check inventory passes for all 1,798 files;
`git diff --quiet` and `git diff --cached --quiet` both exit 0. Evidence is added
only under this untracked review directory. No implementation repair or cookbook
adaptation was made, following the requested no-tracked-modification boundary.

VERDICT: REJECT
