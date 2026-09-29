# Stream D result and verification receipt

Date: 2026-09-29. Branch: `kg/rebuild-codex-data`.
Starting HEAD: `2d9f1cd34a1b23470497c5d255220adc70b6f6df`.
Reference: `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`.
Final commit identity is the Git commit containing this receipt in the local
review bundle described below; source/artifact SHA-256s are in
`stream-d-SHA256SUMS`. The requested worktree's branch could not be advanced.

## Delivered scope

- `scripts/kaggriculture_bc/select_replays.py`: typed standard-library CLI/API;
  reviewed P port retaining the seven seeds, 32/4 counts and manifest fields.
  Duplicate episode rejection moves from preparation to selection. Fail-fast
  short inputs, duplicate ZIP entries, path validation and no-overwrite/copy
  cleanup are deliberate improvements. No engine or model import.
- `tests/scripts/test_kaggriculture_select_replays.py`: 17 synthetic unit cases,
  including isolated CLI execution, reference golden ordering, manifest/payload
  fidelity, episode split separation, source preservation and interrupted copy.
- `../briefs/5.1-bc-data.md`: physical source custody and exact preparation work
  for document v4 / observation schema 3. Preparation itself remains R/deferred.
- `../briefs/7.3-replay-export.md`: Kaggle episode and native trace schemas,
  seed round-trip requirements, terminal/reset boundary and implementation tests.
- `../briefs/7.1-opponents.md`: starter/R04/EcoBot/E776 source/dependency/style
  recommendation; no bot code imported or strength claim made.
- Cookbook Reference + index + log record the adaptation and correct historical
  omission handling. `docs/kaggriculture-contract.md` changes by one blank line
  only, fixing the pre-existing Markdown failure uncovered by full preparation.

## Question, inputs and stopping condition

Prove that a local selector reproduces episode/split identity without an engine
or bulk corpus, and make the pending data/evaluation seams reviewable. Inputs
are reference Git blobs, accepted v4 contract, tiny synthetic ZIPs and the synced
source inventory. Discriminating checks are golden order, all 252 historical
IDs/splits, payload fidelity and duplicate rejection. Stop after source-bound
briefs, required offline unit/preparation checks and a branch-local commit.
No training, panel rollout, native Kaggriculture execution or network request is
part of this change.

## Commands and observed checks

All `just` recipes used this environment:

```sh
UV_TOOL_DIR=/private/tmp/kg-v3-stream-d-uv-tools \
UV_OFFLINE=1 UV_NO_SYNC=1 CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=3 \
uvx --from rust-just just py-prepare

UV_TOOL_DIR=/private/tmp/kg-v3-stream-d-uv-tools \
UV_OFFLINE=1 UV_NO_SYNC=1 CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=3 \
uvx --from rust-just just prepare
```

| Check | Actual outcome | Evidence |
|---|---|---|
| Initial focused selector pytest | 15 passed in 0.18s; two cleanup cases added afterward | Tool output; final coverage included below |
| Final Python preparation | 739 passed, 3 skipped; format, Ruff, Python 3.11 syntax, mypy (49 files), doc freshness passed | `stream-d-py-prepare.log` |
| Initial full preparation | Build/Rust lint/Python lint passed; stopped at existing MD032 in contract line 173 | `stream-d-prepare-initial.log` |
| Final full preparation | 155 Rust passed, 2 ignored; 739 Python passed, 3 skipped; build/format/lint/typing/docs passed | `stream-d-prepare.log` |
| Port versus independent source inventory | 252 ordered IDs, splits and uncompressed sizes match; 224 train / 28 validation | `stream-d-source-audit.json` |
| Native fixture structure | 4 headers / 2,876 transitions inspected; sequential timing, terminal banks/statuses match | 7.3 brief, pinned fixture hashes |
| Opponent footprint | 217,617 source bytes + 117,954 policy-data bytes; selected blobs hashed | 7.1 brief |

The Python skips are two unavailable FlashAttention/CUDA cases and one unavailable
x86 quantization backend on the Mac. The Rust ignored tests are inherited;
no test was disabled to make this change pass. The final full run contains all
17 new selector cases. These are software/structural checks, not learning,
throughput, live telemetry or new Kaggriculture engine parity evidence.

`uvx` initially failed because its default tools directory was outside the
sandbox's writable roots. Setting `UV_TOOL_DIR` to the writable temporary path
allowed cached offline tools to run. Initial focused Ruff reported line-length
and regex-literal issues; formatting and an explicit raw regex fixed them before
the successful preparations. No dependency, lockfile, toolchain or driver changed.

The fresh worktree lacked ignored Orbit fixtures. Three existing files were
copied byte-for-byte from `/Users/poonszesen/kg-v3/tests/fixtures/`; paths, sizes
and SHA-256s are in `stream-d-fixture-custody.json`. They stay ignored and are
not committed. No fixtures were generated or downloaded. Rust builds used
`CARGO_BUILD_JOBS=3` and Cargo offline mode.

## Review and residual gaps

The patch is scoped to the new selector/tests, three requested briefs, compact
evidence/cookbook and the one-line Markdown fix. No retained Isaiah code or Rust
code changed. Existing RL/model/training semantics and lockfiles remain intact.
Review checked action/rule logic duplication (none added), source provenance,
archive no-overwrite/identity behavior, ignored artifact exclusion and doc
freshness. A native subagent independently reviewed selector/test code and found
no blocker; its missing cleanup-test suggestion is covered by two added cases.
A second subagent cross-reviewed preparation/replay requirements against source;
no substantive defect was found. Its evidence-scope comment narrowed the BC
brief's negative filesystem claim to the three retained existence checks.
The owner's separate Codex verification before merge remains pending; no PR,
push or merge is performed by this stream.

### Commit custody and sandbox limit

Both ordinary `git add` and an explicit `--git-dir` invocation failed to create
`/Users/poonszesen/kg-v3/.git/worktrees/kg-v3-codex-data/index.lock` with
`Operation not permitted`, despite that Git directory appearing among the
declared writable roots. No stage or commit changed the original worktree.
Approval escalation is unavailable in this session.

A local `git clone --shared --branch kg/rebuild-codex-data` into
`/private/tmp/kg-v3-stream-d-review-20260929` provides writable independent Git
metadata without changing the protected original metadata. Only the 17 scoped
files are copied there, checked byte-for-byte against `stream-d-SHA256SUMS`,
staged, source/log-linted and committed on **`kg/rebuild-codex-data`**. The commit
is exported as `/private/tmp/kg-v3-stream-d-20260929.bundle`, with the starting
HEAD above as its prerequisite. This is an importable review commit, not a
claim that the original branch moved. The final report supplies its hash and
bundle path. The other Codex session can fetch/import it after review in an
environment that permits repository Git writes.

Open questions for the next stream:

1. **Current data reachability:** source volume `4llk4uaf20` is historically
   EU-RO-1, at `/workspace/kaggriculture-v2/public-episodes-2026-09-14-to-2026-09-27`
   or `/data/...` on the temporary reader. Historical transfer to GPU pod
   `w7ia3zvxqsvs3g` succeeded. Current mounts, retained selected ZIP/arrays and
   reachability are unverified without network. The raw ZIP destination and
   transfer protocol were not found in the tracked receipts; prepared GPU arrays
   were `/workspace/kg-v3/replays/bc-bootstrap/arrays`.
2. **Preparation/admission:** new native explicit-state writer and grammar must
   land before real data is prepared. Reproduce or explain the historical
   158,772 admitted / 22,416 rejected paired turns. No raw payload was rehashed
   or admitted in this task. Correct `None` versus PASS/NONE handling explicitly.
3. **Replay export:** expose diagnostic resolved-game seeds and completed state
   before auto-reset, implement the exporter and execute native round trips;
   structural fixture inspection does not qualify that path.
4. **Opponents:** preserve original notices, import the reviewed dependency
   closure, verify seat/episode lifecycle and action parity, then measure panel
   diversity and strength on the pod. No opponent is qualified by this brief.

The cookbook first-edit gate required reading the new governing Reference before
the contract whitespace edit; the note was read and the exact edit retried.
Final source/log lint and Git whitespace checks are retained with the staged
commit checks. All repository adaptations are inventoried in
`cookbook/references/rebuild-data-preparation-preserves-replay-identity.md`.
