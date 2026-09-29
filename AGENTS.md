<!-- pyml disable first-line-heading -->
<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
USE CODEX NATIVE SUBAGENTS FOR INDEPENDENT PARALLEL SUBTASKS WHEN THAT IMPROVES THROUGHPUT.
<!-- END AUTONOMY DIRECTIVE -->

# Kaggriculture v3

Use this repository as the Isaiah starter adapted to Kaggriculture. Preserve its
license, useful training infrastructure and development workflow. Read the
[cookbook scope](cookbook/decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics.md).
Historical Orbit Wars docs describe the retained starter; they do not establish
Kaggriculture rules or port completion. Keep `scripts/run_ppo.py` the canonical
trainer and extend its shared path instead of creating a second trainer. The
owner explicitly requires the current v3 data pipeline (Rust I/O and PPO): keep
the starter's Rust/PyO3 caller-owned buffers and existing PPO loop canonical;
reuse v2 engine semantics behind the v3 adapter, not its collector/trainer.
**Do not take the v2 model implementation.** Reuse starter transformer blocks and
new game stems/heads. V2 reward shaping (economic penalties, bank difference,
win/loss/draw) may be adapted with explicit coefficients, critic-range and
bootstrap semantics; no unmeasured reward recipe is presumed optimal.

**Architecture and debugging.** Map investment payback, labor allocation, sale
timing and market adaptation through legal observation, representation, choice
and executable action. Make behavior reproducible and localizable. Before a run,
state its supported mechanism or diagnostic question, inputs/code path, expected
discriminating observation and stopping condition. Diagnose information,
decision, execution and architecture errors separately; name unresolved
attribution. When repairs stall, inspect the engine and relevant primary sources
before spending again; record supported limits and reopening conditions. No blind
experiments or uninspectable systems; this is reasoning, not an approval gate.

**Stateless observation-only policy.** Actions use explicit current-observation
tokens. No between-turn hidden state, temporal memory training or new history
tokens. Within-turn autoregressive prefix state resets on each observation.
Opponent identity may not condition actor/critic inputs, embeddings, heads,
losses, rewards, normalization or checkpoint selection. Collection mixes and
panel labels are allowed without exposing identity to the model. Reject
prohibited checkpoint state. Follow the [adopted constraints](cookbook/decisions/the-policy-is-stateless-and-observation-only.md).

**Correct high throughput is required.** Measure equivalent complete work,
including valid learner turns per full update and phase costs; distinguish
startup, hardware, precision and workload changes. Component speed does not
prove end-to-end speed. For NVIDIA CUDA bottlenecks, **Nsight Systems (`nsys`)
is the canonical timeline profiler**; reuse representative captures or profile
a bounded relevant phase. Preserve versions, source/config identity, trace paths,
overhead and limits. Do not disrupt a learner, upgrade drivers or weaken security
for profiling. Missing tools limit performance claims and add no mandatory run
or admission gate. See the [profiling workflow](cookbook/workflows/profile-cuda-bottlenecks-with-nsight-systems.md).

**Evaluate generality and keep custody.** Report win rate and bank margin across
different opponents with version, seed, seat, denominator, legality, completion
and runtime. Distinguish selection from held-out qualification. Preserve stable
experiment identity with separate attempt/checkpoint metadata, compact local
source-bound evidence and W&B telemetry using v3 identifiers. Telemetry outages
remain visible. Keep credentials, bulk weights/corpora and local state out of
git with custody manifests; preserve source and lockfiles. The owner decides
submission. No old v2 score, ranking, lane mandate or pod allocation transfers.

## Retained starter engineering workflow

## Repository Map

- Start with `README.md` for setup, fixture regeneration, and the current
  reference episode IDs.
- Use `docs/rules-engine.md` for rules-engine architecture, current status,
  and known rule risks before changing `src/rules_engine/`.
- Use `docs/rules-parity-coverage.md` as the parity coverage source of truth.
  Update it whenever rules behavior, fixtures, or coverage changes.
- Reference `docs/rl-api-specs.md` before changing `python/owl/rl.py`, `src/rl/`, or
  public RL tensor shapes.
- Reference `docs/model-architecture.md` before changing `python/owl/model/` or model
  config, tensor ordering, actor, critic, or initialization behavior.
- Reference `README.md` and the training config tests before changing
  `python/owl/train/`, `scripts/run_ppo.py`, or `configs/train/`.
- Reference `docs/pr-checklist.md` before creating, recommending, or merging a PR.

## Development Workflow

- Run `just py-prepare` / `just rs-prepare` after any `python` / `rust` code edits, respectively. This handles formatting, linting, static type-checking, and tests.
- Run `just prepare` before creating or recommending a commit or PR.
- Leave touched code in better shape than you found it, while keeping cleanup
  scoped to the task at hand.
- Add dependencies with `uv add` / `cargo add`; don't edit `.toml` or `.lock` files directly when adding dependencies.
- Keep `Cargo.lock` and `uv.lock` tracked. Update lockfiles with package-manager
  commands, not manual edits.
- Avoid jumping through hoops for backwards compatibility - don't be afraid of
  refactoring and breaking old APIs in order to improve them.
- Before creating or recommending a PR, complete `docs/pr-checklist.md` and summarize any residual risks.
- Merge PRs with a regular merge commit by default. Do not squash-merge unless
  the user explicitly requests it.
- For retained Orbit Wars rules changes, keep `docs/rules-engine.md` and
  `docs/rules-parity-coverage.md` current with implementation state, test
  surface, and known gaps.
- `just docs-fresh` requires mapped code changes to update their mapped docs.
  If docs are already current for a small change, rerun the check with
  `DOCS_CURRENT=1` to indicate that the mapped docs were reviewed and are
  still current.

## Error Handling

- Fail fast with explicit, informative errors instead of silent fallbacks.
- When user input is invalid, raise clear exceptions.
- For persisted data schemas owned by this repo, prefer strict key access and explicit validation over backward-compatibility fallbacks.

## Attribute Access

- Avoid `getattr`, `setattr`, and `__dict__` in first-class code paths. They
  undermine the repo's mypy-forward pseudo-typed style; prefer direct attribute
  access and explicit union narrowing with `isinstance`.
- Use dynamic attribute access only for narrow, deliberate cases such as
  iterating over known schema/dataclass fields (`ObsBatch.model_fields`,
  `_OBS_TENSOR_FIELDS`) or test monkeypatching. If backwards compatibility
  seems to require dynamic access, prefer explicit schema migration and record the reason.

## Cookbook contract

**Read first.** Read `cookbook/index.md`, the newest `cookbook/log.md`, and
relevant notes at the start of work. Search concepts, synonyms and negative
evidence before creating knowledge or repeating an approach. Read governing
notes citing an edited file as a `repository:` source; if a first-edit hook
presents one, read it and explicitly retry the intended edit.

**Work in bounded changes.** State a target and observable completion/stopping
condition. Keep change/run artifacts in `ops/`; git preserves implementation
history. Report checks and unknowns without inventing thresholds or results.

**Record every adaptation.** Owner: “every adaption you made to this repo, please
add a cookbook record.” Every material repository adaptation must have an
identified cookbook record linking changed paths, reason, actual checks and
remaining gaps. Coherent adaptations may share a contract record with an explicit
inventory. This overrides the generic progress-only exemption for adaptations;
routine monitoring still belongs in working artifacts.

**Record before finishing.** Material decisions, corrections and reusable
findings require three edits together: note, its folder `index.md`, and a
prepended `cookbook/log.md` entry.

- Keep routine launch, build, checkpoint and monitor output in the change's
  working artifacts. Close result episodes with outcomes, denominators and gaps;
  freeze completed evidence. Progress alone earns no new durable note.
- Revise an existing durable concept when its reusable claim, evidence,
  consequence or limits change. Reconcile its opening, conclusion, metadata and
  index description together. Replace superseded current prose and link history;
  do not append dated operational progress to References.
- Split independent claims when needed. Preserve unique dirty history and incoming
  links before moving it. Keep indexes to links/descriptions and logs to what
  changed, why and a link.

**Declare provenance.** Every concept requires `type`, `title`, `description`,
`tags`, `status`, `generated`; first tag `kaggriculture-v3`. Reserved index/log
files are exempt; a register declaring fields is not. A Decision quotes its
decider and identifies the source. Never make an interpretation into owner
adoption. Verification names an actual check of this version; status alone proves
nothing. Repository sources are literal single-line
`repository:<repo-root-relative-path>` values with forward slashes and no `.`
or `..` segments. Keep external sources in their own namespace. Shape/source
lint cannot establish truth, completeness or applicability.

**Promote supported knowledge.** Name an independent source/check, future
consequence and existing-concept search before promotion. Unsupported findings
stay in working evidence. A Lesson needs at least two supporting episodes.
Review after each episode or monthly, whichever is rarer; repair retrieval and
contradicted claims in place. Add a typed folder only with its first real note.

**One current board, when warranted.** Once there are two results to rank, keep
one living Decision `cookbook/decisions/the-kaggriculture-v3-board.md`: dated
position, ranked options with evidence and loss conditions, scoped tombstones.
Read the full board before opening an episode and say which option is pulled and
why. Changes to its facts require four edits together: note, index, log, board.
Keep the whole board within 12,000 JavaScript UTF-16 code units, replacing current
state and linking history. Preserve byte-exact dirty history and checksums before
removal. No empty day-one board or imported v2 ranking.

**Keep the view derived.** Root `kaggriculture-v3.base` sits beside `cookbook/`;
the vault symlinks them as `cookbook/kaggriculture-v3.base` and
`cookbook/kaggriculture-v3`. Scope by folder and exclude index/log basenames.

Claude/Codex hooks restore context, gate the first governed edit, lint writes and
check corrections; Git pre-commit checks staged note sources plus staged log.
Direct script checks are distinct from harness discovery and trust. See
[setup receipt](ops/cookbook-setup-checks.md) for exact verification limits.
