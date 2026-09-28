# Cookbook setup checks — 2026-09-28

Scope: cookbook, lifecycle and derived view only. Code adaptation has separate evidence. Latest owner instruction overrides further interviewing; the inherited boundary choices are explicitly described in the scope Decision. No v3 empirical result is fabricated.

## Source custody

The selective v2 sources were read from the working tree, not reconstructed from its commit. Their original owner quotations and applicability limits remain in the source notes; the v3 synthesis identifies imported discipline separately from v3 performance evidence. Local external-repository citations require access to the retained v2 checkout. These hashes pin the exact bytes inspected at setup:

| v2 relative source | SHA-256 |
| --- | --- |
| `cookbook/decisions/kaggriculture-v2-starts-with-a-clean-design.md` | `a2e236e7a571a18469d5fa7815b856318dc6b361087adba3b479d065713fd479` |
| `cookbook/decisions/the-architecture-must-express-the-decisions-needed-to-win.md` | `b5934eed388422c29091630096cabc29004cd86dc482120a0d81ae5a409694e1` |
| `cookbook/decisions/failures-must-distinguish-information-decision-execution-and-architecture-errors.md` | `f6c3578e5f9fc530c734c21620ec82d572f86e3ab18d5efc2c043eab7907d80b` |
| `cookbook/decisions/stalled-improvement-requires-attribution-and-explicit-architecture-limits.md` | `a3019b8ff7a8c4617d3a41e200043261253b417da356646795e9bcf9a395ae29` |
| `cookbook/decisions/high-throughput-is-mandatory.md` | `83a82ee3b0fff73f583eb425b55b161dc2bb0dedf5e9ccb1ab97f0a607698e8a` |
| `cookbook/decisions/one-trainer-extend-selfplay-never-fork.md` | `11ccb85dd6718c93fed7b85700b1785576c5085969c477b240a8efb570d69610` |
| `cookbook/decisions/myolie-is-a-stateless-agent.md` | `317e80b6be118ce059dd17e8e412b7060a68e3c97bba7fb78906b944510c8727` |
| `cookbook/references/myolie-opponent-context-needs-explicit-requalification.md` | `c2e90d97b178bb9df5983ade202d5e701dfb08ec050d590d4dc163685fe0b8da` |
| `cookbook/decisions/candidates-are-measured-by-win-rate-and-margin.md` | `368218096a0082b92d3babc1b0c813095e3d55c2d4adecfe1acc646285775c58` |
| `cookbook/decisions/no-candidate-agent-overfits-to-a-single-opponent.md` | `357b5bc685986300d85f2c9d1e652ddc3726a1b8579834fc68209fc8365d177e` |
| `cookbook/decisions/repository-tracking-preserves-evidence-and-excludes-local-state.md` | `4d3b03dfe3019b613ff86bc39766f3374ac601bf102c754c4e352b832e55ef69` |
| `cookbook/decisions/myolie-uses-wandb-for-telemetry-and-monitoring.md` | `7122c6704e51ce507abc81ebd35b1de7256ef685409fe8eb917d41ff6400d98f` |
| `cookbook/references/myolie-native-sampling-grammar.md` | `55f5a912b38102ec594d60dd79d420f77fc6d75c84cfd9d1f17c52ff83fb70ad` |
| `cookbook/references/rust-engine.md` | `a22b7e6dd6dd90c2e0228d8e14fa92e17cde3177ce871be04256912d228fa7ac` |

## Verification

- Current skill asset suites: `node --test .../tests/lifecycle.test.mjs .../tests/source-resources.test.mjs`: **27 passed, 0 failed**.
- Repeated both suites against each installed surface via `COOKBOOK_TEST_PREFLIGHT`, `COOKBOOK_TEST_LINT`, `COOKBOOK_TEST_CORRECTION`, `COOKBOOK_TEST_PRECOMMIT`: **27 passed for Claude and 27 for Codex**, including staged-byte source/log/deletion tests in isolated temporary repositories.
- Direct JSON fixtures on both installed copies: compact SessionStart, UserPromptSubmit, SubagentStart restore real navigation/log; governed `scripts/run_ppo.py` patch denies once then permits retry; six real concept notes lint clean; numeric correction blocks once then permits retry. A first English wording fixture was not recognized; the suite's explicit Chinese correction fixture passed. This is heuristic wording detection, not semantic completeness.
- Both JSON configs parse and required events are present. Installed scripts match current asset bytes. No prior project hooks/config existed; `.git/hooks` contained sample hooks only. `core.hooksPath=ops`; pre-commit installed executable. Real repository index was not staged or changed for verification.
- Internal cookbook wikilinks resolve. Concept frontmatter/source declarations pass installed lint; lint does not prove truth or check remote sources. Source claims were compared to the retained v2 records during synthesis.
- Vault bundle and adjacent Base are symlinks to this repo. First open/query preceded vault indexing and returned file-not-found; `obsidian reload` resolved it. Base opened in Obsidian, `All knowledge` returned **6 concepts**, no reserved index/log or other-subject rows; active DOM displayed the six-row table. `Needs review` returned no rows. After the GPU direction and reward clarification, the final query returns **12 concepts**, with **4 draft port contracts** in Needs review; all scope/exclusion checks pass. The final direct installed fixtures lint/link-check all twelve concepts. Root license bytes match the imported HEAD.
- Runtime discovery/trust for Claude and Codex remains **unverified**: direct execution and parsed config do not establish a live harness event or approval of new command hashes. No reload/trust action was performed on these coding sessions.
- `nsys` is absent on this macOS setup host. No NVIDIA capture, CUDA speedup, training convergence, W&B synchronization or competition result was checked by cookbook setup.

The reusable retrieval path is `cookbook/index.md` → the scoped discipline Decisions and profiling Workflow. No empirical Lesson qualifies and no standing board exists. Writing a durable note remains note + folder index + log together; a later result changing a standing board makes it four.


## Read-only GPU capacity audit — 2026-09-28

Loaded `runpod://skills/runpod` and `runpod://skills/discovery`. No RunPod mutation was called. Existing pod IDs/status supplied by the parent inventory were not re-read here and were left untouched. No pod env payload or secret was printed.

Calls: `get_capacity(gpuCount=4,gpuTypeIds=["5090","PRO 6000"],cudaVersions=["12.8","13.0"],limit=20)`; `list_gpu_types(include=["AVAILABILITY"],product=["POD"],count=4)` (Secure default); `get_gpu_type(id=...,count=4,include=["AVAILABILITY"],product=["POD"],cloud="COMMUNITY")` separately for the three full-GPU types below; then `get_capacity` without a CUDA filter to cover every returned host version.

| GPU identifier | VRAM | Secure/Community four-GPU availability | Catalog USD/GPU-hour Community / Secure |
| --- | --- | --- | --- |
| NVIDIA GeForce RTX 5090 | 32 GB | NONE / NONE | 0.69 / 0.99 |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | NONE / NONE | 1.69 / 2.09 |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 96 GB | NONE / NONE | 1.69 / 2.19 |

All-version capacity reports `UNAVAILABLE` and `pricePerHr:null`: 5090 versions12.8/12.9/13.0/13.1/13.2/13.3; PRO6000 Server13.0/13.2/13.3/13.4; Workstation12.8/13.0/13.2/13.3. No data-center placement was returned and no available candidate or four-GPU booking price is established. Other matching MaxQ/MIG rows were also unavailable and were not silently substituted for the requested full GPUs. Base catalog prices above are not allocation quotes.

The observed create schema (not invoked) is `body: {name,image or templateId,cloud,gpu:{id,count,allowedCudaVersions or minCudaVersion,minRamPerGpu,minVcpuCountPerGpu},disk,mounts,ports,startSsh,env}`. A later deployment must refresh actual stock/price and report billable rate first. Tools handle infrastructure; SSH/file transfer needs its real local execution path. Local `~/.netrc` exists with mode0600; no credential contents were read by this audit.


Reward source audit: `/Users/poonszesen/kaggriculture-v2/ops/myolie-dagger-2026-09-22/selfplay.py` SHA-256 `60bd910a767a6bd2b11365e94b6807fb8694182e1c0185dc9387a6b2fec9e531`. Inspected formula and CLI-default blocks directly, not historical recipe prose.


## Native test evidence received during cookbook close-out

The native implementer ran `cargo +stable test --manifest-path engine_rs/Cargo.toml --lib --locked`. This cookbook agent independently read the final log; it reports 120 passed, 0 failed. The first missing external Rust test module, second missing JSON fixture, and third 118-pass/2-missing-file attempt motivated making four imported test fixtures self-contained. Final build/test scope remains local CPU native behavior, not Python-wheel, CUDA or full-game official parity.

`/tmp/kg-v3-engine-tests.log` SHA-256 `3c833b3c83ba37538b83c9e5d2d995d385ad9adfc2f31a278e1c8eb93c347c6f` (ephemeral source log).

```text
test native_agents::v39::late::independent_oracle_tests::frozen_python_late_helper_oracle ... ok
test ffi::native_parallel_tests::native_parallel_both_schemas_match_serial_through_complete_games ... ok

test result: ok. 120 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.53s

```

`/tmp/kg-v3-engine-tests-attempt002.log` SHA-256 `23d2395b2c1b7d9b62116b605e4814028438108e0f4321007f60d0915b42b377` (ephemeral source log).

`/tmp/kg-v3-engine-tests-attempt003.log` SHA-256 `3314a4042ec2363db2a2b8c26848b66ad20228128c2d86da023d973681b81650` (ephemeral source log).


## Later scoped evidence custody

The later owner preference is two RTX 5090 or two RTX PRO 6000 GPUs first for SPS qualification, superseding the earlier four-rank-first preference. The parent reports two failed allocations (US-NE1/EUR-IS1 despite LOW stock); no pod was created and the existing two-PRO-6000 pod remains EXITED. This cookbook agent made no infrastructure mutation.

Native logs, fixture-regeneration failures/success, model pre-optimization source and CPU benchmark scripts/JSON supplied by the implementer are copied byte-for-byte to `ops/port-evidence/`; `SHA256SUMS` pins each retained artifact. The paired CPU measurement excludes policy inference, PPO and GPU work. This custody supplements the final parent `ops/v3-port-checks.md` receipt and does not independently certify every reported command.

The cookbook agent independently compared AST function bodies with imported starter HEAD `32b3ec9`: `_run_training_loop`, `_resolve_resume_launch`, `_next_periodic_checkpoint_step`, `_validate_last_best_run_id` are identical. New `tests/kaggriculture/test_last_best.py` runs genuine native two-seat evaluation and PPO with a tiny model, covering initial incumbent, .699/.7 promotion boundaries and current/incumbent resume pairing/cadence. Final scoped run: **3 passed in 1.63s**; Ruff check/format pass. The test starts a logical counter at996 to cross a valid1000 checkpoint boundary with four actual transitions, not1000 executed transitions. Selection scalar is controlled only after real evaluation to make branches deterministic.

Canonical command: `OWL_NATIVE_MODULE_DIR=/tmp/kg-v3-native-only PYTHONPATH=/Users/poonszesen/.cache/uv/archive-v0/AHnJO_RPkjOJuSUHa3sP7:python:. /tmp/kg-v3-check-venv/bin/python -m pytest -q tests/kaggriculture/test_last_best.py`.

After these note updates, direct fixtures on both installed hook surfaces, all12concept shapes/internal links, config parsing and asset identity pass again. Runtime harness trust remains unverified.


## Final scoped reconciliation

`ops/v3-port-checks.md` is now the executed code-check receipt. All four provisional implementation References are reconciled to `verified-scoped`, preserving explicit GPU/container/submission/learning gaps and live-hook trust limits. The receipt reports772passing Python cases/3hardware-backend skips,155starter Rust passes/2existing ignored,120vendored Rust passes and actual CPU CLI/DDP updates. Validation uses the documented separately prepared Python environment; normal default dependency sync was still completing downloads at close-out. No default-install success is inferred.

Final inventory audit: every currently modified tracked file appears as an explicit `repository:` source in the cookbook; every such source path exists. New native/model/training/test/config files are grouped under the five adaptation References; lifecycle artifacts are under the starter/history Reference. This includes both Dockerfiles/container docs, `justfile`, `scripts/regenerate_test_fixtures.sh`, optional reference dependencies, two/four-rank configs and CUDA measurement. The model profiling JSON specifies means; final prose was corrected from medians. `ops/port-evidence/SHA256SUMS` now verifies25retained artifacts, including the final Python preparation, CLI, CUDA-guard and lock logs.

Final live Obsidian queries: `base:query path="cookbook/kaggriculture-v3.base" view="All knowledge" format=json` returns12subject concepts, with five verified-scoped References and seven stable Decisions/Workflow, no index/log or foreign rows. `Needs review` returns[] after scoped reconciliation. This verifies current vault retrieval, not runtime hook trust or semantic completeness. Installed fixtures, all12note shapes/internal links, config parsing and asset identity pass again.


## Locked project environment follow-up

The default-install gap above is now closed: normal `uv sync --locked` completed its native release build and package installation. This reviewer independently read the retained project-env test log (232passed in5.40s) and mypy log (all58source files clean). The parent also retains the normal-env actual native PPO CLI/checkpoint log. The final full preparation log still reports772passed/3hardware-backend skips. Source notes name the pytest root import configuration, grammar scalar rename and finite-metric/effective-config measurement hardening. These do not extend qualification to NVIDIA execution, container builds, learning strength or live hook trust.

`SHA256SUMS` refreshed over all29retained evidence artifacts, including four new locked-environment logs and the refreshed preparation result. Source inventory/existence, internal links, note shapes, installed hook fixtures, configs and asset identity were rechecked. Earlier receipt sections preserve the state at the time; this follow-up supersedes only the incomplete-default-sync gap.


## Original pinned Rust toolchain restored

The reviewer independently read `kg-v3-nightly-repair.log`: Rustup recovered a partially installed toolchain and installed `1.97.0-nightly (e9e32aca5 2026-04-17)`. The retained `kg-v3-rs-prepare-pinned.log` finishes155passed/0failed/2existing ignored and documentation freshness; the parent ran `CARGO_NET_OFFLINE=true UV_NO_SYNC=1 uvx --from rust-just just rs-prepare`, including formatting and all-target Clippy. Stable verification remains separately retained. The earlier missing-driver failure is preserved as failure provenance, not a current blocker.

The final parent pod inventory confirms only the two preexisting EXITED pods (`0yihpugnavlg7e`, `p0wdnd40cbmrej`), with no new GPU resource or execution. Current evidence custody now contains31artifacts; SHA256SUMS, source inventory/existence, internal links, note shapes, installed direct hook fixtures, config/asset identity and diff checks pass. Linux/CUDA/container execution, competitive strength and live hook trust remain outside the checks.
