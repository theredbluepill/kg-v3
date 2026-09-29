# Task 1.5 Stage 1 implementation receipt

Base: `e197528820ab7cfb429e21259000957370abf1c6`, branch `kg/rebuild-adapter`.
Date: 2026-09-29. No Git mutations, other-worktree writes, Rust/engine changes,
builds, network, training, GPU workload or throughput experiment.

Target: implement the Python reward/configuration/adapter/factory/codec boundary
without Task 1.4. The discriminating observations are independent numerical
oracles, exact fake signatures, stable caller-owned tensor/NumPy identities,
real header-encoder privacy, and factory seed arithmetic. Completion condition:
all runnable Stage 1 contracts and `just py-prepare` pass, required future tests
exist with explicit skips, and protected paths remain byte-unchanged.

## Changed and added files

- `python/owl/kaggriculture/rewards.py`: sole explicit coefficient definition and independent f64/f32 reward oracles.
- `python/owl/kaggriculture/config.py`: imports reward definition, strict seed/count fields, required native threads and reward shaping, default game envelope.
- `python/owl/kaggriculture/types.py`: game envelope, aliases/admission/canonicalization and named recursive JSON type.
- `python/owl/kaggriculture/env.py`: 29+6 caller buffers, persistent NumPy holder, exact native calls, input checks, reuse fence and diagnostics.
- `python/owl/kaggriculture/codec.py`: four cold native-only wrappers and atomic private batch output.
- `python/owl/game.py`: three transport unions, checked factory dispatch and rank streams.
- `python/owl/rs.pyi`: verbatim reviewed Task 1.4 declarations, deduplicated imports and formatting/line-length preservation directives.
- `tests/kaggriculture/test_rewards.py`: shared admission cases, independent rewards/rounding/bounds, native comparisons and overflow regression.
- `tests/kaggriculture/test_configs.py`: new reward import, explicit coefficient inputs and expected seed/config fields, preserved Task 3.4 assertions.
- `tests/kaggriculture/test_game.py`: game envelope, real encoder canonicalization, unchanged config union, exact Orbit dispatch and rank-stream tests.
- `tests/kaggriculture/fake_env.py`: eight exact native signatures, test-only writer/failure/seed observations.
- `tests/kaggriculture/test_env.py`: 35-buffer contracts, ownership, strict inputs, pin/fence/error checks and real header privacy.
- `tests/kaggriculture/test_env_cuda_fence.py`: deferred real pending-DMA test with fence-removal mutation.
- `tests/kaggriculture/test_codec.py`: native transport/derivation/atomicity checks and deferred hashed reference fixture replay.
- `configs/kaggriculture.yaml`: explicit inactive ineffective cap .1.
- `configs/kaggriculture_2rank.yaml`: explicit inactive ineffective cap .1.
- `configs/kaggriculture_4rank.yaml`: explicit inactive ineffective cap .1.
- `docs/rl-api-specs.md`: explicit Stage 1 pending-native status, config/factory/reward/codec/buffer contracts.
- `ops/rebuild-2026-09-29/briefs/1.5.md`: appended owner-directed Stage 1 reconciliation/deviation history.
- `cookbook/references/native-game-semantics-use-v3-owned-buffers.md`: current Stage 1 scope/inventory/checks/gaps alongside scoped historical evidence.
- `cookbook/references/reward-reuse-preserves-objective-and-critic-semantics.md`: current reward config/oracle and checked numerical correction; old recipes explicitly historical.
- `cookbook/references/index.md`: current rebuild descriptions for both revised References.
- `cookbook/log.md`: one prepended adaptation entry.
- `ops/rebuild-2026-09-29/stage1-adapter/check_contract.py`: read-only stub/signature/protected-path checker.
- `ops/rebuild-2026-09-29/stage1-adapter/verify.py`: bounded offline check driver preserving real return codes.
- `ops/rebuild-2026-09-29/stage1-adapter/results.md`: this receipt.
- `ops/rebuild-2026-09-29/stage1-adapter/commands.json`: full verification command/exit ledger, including intermediate failures.
- `ops/rebuild-2026-09-29/stage1-adapter/source-manifest.json`: SHA-256 custody for the final changed implementation, tests, configs, docs and notes.
- `ops/rebuild-2026-09-29/stage1-adapter/files.txt`: complete changed/added file inventory, including every log and supporting receipt.
- Other files in this receipt directory are command logs, check JSON and the codec subtask receipt; `files.txt` lists each separately.

## Commands and outcomes

Every execution check used `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2
CARGO_NET_OFFLINE=true UV_OFFLINE=true`; uv/uvx commands were offline. Direct
subprocess return codes or shell `$?` were logged; no pipeline masks failures.
Read-only source inspections (`cat`, `sed`, `rg`, `git show/status/diff`) preceded
implementation. No build command was run.

The machine-readable complete command ledger is `commands.json`. Main pytest
sequence (P=passed, F=failed, S=skipped, E=collection error, D=deselected):

| Section / command (all `uv run --offline`) | Red / intermediate | Green / final |
| --- | --- | --- |
| A: `pytest tests/kaggriculture/test_rewards.py -q` | exit 2, 1E: module absent | exit 0, 25P/11S before final overflow regression |
| A migration: `pytest tests/kaggriculture/test_configs.py -q` | exit 1, 25F/14P: YAML omitted now-required coefficient | exercised in combined final below |
| A: `pytest tests/kaggriculture/test_rewards.py tests/kaggriculture/test_configs.py -q` | — | exit 0, 64P/11S twice; final 65P/11S after overflow fix |
| A correction: `pytest tests/kaggriculture/test_rewards.py -k overflow_rescaling -q` | exit 1, 1F/36D: .7 cap became .699999988079071 | exit 0, 1P/36D with float64 tensor branches |
| B/C: `pytest tests/kaggriculture/test_game.py -q` | exit 1, 25F/1P/1S, including corrected helper-argument typo | exit 0, 26P/1S |
| B: `pytest tests/kaggriculture/test_game.py -k 'envelope or hire_limit or config_union' -q` | exit 1, 3F/1P/23D; intermediate exit 2, 1E while rewards module was pending | exit 0, 4P/23D; existing union already passed red phase |
| D: `pytest tests/kaggriculture/test_env.py -q` | exit 2, 1E for test import typo; corrected rerun exit 2, 1E for missing adapter | final included below |
| D: `pytest tests/kaggriculture/test_env.py tests/kaggriculture/test_env_cuda_fence.py -q` | exit 1, 7F/22P/2S: fake step/day inconsistency, corrected without weakening native check | exit 0, 29P/2S |
| D: `pytest tests/kaggriculture/test_env.py -k checks_contract -q` | adapter absent in D import-red | exit 0, 1P/29D (both final rounds) |
| D: `pytest tests/kaggriculture/test_env.py -k fence -q` | adapter absent in D import-red | exit 0, 5P/25D (both final rounds) |
| E: `pytest tests/kaggriculture/test_codec.py -q` | exit 2, 1E module absent; intermediate exit 2, 1E while JsonValue alias was pending | exit 0, 6P/1S |
| All focused: `pytest tests/kaggriculture/test_rewards.py tests/kaggriculture/test_configs.py tests/kaggriculture/test_game.py tests/kaggriculture/test_env.py tests/kaggriculture/test_env_cuda_fence.py tests/kaggriculture/test_codec.py -q` | — | first 125P/15S; final 126P/15S, exit 0 |

Static/full commands:

- `uvx --offline ruff format python/owl/kaggriculture/rewards.py tests/kaggriculture/test_rewards.py tests/kaggriculture/test_configs.py`: exit 0, three files unchanged.
- `uvx --offline ruff check` with those same three paths: exit 1, three lint findings; corrected repeat exit 0.
- `uv run --offline ruff format python/owl/kaggriculture/codec.py tests/kaggriculture/test_codec.py`: exit 0, two files reformatted (tool output).
- `uv run --offline ruff check` with those same two codec paths: exit 1, five style findings; corrected repeat exit 0.
- `uv run --offline mypy python/owl/kaggriculture/codec.py`: exit 0, one source file.
- `uv run --offline mypy python/owl/kaggriculture/types.py python/owl/kaggriculture/config.py python/owl/game.py`: exit 0, three source files.
- `uvx --offline ruff check python/owl/kaggriculture/types.py python/owl/kaggriculture/config.py python/owl/game.py tests/kaggriculture/test_game.py`: exit 0.
- `uvx --offline --from rust-just just py-prepare`: first exit 1 at lint (14 findings); next exit 0, 1,704P/22S; first final exit 0, 1,704P/22S; after independent overflow correction final exit 0, 1,705P/22S. Each successful invocation passed format, syntax/lint, mypy (67 source files), `pytest tests/ -m "not slow"` and docs freshness. Earlier final logs are preserved with `-before-overflow-fix` suffixes.
- `uv run --offline python ops/rebuild-2026-09-29/stage1-adapter/check_contract.py`: exit 0 on all three runs; reviewed declarations byte-identical, all eight fake signatures match, protected paths unchanged.
- `uvx --offline pymarkdownlnt scan docs/rl-api-specs.md`: exit 0 in both final rounds.
- `git diff --stat -- tests/owl tests/scripts tests/tools`: exit 0, empty in both final rounds.
- `git diff --check`: exit 0 in both final rounds; repeated after receipt updates.
- The verification driver `python3 ops/rebuild-2026-09-29/stage1-adapter/verify.py` exited 0 twice. It preserves each nested command's return code in `final-checks*.json`.
- Cookbook direct-lint and final custody/diff commands are recorded in `commands.json`; this is script invocation evidence, not harness discovery/trust evidence.

A tiny NumPy probe (exit 0, tool output) selected the f32 rounding discriminator:
`uv run --offline python -` tested `np.float32(np.float64(np.float32(-x))+.65)`
versus `np.float32(.65-x)` for seven candidate x values, then the first 9,999
positive millionths. Its exact body is recorded in `commands.json`.

## Skipped tests

New Stage 1 tests, 15 cases:

- `test_rewards.py::test_python_and_native_reward_admission_agree` (the ten shared cases): `needs Task 1.4 binding`.
- `test_rewards.py::test_native_fixture_rewards_match_independent_oracle`: `needs Task 1.4 binding`.
- `test_game.py::test_real_binding_seed_streams`: `needs Task 1.4 binding`.
- `test_env.py::test_seat_private_isolation_live_actions`: `needs Task 1.4 binding`.
- `test_codec.py::test_reference_programs_round_trip`: `needs Task 1.4 binding`.
- `test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma`: `CUDA is unavailable` on this Mac; also carries `needs Task 1.4 binding` for Stage 1 on a CUDA host.

Inherited skips, seven cases, unchanged:

- `test_gpu_grammar.py::test_native_tables_match_expected_tables`: `Task 1.2/1.4 native grammar binding not available: native grammar tables arrive with Task 1.2/1.4 (owl.rs.kaggriculture_grammar_tables); use expected_grammar_tables until the binding exists`.
- `test_observe.py::test_native_writer_uses_real_schema_and_keeps_all_pointers[pinned-1]`: `pinned host memory requires CUDA`.
- `test_observe.py::test_native_writer_uses_real_schema_and_keeps_all_pointers[pinned-2]`: `pinned host memory requires CUDA`.
- `tests/owl/model/test_attn.py::test_varlen_attention_flash_backend`: `flash-attn CUDA backend is not available`.
- `tests/owl/model/test_attn.py::test_varlen_attention_matches_torch_sdpa_per_sequence`: `flash-attn CUDA backend is not available`.
- `tests/owl/test_int8_emulation.py::test_int8_emulated_model_matches_x86_dynamic_quantization_except_output`: `x86 torch quantized backend is unavailable: qnnpack`.
- `tests/scripts/test_run_ppo.py::test_kaggriculture_native_evaluations_draw_fresh_reproducible_worlds`: `Needs the native Kaggriculture environment and its EnvConfig seed (rebuild Tasks 1.4/1.5).`

## Reconciliations and verified base facts

1. Task 3.4's callable obs-tag `GameEnvConfig` remains the sole config union; no `game` literals or extra validator, no concrete discriminator defect, and no trainer/config or Isaiah EnvConfig edit.
2. `KaggricultureEnvConfig` stays in `config.py`; `reward_shaping` and its top-level `reward_mode` stay. Adapter gets the required extra `reward_mode` keyword; reward serialization accepts it, preserving exact native keys/ABI.
3. All coefficients and native threads are explicit; shipped ineffective cap .1 only materializes an unchanged resolved value. There is one reward class, imported from rewards.py without an alias/re-export shim.
4. Stage 1 implements native-dependent test bodies but applies only the stipulated binding skips and the pod hardware skip. Real header checks and all fake/CPU checks run now. Native tables/model default remain untouched for Stage 2.
5. `W:` runner line numbers and pre-3.4 FullConfig prose were stale; inspected by content. The run_ppo stop, require_orbit_env and all trainer/distributed files stay unchanged.
6. Native stub additions are verbatim (imports deduplicated); a fmt-off block and E501 exemption preserve the two 90-character reviewed signatures. Runtime names are absent until Task 1.4.
7. JsonValue uses existing Pydantic recursive JSON typing; framework metadata is explicit, serialized only when non-null, and cannot drive native seeds.
8. Pinning checks availability before touching the unsafe Mac allocation path; then checks actual pinning. No fallback. Pod test preallocates device outputs before delaying/copying and includes a fence-removal control.
9. The oracle safely handles huge coefficients; independent review's scalar-branch f32 cap error was reproduced red and corrected to retain f64. Native extremes still need paired Stage 2 execution.
10. The owner limited final verification to Python; no Cargo/maturin build/test or `just prepare` was run.

Verified: clean starting checkout/branch/base; extension `dir(rs)` exposes only
`encode_kaggriculture_headers_into` for this game; grammar encode/decode/tables
are Rust-internal; reward coefficients had defaults and caps were strictly
positive; all three shipped YAMLs omitted the ineffective cap.

One supplied base fact was imprecise: its inert-shaping check was W>0 with raw
starvation=drought=0, irrespective of ineffective weight. It was not the
reference's combined ineffective-weight exception. The specified exact shared
binary64 admission predicate is now implemented instead.

## Open questions and residual scope

No Stage 1 blocker remains. After Task 1.4 merges, remove the binding skips and
confirm runtime/stub identity, full native game/seed/terminal/output rollback,
selected-only truncation and terminal/autoreset timing, real rank consumption,
private action effects, reference codec programs and Python/native admission.
The pending NPZ reward comparison uses the brief's named arrays and two recipes;
Stage 2 must bind it to the actual fixture manifest/hash validator before loading,
confirm exact array keys/custody and compare both routine and extreme rewards.
There is no current fixture to invent a manifest schema for.

Stage 2 must implement/qualify native grammar-table loading and switch the model
default. Task 3.1 must adopt the factory in the existing trainer/runner, transport
contiguous CPU actions, preserve caller-buffer lifetime, clone aliased CPU
transition values before artificial edits, and connect the existing truncation,
raw-bank evaluation and seed-band semantics. The factory enforces i64 bounds;
training/evaluation stream-band policy remains a downstream integration concern.

Only the pod can qualify pinned DMA and the fence-removal control, CUDA/model
execution, canonical functional smoke and complete-work throughput. Fake tests
prove forwarding/ownership and absence of adapter writes, not native rollback.
The actual header encoder proves observation-private isolation, not live action
semantics. No outcome, speed or policy-quality claim follows these unit checks.

VERDICT: STAGE1-COMPLETE
