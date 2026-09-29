Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: BC best to PPO warm start (kg/rebuild-bc-handoff), round 1

Date: 2026-09-30. CPU only, on the Mac. No training, no GPU.

## Scope

- Branch `kg/rebuild-bc-handoff` at its committed tip `a814544`. Diff reviewed: `git diff 218a05b...a814544` (commits `5349e96`, `ca51222`, `a814544`; 21 files, +849/-20).
- Target: `--load-model-weights-mode model_fresh_critic_head` (`scripts/run_ppo.py`), `PPOTrainer.load_model_weights(..., fresh_state_keys=...)` and `reject_unknown_checkpoint_keys` (`python/owl/train/ppo.py`), the tests in `tests/kaggriculture/test_bc.py`, and the critic decision with its docs and cookbook claims.
- Governing sources read: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md` (the spec), Codex r1 `ops/rebuild-2026-09-29/codex/verify-bc-handoff.md` (REQUEST CHANGES, fixed in `ca51222`), the recipe-choices-align-to-Isaiah Decision, the preserve-the-PPO-recipe-across-BC-bootstrap Decision, the stateless-policy rules in `CLAUDE.md`, and `docs/write-up.md` (Isaiah quotes).
- Out of scope: the reviewed worktree `/Users/poonszesen/kg-v3-bchandoff` also has **uncommitted** edits to 10 tracked files (warm-start `warm_start.json` recording, new `test_run_ppo.py` wiring tests, sbatch/containerization mode support, note edits) plus an untracked `py-prepare-r1fix.log`. They are not at the tip, so this round does not review them. Several findings below look like what those edits address. They need their own review once committed.

## Checks (scratch worktree `/tmp/cv-bchandoff-r1` at `a814544`, owl.rs built there by `uv sync`)

| Check | Result |
|---|---|
| `pytest tests/kaggriculture/test_bc.py tests/scripts/test_run_ppo.py tests/kaggriculture/test_teacher.py` | 193 passed, 6 skipped (the 6 are the opt-in real-checkpoint test and 5 existing Task 1.4/3.1 gaps) |
| Same handoff tests with `KG_V3_BC_BEST`/`KG_V3_BC_SHARDS` set (`-k "bc_best or ranked or prohibited or fresh_critic"`) | 13 passed, 31 deselected, 0 skipped. The real-checkpoint test ran |
| `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare` | exit 0: ruff (isort, format, lint) clean; mypy no issues in 70 files; 2,193 passed, 12 skipped; `check_doc_freshness.py` passed. This matches the note's 2,193/12 |
| Custody hashes in `/tmp/kg-v3-bc-best` | checkpoint `fd854587…6f51`, manifest `ba5fe1c4…3036`, shard `ab4e6261…17a3`, and the 4 sidecar files all match `SHA256SUMS`, the checkpoint record and the manifest |
| Manifest recount (independent script) | 523 episodes: 523 imitated-seat wins, 0 losses, 0 draws. The policy seat is seat 0 in 281 episodes and seat 1 in 242; 507 train, 16 validation |
| `critic_probe.py` rerun | Reproduced exactly: BC head saturation fraction 0.9722, mean abs value 0.9936, grad norm 0.118016; fresh head 0.0, 0.4682, grad norm 12.364379; actor log-probs identical `True` |
| Cookbook lint (`.claude/hooks/cookbook-lint.mjs`) on the 3 touched concepts | 3 passed. Every `repository:` source in the new Reference exists at the tip |
| Isaiah quotes | `docs/write-up.md:12` (no imitation-learning initialization) and `:77` (warm restarts) are verbatim |
| Source review of `run_ppo.main` ordering | Order: `reset_parameters`, then compile, then DDP adapter construction (initial parameter sync, `broadcast_buffers=False`), then `load_model_weights`, which captures the post-sync fresh head and restores it. Restoration uses in-place `load_state_dict`, so the optimizer's parameter references stay valid. A last-best teacher is copied from the loaded model. LoRA cannot combine with this mode: `lora_config_for_model` returns `None` for the Kaggriculture config, and the mode raises for any non-`KaggricultureTransformer` |

Resource note: the `critic_probe.py` rerun peaked at 8.8 GB RSS (`/usr/bin/time -l`), well over the 1 GB tiny-check budget. It finished normally; I record it here rather than omit it. Every other check stayed inside the budget.

## Mutations (each applied in the scratch copy, the 3 test files run, then restored byte-for-byte)

| # | Mutation | Result |
|---|---|---|
| M1 | Drop the fresh-state restore in `PPOTrainer.load_model_weights` | CAUGHT: 3 failed (`model_fresh_critic_head` × 3 configs) |
| M2 | Capture fresh state without `.clone()` (aliasing) | CAUGHT: 3 failed |
| M3 | Drop `reject_unknown_checkpoint_keys` from `ppo._checkpoint_metadata` | CAUGHT: `test_ppo_load_rejects_prohibited_checkpoint_state` |
| M4 | Drop `reject_unknown_checkpoint_keys` from `run_ppo._load_model_weights` (the teacher_init loader) | CAUGHT: same test |
| M5 | `_fresh_state_keys_for_mode` always returns an empty set | CAUGHT: 4 failed |
| M6 | Disable the unknown-`fresh_state_keys` guard | CAUGHT: `test_fresh_critic_head_mode_is_explicit` |
| M7 | Disable the KaggricultureTransformer-only guard | CAUGHT: same test |
| M8 | `run_ppo.main` passes `"model_only"` instead of the launch mode to `_fresh_state_keys_for_mode` | **SURVIVED**: 194 passed |
| M9 | `run_ppo.main` loads optimizer state for any mode other than `model_only` (BC optimizer moments leak into the fresh-critic mode) | **SURVIVED**: 194 passed |
| M10 | Add `"opponent_id"` to `_CHECKPOINT_KEYS` | CAUGHT: prohibited-state test |
| M11 | `write_checkpoint` emits an extra `"hidden_state"` key (writer and allow-list drift) | CAUGHT: 6 existing `test_run_ppo.py` round-trip tests |

9 of 11 were caught. Both survivors are in the one place that turns the CLI mode into a load: `scripts/run_ppo.py:278-287`.

## Findings

### P2-1: the `main` wiring of the new mode is untested (M8 and M9 survive)
- Where: `scripts/run_ppo.py:278-287`. The fake trainers in `tests/scripts/test_run_ppo.py:1001-1008` and `tests/kaggriculture/test_teacher.py:1575-1583` only assert `fresh_state_keys == frozenset()` for `model_only`.
- Issue: every new test calls `PPOTrainer.load_model_weights` or `_fresh_state_keys_for_mode` directly. Nothing checks that a `--load-model-weights-mode model_fresh_critic_head` launch reaches the trainer with the critic keys and `load_optimizer=False`. M8 (the critic is silently loaded from BC, which is the exact outcome the decision rejects) and M9 (the BC optimizer state is loaded, against the BC-bootstrap Decision's "reset optimizer state") both pass the whole suite. The note says "The tests call the function its fresh-launch branch calls", but no test checks that the branch passes the right arguments.
- Fix: add a `run_ppo.main` (or `_parse_args`+`_resolve_launch`+launch-branch) test with a fake trainer that asserts `fresh_state_keys == {critic_head.*}` and `load_optimizer is False` for `model_fresh_critic_head`, and `load_optimizer is True` only for `model_and_optimizer`. Monkeypatch `_fresh_state_keys_for_mode` or the model so that a Linear stand-in works. The uncommitted `test_run_ppo.py` edits look aimed at this. Re-run M8/M9 against them once committed.

### P2-2: the tip does not record which checkpoint and mode started the run
- Where: `scripts/run_ppo.py:278-300`. `config.yaml` is written from `cfg` and does not include the launch arguments. No W&B or run-directory record of `load_model_weights_path`, its SHA-256 or the mode exists at the tip.
- Issue: the BC-bootstrap Decision requires a weights-only PPO initialization to "explicitly name the new experiment" and keep artifacts hash-bound. `CLAUDE.md` requires checkpoint metadata with local source-bound evidence and W&B telemetry. A Phase 6.2 run launched at this tip would leave no in-run record that it started from `fd854587…` with a fresh critic head. The gap is pre-existing for `model_only`, but this branch exists to support that handoff.
- Fix: on the main rank, write the resolved path, SHA-256 and mode to the run directory and the metric-run summary before training. Test it. The uncommitted working tree appears to add `warm_start.json` and `warm_start/*` summary keys. That code needs its own review and mutation test once committed.

### P3-1: the SLURM env-var path rejects the new mode
- Where: `scripts/slurm/launch-train.sbatch:210-215`.
- Issue: `ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE=model_fresh_critic_head` exits with "must be model_only or model_and_optimizer", while the README presents the mode as available. The Phase 6.2 launch uses `torchrun` directly, so this path is secondary.
- Fix: add the mode to the allow-list and update `docs/containerization.md`. The uncommitted tree does this.

### P3-2: loader-coverage claims are broader than the code
- Where: the Reference description (line 4, "Every PPO checkpoint loader now also rejects unknown top-level checkpoint keys"), `cookbook/references/index.md:5` ("All PPO checkpoint loaders"), and the Reference body line 19 ("Isaiah's minimal model-weights checkpoints … still load").
- Issue: `python/owl/agent/agent.py:212`, `scripts/benchmark_checkpoints.py:353` and `scripts/roundtrip_checkpoint_quantization.py:84` load PPO checkpoints without the check. All three are Orbit-Wars-only today. A minimal `{"model": …}` checkpoint, such as `scripts/extract_model_weights.py` output, loads only through the `teacher_init` loader. Through `--load-model-weights` it still fails with a bare `KeyError: 'env_steps'` from `ppo._checkpoint_metadata` (pre-existing, and not an informative error).
- Fix: say "every `run_ppo` checkpoint loader". Scope the minimal-checkpoint sentence to `teacher_init`/the initial last-best loader. Optionally make `ppo._checkpoint_metadata` raise a named `ValueError` for missing keys.

### P3-3: two allow-lists for one schema
- Where: `python/owl/train/ppo.py:2654` (`_CHECKPOINT_KEYS`) and `scripts/run_ppo.py:1295-1306` (`required_keys` | `optional_keys`).
- Issue: the same writer schema is declared twice. M11 shows that writer drift is caught, but drift between the two lists is not.
- Fix: have `run_ppo._checkpoint_metadata` derive its sets from `ppo._CHECKPOINT_KEYS`, or export one constant.

### P3-4: the 4-rank real-checkpoint claim is not in the branch's own evidence
- Where: the Reference description line 4, `cookbook/log.md` (2026-09-29 entry) and `index.md:5` say the real checkpoint loads into "eager, 2-, 4- and 8-rank" models with equal outputs. The Reference body (line 18) and `ops/.../bc-handoff/real_check.txt` cover eager, 2-rank and 8-rank only. The opt-in test uses `kaggriculture_2rank.yaml` only.
- Issue: the 4-rank case follows from the config-equality test, and Codex r1 ran 4 configs × 3 modes on the real checkpoint (its report is in the main worktree). Still, the note's own evidence and its description disagree.
- Fix: cite Codex r1's 12-case run for the 4-rank claim, or phrase it as following from equal model sections.

### No finding (checked)
- Fresh-head semantics, correctness and tests: M1, M2, M5, M6 and M7 are all caught, and the tests are non-vacuous. Source and target seeds differ (3 vs 5), and the tests assert changed values in fresh mode and equal values otherwise.
- Prohibited-state rejection covers the PPOTrainer weights and resume path, the run_ppo resume-metadata loader and the teacher_init loader. Codex r1 P2 is fixed (M3, M4 and M10 caught).
- Stateless policy: the change adds no identity or temporal state to inputs, heads, losses or checkpoints.
- The canonical trainer stays `scripts/run_ppo.py`. No v2 model code enters.
- The critic decision's evidence reproduces. It is recorded as an implementer choice under the recipe Decision, with the Isaiah precedent stated accurately. Its improvement claim is correctly labelled as inferred, not measured.

Two P2s remain at the committed tip: a surviving wiring mutation and missing warm-start custody. As in Codex r1, where one P2 meant REQUEST CHANGES, these block approval. The uncommitted working-tree edits may resolve both, but they need review once committed.

VERDICT: REQUEST CHANGES
