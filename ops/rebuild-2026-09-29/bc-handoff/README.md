# BC best to PPO handoff (Phase 6.2 prerequisite), 2026-09-29

Branch `kg/rebuild-bc-handoff` from `kg/rebuild-bc-now` `218a05b`. CPU only, on
the Mac; no training.

## Questions

1. Does `checkpoint_bc_best.pt` load through `run_ppo`'s `--load-model-weights`
   path into the model built from the ranked PPO configs, with identical actor
   and critic outputs, and is prohibited state rejected?
2. What should PPO do with the BC critic?

Expected discriminating observations: exact tensor equality of `evaluate_actions`
outputs after the load; a raised error for identity-bearing checkpoint state;
the BC critic's value distribution on a real held-out game.

## Inputs and custody

- Checkpoint copied read-only from the pod
  (`/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/`) to
  `/tmp/kg-v3-bc-best/`. `SHA256SUMS` lists the copied files; the checkpoint's
  SHA-256 `fd854587…6f51` matches the pod's `sha256sum` and
  `checkpoint_bc_best.json`.
- One validation shard `validation/2026-09-22-112208626.npz` (SHA-256
  `ab4e6261…17a3`, as in the manifest) and `manifest.json` (SHA-256
  `ba5fe1c4…3036`, the checkpoint record's `dataset_manifest_sha256`) in
  `/tmp/kg-v3-bc-best/shards-top1/`. Neither the checkpoint nor the shard is
  committed.
- Manifest count: all 523 episodes have one policy seat and that seat's final
  bank is larger (523 won, 0 lost, 0 drawn).

## Results

- Model sections: `configs/kaggriculture_1gpu_eager.yaml` (the BC run's PPO
  config), `kaggriculture_2rank.yaml`, `kaggriculture_4rank.yaml`, and the
  integration branch's `kaggriculture_8rank.yaml` (`kg/isaiah-gap-closure`
  `faed717` and `5b43062`) have equal `model`, `obs_spec` and `action_spec`.
  `force_flash_attn` must be false on CPU; it changes no parameter.
- `real_check.txt`: the real checkpoint loads through
  `PPOTrainer.load_model_weights` into eager, 2-rank and 8-rank models; values,
  winner probabilities and log-probs on 8 real rows are equal across the three
  and finite; policy-seat turn NLL 0.419 on those rows.
- `critic_probe.txt` (saturation over all 360 rows of the shard): with the BC
  critic head, 700 of 720 seat values (97.2 %) have |value| > 1 − 2e-6 (both
  seats on 345 of 360 turns), and the values are already [−1, 1] at turn 20 of
  720 (row 10, the first printed row after turn 0; turns 1–19 were not
  printed). On a 30-row sample (every 12th row), with each model's
  target set opposite to its own prediction, the MSE value loss's gradient norm
  on `critic_head` is 0.118 with the BC head and 12.4 with a fresh head. With a
  fresh head, no row saturates (mean |value| 0.468) and the actor log-probs are
  bit-identical to the BC-head model's.
- Prohibited state: before this change `PPOTrainer.load_model_weights` ignored
  unknown top-level checkpoint keys (such as `opponent_id` or `hidden_state`).
  `ppo.reject_unknown_checkpoint_keys` now rejects them, in
  `ppo._checkpoint_metadata` and in `run_ppo._load_model_weights` (the
  `teacher_init` loader, which also ignored them: Codex r1 P2). Unknown model tensors were
  already rejected by `load_model_state_dict_allowing_lora`. A minimal
  `{"model": ...}` checkpoint loads only through the `teacher_init` loader;
  through `--load-model-weights` it now fails with a named
  `checkpoint is missing keys [...]` error instead of a bare `KeyError`.

## Decision and launch

`--load-model-weights-mode model_fresh_critic_head` (new): every model tensor
from the BC best except `critic_head.*`, which keeps the fresh launch's
`reset_parameters` values (identical on every rank, since DDP broadcasts rank
0's parameters before the load); fresh optimizer and scheduler as `model_only`.
Reason and Isaiah evidence: see the cookbook Reference
`cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md`.

Phase 6.2 launch (two ranks), once `run_ppo` runs Kaggriculture (Task 3.1):

```sh
torchrun --nproc-per-node 2 scripts/run_ppo.py configs/kaggriculture_2rank.yaml runs \
  --load-model-weights <run>/checkpoint_bc_best.pt \
  --load-model-weights-mode model_fresh_critic_head
```

The run directory's `warm_start.json` (and the `warm_start/*` metric-run summary
keys) must then show SHA-256 `fd854587…6f51` and mode `model_fresh_critic_head`.

## Checks

- `targeted-tests.log`: "14 passed, 30 deselected" (the new handoff tests,
  3 configs × 3 modes, and the opt-in real-checkpoint test with
  `KG_V3_BC_BEST` and `KG_V3_BC_SHARDS` set). The log records no command. The
  stated selection, `uv run pytest -q tests/kaggriculture/test_bc.py -k "bc_best
  or ranked or prohibited or fresh_critic"`, collects 13 of 44 tests at
  `5349e96`, `ca51222` and later (1 + 9 + 1 + 1 + 1), and no test was added to
  or removed from `test_bc.py` between them that matches it. The 14th test is
  unidentified; the reproducible figure is 13, with its exact command in
  `targeted-tests-r1fix.log`.
- `py-prepare.log`: `uvx --from rust-just just py-prepare`, 2,193 passed,
  12 skipped, docs fresh. It needed two pre-existing docstring fixes
  (`bc_data.py`, `bc.py`) for ruff D205/D209.

## Claude verification r1 fixes

Report: `ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r1.md` (main
checkout; REQUEST CHANGES, two P2s, four P3s).

- P2-1 (mode wiring untested; M8/M9 survived):
  `test_fresh_launch_from_checkpoint_uses_starting_checkpoint_as_teacher` now
  runs `main` once per mode through the fake trainer and checks the mode
  passed to `_fresh_state_keys_for_mode`, the `fresh_state_keys` and the
  `load_optimizer` flag.
- P2-2 (no warm-start custody): on a fresh launch with `--load-model-weights`
  the main rank writes `warm_start.json` (resolved path, SHA-256, mode) and
  sets `warm_start/*` metric-run summary keys; tested in the same `main` test
  and in `test_run_training_session_sets_launch_summaries`.
- `mutations-r1fix.txt`: M8 2 failed, M9 1 failed; dropping the
  `warm_start.json` write 3 failed, the summaries 1 failed, a constant digest
  3 failed. The test checkpoint was then empty, so a digest that read no
  bytes or only the first chunk still passed (Claude r2, fixed below).
- P3-1: `launch-train.sbatch` accepts `model_fresh_critic_head`;
  `docs/containerization.md` documents it.
- P3-2: loader-scope wording narrowed to `run_ppo`/`PPOTrainer`, the Orbit
  inference/tooling loaders named as unchecked, and the minimal-checkpoint
  sentence scoped to `teacher_init`; `ppo._checkpoint_metadata` now raises a
  named `ValueError` for missing metadata keys (test in `test_bc.py`).
- P3-3: one exported `ppo.CHECKPOINT_KEYS`/`OPTIONAL_CHECKPOINT_KEYS` pair
  feeds `reject_unknown_checkpoint_keys` and `run_ppo._checkpoint_metadata`,
  with tests that `write_checkpoint` saves exactly that set and that
  `run_ppo._checkpoint_metadata` follows it.
- P3-4: the 4-rank claim is now stated as following from equal model
  sections (`real_check.txt` covers eager, 2-rank and 8-rank). The critic
  gradient figure is labelled as a 30-row sample (every 12th row).
- `py-prepare-r1fix.log`: `just py-prepare` after these fixes (2,196 passed, 12 skipped,
  docs fresh).
- `targeted-tests-r1fix.log`: the handoff tests with the real checkpoint and
  shard after these fixes, 13 passed, 0 skipped (peak RSS 1.54 GB, over the
  1 GB tiny-check budget).

## Claude verification r2 fixes

Report: `ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r2.md` (main
checkout; no P1 or P2; both r1 P2s resolved; four P3s).

- P3-1 (vacuous digest oracle) and P3-2 (path resolution untested): the `main`
  test writes (1 << 20) + 17 random bytes into the checkpoint, passes it as
  the relative path `checkpoint.pt` after `monkeypatch.chdir(tmp_path)`, and
  asserts the resolved absolute path and the SHA-256 of those bytes.
  `mutations-r2fix.txt`: R3 (digest reads no bytes), R4 (unresolved path) and
  R14 (first 1 MiB only), which survived r2, each give 3 failures.
- P3-3: the top-1 team BC note now says eager, 2- and 8-rank, with 4-rank by
  equal model sections.
- P3-4: the "14 targeted tests" figure is restated above; the 14th test is
  unidentified.
- Checks: the three-file shard (`test_run_ppo.py`, `test_bc.py`,
  `test_teacher.py`) gives 196 passed, 6 skipped; `py-prepare-r2fix.log`
  records `just py-prepare`. `test_bc.py` did not change, so the
  real-checkpoint selection was not rerun (its peak RSS, 1.54 GB, is over the
  1 GB tiny-check budget).

## Limits

- `run_ppo` still stops before the environment for Kaggriculture; `main`'s
  fresh-launch branch runs only through a fake trainer, and the handoff tests
  call the load function it calls on Kaggriculture models. No PPO update ran
  from the BC checkpoint, so whether the fresh head learns faster than the BC
  head is inferred from the gradient probe, not measured.
- One real validation game; the saturation fraction is not a corpus statistic.
- The integration branch's model now loads native grammar tables; they are
  non-persistent buffers, so the checkpoint keys are unchanged, but the load was
  not rerun on that tree.

## Verification

- Codex r1 (`ops/rebuild-2026-09-29/codex/verify-bc-handoff.md` in the main
  worktree): REQUEST CHANGES, one P2 (the `teacher_init` loader accepted unknown
  top-level keys) and one P3 (evidence wording). Both fixed in `ca51222`. A
  local mutation that removes the new call makes the targeted test fail.
- Codex r2 did not run: the Codex CLI hit its usage limit after 39k tokens
  (`verify-bc-handoff-r2-transcript.log`, no report, no VERDICT). The r1 fixes
  are therefore not independently re-verified.
- Claude r1 and r2 (independent Claude subagents standing in for Codex during
  its usage limit; not Codex verdicts): r1 REQUEST CHANGES (two P2s, four
  P3s); r2 found both P2s resolved and raised four P3s, fixed above. The r2
  fixes are not re-verified by a separate reviewer.
