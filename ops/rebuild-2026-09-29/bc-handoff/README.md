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
- `critic_probe.txt` (all 360 rows of the shard): with the BC critic head,
  700 of 720 seat values (97.2 %) have |value| > 1 − 2e-6 (both seats on 345
  of 360 turns), and the values are already [−1, 1] at turn 20 of 720. With
  each model's target set opposite to its own prediction, the MSE value loss's
  gradient norm on `critic_head` is 0.118 with the BC head and 12.4 with a
  fresh head. With a
  fresh head, no row saturates (mean |value| 0.468) and the actor log-probs are
  bit-identical to the BC-head model's.
- Prohibited state: before this change `PPOTrainer.load_model_weights` ignored
  unknown top-level checkpoint keys (such as `opponent_id` or `hidden_state`).
  `ppo.reject_unknown_checkpoint_keys` now rejects them, in
  `ppo._checkpoint_metadata` and in `run_ppo._load_model_weights` (the
  `teacher_init` loader, which also ignored them: Codex r1 P2). Unknown model tensors were
  already rejected by `load_model_state_dict_allowing_lora`.

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

## Checks

- `targeted-tests.log`: 14 passed (the new handoff tests, 3 configs × 3 modes,
  and the opt-in real-checkpoint test with `KG_V3_BC_BEST` and
  `KG_V3_BC_SHARDS` set).
- `py-prepare.log`: `uvx --from rust-just just py-prepare`, 2,193 passed,
  12 skipped, docs fresh. It needed two pre-existing docstring fixes
  (`bc_data.py`, `bc.py`) for ruff D205/D209.

## Limits

- `run_ppo` still stops before the environment for Kaggriculture; the tests call
  the load function the fresh-launch branch calls, not `main`. No PPO update ran
  from the BC checkpoint, so whether the fresh head learns faster than the BC
  head is inferred from the gradient probe, not measured.
- One real validation game; the saturation fraction is not a corpus statistic.
- The integration branch's model now loads native grammar tables; they are
  non-persistent buffers, so the checkpoint keys are unchanged, but the load was
  not rerun on that tree.
