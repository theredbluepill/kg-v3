"""No-update probe of extra off-cadence evaluations across a BC budget extension.

Uses the real train_bc control flow, checkpoint/state writes, compatibility check,
and CPU synthetic fixtures. _train_step is replaced by a no-op, and validation
NLL is prescribed by step: 3 at initialization and 4 at every later step.
No optimizer step, real training, GPU, dataset corpus or W&B call is executed.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import torch
from owl.train import bc
from owl.kaggriculture.bc_data import load_bc_dataset
from tests.kaggriculture import test_bc as fixture

torch.set_num_threads(1)

def history(path):
    return [json.loads(line)["bc/step"] for line in (path / bc.BC_HISTORY).read_text().splitlines()]

def run(path, data, config, *, resume=False):
    position = {"step": bc.load_bc_state(path / bc.BC_STATE).step if resume else 0}
    def no_update(**kwargs):
        position["step"] = kwargs["step"] + 1
        kwargs["losses"][3] += 4
    def prescribed_evaluation(*args, **kwargs):
        return bc.RowMetrics(
            turn_nll=3.0 if position["step"] == 0 else 4.0,
            frame_nll=3.0 if position["step"] == 0 else 4.0,
            value_ce=1.0,
            seat_rows=6,
        )
    with patch.object(bc, "_train_step", no_update), patch.object(bc, "evaluate_rows", prescribed_evaluation):
        result, _ = fixture._run(path, data, config, resume_from=(path / bc.BC_STATE if resume else None))
    return asdict(result)

with TemporaryDirectory(prefix="bc-restart-cadence-") as tmp:
    root = Path(tmp)
    data = fixture._dataset_root(root)
    config = fixture._bc_config(max_steps=4, eval_interval_steps=2, patience_evals=2, preflight_train_replay=False)
    straight = run(root / "straight", data, config)
    first = run(root / "resumed", data, config.model_copy(update={"max_steps": 1}))
    state = bc.load_bc_state(root / "resumed" / bc.BC_STATE)
    bc.check_resume_compatible(state, dataset=load_bc_dataset(data), world_size=1, config=config, ppo_config=fixture._ppo_config())
    resumed = run(root / "resumed", data, config, resume=True)
    output = {
        "scope": "No-op updates; scripted validation; real BC selection/checkpoint/restart flow",
        "validation_nll_by_step": {"0": 3.0, "every_positive_step": 4.0},
        "config": {"eval_interval_steps": 2, "patience_evals": 2, "min_delta": 0.0},
        "uninterrupted": {"result": straight, "evaluation_steps": history(root / "straight")},
        "first_attempt_max_steps_1": first,
        "resumed_with_max_steps_4": {"result": resumed, "evaluation_steps": history(root / "resumed")},
        "compatibility_check": "passed",
    }
    assert straight["steps"] == 4 and resumed["steps"] == 2
    assert straight["stop_reason"] == resumed["stop_reason"] == "no_held_out_improvement"
    print(json.dumps(output, indent=2, sort_keys=True))
