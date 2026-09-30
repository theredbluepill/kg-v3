"""No-update runtime-stop cadence probe: fake clock, scripted NLL, real state I/O.

Diagnostic: a runtime cap after step 1 must leave scheduled patience untouched.
Expected: restart matches the uninterrupted step-4 patience stop, while retaining
an optional lower step-1 checkpoint and recording scheduled flags [1,0,1,1].
Bound: two NLL variants, no optimizer updates, no GPU/corpus/W&B.
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
    return [json.loads(line) for line in (path / bc.BC_HISTORY).read_text().splitlines()]

def run(path, data, config, *, resume=False, runtime_cap=False, off_nll=4.0):
    position = {"step": bc.load_bc_state(path / bc.BC_STATE).step if resume else 0, "time": 0.0}
    def clock():
        position["time"] += 0.1
        return position["time"]
    def no_update(**kwargs):
        position["step"] = kwargs["step"] + 1
        position["time"] += 100.0
        kwargs["losses"][3] += 4
    def prescribed_evaluation(*args, **kwargs):
        nll = 3.0 if position["step"] == 0 else off_nll if position["step"] == 1 else 4.0
        return bc.RowMetrics(turn_nll=nll, frame_nll=nll, value_ce=1.0, seat_rows=6)
    def timed_train(**kwargs):
        return bc.train_bc(**kwargs, max_runtime_seconds=50.0 if runtime_cap else None, clock=clock)
    with patch.object(bc, "_train_step", no_update), patch.object(bc, "evaluate_rows", prescribed_evaluation), patch.object(fixture, "train_bc", timed_train):
        result, _ = fixture._run(path, data, config, resume_from=path / bc.BC_STATE if resume else None)
    return asdict(result)

with TemporaryDirectory(prefix="bc-r3-runtime-") as tmp:
    root = Path(tmp)
    data = fixture._dataset_root(root)
    config = fixture._bc_config(max_steps=8, eval_interval_steps=2, patience_evals=2, preflight_train_replay=False)
    output=[]
    for off_nll in (4.0, 2.5):
        case=root/str(off_nll)
        straight = run(case/"straight", data, config)
        first = run(case/"resumed", data, config, runtime_cap=True, off_nll=off_nll)
        state=bc.load_bc_state(case/"resumed"/bc.BC_STATE)
        assert (first["steps"],first["stop_reason"])==(1,"max_runtime")
        assert state.selection.evals_since_improvement == 0
        assert state.selection.improvement_nll == 3.0
        bc.check_resume_compatible(state,dataset=load_bc_dataset(data),world_size=1,config=config,ppo_config=fixture._ppo_config())
        resumed=run(case/"resumed",data,config,resume=True,off_nll=off_nll)
        lines=history(case/"resumed")
        assert straight["steps"]==resumed["steps"]==4
        assert straight["stop_reason"]==resumed["stop_reason"]=="no_held_out_improvement"
        assert resumed["best_validation_nll"]==min(3.0,off_nll)
        assert [line["bc/step"] for line in lines]==[0.0,1.0,2.0,4.0]
        assert [line["bc/scheduled_evaluation"] for line in lines]==[1.0,0.0,1.0,1.0]
        output.append({"off_cadence_nll":off_nll,"uninterrupted":straight,"first_runtime_stop":first,"resumed":resumed,"history":[{key:line[key] for key in ("bc/step","bc/scheduled_evaluation","bc/validation_nll","bc/evals_since_improvement")} for line in lines]})
    print(json.dumps({"scope":"No optimizer updates; scripted validation and clock; real selection/state/history flow","cases":output},indent=2))
