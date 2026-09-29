"""No-training probe of the real train_bc.main resume/provenance control flow.

All model, optimizer, dataset I/O and training functions are stubbed. Config
loading, resume compatibility, launch.json reading and main's provenance plumbing
remain real. The checkpoint fixture represents an older source/config attempt.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from owl.train.bc import BCResumeState, BCResult, HeldOutSelection, TrainSharding, load_bc_configs
from owl.train.distributed import DistributedContext
from owl.train.logging import DebugLogger, LogMode

spec = importlib.util.spec_from_file_location("identity_probe_train_bc", REPO / "scripts/train_bc.py")
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
old_source = "0" * 40
context = DistributedContext.single_process_cpu()
dataset = SimpleNamespace(manifest_sha256="fixture-manifest")
state = BCResumeState(
    step=2,
    selection=HeldOutSelection(patience_evals=3, min_delta=0, best_nll=1, best_step=2, last_nll=1, last_step=2),
    wandb_run_id=None,
    dataset_manifest_sha256="fixture-manifest",
    world_size=1,
    model={},
    optimizer={},
    lr_scheduler=None,
)
original, _ = load_bc_configs(REPO / "configs/bc/kaggriculture_2rank.yaml")
changed = original.model_copy(update={"seed": original.seed + 1, "rows_per_rank": original.rows_per_rank // 2, "ppo_config": REPO / "configs/kaggriculture.yaml"})
captured = {}

def no_training(**kwargs):
    captured.update({"provenance": kwargs["provenance"], "seed": kwargs["config"].seed, "rows_per_rank": kwargs["config"].rows_per_rank, "resume_step": kwargs["resume"].step})
    return BCResult("max_steps", 2, 2, 1.0, 1.0)

original_rows = TrainSharding(original.seed, original.rows_per_rank, 1, (2048,)).rows(step=state.step, micro=0, rank=0)
seed_only_rows = TrainSharding(changed.seed, original.rows_per_rank, 1, (2048,)).rows(step=state.step, micro=0, rank=0)
batch_only_rows = TrainSharding(original.seed, changed.rows_per_rank, 1, (2048,)).rows(step=state.step, micro=0, rank=0)

with tempfile.TemporaryDirectory(prefix="bc-resume-identity-") as tmp:
    run_dir = Path(tmp)
    changed.to_file(run_dir / "bc_config.yaml")
    (run_dir / "launch.json").write_text(json.dumps({"source_commit": old_source, "world_size": 1, "dataset_manifest_sha256": "fixture-manifest"}))
    args = Namespace(output_dir=None, target=run_dir, data=run_dir / "stub-dataset", log_mode=LogMode.DEBUG, wandb_mode="offline", max_runtime_hours=None)
    with (
        patch.object(script, "_parse_args", return_value=args),
        patch.object(script, "configure_torch"),
        patch.object(script, "distributed_session", return_value=nullcontext(context)),
        patch.object(script, "load_bc_dataset", return_value=dataset),
        patch.object(script, "load_bc_state", return_value=state),
        patch.object(script, "build_bc_model", return_value=object()),
        patch.object(script, "configure_model_compile", return_value=0),
        patch.object(script, "gemm_backend_claim", return_value=None),
        patch.object(script, "create_optimizer", return_value=object()),
        patch.object(script, "create_lr_scheduler", return_value=None),
        patch.object(script, "restore_bc_state"),
        patch.object(script, "wrap_bc_model_for_distributed", side_effect=lambda model, _: model),
        patch.object(script, "create_bc_logger", return_value=DebugLogger()),
        patch.object(script, "train_bc", side_effect=no_training),
        patch.object(script, "_git_head", return_value=actual_head) as git_head,
    ):
        script.main()
        observed = {
            "training_executed": False,
            "actual_current_head": actual_head,
            "original_attempt_source": old_source,
            "git_head_calls_during_resume": git_head.call_count,
            "original_attempt_seed": original.seed,
            "edited_resume_seed": changed.seed,
            "original_attempt_rows_per_rank": original.rows_per_rank,
            "edited_resume_rows_per_rank": changed.rows_per_rank,
            "same_resume_step_original_row_prefix": original_rows[:8].tolist(),
            "same_resume_step_seed_only_row_prefix": seed_only_rows[:8].tolist(),
            "same_resume_step_batch_only_row_prefix": batch_only_rows[:8].tolist(),
            "captured_at_training_boundary": captured,
        }
        assert git_head.call_count == 0
        assert captured["provenance"]["source_commit"] == old_source != actual_head
        assert captured["seed"] == changed.seed != original.seed
        assert captured["rows_per_rank"] == changed.rows_per_rank != original.rows_per_rank
        assert original_rows[:8].tolist() != seed_only_rows[:8].tolist()
        assert original_rows[:8].tolist() != batch_only_rows[:8].tolist()
        print(json.dumps(observed, indent=2, sort_keys=True))
