"""Pull one W&B run's state and history (receipt only). Usage: pull_wandb_run.py RUN_ID."""

import json
import sys

import wandb

api = wandb.Api()
run = api.run(f"spoon/kg-v3/{sys.argv[1]}")
rows = list(run.scan_history())
print(
    json.dumps(
        {
            "state": run.state,
            "name": run.name,
            "project": run.project,
            "entity": run.entity,
            "tags": run.tags,
            "group": run.group,
            "job_type": run.job_type,
            "url": run.url,
            "summary": {k: run.summary.get(k) for k in run.summary.keys() if not k.startswith("_wandb")},
            "rows": rows,
        },
        indent=1,
        default=str,
    )
)
