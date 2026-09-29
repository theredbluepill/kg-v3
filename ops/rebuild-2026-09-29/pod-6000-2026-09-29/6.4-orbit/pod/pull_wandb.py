import json, wandb
api = wandb.Api()
run = api.run("spoon/kg-v3/fmoj4eu9")
rows = list(run.scan_history())
keep = lambda k: k.startswith(("perf/", "time/", "eval/", "train/env_steps", "train/player_step_total", "train/total_games_played", "teacher/", "_step", "_runtime", "_timestamp", "losses/", "train/"))
out = {"state": run.state, "name": run.name, "project": run.project, "entity": run.entity,
       "tags": run.tags, "group": run.group, "url": run.url,
       "summary_compile": {k: run.summary.get(k) for k in run.summary.keys() if "compile" in k},
       "rows": [{k: v for k, v in r.items() if keep(k)} for r in rows]}
print(json.dumps(out, indent=1, default=str))
