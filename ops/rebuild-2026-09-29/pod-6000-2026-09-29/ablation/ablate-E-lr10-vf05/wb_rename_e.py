import json, wandb
api = wandb.Api()
r = api.run("spoon/kg-v3/hi2lqsrx")
orig = [r.name, r.group]
r.name = "ablate-E-lr10-vf05"
r.group = "kg-v3-ppo-collapse-ablation"
r.update()
r = api.run("spoon/kg-v3/hi2lqsrx")
out = {"state": r.state, "name": r.name, "group": r.group, "job_type": r.job_type, "tags": r.tags,
       "url": r.url, "original_name_group": orig,
       "summary": {k: v for k, v in r.summary._json_dict.items() if not isinstance(v, dict)}}
print(json.dumps(out, indent=1, default=str))
