import subprocess, shutil, sys
from pathlib import Path
root = Path("/Users/poonszesen/kg-v3-t31")
TESTS = ["tests/owl/train/test_ppo_observation_mapping.py", "tests/kaggriculture/test_training_smoke.py",
         "tests/scripts/test_run_ppo.py"]
muts = [
 ("S5 seat swap bootstrap", "python/owl/train/ppo.py", "self.bootstrap_values[step].copy_(bootstrap_values)", "self.bootstrap_values[step].copy_(bootstrap_values.flip(-1))"),
 ("N1 guard disabled", "scripts/run_ppo.py", "    if loaded != read_at_startup:", "    if False:"),
 ("N7 budget start 0", "scripts/run_ppo.py", "            start_env_steps=rollout_start_env_steps,\n        )\n        if isinstance(cfg.env, KaggricultureEnvConfig) and cfg.rl.eval", "            start_env_steps=0,\n        )\n        if isinstance(cfg.env, KaggricultureEnvConfig) and cfg.rl.eval"),
 ("N8 drop transition reward", "python/owl/train/ppo.py", "keep_transition_reward=_truncation_keeps_transition_reward(next_obs),", "keep_transition_reward=False,"),
 ("L1 load-weights no offset", "scripts/run_ppo.py", "    return launch.load_model_weights_path\n", "    return None\n"),
 ("L2 load-weights guard removed", "scripts/run_ppo.py", "                    launch.load_model_weights_path,\n                    read_at_startup=rollout_start_env_steps,\n                    loaded=start_env_steps,", "                    launch.load_model_weights_path,\n                    read_at_startup=start_env_steps,\n                    loaded=start_env_steps,"),
]
for name, f, old, new in muts:
    path = root / f; orig = path.read_text()
    assert orig.count(old) == 1, (name, orig.count(old))
    path.write_text(orig.replace(old, new))
    try:
        r = subprocess.run(["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS, "-k",
            "keeps_each_seat or truncation_bootstraps or seed or startup or load_weights or resume"],
            cwd=root, capture_output=True, text=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "2"})
        last = r.stdout.strip().splitlines()[-1]
        print(("KILLED " if r.returncode else "SURVIVED ") + name + " :: " + last, flush=True)
    finally:
        path.write_text(orig)
