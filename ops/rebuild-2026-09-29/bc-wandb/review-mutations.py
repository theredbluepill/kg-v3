"""Adversarial mutations of the BC W&B wiring (independent review pass)."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
M = [
 ("M1 gate skipped on rank 0", "scripts/train_bc.py",
  "    if not is_main_process:\n        return telemetry_mode(args.log_mode, args.wandb_mode)\n",
  "    if True:\n        return telemetry_mode(args.log_mode, args.wandb_mode)\n"),
 ("M2 configs load before the gate", "scripts/train_bc.py",
  "        # Before any config or data: online W&B without a key fails here.\n",
  "        # Before any config or data: online W&B without a key fails here.\n        _early = load_bc_configs(args.target, _parse_overrides(args.overrides)) if args.output_dir is not None else None\n"),
 ("M3 telemetry_mode dropped from bc_attempts/provenance", "scripts/train_bc.py",
  '        "telemetry_mode": str(identity.telemetry),\n', ""),
 ("M4 offline restart accepted", "scripts/train_bc.py",
  "        if args.wandb_mode == WandbMode.OFFLINE:\n            raise ValueError(\n", "        if False:\n            raise ValueError(\n"),
 ("M5 debug-origin restart accepted", "scripts/train_bc.py",
  "    if resume.wandb_run_id is None:\n", "    if False:\n"),
 ("M6 wrong game tag", "scripts/train_bc.py", "                game=BC_GAME,\n", '                game="orbit",\n'),
 ("M7 receipt agreement unchecked", "scripts/train_bc.py",
  "    if len(earlier) != identity.attempt or [\n", "    if False and [\n"),
 ("M8 wandb_run_id dropped from bc_result", "python/owl/train/bc.py",
  '                    "wandb_run_id": wandb_run_id,\n                    "dataset_manifest_sha256": dataset.manifest_sha256,\n                    "world_size": context.world_size,\n',
  '                    "dataset_manifest_sha256": dataset.manifest_sha256,\n                    "world_size": context.world_size,\n'),
 ("M9 recorded-outage line not printed", "scripts/train_bc.py",
  "                announce_recorded_outage(identity, receipt, run_dir)\n", "                pass\n"),
 ("M10 --source-commit overrides git (old behaviour)", "scripts/train_bc.py",
  "            resolve_source_commit(_SCRIPT_DIR, override=args.source_commit)\n",
  "            (args.source_commit or resolve_source_commit(_SCRIPT_DIR, override=None))\n"),
 ("M11 shared receipt not written", "scripts/train_bc.py",
  "                receipt = record_attempt(run_dir, identity, logger, start_env_steps=0)\n",
  "                receipt = run_dir / 'attempts.jsonl'\n"),
 ("M12 experiment id not forwarded", "scripts/train_bc.py",
  "                experiment_id=args.experiment_id,\n", "                experiment_id=None,\n"),
 ("M13 hash includes the ppo_config path", "python/owl/train/bc.py",
  '            "bc": config.model_dump(mode="json", exclude={"ppo_config"}),\n',
  '            "bc": config.model_dump(mode="json"),\n'),
 ("M14 restart --experiment-id accepted", "scripts/train_bc.py",
  "        if args.experiment_id is not None:\n            raise ValueError(\"resume launches keep the recorded --experiment-id\")\n", ""),
 ("M15 debug+offline parse guard removed (redundant with gate)", "scripts/train_bc.py",
  "    telemetry_mode(args.log_mode, args.wandb_mode)\n    if args.max_runtime_hours", "    if args.max_runtime_hours"),
]
results = []
for name, rel, old, new in M:
    path = ROOT / rel
    text = path.read_text()
    if text.count(old) != 1:
        results.append({"mutation": name, "status": "NOT APPLIED", "count": text.count(old)})
        continue
    path.write_text(text.replace(old, new))
    try:
        proc = subprocess.run(["uv", "run", "pytest", "tests/kaggriculture/test_bc.py", "-q", "-x", "-p", "no:cacheprovider"],
                              cwd=ROOT, capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
        failed = [l for l in proc.stdout.splitlines() if l.startswith("FAILED")]
        results.append({"mutation": name, "status": "KILLED" if proc.returncode else "SURVIVED",
                        "by": failed[:1], "tail": tail})
    finally:
        subprocess.run(["git", "checkout", "--", rel], cwd=ROOT, check=True)
    print(json.dumps(results[-1]), flush=True)
clean = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout
print(json.dumps({"restored_clean": clean == ""}))
