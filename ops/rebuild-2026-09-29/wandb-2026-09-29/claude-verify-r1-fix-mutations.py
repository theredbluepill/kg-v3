import subprocess, sys, pathlib
root = pathlib.Path("/Users/poonszesen/kg-v3-wandb")
M = [
 ("M1", "python/owl/train/logging.py", "if (run is None) != (identity.telemetry is TelemetryMode.DISABLED):", "if run is None and identity.telemetry is not TelemetryMode.DISABLED:"),
 ("M2", "scripts/run_ppo.py", "resume=isinstance(launch, ResumeLaunch),", "resume=False,"),
 ("M3", "scripts/run_ppo.py", "experiment_id=args.experiment_id,", "experiment_id=None,"),
 ("M4", "scripts/run_ppo.py", "config_sha256=config_sha256(cfg),", "config_sha256='0'*64,"),
 ("M5", "python/owl/train/logging.py", "        return override\n    if override", "        return 'ignored'\n    if override"),
 ("M5b", "python/owl/train/logging.py", "    if override is not None and override != commit:", "    if False:"),
 ("M6", "python/owl/train/logging.py", 'return f"{commit}-dirty" if status else commit', "return commit"),
 ("M7", "python/owl/train/logging.py", '"started_at": _is_aware_timestamp(record["started_at"]),', '"started_at": True,'),
 ("M7b", "python/owl/train/logging.py", "    return parsed.tzinfo is not None", "    return True"),
 ("M19", "python/owl/train/logging.py", 'login = login or "user"', "pass"),
 ("F1a", "python/owl/train/logging.py", "        wandb_host(environ)\n        _check_environment_key(environ)\n", "        pass\n"),
 ("F1b", "python/owl/train/logging.py", "        _check_environment_key(environ)\n    if mode", "        pass\n    if mode"),
 ("F1c", "python/owl/train/logging.py", 'elif not (base_url := environ["WANDB_BASE_URL"]):', 'elif not (base_url := environ["WANDB_BASE_URL"] or DEFAULT_WANDB_BASE_URL):'),
 ("F6a", "python/owl/train/logging.py", "        if observed != _wandb_init_mode(mode):", "        if False:"),
 ("F6b", "python/owl/train/logging.py", "    if run.disabled:\n        return \"disabled\"\n", ""),
 ("F6c", "python/owl/train/logging.py", "            self._run.finish(exit_code=1)\n", ""),
]
only = sys.argv[1:]
for name, rel, old, new in M:
    if only and name not in only: continue
    p = root/rel; orig = p.read_bytes(); t = orig.decode()
    assert t.count(old) == 1, (name, old)
    p.write_text(t.replace(old, new))
    try:
        r = subprocess.run(["uv","run","pytest","tests/owl/train/test_logging.py","tests/scripts/test_run_ppo.py","tests/scripts/test_export_wandb_netrc_entry.py","-m","not slow","-q","-p","no:cacheprovider"],cwd=root,capture_output=True,text=True,env={**__import__("os").environ,"OMP_NUM_THREADS":"2"})
        last = r.stdout.strip().splitlines()[-1]
        print(name, "KILLED" if r.returncode else "SURVIVED", "|", last)
    finally:
        p.write_bytes(orig)
        assert p.read_bytes() == orig
