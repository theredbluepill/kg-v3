import importlib.util,os,signal,tempfile,time
from pathlib import Path
p=Path('/private/tmp/kg-gpu-verify-r2-qmsanrkg/bundle/scripts/test_driver_cleanup.py')
spec=importlib.util.spec_from_file_location('cleanup',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
# ps/pgrep process inventory is sandbox-denied. Only exact child PID probes.
m.alive=alive
m.survivors=lambda token: []
print('LIMIT: sandbox rejects ps/pgrep. Exact recorded child PIDs checked via kill(pid,0); no token-scan claim.',flush=True)
raise SystemExit(m.main())
