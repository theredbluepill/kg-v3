"""Offline command receipts; preserve child exit codes without shell pipelines."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent
ENV = os.environ | {'OMP_NUM_THREADS': '2', 'CARGO_BUILD_JOBS': '2',
                    'CARGO_NET_OFFLINE': 'true', 'UV_OFFLINE': 'true'}

def group_rss(group: int) -> int:
    # Same macOS libproc process-group accounting as the fixture recorder;
    # excludes unrelated agent/caller processes and includes command children.
    libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    pids = (ctypes.c_int * 4096)()
    count = libproc.proc_listpgrppids(group, pids, ctypes.sizeof(pids))
    if not 0 <= count < len(pids):
        raise RuntimeError('process-group inventory unavailable')
    total = 0
    for pid in pids[:count]:
        info = ctypes.create_string_buffer(96)
        if libproc.proc_pidinfo(pid, 4, 0, info, 96) == 96:
            total += int((ctypes.c_uint64 * 2).from_buffer(info)[1])
    return total

def run(name: str, command: list[str], *, cwd: Path = ROOT, timeout: int = 115) -> int:
    start = time.monotonic()
    path = ARTIFACTS / f'{name}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log:
        log.write('COMMAND: ' + shlex.join(command) + '\nCWD: ' + str(cwd) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=ENV, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        peak, reason = 0, None
        try:
            while process.poll() is None:
                peak = max(peak, group_rss(process.pid))
                if peak >= 960 * 1024**2 or time.monotonic() - start >= timeout:
                    reason = 'memory' if peak >= 960 * 1024**2 else 'wall_time'
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.05)
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.write(f'\nEXIT_CODE={code}\nSTOP_REASON={reason}\nPEAK_RSS_BYTES={peak}\n')
    entry = {'name': name, 'command': shlex.join(command), 'cwd': str(cwd),
             'exit_code': code, 'wall_seconds': round(time.monotonic()-start, 3),
             'log': str(path.relative_to(ARTIFACTS)), 'peak_rss_bytes': peak, 'stop_reason': reason}
    ledger = ARTIFACTS / 'commands.jsonl'
    with ledger.open('a') as out:
        out.write(json.dumps(entry) + '\n')
    print(json.dumps(entry), flush=True)
    print('\n'.join(path.read_text().splitlines()[-10:]), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1], sys.argv[2:]))
