"""Local check receipt runner; sampled aggregate RSS and a hard wall timeout."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=120)
parser.add_argument('--name', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command
if command and command[0] == '--':
    command = command[1:]
out = Path(__file__).parent / args.name
libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
start = time.monotonic()
peak = 0
stopped = None
with out.with_suffix('.log').open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        pids = (ctypes.c_int * 4096)()
        count = libproc.proc_listpgrppids(process.pid, pids, ctypes.sizeof(pids))
        if count < 0 or count >= len(pids):
            os.killpg(process.pid, signal.SIGKILL)
            raise RuntimeError('process-group RSS inventory unavailable')
        rss = 0
        for pid in list(pids)[:count]:
            info = ctypes.create_string_buffer(96)
            size = libproc.proc_pidinfo(pid, 4, 0, info, 96)
            if size == 96:
                rss += (ctypes.c_uint64 * 2).from_buffer(info)[1]
        peak = max(peak, rss)
        elapsed = time.monotonic() - start
        if rss >= 1_000_000_000 or elapsed >= args.seconds:
            stopped = 'memory' if rss >= 1_000_000_000 else 'wall_time'
            os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    code = process.wait()
result = {'argv': command, 'exit_status': code, 'wall_seconds': time.monotonic() - start,
          'sampled_process_group_peak_rss_bytes': peak, 'stop_reason': stopped,
          'time_limit_seconds': args.seconds, 'rss_limit_bytes': 1_000_000_000,
          'rss_sampling_seconds': 0.1}
out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
print(out.with_suffix('.log').read_text()[-6000:])
raise SystemExit(code if code >= 0 else 128 - code)
