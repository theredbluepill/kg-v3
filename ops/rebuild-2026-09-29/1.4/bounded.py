"""Local check receipt runner; sampled aggregate RSS and a hard wall timeout."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=115)
parser.add_argument('--name', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command
if command and command[0] == '--':
    command = command[1:]
out = Path(__file__).parent / args.name
libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True) if sys.platform == 'darwin' else None

def group_rss(group):
    if libproc is not None:
        pids = (ctypes.c_int * 4096)()
        count = libproc.proc_listpgrppids(group, pids, ctypes.sizeof(pids))
        if count < 0 or count >= len(pids):
            raise RuntimeError('process-group RSS inventory unavailable')
        rss = 0
        for pid in list(pids)[:count]:
            info = ctypes.create_string_buffer(96)
            if libproc.proc_pidinfo(pid, 4, 0, info, 96) == 96:
                rss += (ctypes.c_uint64 * 2).from_buffer(info)[1]
        return rss
    if sys.platform == 'linux':
        rss = 0
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            try:
                stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(stat[2]) == group:
                    rss += int((path / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            except (FileNotFoundError, ProcessLookupError):
                continue
        return rss
    raise RuntimeError('watchdog requires macOS libproc or Linux /proc')
start = time.monotonic()
peak = 0
stopped = None
with out.with_suffix('.log').open('w') as log:
    # Nested watchdogs must finish killing their own child groups before this
    # outer watchdog kills them. Mac Python 3.9's monotonic origin is process
    # local, so export a Unix deadline; children convert remaining wall time to
    # their own monotonic deadline. This includes any uv startup time.
    child_env = os.environ | {'KG_BOUNDED_DEADLINE_UNIX': str(time.time() + args.seconds)}
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env)
    while process.poll() is None:
        try:
            rss = group_rss(process.pid)
        except Exception:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        peak = max(peak, rss)
        elapsed = time.monotonic() - start
        if rss >= 960 * 1024**2 or elapsed >= args.seconds:
            stopped = 'memory' if rss >= 960 * 1024**2 else 'wall_time'
            os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    code = process.wait()
result = {'argv': command, 'exit_status': code, 'wall_seconds': time.monotonic() - start,
          'sampled_process_group_peak_rss_bytes': peak, 'stop_reason': stopped,
          'time_limit_seconds': args.seconds, 'rss_limit_bytes': 960 * 1024**2,
          'rss_sampling_seconds': 0.1}
out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
print(out.with_suffix('.log').read_text()[-6000:])
raise SystemExit(code if code >= 0 else 128 - code)
