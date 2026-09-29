"""Receipt-only pytest watchdog for the owner's Mac; no tests are altered.

Enable with PYTEST_PLUGINS=bounded_pytest and this directory on PYTHONPATH.
This samples pytest's own RSS (descendant enumeration is sandbox-blocked).
"""

import json
import os
import resource
import sys
import threading
import time

import psutil

_START = time.monotonic()
_DONE = threading.Event()
_PEAK = 0
_LIMIT = 900 * 1024**2
_SECONDS = 115
_LOG_FD = os.dup(2)


def _watch():
    global _PEAK
    process = psutil.Process()
    while not _DONE.wait(0.01):
        rss = process.memory_info().rss
        _PEAK = max(_PEAK, rss)
        elapsed = time.monotonic() - _START
        if rss >= _LIMIT or elapsed >= _SECONDS:
            message = (
                f"\nBOUNDED CHECK STOP: rss={rss} bytes, elapsed={elapsed:.3f}s; "
                f"limits={_LIMIT} bytes/{_SECONDS}s; rerun in smaller batches\n"
            )
            os.write(_LOG_FD, message.encode())
            os._exit(86)


def pytest_sessionstart(session):
    print(f"Bounded pytest: {sys.argv!r}", flush=True)
    threading.Thread(target=_watch, daemon=True).start()


def pytest_sessionfinish(session, exitstatus):
    _DONE.set()
    print("\nRESOURCE RECEIPT " + json.dumps({
        "elapsed_seconds": time.monotonic() - _START,
        "sampled_pytest_rss_bytes": _PEAK,
        "pytest_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exitstatus": int(exitstatus),
        "scope": "pytest process only, no descendants; 10ms RSS polling",
    }), flush=True)
