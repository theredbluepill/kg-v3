
import json, os, signal, subprocess, sys, time
argv = sys.argv[1:]
backends = argv[argv.index("--backends") + 1]
record = argv[argv.index("--record") + 1]
mode = argv[argv.index("--mode") + 1]
token = argv[argv.index("--token") + 1]
want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
with open(record, "w") as fh:
    json.dump({"value_at_start": want, "value_at_end": want}, fh)
if mode == "quick":
    time.sleep(0.3)
    sys.exit(0)
ignore = mode == "long_ignore_term"
if ignore:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = ("import signal, sys, time; "
              "sys.argv[1] == 'ignore' and signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "time.sleep(300)")
gc = subprocess.Popen([sys.executable, "-c", child_code,
                       "ignore" if ignore else "default", token])
piddir = os.path.join(os.path.dirname(record), "pids")
os.makedirs(piddir, exist_ok=True)
for role, pid in (("leader", os.getpid()), ("grandchild", gc.pid)):
    open(os.path.join(piddir, f"{role}-{pid}"), "w").close()
time.sleep(300)
