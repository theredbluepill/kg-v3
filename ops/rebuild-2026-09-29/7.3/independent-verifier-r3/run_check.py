import json, pathlib, subprocess, sys, time
out = pathlib.Path(__file__).resolve().parent
name, *cmd = sys.argv[1:]
start = time.time()
with (out / (name + ".log")).open("w") as f:
    run = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
receipt = {"command": cmd, "exit_code": run.returncode, "wall_seconds": time.time()-start}
(out / (name + ".json")).write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
print("".join((out/(name+".log")).read_text().splitlines(keepends=True)[-14:]))
sys.exit(run.returncode)
