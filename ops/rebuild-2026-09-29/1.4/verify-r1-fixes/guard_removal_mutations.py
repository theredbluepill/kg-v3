import subprocess, sys, shutil, json
from pathlib import Path
src = Path("scripts/record_kaggriculture_env_reference.py")
orig = Path(sys.argv[1]).read_text()
msgs = ["program lengths outside 1..252", "program offset inventory differs",
 "packed token inventory differs", "transition index inventory differs",
 "seed consumption differs", "terminal steps differ", "terminal done schedule differs",
 "terminal values differ", "terminal winner differs", "economic transition continuity differs",
 "bank transition continuity differs", 'f"nonfinite {name}"', "economic counters decrease",
 "array hash/shape custody differs"]
out = {}
try:
    for m in msgs:
        key = m if m.startswith("f\"") else f'"{m}"'
        i = orig.index(key)
        j = orig.rindex("require(", 0, i)
        mutated = orig[:j] + "(lambda *_: None)(" + orig[j + len("require("):]
        src.write_text(mutated)
        r = subprocess.run(["uv","run","pytest","tests/tools/test_record_kaggriculture_env_reference.py","-q","-k","semantic or hash_custody","-p","no:cacheprovider"],capture_output=True,text=True)
        last = r.stdout.strip().splitlines()[-1]
        out[m] = {"returncode": r.returncode, "summary": last}
        print(m, "->", r.returncode, last, flush=True)
finally:
    src.write_text(orig)
print(json.dumps(out))
