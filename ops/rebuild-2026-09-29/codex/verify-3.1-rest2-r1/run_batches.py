import json, os, subprocess, sys, re
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
plan=json.loads((out/sys.argv[1]).read_text())
results=[]
for item in plan:
    print("START "+item["name"],flush=True)
    env=dict(os.environ,VERIFY_RECEIPT=str(out/(item["name"]+".json")))
    with (out/item["log"]).open("w") as log:
        try:
            result=subprocess.run(item["command"],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=118)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    text=(out/item["log"]).read_text()
    lines=[s for s in text.splitlines() if (re.search(r"(?:passed|failed|skipped|deselected).* in [0-9]",s) or "RESOURCE RECEIPT" in s or "BOUNDED CHECK STOP" in s)]
    print("END "+item["name"]+" exit="+str(code)+" "+repr(lines),flush=True)
    results.append(dict(name=item["name"],command=item["command"],exit_code=code,printed=lines))
    (out/(sys.argv[1]+"-results.json")).write_text(json.dumps(results,indent=2))
    if code: sys.exit(code)
