"""Run each requested bounded validation batch sequentially; retain exact output."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[2]
plan = json.loads((out / (sys.argv[1] if len(sys.argv) > 1 else "kg-batches-plan.json")).read_text())
results = json.loads((out / "kg-batches-results.json").read_text()) if len(sys.argv) > 1 else []
for batch in plan:
    print("START " + batch["name"], flush=True)
    with (out / batch["log"]).open("w") as log:
        result = subprocess.run(batch["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
    text = (out / batch["log"]).read_text()
    completed = re.findall(r"^(tests/.*?::.*?) (PASSED|SKIPPED|FAILED|ERROR)(?: |$)", text, re.MULTILINE)
    summaries = [line for line in text.splitlines() if re.search(r"\b(?:passed|failed|skipped|deselected)\b.*\bin \d", line)]
    resources = [json.loads(line.removeprefix("RESOURCE RECEIPT ")) for line in text.splitlines() if line.startswith("RESOURCE RECEIPT ")]
    receipt = {**batch, "exit_code": result.returncode, "completed": [{"nodeid":node,"outcome":outcome} for node,outcome in completed], "summaries":summaries, "resources":resources}
    results.append(receipt)
    (out / "kg-batches-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("END " + batch["name"] + " exit=" + str(result.returncode) + " " + repr(summaries) + " " + repr(resources), flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
