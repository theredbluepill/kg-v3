"""Independently recount Rust producer bytes and preserve compact working evidence."""
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "oracle_driver", ROOT / "scripts/kaggriculture_observation_oracle/regenerate.py"
)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
inputs = ROOT / ".codex-tmp/g-observation-inputs"
report = driver.parse((inputs/"generation.json").read_bytes())
checked = []
for run in report["seed_runs"]:
    path = inputs / f"actions-{run['seed']}.jsonl"
    assert driver.hash_file(path) == run["action_sha256"]
    with path.open("rb") as stream:
        assert sum(1 for _ in stream) == 95
    checked.append({"seed": run["seed"], "action_sha256": run["action_sha256"], "action_rows": 95})
try:
    driver.validate_inputs(inputs/"states.jsonl", report)
except ValueError as error:
    assert "actor_gt16_states: actual 0, required 4" in str(error)
    outcome = str(error)
else:
    raise AssertionError("the fixed R1 recipe must not be admitted")
result = {"input_states_sha256": driver.hash_file(inputs/"states.jsonl"),
          "input_states_bytes": (inputs/"states.jsonl").stat().st_size,
          "independent_recount": outcome, "action_files": checked,
          "final_fixture_present": (ROOT/"tests/fixtures/kaggriculture/observation-v3").exists()}
assert not result["final_fixture_present"]
directory = Path(__file__).parent
shutil.copyfile(inputs/"generation.json", directory/"g-input-generation.json")
(directory/"g-input-audit.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
