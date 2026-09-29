"""Bounded source-only compile; never records features or admits a corpus."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "oracle_driver", ROOT / "scripts/kaggriculture_observation_oracle/regenerate.py"
)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
out = Path(__file__).with_name("g-recorder-compile.json")
start = time.monotonic()
export = Path(tempfile.mkdtemp(prefix="g-recorder-check-"))
hashes = driver.export_reference(export)
recorder = ROOT / "scripts/kaggriculture_observation_oracle/record.rs"
example = export / "engine_rs/examples/observation_v3_oracle.rs"
assert not example.exists()
example.parent.mkdir(exist_ok=True)
shutil.copyfile(recorder, example)
argv = ["cargo", "check", "--locked", "--offline", "--manifest-path",
        str(export / "engine_rs/Cargo.toml"), "--example", "observation_v3_oracle"]
report = {
    "purpose": "source-only recorder compile; no inputs or features generated",
    "reference_commit": driver.PIN,
    "export": str(export), "reference_file_hashes": hashes,
    "recorder_sha256": driver.hash_file(recorder),
    "argv": argv, "status": "running", "existing_reference_bytes_unchanged": None,
    "rustc": driver.run(["rustc", "--version"]).decode().strip(),
    "cargo": driver.run(["cargo", "--version"]).decode().strip(),
}
out.write_text(json.dumps(report, indent=2) + "\n")
try:
    driver.run(argv)
    report["status"] = "compiled"
except BaseException as error:
    report["status"] = "failed"
    report["error"] = str(error)
    raise
finally:
    report["command_receipt"] = driver.COMMAND_RECEIPTS[-1]
    report["existing_reference_bytes_unchanged"] = all(
        driver.hash_file(export/path) == value for path, value in hashes.items()
    )
    report["wall_seconds"] = time.monotonic()-start
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "reference_file_hashes"}, indent=2))
    assert report["existing_reference_bytes_unchanged"]
