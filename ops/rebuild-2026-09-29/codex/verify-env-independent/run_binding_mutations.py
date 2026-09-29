"""Build/test one restored scratch mutation at a time; never install a wheel.

Prepare-only artifact. Run explicitly from the repository's Python interpreter.
UV_NO_SYNC prevents uv run from changing the shared environment's metadata.
Every build and test has its own bounded.py receipt. A test must fail normally
with pytest exit 1 to count as killed; compiler/resource/import failures do not.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCRATCH = ROOT / ".codex-tmp/verify-env-independent"
PYTHON = ROOT / ".venv/bin/python"
BOUNDED = HERE / "bounded.py"
WHEELS = SCRATCH / "mutation-wheels"
BINDINGS = "src/kaggriculture/bindings.rs"
ENV = "src/kaggriculture/env.rs"
NATIVE_TEST = "tests/kaggriculture/test_native_env.py"
GRAMMAR_TEST = "tests/kaggriculture/test_native_grammar_bindings.py"
ORACLE_TEST = "tests/kaggriculture/test_env_reference.py"
SHAPE_TEST = "tests/kaggriculture/test_independent_binding_probe.py"
EXTENSION = SCRATCH / "python/owl/rs.abi3.so"

# One exact replacement per mutation. Selectors exercise existing tests except
# the extra same-cell-count shape probe, which distinguishes shape from length.
MUTATIONS = [
    (
        "strict-seeds", BINDINGS,
        "if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {",
        "if false && (value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>()) {",
        NATIVE_TEST, "strict_seed_admission",
    ),
    (
        "reward-plain-dict", BINDINGS,
        "if !value.is_exact_instance_of::<PyDict>() || value.len() != KEYS.len() {",
        "if value.len() != KEYS.len() {",
        NATIVE_TEST, "reward_config_requires_plain_dict",
    ),
    (
        "reward-exact-keys", BINDINGS,
        "if !value.is_exact_instance_of::<PyDict>() || value.len() != KEYS.len() {",
        "if !value.is_exact_instance_of::<PyDict>() {",
        NATIVE_TEST, "reward_dict_rejects_invalid_config",
    ),
    (
        "overlap", BINDINGS,
        "if left.start < right.end && right.start < left.end {",
        "if false && left.start < right.end && right.start < left.end {",
        NATIVE_TEST, "distinct_numpy_bases_with_shared_storage_are_rejected",
    ),
    (
        "table-bit", BINDINGS,
        "bool_table(py, &[20], tables.unit_kind)?",
        "bool_table(py, &[20], tables.unit_kind.map(|v| !v))?",
        GRAMMAR_TEST, "tables_are_copied_bool_support",
    ),
    (
        "constants-version", BINDINGS,
        "grammar::GRAMMAR_TABLES_VERSION\n",
        "(grammar::GRAMMAR_TABLES_VERSION + 1)\n",
        GRAMMAR_TEST, "exact_grammar_constants",
    ),
    (
        "c-layout", BINDINGS,
        "if !array.is_c_contiguous() || !array.is_aligned() {",
        "if !array.is_aligned() {",
        NATIVE_TEST, "every_destination and fortran",
    ),
    (
        "alignment", BINDINGS,
        "if !array.is_c_contiguous() || !array.is_aligned() {",
        "if !array.is_c_contiguous() {",
        NATIVE_TEST, "every_destination and unaligned",
    ),
    (
        "shape", BINDINGS,
        "if array.shape() != shape {",
        "if false && array.shape() != shape {",
        SHAPE_TEST, "same_cell_count",
    ),
    (
        "codec-tail", BINDINGS,
        'grammar::encode(&plan, &action, output_slice(&mut out, "out")?).map_err(PyValueError::new_err)',
        'let length = grammar::encode(&plan, &action, output_slice(&mut out, "out")?).map_err(PyValueError::new_err)?;\n'
        '    output_slice(&mut out, "out")?[3023] = 1;\n    Ok(length)',
        GRAMMAR_TEST, "encode_decode_preserves_canonical_action",
    ),
    (
        "l6-econ-after", ENV,
        "        out.transition_econ_after\n            .copy_from_slice(&self.transition_econ_after);",
        "        // MUTANT: omit publication of transition_econ_after.",
        NATIVE_TEST, "calls_overwrite_every_output_byte_of_the_same_buffer_set",
    ),
    (
        "seed-stride", ENV,
        "next.checked_add(self.stride)",
        "next.checked_add(1)",
        NATIVE_TEST, "seed_partition_across_constructor_full_partial_and_terminal_resets",
    ),
    (
        "native-dones-oracle", ENV,
        "transition.dones[i * 2..i * 2 + 2].fill(row.done);",
        "transition.dones[i * 2..i * 2 + 2].fill(!row.done);",
        ORACLE_TEST, "native_matches_training_batch_16_complete_games",
    ),
]

PROBE = '''"""Independent exact-shape probe kept only in the mutation scratch tree."""
import pytest
from .test_native_env import buffers, diagnostics, make_env, snapshot


def test_same_cell_count_shape_is_rejected_transactionally():
    env = make_env(2)
    arrays = buffers(2)
    env.observe(**arrays)
    arrays["tiles_int"] = arrays["tiles_int"].reshape(2, 2, 7, 200)
    before = snapshot(arrays), diagnostics(env, 2)
    with pytest.raises(ValueError, match="tiles_int"):
        env.reset(**arrays)
    assert (snapshot(arrays), diagnostics(env, 2)) == before
'''

ORIGIN_PLUGIN = '''"""Abort collection if pytest would exercise the installed/main extension."""
from pathlib import Path
from owl import rs


def pytest_sessionstart(session):
    expected = Path(__file__).resolve().parent / "python/owl/rs.abi3.so"
    actual = Path(rs.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"wrong native extension: {actual}, expected {expected}")
    print(f"INDEPENDENT_NATIVE_EXTENSION={actual}")
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        GIT_DIR=subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True).strip(),
        GIT_WORK_TREE=str(SCRATCH),
        CARGO_BUILD_JOBS="2",
        CARGO_NET_OFFLINE="true",
        CARGO_TARGET_DIR=str(SCRATCH / "target"),
        UV_OFFLINE="true",
        UV_NO_SYNC="1",
        UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"),
        VIRTUAL_ENV=str(ROOT / ".venv"),
        PYTHONPATH=os.pathsep.join((str(SCRATCH / "python"), str(SCRATCH))),
        OWL_NATIVE_MODULE_DIR=str(SCRATCH / "python/owl"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    return result


def bounded(name: str, command: list[str], seconds: float) -> dict:
    receipt = HERE / f"binding-{name}.json"
    log = HERE / f"binding-{name}.log"
    # Distinct names in repeat runs still overwrite the receipt deliberately;
    # copy the entire evidence directory first if preserving a prior attempt.
    receipt.unlink(missing_ok=True)
    process = subprocess.run(
        [str(PYTHON), str(BOUNDED), "--seconds", str(seconds),
         "--name", f"binding-{name}", "--", *command],
        cwd=SCRATCH,
        env=environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not receipt.exists():
        raise RuntimeError(f"watchdog produced no receipt: {process.stdout}")
    result = json.loads(receipt.read_text())
    result.update(receipt=str(receipt.relative_to(ROOT)), log=str(log.relative_to(ROOT)))
    print(json.dumps({"phase": name, "exit_status": result["exit_status"],
                      "stop_reason": result["stop_reason"]}), flush=True)
    return result


def build(name: str, seconds: float) -> tuple[dict, dict | None]:
    WHEELS.mkdir(parents=True, exist_ok=True)
    # Isolate this build's output so a failed build cannot reuse a prior wheel.
    out = WHEELS / name
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()
    receipt = bounded(
        f"{name}-build",
        ["uv", "run", "--offline", "maturin", "build", "--profile", "dev",
         "--locked", "--out", str(out)],
        seconds,
    )
    if receipt["exit_status"] != 0 or receipt["stop_reason"] is not None:
        return receipt, None
    wheels = list(out.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one freshly built wheel: {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = [n for n in archive.namelist()
                   if n.startswith("owl/rs.") and n.endswith(".so")]
        if members != ["owl/rs.abi3.so"]:
            raise RuntimeError(f"unexpected native wheel members: {members}")
        extension = archive.read(members[0])
    EXTENSION.write_bytes(extension)
    return receipt, {"path": str(wheel.relative_to(ROOT)),
                     "sha256": sha(wheel.read_bytes()),
                     "extension_sha256": sha(extension)}


def test(name: str, path: str, selector: str, seconds: float) -> dict:
    # --rootdir does not itself stop ancestor conftest discovery. confcutdir
    # confines it to scratch; its tests/conftest.py is a byte-identical copy.
    return bounded(
        f"{name}-test",
        [str(PYTHON), "-m", "pytest", "--rootdir", str(SCRATCH),
         "--confcutdir", str(SCRATCH), "-p", "independent_native_origin",
         path, "-k", selector, "-q"],
        seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="mutation names; omit for all 13")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seconds", type=float, default=115)
    parser.add_argument("--baseline", action="store_true",
                        help="build the unmutated wheel and run each selected probe first")
    args = parser.parse_args()
    unknown = set(args.names) - {m[0] for m in MUTATIONS}
    if unknown:
        parser.error(f"unknown mutation names: {sorted(unknown)}")
    selected = [m for m in MUTATIONS if not args.names or m[0] in args.names]
    if args.list:
        print(json.dumps([{"name": m[0], "source": m[1], "test": m[4],
                           "selector": m[5]} for m in selected], indent=2))
        return 0
    if not SCRATCH.is_dir() or not PYTHON.exists() or not BOUNDED.is_file():
        raise RuntimeError("expected pre-existing scratch, root Python, and watchdog")
    originals = {path: (SCRATCH / path).read_bytes() for path in {m[1] for m in selected}}
    for path, data in originals.items():
        if data != (ROOT / path).read_bytes():
            raise RuntimeError(f"scratch source differs from worktree before mutation: {path}")
    (SCRATCH / SHAPE_TEST).write_text(PROBE)
    (SCRATCH / "independent_native_origin.py").write_text(ORIGIN_PLUGIN)
    extension_before = EXTENSION.read_bytes() if EXTENSION.exists() else None
    summary = {"scratch": str(SCRATCH), "started_unix": time.time(),
               "original_source_sha256": {p: sha(b) for p, b in originals.items()},
               "mutations": []}
    summary_path = HERE / "binding-mutations.json"
    try:
        if args.baseline:
            build_receipt, wheel = build("baseline", args.seconds)
            summary["baseline_build"] = build_receipt
            summary["baseline_wheel"] = wheel
            write_json(summary_path, summary)
            if wheel is None:
                return 2
            for name, _, _, _, path, selector in selected:
                receipt = test(f"baseline-{name}", path, selector, args.seconds)
                summary.setdefault("baseline_tests", {})[name] = receipt
                write_json(summary_path, summary)
                if receipt["exit_status"] != 0 or receipt["stop_reason"]:
                    return 2
        for name, path, old, new, test_path, selector in selected:
            data = originals[path]
            old_bytes, new_bytes = old.encode(), new.encode()
            if data.count(old_bytes) != 1:
                raise RuntimeError(f"{name}: replacement matched {data.count(old_bytes)} times")
            mutant = data.replace(old_bytes, new_bytes, 1)
            diff = "".join(difflib.unified_diff(data.decode().splitlines(True),
                                             mutant.decode().splitlines(True),
                                             fromfile=path, tofile=path + ".mutant"))
            (HERE / f"binding-{name}.patch").write_text(diff)
            entry = {"name": name, "source": path, "before_sha256": sha(data),
                     "mutant_sha256": sha(mutant), "test": test_path,
                     "selector": selector, "status": "running"}
            summary["mutations"].append(entry)
            write_json(summary_path, summary)
            try:
                (SCRATCH / path).write_bytes(mutant)
                build_receipt, wheel = build(name, args.seconds)
                entry.update(build=build_receipt, wheel=wheel)
                if build_receipt["stop_reason"]:
                    entry["status"] = "build_resource_or_timeout"
                elif wheel is None:
                    entry["status"] = "compile_error"
                else:
                    receipt = test(name, test_path, selector, args.seconds)
                    entry["test_receipt"] = receipt
                    if receipt["stop_reason"]:
                        entry["status"] = "test_resource_or_timeout"
                    elif receipt["exit_status"] == 1:
                        entry["status"] = "killed"
                    elif receipt["exit_status"] == 0:
                        entry["status"] = "survived"
                    else:
                        entry["status"] = "test_infrastructure_error"
            finally:
                (SCRATCH / path).write_bytes(data)
                entry["restored_sha256"] = sha((SCRATCH / path).read_bytes())
                entry["restored_byte_for_byte"] = (SCRATCH / path).read_bytes() == data
                write_json(summary_path, summary)
            print(json.dumps({"name": name, "status": entry["status"],
                              "restored": entry["restored_byte_for_byte"]}), flush=True)
    finally:
        for path, data in originals.items():
            (SCRATCH / path).write_bytes(data)
        if extension_before is None:
            EXTENSION.unlink(missing_ok=True)
        else:
            EXTENSION.write_bytes(extension_before)
        summary["finished_unix"] = time.time()
        summary["all_scratch_sources_restored"] = all(
            (SCRATCH / p).read_bytes() == data for p, data in originals.items())
        summary["all_worktree_sources_unchanged"] = all(
            (ROOT / p).read_bytes() == data for p, data in originals.items())
        summary["extension_restored"] = (
            not EXTENSION.exists() if extension_before is None
            else EXTENSION.read_bytes() == extension_before)
        write_json(summary_path, summary)
    return 0 if all(m["status"] == "killed" for m in summary["mutations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
