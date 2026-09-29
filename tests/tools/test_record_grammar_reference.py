from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"
SPEC = importlib.util.spec_from_file_location("grammar_reference", PATH)
assert SPEC is not None
assert SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def table_shape(shape):
    return (
        [False] * shape[0]
        if len(shape) == 1
        else [table_shape(shape[1:]) for _ in range(shape[0])]
    )


def example() -> tuple[bytes, dict]:
    header = {
        "type": "header",
        "schema_version": 1,
        "reference_commit": recorder.PIN,
        "slot_names": recorder.NAMES,
        "slot_widths": recorder.WIDTHS,
        "tables": {k: table_shape(v) for k, v in recorder.TABLE_SHAPES.items()},
    }
    rows = []
    for j, shape in enumerate(recorder.SHAPES):
        for i in range(32):
            actors, orders, hire = shape
            tokens = [v for a in range(actors) for v in [a, 1] + [0] * 10]
            count = orders if i % 2 else 0
            tokens += ([0] * 7 + [7] + [0] * 4) * count + [0] * 11 + [1]
            action = {
                "farmer": ["PASS"],
                "hands": [["PASS"]] * (actors - 1),
                "market": [[]] * count,
            }
            result = {"accepted": True, "action": action, "error": None}
            rows.append(
                {
                    "type": "program",
                    "id": f"sampled-{j}-{i}",
                    "source": "sampled",
                    "shape": {
                        "actors": actors,
                        "order_limit": orders,
                        "hire_limit": hire,
                    },
                    "tokens": tokens,
                    "length": len(tokens) // 12,
                    "padding_edits": [],
                    "sampler": result,
                    "training_decoder": result,
                    "python_codec": result,
                    "v4_expected": {
                        "accepted": True,
                        "action": action,
                        "reason": "sampler canonical prefix",
                    },
                    "seed": 2026092900 + j * 32 + i,
                    "selection_mode": "full_market" if i % 2 else "uniform",
                }
            )
    for e in recorder.EPISODES:
        for s in range(4):
            for i in range(4):
                row = copy.deepcopy(rows[0])
                del row["seed"], row["selection_mode"]
                row.update(
                    id=f"replay-{e}-{s}-{i}",
                    source="replay",
                    episode=e,
                    from_step=180 * s + i // 2,
                    seat=i % 2,
                    raw_action=row["sampler"]["action"],
                )
                rows.append(row)
    for i in range(40):
        row = copy.deepcopy(rows[0])
        del row["seed"], row["selection_mode"]
        row.update(
            id=f"mutation-{i}",
            source="mutation",
            category="syntax",
            tokens=[-1, *row["tokens"][1:]],
        )
        result = {"accepted": False, "action": None, "error": "invalid token"}
        row.update(
            sampler=result,
            training_decoder=result,
            python_codec=result,
            v4_expected={"accepted": False, "action": None, "reason": "syntax"},
        )
        rows.append(row)
    payload = recorder.jsonl([header, *rows])
    return payload, recorder.fixture_manifest(payload)


def test_fixture_validation_and_compression() -> None:
    payload, manifest = example()
    recorder.validate_fixture(payload, manifest)
    first = recorder.deterministic_gzip(payload)
    assert first == recorder.deterministic_gzip(payload)
    assert gzip.decompress(first) == payload
    assert first[3] == 0
    assert first[4:8] == bytes(4)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "source",
        "action",
        "duplicate",
        "dense",
        "full",
        "corrupt",
        "size",
    ],
)
def test_fixture_rejects_corruption(change: str) -> None:
    payload, manifest = example()
    records = [json.loads(line) for line in payload.splitlines()]
    if change == "missing":
        del records[1]["tokens"]
    elif change == "extra":
        records[1]["unexpected"] = True
    elif change == "source":
        manifest["source_sha256"]["engine_rs/src/myolie_sampler.rs"] = "0" * 64
    elif change == "action":
        records[1]["v4_expected"]["action"]["farmer"] = ["NORTH"]
    elif change == "duplicate":
        records[2]["id"] = records[1]["id"]
    elif change == "dense":
        records = [r for r in records if r.get("shape", {}).get("actors") != 241]
    elif change == "full":
        records = [r for r in records if r.get("selection_mode") != "full_market"]
    elif change == "corrupt":
        payload = bytes([payload[0] ^ 1]) + payload[1:]
    else:
        payload = b" " * (4 * 1024 * 1024 + 1)
    if change not in {"source", "corrupt", "size"}:
        payload = recorder.jsonl(records)
        manifest.update(recorder.output_metadata(payload))
    with pytest.raises(ValueError, match=r".+"):
        recorder.validate_fixture(payload, manifest)


@pytest.mark.parametrize("location", ["header", "shape", "tables", "oracle"])
@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_nested_schema_is_strict(location: str, mode: str) -> None:
    payload, manifest = example()
    records = [json.loads(line) for line in payload.splitlines()]
    target = {
        "header": records[0],
        "shape": records[1]["shape"],
        "tables": records[0]["tables"],
        "oracle": records[1]["sampler"],
    }[location]
    if mode == "missing":
        del target[next(iter(target))]
    else:
        target["unexpected"] = 0
    payload = recorder.jsonl(records)
    manifest.update(recorder.output_metadata(payload))
    with pytest.raises(ValueError, match=r".+"):
        recorder.validate_fixture(payload, manifest)


def test_recorded_fixture_provenance_and_expected_action_mutation() -> None:
    fixture = ROOT / "tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz"
    manifest_path = fixture.with_name("grammar-v4-reference.manifest.json")
    payload = gzip.decompress(fixture.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    recorder.validate_fixture(payload, manifest)
    records = [json.loads(line) for line in payload.splitlines()]
    records[1]["v4_expected"]["action"]["farmer"] = ["FAULT_IN_EXPECTATION"]
    changed = recorder.jsonl(records)
    manifest.update(recorder.output_metadata(changed))
    with pytest.raises(ValueError, match="accepted expected JSON differs"):
        recorder.validate_fixture(changed, manifest)


@pytest.mark.parametrize(
    "field",
    [
        "append_patch_sha256",
        "harness_sha256",
        "recorder_sha256",
        "cargo_lock_sha256",
        "toolchain_sha256",
    ],
)
def test_recording_provenance_hashes_are_enforced(field: str) -> None:
    payload, manifest = example()
    manifest[field] = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        recorder.validate_fixture(payload, manifest)


def test_duplicate_json_key_rejected() -> None:
    payload, manifest = example()
    payload = payload.replace(b'"type":"header"', b'"type":"header","type":"header"', 1)
    manifest.update(recorder.output_metadata(payload))
    with pytest.raises(ValueError, match="invalid JSONL"):
        recorder.validate_fixture(payload, manifest)
