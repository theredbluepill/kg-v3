"""Cold native codec/table ABI; the Task 1.2 recorded corpus is the oracle."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from owl import rs
from owl.kaggriculture.gpu_grammar import TABLE_SHAPES, expected_grammar_tables

NAMES = (
    "unit_actor",
    "unit_kind",
    "unit_target",
    "unit_item",
    "unit_quantity_present",
    "unit_quantity_high",
    "unit_quantity",
    "market_kind",
    "market_item",
    "market_quantity_high",
    "market_quantity",
    "stop",
)
WIDTHS = (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/kaggriculture"


def _pass_tokens() -> np.ndarray:
    tokens = np.zeros((252, 12), dtype=np.int64)
    tokens[0, 1] = 1
    tokens[1, 11] = 1
    return tokens


def test_exact_grammar_constants() -> None:
    constants = rs.kaggriculture_grammar_constants()
    assert constants == (1, NAMES, WIDTHS)
    assert isinstance(constants, tuple)
    assert isinstance(constants[1], tuple)
    assert isinstance(constants[2], tuple)


def test_tables_are_copied_bool_support() -> None:
    actual = rs.kaggriculture_grammar_tables()
    expected = expected_grammar_tables().as_dict()
    assert actual.keys() == expected.keys() == TABLE_SHAPES.keys()
    assert sum(array.size for array in actual.values()) == 964
    second = rs.kaggriculture_grammar_tables()
    for key, array in actual.items():
        assert array.dtype == np.bool_
        assert array.flags.c_contiguous
        assert array.shape == TABLE_SHAPES[key]
        np.testing.assert_array_equal(array, expected[key].numpy())
        assert not np.shares_memory(array, second[key])
        array[:] = ~array
        np.testing.assert_array_equal(second[key], expected[key].numpy())
    third = rs.kaggriculture_grammar_tables()
    for key, array in third.items():
        np.testing.assert_array_equal(array, expected[key].numpy())


@pytest.mark.parametrize(
    ("action", "actors", "length"),
    [
        (PASS, 1, 2),
        ({**PASS, "market": [[], []]}, 1, 4),
        ({**PASS, "market": [[], ["BUY_PRODUCT", "WHEAT", 0]]}, 1, 4),
        ({**PASS, "farmer": ["PICKUP", "GOOSE"]}, 1, 2),
        ({**PASS, "farmer": ["PLACE", "WHEAT"]}, 1, 2),
        ({**PASS, "farmer": ["PICKUP", "WHEAT", 1]}, 1, 2),
        ({**PASS, "hands": [["EAST"], ["PASS"]]}, 3, 4),
    ],
)
def test_encode_decode_preserves_canonical_action(
    action: dict[str, Any], actors: int, length: int
) -> None:
    out = np.full((252, 12), 91, dtype=np.int64)
    assert rs.kaggriculture_encode(json.dumps(action), actors, 10, 241, out) == length
    assert (out[length:] == 0).all()
    assert json.loads(rs.kaggriculture_decode(out, length, actors, 10, 241)) == action
    out.setflags(write=False)
    assert json.loads(rs.kaggriculture_decode(out, length, actors, 10, 241)) == action
    if action == PASS:
        np.testing.assert_array_equal(out, _pass_tokens())


@pytest.mark.parametrize(
    "action_json",
    [
        "{",
        "null",
        "[]",
        "{}",
        '{"farmer":null,"hands":[],"market":[]}',
        '{"farmer":["PASS"],"hands":[["PASS"]],"market":[]}',
        '{"farmer":["PASS"],"hands":[],"market":[],"extra":0}',
        '{"farmer":["PICKUP","WHEAT",0],"hands":[],"market":[]}',
    ],
)
def test_failed_encode_preserves_output(action_json: str) -> None:
    out = np.full((252, 12), 91, dtype=np.int64)
    before = out.tobytes()
    with pytest.raises(
        ValueError,
        match=(
            r"action_json|action keys|unsupported unit command"
            r"|hand commands|quantity range"
        ),
    ):
        rs.kaggriculture_encode(action_json, 1, 10, 241, out)
    assert out.size == 3024
    assert out.tobytes() == before


@pytest.mark.parametrize(
    ("actors", "order_limit", "hire_limit"),
    [
        (0, 10, 241),
        (242, 10, 241),
        (1, 11, 241),
        (1, -1, 241),
        (1, 10, 0),
        (1, 10, 242),
    ],
)
def test_codec_shape_rejection_is_transactional(
    actors: int, order_limit: int, hire_limit: int
) -> None:
    out = np.full((252, 12), 91, dtype=np.int64)
    before = out.tobytes()
    with pytest.raises(ValueError, match="shape"):
        rs.kaggriculture_encode(json.dumps(PASS), actors, order_limit, hire_limit, out)
    assert out.tobytes() == before
    with pytest.raises(ValueError, match="shape"):
        rs.kaggriculture_decode(_pass_tokens(), 2, actors, order_limit, hire_limit)


def _invalid_array(kind: str) -> np.ndarray:
    if kind == "dtype":
        return _pass_tokens().astype(np.int32)
    if kind == "shape":
        return np.zeros((251, 12), dtype=np.int64)
    if kind == "stride":
        return np.zeros((252, 24), dtype=np.int64)[:, ::2]
    if kind == "fortran":
        return np.asfortranarray(_pass_tokens())
    if kind == "unaligned":
        return np.ndarray(
            (252, 12), dtype=np.int64, buffer=bytearray(3024 * 8 + 1), offset=1
        )
    if kind == "endian":
        return _pass_tokens().astype(np.dtype(np.int64).newbyteorder("S"))
    assert kind == "alias"
    # Internal row overlap is invalid even when NumPy exposes a writable view.
    return np.lib.stride_tricks.as_strided(
        np.zeros(12, dtype=np.int64), shape=(252, 12), strides=(0, 8), writeable=True
    )


@pytest.mark.parametrize(
    "kind", ["dtype", "shape", "stride", "fortran", "unaligned", "endian", "alias"]
)
def test_codec_array_admission_before_any_write(kind: str) -> None:
    out = _invalid_array(kind)
    before = out.tobytes()
    with pytest.raises(ValueError, match="out"):
        rs.kaggriculture_encode(json.dumps(PASS), 1, 10, 241, out)
    assert out.tobytes() == before
    with pytest.raises(ValueError, match="tokens"):
        rs.kaggriculture_decode(out, 2, 1, 10, 241)
    assert out.tobytes() == before


def test_encode_rejects_readonly_storage() -> None:
    out = np.full((252, 12), 91, dtype=np.int64)
    out.setflags(write=False)
    before = out.tobytes()
    with pytest.raises(ValueError, match="out"):
        rs.kaggriculture_encode(json.dumps(PASS), 1, 10, 241, out)
    assert out.tobytes() == before


@pytest.mark.parametrize("length", [-1, 0, 1, 3, 253, 2**63 - 1])
def test_decode_rejects_invalid_length(length: int) -> None:
    with pytest.raises(ValueError, match=r"length|STOP"):
        rs.kaggriculture_decode(_pass_tokens(), length, 1, 10, 241)


@pytest.mark.parametrize(
    ("frame", "slot", "value"),
    [(2, 0, 1), (251, 11, 1), (252 - 1, 1, -1), (0, 1, 2**62)],
)
def test_decode_rejects_dirty_tail_and_wide_tokens(
    frame: int, slot: int, value: int
) -> None:
    tokens = _pass_tokens()
    tokens[frame, slot] = value
    before = tokens.tobytes()
    with pytest.raises(ValueError, match=r"padding|grammar|vocabulary"):
        rs.kaggriculture_decode(tokens, 2, 1, 10, 241)
    assert tokens.tobytes() == before


def test_recorded_grammar_corpus_including_dense_and_rejection_classes() -> None:
    compressed = (FIXTURES / "grammar-v4-reference.jsonl.gz").read_bytes()
    manifest = json.loads((FIXTURES / "grammar-v4-reference.manifest.json").read_text())
    assert hashlib.sha256(compressed).hexdigest() == manifest["compressed_sha256"]
    payload = gzip.decompress(compressed)
    assert hashlib.sha256(payload).hexdigest() == manifest["uncompressed_sha256"]
    rows = [json.loads(line) for line in payload.splitlines()]
    assert rows[0]["type"] == "header"
    assert rows[0]["slot_names"] == list(NAMES)
    assert rows[0]["slot_widths"] == list(WIDTHS)
    assert rows[0]["reference_commit"] == manifest["reference_commit"]
    accepted = scheduled = dense = dense_full = 0
    rejected: set[str] = set()
    for row in rows[1:]:
        assert row["type"] == "program"
        assert row["source"] != "replay_rejected"
        shape = row["shape"]
        actors, orders, hire = (
            shape["actors"],
            shape["order_limit"],
            shape["hire_limit"],
        )
        tokens = np.zeros((252, 12), dtype=np.int64)
        tokens.reshape(-1)[: len(row["tokens"])] = row["tokens"]
        for frame, slot, value in row["padding_edits"]:
            tokens[frame, slot] = value
        length = row["length"]
        if row["v4_expected"]["accepted"]:
            expected = row["v4_expected"]["action"]
            actual = json.loads(
                rs.kaggriculture_decode(tokens, length, actors, orders, hire)
            )
            assert actual == expected, row["id"]
            for source in ("sampler", "training_decoder", "python_codec"):
                assert row[source]["accepted"], (row["id"], source)
                assert actual == row[source]["action"], (row["id"], source)
            out = np.full((252, 12), 91, dtype=np.int64)
            assert (
                rs.kaggriculture_encode(json.dumps(expected), actors, orders, hire, out)
                == length
            ), row["id"]
            np.testing.assert_array_equal(out, tokens, err_msg=row["id"])
            accepted += 1
            if row["source"] in {"sampled", "replay"}:
                scheduled += 1
                dense += actors == 241
                dense_full += actors == 241 and length == 252
        else:
            with pytest.raises(
                ValueError,
                match=r"grammar|padding|STOP|length|vocabulary|hire capacity",
            ):
                rs.kaggriculture_decode(tokens, length, actors, orders, hire)
            rejected.add(row["category"])
    assert (accepted, scheduled, dense, dense_full, len(rejected)) == (
        321,
        320,
        64,
        22,
        43,
    )


def test_encode_accepts_disjoint_contiguous_view_of_caller_storage() -> None:
    backing = np.full(3024 + 8, 91, dtype=np.int64)
    out = backing[4:-4].reshape(252, 12)
    assert not out.flags.owndata
    assert out.flags.c_contiguous
    assert rs.kaggriculture_encode(json.dumps(PASS), 1, 10, 241, out) == 2
    np.testing.assert_array_equal(out, _pass_tokens())
    assert (backing[:4] == 91).all()
    assert (backing[-4:] == 91).all()
