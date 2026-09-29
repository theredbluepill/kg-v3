"""Cold codec transport checks; native grammar qualification waits for Task 1.4."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from owl import rs
from owl.kaggriculture import codec
from owl.kaggriculture.types import (
    _SCHEMA,
    MAX_ACTORS,
    MAX_FRAMES,
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureObsBatch,
)


def _observation() -> KaggricultureObsBatch:
    tensors = {
        name: torch.zeros((2, 2, *shape), dtype=dtype)
        for name, (dtype, shape, _, _) in _SCHEMA.items()
    }
    obs = KaggricultureObsBatch(
        **tensors,
        action_mask=KaggricultureActionMask(
            can_act=torch.zeros((2, 2, MAX_FRAMES), dtype=torch.bool)
        ),
    )
    for env, seat, actors in ((0, 0, 1), (0, 1, 3), (1, 0, 2), (1, 1, 4)):
        obs.actor_mask[env, seat, :actors] = True
    obs.actor_mask[..., MAX_ACTORS:] = True
    obs.order_limits.copy_(torch.tensor([[1, 3], [2, 4]]))
    obs.check_contract()
    return obs


def _programs() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (
            {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": [[]]},
            {
                "farmer": ["PLACE", "MILK", 1],
                "hands": [["PASS"], ["EAST"]],
                "market": [["SELL", "WHEAT", 0]],
            },
        ),
        (
            {"farmer": ["WEST"], "hands": [["PASS"]], "market": []},
            {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [["PASS"], ["NORTH"], ["SOUTH"]],
                "market": [["HIRE"], [], ["BUY_SEED", "WHEAT", 33]],
            },
        ),
    ]


def test_codec_single_rows_serialize_and_parse_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _programs()[0][1]
    out = np.full((252, 12), -7, dtype=np.int64)
    encode_calls = []
    decode_calls = []
    dumps_calls = []
    loads_calls = []
    original_dumps, original_loads = json.dumps, json.loads

    def encode(
        action_json: str,
        actors: int,
        order_limit: int,
        hire_limit: int,
        out: NDArray[np.int64],
    ) -> int:
        encode_calls.append(
            (original_loads(action_json), actors, order_limit, hire_limit, out)
        )
        out.fill(0)
        out[0, 1] = 7
        return 5

    def decode(
        tokens: NDArray[np.int64],
        length: int,
        actors: int,
        order_limit: int,
        hire_limit: int,
    ) -> str:
        decode_calls.append((tokens, length, actors, order_limit, hire_limit))
        return original_dumps(action)

    def dumps(value: Any, **kwargs: Any) -> str:
        dumps_calls.append(value)
        return original_dumps(value, **kwargs)

    def loads(value: str) -> Any:
        loads_calls.append(value)
        return original_loads(value)

    monkeypatch.setattr(rs, "kaggriculture_encode", encode, raising=False)
    monkeypatch.setattr(rs, "kaggriculture_decode", decode, raising=False)
    monkeypatch.setattr(codec.json, "dumps", dumps)
    monkeypatch.setattr(codec.json, "loads", loads)
    length = codec.encode_action_into(
        action, actors=3, order_limit=6, hire_limit=17, out=out
    )
    assert length == 5
    assert encode_calls[0][:4] == (action, 3, 6, 17)
    assert encode_calls[0][4] is out
    assert (
        codec.decode_action(out, length, actors=3, order_limit=6, hire_limit=17)
        == action
    )
    assert decode_calls[0][0] is out
    assert decode_calls[0][1:] == (5, 3, 6, 17)
    assert dumps_calls == [action]
    assert len(loads_calls) == 1


def test_codec_batch_uses_each_seats_public_grammar_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs, programs = _observation(), _programs()
    spec = KaggricultureActionConfig(hire_limit=19)
    encode_calls = []
    decode_calls = []
    source = [program for pair in programs for program in pair]

    def encode(
        action_json: str,
        actors: int,
        order_limit: int,
        hire_limit: int,
        out: NDArray[np.int64],
    ) -> int:
        index = len(encode_calls)
        encode_calls.append((json.loads(action_json), actors, order_limit, hire_limit))
        assert out.dtype == np.int64
        assert out.shape == (252, 12)
        assert out.flags.c_contiguous
        out.fill(0)
        out[0, 1] = index + 1
        return actors + 1

    def decode(
        tokens: NDArray[np.int64],
        length: int,
        actors: int,
        order_limit: int,
        hire_limit: int,
    ) -> str:
        decode_calls.append((length, actors, order_limit, hire_limit))
        return json.dumps(source[int(tokens[0, 1]) - 1])

    monkeypatch.setattr(rs, "kaggriculture_encode", encode, raising=False)
    monkeypatch.setattr(rs, "kaggriculture_decode", decode, raising=False)
    result = codec.encode_actions(programs, obs, action_spec=spec)
    assert result.tokens.dtype == result.lengths.dtype == torch.int64
    assert result.tokens.device.type == result.lengths.device.type == "cpu"
    assert result.tokens.is_contiguous()
    assert result.lengths.is_contiguous()
    assert result.tokens.shape == (2, 2, 252, 12)
    assert result.lengths.tolist() == [[2, 4], [3, 5]]
    assert encode_calls == [
        (source[0], 1, 1, 19),
        (source[1], 3, 3, 19),
        (source[2], 2, 2, 19),
        (source[3], 4, 4, 19),
    ]
    assert codec.decode_actions(result, obs, action_spec=spec) == programs
    assert decode_calls == [(2, 1, 1, 19), (4, 3, 3, 19), (3, 2, 2, 19), (5, 4, 4, 19)]


def test_codec_batch_failure_never_publishes_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs, programs = _observation(), _programs()
    before = copy.deepcopy(programs)
    calls = []
    published = []

    def encode(
        action_json: str,
        actors: int,
        order_limit: int,
        hire_limit: int,
        out: NDArray[np.int64],
    ) -> int:
        calls.append((action_json, actors, order_limit, hire_limit))
        if len(calls) == 2:
            raise ValueError("native rejected second seat")
        out.fill(0)
        return 2

    def publish(tokens: torch.Tensor, lengths: torch.Tensor) -> KaggricultureActions:
        published.append((tokens, lengths))
        return KaggricultureActions(tokens=tokens, lengths=lengths)

    monkeypatch.setattr(rs, "kaggriculture_encode", encode, raising=False)
    monkeypatch.setattr(codec, "KaggricultureActions", publish)
    with pytest.raises(ValueError, match="native rejected second seat"):
        codec.encode_actions(programs, obs, action_spec=KaggricultureActionConfig())
    assert len(calls) == 2
    assert not published
    assert programs == before


def test_codec_propagates_native_failure_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = np.full((252, 12), 23, dtype=np.int64)
    before = out.copy()

    def encode(
        action_json: str,
        actors: int,
        order_limit: int,
        hire_limit: int,
        out: NDArray[np.int64],
    ) -> int:
        assert (json.loads(action_json), actors, order_limit, hire_limit) == (
            {},
            1,
            10,
            241,
        )
        assert np.array_equal(out, before)
        raise ValueError("native grammar rejection")

    monkeypatch.setattr(rs, "kaggriculture_encode", encode, raising=False)
    with pytest.raises(ValueError, match="native grammar rejection"):
        codec.encode_action_into({}, actors=1, order_limit=10, hire_limit=241, out=out)
    np.testing.assert_array_equal(out, before)


def test_codec_missing_native_binding_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(rs, "kaggriculture_encode", raising=False)
    monkeypatch.delattr(rs, "kaggriculture_decode", raising=False)
    out = np.zeros((252, 12), dtype=np.int64)
    with pytest.raises(AttributeError, match="kaggriculture_encode"):
        codec.encode_action_into({}, actors=1, order_limit=10, hire_limit=241, out=out)
    with pytest.raises(AttributeError, match="kaggriculture_decode"):
        codec.decode_action(out, 1, actors=1, order_limit=10, hire_limit=241)


def test_codec_batch_rejects_mismatched_environment_count() -> None:
    with pytest.raises(ValueError, match=r"actions.*environments"):
        codec.encode_actions(
            _programs()[:1], _observation(), action_spec=KaggricultureActionConfig()
        )


@pytest.mark.skip(reason="needs Task 1.4 binding")
def test_reference_programs_round_trip() -> None:
    fixtures = Path(__file__).parents[1] / "fixtures" / "kaggriculture"
    fixture = fixtures / "grammar-v4-reference.jsonl.gz"
    manifest = json.loads((fixtures / "grammar-v4-reference.manifest.json").read_text())
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest()
        == manifest["compressed_sha256"]
    )
    accepted = rejected = 0
    with gzip.open(fixture, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["type"] == "header":
                continue
            shape = record["shape"]
            rows = np.zeros((252, 12), dtype=np.int64)
            prefix = np.asarray(record["tokens"], dtype=np.int64)
            rows.flat[: prefix.size] = prefix
            for frame, slot, value in record["padding_edits"]:
                rows[frame, slot] = value
            if not record["v4_expected"]["accepted"]:
                original = rows.copy()
                with pytest.raises(ValueError, match=r".+"):
                    codec.decode_action(rows, record["length"], **shape)
                np.testing.assert_array_equal(rows, original)
                rejected += 1
                continue
            expected = record["v4_expected"]["action"]
            assert codec.decode_action(rows, record["length"], **shape) == expected
            encoded = np.full((252, 12), -7, dtype=np.int64)
            length = codec.encode_action_into(expected, out=encoded, **shape)
            assert length == record["length"]
            np.testing.assert_array_equal(encoded, rows)
            accepted += 1
    assert accepted == manifest["counts"]["accepted"] + 1
    assert rejected == manifest["counts"]["mutations"] - 1
