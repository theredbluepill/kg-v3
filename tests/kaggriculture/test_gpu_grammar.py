"""Task 2.3: typed grammar support tables (brief §2, §7.2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt


def _support(mask: torch.Tensor) -> list[int]:
    return mask.nonzero().flatten().tolist()


def _unit_kind(name: str) -> int:
    return kt.UNIT_KINDS.index(name)


def _market_kind(name: str) -> int:
    return kt.MARKET_KINDS.index(name)


def test_expected_tables_have_the_native_shapes_and_names() -> None:
    tables = gg.expected_grammar_tables()
    fields = tables.as_dict()
    assert list(fields) == list(gg.TABLE_SHAPES)
    assert [f.name for f in dataclasses.fields(tables)] == list(gg.TABLE_SHAPES)
    for name, shape in gg.TABLE_SHAPES.items():
        assert fields[name].dtype == torch.bool, name
        assert tuple(fields[name].shape) == shape, name
    # 964 booleans, as the Task 1.2 native GrammarTables.
    assert sum(t.numel() for t in fields.values()) == 964


def test_unit_supports_follow_the_grammar_rules() -> None:
    tables = gg.expected_grammar_tables()
    assert _support(tables.unit_kind) == list(range(1, 19))
    for kind in range(20):
        item = _support(tables.unit_item[kind])
        present = _support(tables.unit_quantity_present[kind])
        if kind in (_unit_kind("PICKUP"), _unit_kind("PLACE")):
            assert item == list(range(1, 13)), kind
            assert present == [0, 1], kind
        elif kind == _unit_kind("PLANT"):
            assert item == [1, 2, 3, 4, 5], kind
            assert present == [0], kind
        else:
            assert item == [0], kind
            assert present == [0], kind
    assert _support(tables.unit_quantity_high[0]) == [0]
    assert _support(tables.unit_quantity_high[1]) == list(range(32))
    # Rows are indexed by present, then by whether the high digit is zero.
    assert _support(tables.unit_quantity_low[0, 0]) == [0]
    assert _support(tables.unit_quantity_low[0, 1]) == [0]
    assert _support(tables.unit_quantity_low[1, 0]) == list(range(32))
    assert _support(tables.unit_quantity_low[1, 1]) == list(range(1, 32))


def test_market_supports_follow_the_grammar_rules() -> None:
    tables = gg.expected_grammar_tables()
    assert _support(tables.market_kind) == list(range(8))
    items = {
        _market_kind("BUY_SEED"): list(range(1, 6)),
        _market_kind("BUY_PRODUCT"): [1, 9],
        _market_kind("BUY_ANIMAL"): [10, 11, 12],
        _market_kind("SELL"): list(range(1, 10)),
    }
    for kind in range(8):
        assert _support(tables.market_item[kind]) == items.get(kind, [0]), kind
        digits = list(range(32)) if kind in items else [0]
        assert _support(tables.market_quantity[kind]) == digits, kind


def _unit_quantity_admitted(
    tables: gg.GrammarTables, present: int, high: int, low: int
) -> bool:
    return bool(
        tables.unit_quantity_high[present, high]
        and tables.unit_quantity_low[present, int(high == 0), low]
    )


@pytest.mark.parametrize(
    ("present", "value", "admitted"),
    [
        (0, 0, True),  # omitted quantity
        (0, 1, False),  # absent quantity must keep both digits zero
        (1, 0, False),  # explicit unit zero is rejected
        (1, 1, True),
        (1, 31, True),
        (1, 32, True),
        (1, 1023, True),
    ],
)
def test_unit_quantity_cases(present: int, value: int, admitted: bool) -> None:
    tables = gg.expected_grammar_tables()
    high, low = divmod(value, 32)
    assert _unit_quantity_admitted(tables, present, high, low) is admitted


@pytest.mark.parametrize("value", [0, 1, 31, 32, 1023])
def test_market_quantity_accepts_zero_through_1023(value: int) -> None:
    tables = gg.expected_grammar_tables()
    high, low = divmod(value, 32)
    for kind in (3, 4, 5, 6):
        assert bool(tables.market_quantity[kind, high])
        assert bool(tables.market_quantity[kind, low])
    for kind in (0, 1, 2, 7):
        assert bool(tables.market_quantity[kind, high]) is (high == 0)


def _fields() -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in gg.expected_grammar_tables().as_dict().items()}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda f: f.update(unit_kind=f["unit_kind"].long()), "torch.bool"),
        (lambda f: f.update(unit_item=f["unit_item"][:, :15]), "shape"),
        (lambda f: f["market_item"][3].fill_(False), "no admitted value"),
        (lambda f: f["unit_quantity_low"][0, 0].fill_(False), "no admitted value"),
        (lambda f: f["market_kind"].__setitem__(0, False), "NONE"),
        (
            lambda f: f.update(market_kind=torch.arange(8) == gg.MARKET_HIRE),
            "NONE",
        ),
    ],
)
def test_validation_rejects_malformed_tables(edit: object, message: str) -> None:
    fields = _fields()
    assert callable(edit)
    edit(fields)
    with pytest.raises(ValueError, match=message):
        gg.validate_grammar_tables(gg.GrammarTables(**fields))


def test_validation_requires_a_non_hire_market_kind() -> None:
    fields = _fields()
    fields["market_kind"] = torch.zeros(8, dtype=torch.bool)
    fields["market_kind"][gg.MARKET_NONE] = True
    gg.validate_grammar_tables(gg.GrammarTables(**fields))  # NONE alone is fine


def test_from_arrays_round_trips_numpy_and_rejects_bad_keys_or_dtypes() -> None:
    expected = gg.expected_grammar_tables()
    arrays = {k: v.numpy().copy() for k, v in expected.as_dict().items()}
    rebuilt = gg.grammar_tables_from_arrays(arrays)
    for name, table in rebuilt.as_dict().items():
        assert torch.equal(table, expected.as_dict()[name]), name
    with pytest.raises(ValueError, match="missing"):
        gg.grammar_tables_from_arrays(
            {k: v for k, v in arrays.items() if k != "unit_kind"}
        )
    with pytest.raises(ValueError, match="unexpected"):
        gg.grammar_tables_from_arrays({**arrays, "extra": arrays["unit_kind"]})
    with pytest.raises(ValueError, match="must be bool"):
        gg.grammar_tables_from_arrays(
            {**arrays, "unit_kind": arrays["unit_kind"].astype(np.int8)}
        )


def test_native_tables_match_expected_tables() -> None:
    """Brief §7.2: every table row matches the native plans (after Task 1.2)."""
    try:
        native = gg.native_grammar_tables()
    except NotImplementedError as error:
        pytest.skip(f"Task 1.2/1.4 native grammar binding not available: {error}")
    expected = gg.expected_grammar_tables()
    for name, table in native.as_dict().items():
        assert torch.equal(table, expected.as_dict()[name]), name
