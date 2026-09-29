"""Task 1.5: strict native table admission and the model's native default."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from owl import rs
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_native_tables_equal_expected(device: torch.device | str) -> None:
    actual = gg.native_grammar_tables(device=device).as_dict()
    expected = gg.expected_grammar_tables(device=device).as_dict()
    assert actual.keys() == expected.keys()
    for name, table in actual.items():
        assert table.device == torch.device(device)
        assert table.dtype == torch.bool
        assert torch.equal(table, expected[name]), name


@pytest.mark.parametrize(
    "constants",
    [
        (2, kt.SLOT_NAMES, kt.SLOT_WIDTHS),
        (True, kt.SLOT_NAMES, kt.SLOT_WIDTHS),
        (1, ("wrong", *kt.SLOT_NAMES[1:]), kt.SLOT_WIDTHS),
        (1, tuple(reversed(kt.SLOT_NAMES)), kt.SLOT_WIDTHS),
        (1, kt.SLOT_NAMES, (242, *kt.SLOT_WIDTHS[1:])),
        (1, kt.SLOT_NAMES, kt.SLOT_WIDTHS[:-1]),
        (1, list(kt.SLOT_NAMES), kt.SLOT_WIDTHS),
        (1, kt.SLOT_NAMES, tuple(float(n) for n in kt.SLOT_WIDTHS)),
    ],
    ids=[
        "version",
        "bool-version",
        "name",
        "name-order",
        "width",
        "arity",
        "name-list",
        "float-widths",
    ],
)
def test_native_constants_rejected_before_tables_are_requested(
    monkeypatch: pytest.MonkeyPatch, constants: object
) -> None:
    def unexpected_tables() -> dict[str, np.ndarray]:
        pytest.fail("tables must not be fetched before constants are admitted")

    monkeypatch.setattr(rs, "kaggriculture_grammar_constants", lambda: constants)
    monkeypatch.setattr(rs, "kaggriculture_grammar_tables", unexpected_tables)
    with pytest.raises(ValueError, match="grammar constants"):
        gg.native_grammar_tables()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda a: a.pop("unit_kind"), "missing"),
        (lambda a: a.update(extra=a["unit_kind"]), "unexpected"),
        (lambda a: a.update(unit_kind=a["unit_kind"].astype(np.int8)), "bool"),
        (lambda a: a.update(unit_kind=torch.from_numpy(a["unit_kind"])), "NumPy"),
        (lambda a: a.update(unit_item=a["unit_item"][:, :15].copy()), "shape"),
        (
            lambda a: a.update(unit_item=np.asfortranarray(a["unit_item"])),
            "C-contiguous",
        ),
        (lambda a: a.update(unit_kind=a["unit_kind"][::-1]), "C-contiguous"),
    ],
    ids=[
        "missing-key",
        "extra-key",
        "dtype",
        "non-array",
        "shape",
        "fortran",
        "strided",
    ],
)
def test_native_arrays_reject_corruption(
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    arrays = dict(rs.kaggriculture_grammar_tables())
    edit(arrays)
    monkeypatch.setattr(rs, "kaggriculture_grammar_tables", lambda: arrays)
    with pytest.raises(ValueError, match=message):
        gg.native_grammar_tables()


def test_native_loader_checks_constants_then_tables_and_forwards_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    arrays = rs.kaggriculture_grammar_tables()
    constants = rs.kaggriculture_grammar_constants()
    convert = gg.grammar_tables_from_arrays
    requested_device = torch.device("cpu")

    def get_constants() -> tuple[int, tuple[str, ...], tuple[int, ...]]:
        events.append("constants")
        return constants

    def get_tables() -> dict[str, np.ndarray]:
        events.append("tables")
        return arrays

    def from_arrays(
        passed_arrays: dict[str, np.ndarray], *, device: torch.device | str
    ) -> gg.GrammarTables:
        events.append("convert")
        assert passed_arrays is arrays
        assert device is requested_device
        return convert(passed_arrays, device=device)

    monkeypatch.setattr(rs, "kaggriculture_grammar_constants", get_constants)
    monkeypatch.setattr(rs, "kaggriculture_grammar_tables", get_tables)
    monkeypatch.setattr(gg, "grammar_tables_from_arrays", from_arrays)
    gg.native_grammar_tables(device=requested_device)
    assert events == ["constants", "tables", "convert"]


@pytest.mark.parametrize(
    "binding", ["kaggriculture_grammar_constants", "kaggriculture_grammar_tables"]
)
def test_failed_native_binding_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    def fail() -> None:
        raise RuntimeError("native table probe failed")

    monkeypatch.setattr(rs, binding, fail)
    with pytest.raises(RuntimeError, match="native table probe failed"):
        gg.native_grammar_tables()


def _tiny_model(tables: gg.GrammarTables | None = None) -> km.KaggricultureTransformer:
    return km.KaggricultureTransformer(
        km.KaggricultureTransformerConfig(
            embed_dim=32,
            depth=1,
            n_heads=1,
            mlp_ratio=1,
            n_scratch_tokens=1,
            force_flash_attn=False,
        ),
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
        grammar_tables=tables,
    )


def test_model_default_loads_native_tables_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = gg.grammar_tables_from_arrays(rs.kaggriculture_grammar_tables())
    native_tables = rs.kaggriculture_grammar_tables
    calls = 0

    def load() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return native_tables()

    monkeypatch.setattr(rs, "kaggriculture_grammar_tables", load)
    model = _tiny_model()
    assert calls == 1
    for name, table in model.actor.tables().as_dict().items():
        assert torch.equal(table, expected.as_dict()[name]), name
    model.to(device="cpu")
    assert calls == 1


def test_model_explicit_tables_bypass_native_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = gg.expected_grammar_tables()

    def fail() -> None:
        pytest.fail("injected tables must not call the native loader")

    monkeypatch.setattr(rs, "kaggriculture_grammar_constants", fail)
    monkeypatch.setattr(rs, "kaggriculture_grammar_tables", fail)
    model = _tiny_model(expected)
    for name, table in model.actor.tables().as_dict().items():
        assert torch.equal(table, expected.as_dict()[name]), name
