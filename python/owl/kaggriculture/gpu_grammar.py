"""Kaggriculture grammar support tables for the batched action heads.

``GrammarTables`` holds the eight factored boolean tables of contract v4
"Action" (``docs/kaggriculture-contract.md``). The action heads index them with
the frame's already-chosen values to get each slot's prefix-dependent support
mask; the runtime overlays (unit liveness, queue availability, HIRE capacity and
STOP marginalization) are applied by the heads, not stored here.

The native grammar (Task 1.2's ``grammar_tables()`` exposed by Task 1.4's
``kaggriculture_grammar_tables`` binding) is the single source of truth.
:func:`native_grammar_tables` validates that ABI before copying its tables.
:func:`expected_grammar_tables` independently builds the same tables from the
reference support rules as a test oracle only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from owl import rs
from owl.kaggriculture import types as kt

UNIT_KIND_WIDTH = kt.SLOT_WIDTHS[1]
ITEM_WIDTH = kt.SLOT_WIDTHS[3]
PRESENT_WIDTH = kt.SLOT_WIDTHS[4]
DIGIT_WIDTH = kt.SLOT_WIDTHS[5]
MARKET_KIND_WIDTH = kt.SLOT_WIDTHS[7]

MARKET_NONE = kt.MARKET_KINDS.index("NONE")
MARKET_HIRE = kt.MARKET_KINDS.index("HIRE")
MARKET_EMPTY = kt.MARKET_KINDS.index("EMPTY")

# Field name -> exact table shape. Names match Task 1.2's native ``GrammarTables``.
TABLE_SHAPES: dict[str, tuple[int, ...]] = {
    "unit_kind": (UNIT_KIND_WIDTH,),
    "unit_item": (UNIT_KIND_WIDTH, ITEM_WIDTH),
    "unit_quantity_present": (UNIT_KIND_WIDTH, PRESENT_WIDTH),
    "unit_quantity_high": (PRESENT_WIDTH, DIGIT_WIDTH),
    "unit_quantity_low": (PRESENT_WIDTH, 2, DIGIT_WIDTH),
    "market_kind": (MARKET_KIND_WIDTH,),
    "market_item": (MARKET_KIND_WIDTH, ITEM_WIDTH),
    "market_quantity": (MARKET_KIND_WIDTH, DIGIT_WIDTH),
}


@dataclass(frozen=True)
class GrammarTables:
    """Factored grammar support tables (``torch.bool``, one device).

    Indexing (Task 2.3 brief §2):

    - ``unit_item[kind]``, ``unit_quantity_present[kind]``
    - ``unit_quantity_high[present]``
    - ``unit_quantity_low[present, high == 0]`` (index 1 = the high digit is 0)
    - ``market_item[kind]``; ``market_quantity[kind]`` serves both digit slots

    Construct through :func:`validate_grammar_tables` or a builder; the class
    itself carries no checks so the compiled head core can rebuild it freely.
    """

    unit_kind: torch.Tensor
    unit_item: torch.Tensor
    unit_quantity_present: torch.Tensor
    unit_quantity_high: torch.Tensor
    unit_quantity_low: torch.Tensor
    market_kind: torch.Tensor
    market_item: torch.Tensor
    market_quantity: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "unit_kind": self.unit_kind,
            "unit_item": self.unit_item,
            "unit_quantity_present": self.unit_quantity_present,
            "unit_quantity_high": self.unit_quantity_high,
            "unit_quantity_low": self.unit_quantity_low,
            "market_kind": self.market_kind,
            "market_item": self.market_item,
            "market_quantity": self.market_quantity,
        }


def validate_grammar_tables(tables: GrammarTables) -> GrammarTables:
    """Check dtypes, shapes, device and the invariants the heads rely on.

    - every table row admits at least one value, so each masked categorical
      has support (placeholder rows for unreachable prefixes included)
    - market ``NONE`` is admitted, so STOP is reachable at every queue position
    - market ``HIRE`` is not the only admitted kind, so the capacity correction
      always has a non-HIRE outcome
    """
    fields = tables.as_dict()
    device = tables.unit_kind.device
    for name, shape in TABLE_SHAPES.items():
        table = fields[name]
        if table.dtype != torch.bool:
            raise ValueError(
                f"grammar table {name} must be torch.bool, got {table.dtype}"
            )
        if tuple(table.shape) != shape:
            raise ValueError(
                f"grammar table {name} must have shape {shape}, "
                f"got {tuple(table.shape)}"
            )
        if table.device != device:
            raise ValueError(
                f"grammar table {name} is on {table.device}, expected {device}"
            )
        if not bool(table.any(dim=-1).all()):
            raise ValueError(f"grammar table {name} has a row with no admitted value")
    if not bool(tables.market_kind[MARKET_NONE]):
        raise ValueError("grammar table market_kind must admit NONE (STOP)")
    others = tables.market_kind.clone()
    others[MARKET_HIRE] = False
    if not bool(others.any()):
        raise ValueError("grammar table market_kind must admit a kind other than HIRE")
    return tables


def grammar_tables_digest(tables: GrammarTables) -> str:
    """SHA-256 over every table's name, shape and values, in ``TABLE_SHAPES`` order.

    A host-side identity of the grammar: two models whose digests (and
    ``hire_limit``) agree build identical replay masks. Reading the tables
    copies them to the host, so callers take it once, at construction.
    """
    digest = hashlib.sha256()
    fields = tables.as_dict()
    for name in TABLE_SHAPES:
        table = fields[name].detach().to(device="cpu", dtype=torch.bool)
        digest.update(name.encode())
        digest.update(repr(tuple(table.shape)).encode())
        digest.update(table.contiguous().numpy().tobytes())
    return digest.hexdigest()


def grammar_tables_from_arrays(
    arrays: Mapping[str, np.ndarray | torch.Tensor],
    *,
    device: torch.device | str = "cpu",
) -> GrammarTables:
    """Typed tables from exactly the eight named boolean arrays.

    This is the conversion the native binding feeds (Task 1.2 brief: C-contiguous
    ``numpy.bool_`` arrays, no env/seat dimension). Non-bool input is rejected
    rather than cast.
    """
    missing = sorted(set(TABLE_SHAPES) - set(arrays))
    extra = sorted(set(arrays) - set(TABLE_SHAPES))
    if missing or extra:
        raise ValueError(
            f"grammar tables need exactly {sorted(TABLE_SHAPES)}; "
            f"missing {missing}, unexpected {extra}"
        )
    converted: dict[str, torch.Tensor] = {}
    for name in TABLE_SHAPES:
        value = arrays[name]
        tensor = (
            value if isinstance(value, torch.Tensor) else torch.from_numpy(value.copy())
        )
        if tensor.dtype != torch.bool:
            raise ValueError(f"grammar table {name} must be bool, got {tensor.dtype}")
        converted[name] = tensor.to(device=device)
    return validate_grammar_tables(GrammarTables(**converted))


def _mask(width: int, allowed: Collection[int]) -> list[bool]:
    return [index in allowed for index in range(width)]


def expected_grammar_tables(device: torch.device | str = "cpu") -> GrammarTables:
    """The tables the Task 2.3 brief derives from the reference grammar rules.

    Independent test oracle for :func:`native_grammar_tables`:

    - unit kind ``1..18`` (PASS … DIG)
    - unit item ``1..12`` for PICKUP/PLACE (6/7), ``1..5`` for PLANT (8), else
      ``{0}``; rows 0 and 19 are ``{0}`` placeholders (kind admission rejects
      them first)
    - only PICKUP/PLACE may set ``present = 1``; an absent quantity forces both
      digits to 0; a present quantity excludes low digit 0 only when the high
      digit is 0 (so an explicit unit zero is rejected, 1..1023 accepted)
    - market item ``1..5`` / ``{1, 9}`` / ``10..12`` / ``1..9`` for BUY_SEED /
      BUY_PRODUCT / BUY_ANIMAL / SELL, else ``{0}``
    - market quantity digits ``0..31`` for kinds 3..6 (explicit zero accepted),
      else ``{0}``
    """
    unit_kind = _mask(UNIT_KIND_WIDTH, range(1, 19))
    transfer = (kt.UNIT_KINDS.index("PICKUP"), kt.UNIT_KINDS.index("PLACE"))
    plant = kt.UNIT_KINDS.index("PLANT")
    unit_item: list[list[bool]] = []
    unit_present: list[list[bool]] = []
    for kind in range(UNIT_KIND_WIDTH):
        if kind in transfer:
            unit_item.append(_mask(ITEM_WIDTH, range(1, kt.ITEM_COUNT + 1)))
            unit_present.append(_mask(PRESENT_WIDTH, (0, 1)))
        elif kind == plant:
            unit_item.append(_mask(ITEM_WIDTH, range(1, len(kt.CROPS))))
            unit_present.append(_mask(PRESENT_WIDTH, (0,)))
        else:
            unit_item.append(_mask(ITEM_WIDTH, (0,)))
            unit_present.append(_mask(PRESENT_WIDTH, (0,)))
    digits = range(DIGIT_WIDTH)
    unit_high = [_mask(DIGIT_WIDTH, (0,)), _mask(DIGIT_WIDTH, digits)]
    unit_low = [
        [_mask(DIGIT_WIDTH, (0,)), _mask(DIGIT_WIDTH, (0,))],
        [_mask(DIGIT_WIDTH, digits), _mask(DIGIT_WIDTH, range(1, DIGIT_WIDTH))],
    ]
    market_kind = _mask(MARKET_KIND_WIDTH, range(MARKET_KIND_WIDTH))
    item_support = {
        kt.MARKET_KINDS.index("BUY_SEED"): range(1, 6),
        kt.MARKET_KINDS.index("BUY_PRODUCT"): (1, 9),
        kt.MARKET_KINDS.index("BUY_ANIMAL"): range(10, 13),
        kt.MARKET_KINDS.index("SELL"): range(1, 10),
    }
    market_item = [
        _mask(ITEM_WIDTH, item_support.get(kind, (0,)))
        for kind in range(MARKET_KIND_WIDTH)
    ]
    market_quantity = [
        _mask(DIGIT_WIDTH, digits if kind in item_support else (0,))
        for kind in range(MARKET_KIND_WIDTH)
    ]

    def tensor(values: object) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.bool, device=device)

    return validate_grammar_tables(
        GrammarTables(
            unit_kind=tensor(unit_kind),
            unit_item=tensor(unit_item),
            unit_quantity_present=tensor(unit_present),
            unit_quantity_high=tensor(unit_high),
            unit_quantity_low=tensor(unit_low),
            market_kind=tensor(market_kind),
            market_item=tensor(market_item),
            market_quantity=tensor(market_quantity),
        )
    )


def native_grammar_tables(device: torch.device | str = "cpu") -> GrammarTables:
    """Validate and copy the native grammar's tables to the requested device.

    Constants are admitted before table retrieval. The native arrays must
    match the exact boolean C layout; malformed or missing bindings fail.
    Call once at model construction, not for individual observations.
    """
    version, names, widths = rs.kaggriculture_grammar_constants()
    if (
        type(version) is not int
        or version != 1
        or not isinstance(names, tuple)
        or names != kt.SLOT_NAMES
        or not isinstance(widths, tuple)
        or any(type(width) is not int for width in widths)
        or widths != kt.SLOT_WIDTHS
    ):
        raise ValueError(
            "native grammar constants must match version 1 and the exact "
            f"slot names/widths; got {(version, names, widths)!r}"
        )
    arrays = rs.kaggriculture_grammar_tables()
    missing = sorted(set(TABLE_SHAPES) - set(arrays))
    extra = sorted(set(arrays) - set(TABLE_SHAPES))
    if missing or extra:
        raise ValueError(
            f"native grammar tables need exactly {sorted(TABLE_SHAPES)}; "
            f"missing {missing}, unexpected {extra}"
        )
    for name, shape in TABLE_SHAPES.items():
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ValueError(f"native grammar table {name} must be a NumPy array")
        if array.dtype != np.bool_:
            raise ValueError(f"native grammar table {name} must be bool")
        if array.shape != shape:
            raise ValueError(
                f"native grammar table {name} must have shape {shape}, "
                f"got {array.shape}"
            )
        if not array.flags.c_contiguous:
            raise ValueError(f"native grammar table {name} must be C-contiguous")
    return grammar_tables_from_arrays(arrays, device=device)
