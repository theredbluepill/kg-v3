"""The ``TeacherTargets`` protocol on Isaiah's cached teacher targets (Phase 4 prep)."""

from __future__ import annotations

import pytest
import torch
from owl.model import CachedTeacherDistillationTargets, TeacherTargets
from owl.model.stateless_transformer_v1 import DiscreteTargetPolicyParams

_N_SEGMENTS = 5
_HORIZON = 3


def _params(
    generator: torch.Generator, *, n: int, with_continue: bool
) -> DiscreteTargetPolicyParams:
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn((n, _HORIZON, *shape), generator=generator)

    return DiscreteTargetPolicyParams(
        target_logits=rand(4, 7, 7),
        size_mix_logits=rand(4, 7, 7, 2),
        size_mu=rand(4, 7, 7, 2),
        size_scale=rand(4, 7, 7, 2),
        continue_logits=rand(4, 7) if with_continue else None,
    )


def _targets(
    *,
    seed: int = 0,
    n: int = _N_SEGMENTS,
    with_action_params: bool = True,
    with_continue: bool = True,
    with_winner: bool = True,
) -> CachedTeacherDistillationTargets:
    generator = torch.Generator().manual_seed(seed)
    return CachedTeacherDistillationTargets(
        action_params=(
            _params(generator, n=n, with_continue=with_continue)
            if with_action_params
            else None
        ),
        winner_probabilities=(
            torch.softmax(torch.randn((n, _HORIZON, 4), generator=generator), dim=-1)
            if with_winner
            else None
        ),
    )


def _tensors(
    targets: CachedTeacherDistillationTargets,
) -> dict[str, torch.Tensor | None]:
    params = targets.action_params
    return {
        "winner_probabilities": targets.winner_probabilities,
        "target_logits": None if params is None else params.target_logits,
        "size_mix_logits": None if params is None else params.size_mix_logits,
        "size_mu": None if params is None else params.size_mu,
        "size_scale": None if params is None else params.size_scale,
        "continue_logits": None if params is None else params.continue_logits,
    }


def _assert_equal(
    actual: CachedTeacherDistillationTargets,
    expected: CachedTeacherDistillationTargets,
) -> None:
    assert type(actual) is type(expected)
    actual_tensors = _tensors(actual)
    for name, expected_tensor in _tensors(expected).items():
        actual_tensor = actual_tensors[name]
        if expected_tensor is None:
            assert actual_tensor is None, name
        else:
            assert actual_tensor is not None, name
            assert actual_tensor.dtype == expected_tensor.dtype, name
            assert torch.equal(actual_tensor, expected_tensor), name


_LAYOUTS = [
    pytest.param(True, True, True, id="all"),
    pytest.param(True, False, True, id="no-continue"),
    pytest.param(False, False, True, id="value-only"),
    pytest.param(True, True, False, id="action-only"),
    pytest.param(False, False, False, id="empty"),
]


def test_cached_targets_declare_teacher_targets_protocol() -> None:
    # Explicit subclassing makes mypy check the signatures; ``issubclass`` is
    # unavailable because the protocol is deliberately not runtime-checkable.
    assert TeacherTargets in CachedTeacherDistillationTargets.__mro__


@pytest.mark.parametrize(("with_action", "with_continue", "with_winner"), _LAYOUTS)
def test_index_slices_every_tensor_along_segments(
    with_action: bool, with_continue: bool, with_winner: bool
) -> None:
    targets = _targets(
        with_action_params=with_action,
        with_continue=with_continue,
        with_winner=with_winner,
    )
    idx = torch.tensor([4, 1, 1])

    indexed = targets.index(idx)

    assert type(indexed) is CachedTeacherDistillationTargets
    expected_tensors = {
        name: None if tensor is None else tensor[idx]
        for name, tensor in _tensors(targets).items()
    }
    assert set(_tensors(indexed)) == set(expected_tensors)
    for name, tensor in _tensors(indexed).items():
        expected = expected_tensors[name]
        if expected is None:
            assert tensor is None, name
        else:
            assert tensor is not None, name
            assert torch.equal(tensor, expected), name


@pytest.mark.parametrize(("with_action", "with_continue", "with_winner"), _LAYOUTS)
@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_index_then_concat_round_trips(
    with_action: bool, with_continue: bool, with_winner: bool, chunk_size: int
) -> None:
    targets = _targets(
        with_action_params=with_action,
        with_continue=with_continue,
        with_winner=with_winner,
    )
    chunks = [
        targets.index(torch.arange(start, min(start + chunk_size, _N_SEGMENTS)))
        for start in range(0, _N_SEGMENTS, chunk_size)
    ]

    joined = type(chunks[0]).concat(chunks)

    _assert_equal(joined, targets)


def test_concat_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="empty list of teacher targets"):
        CachedTeacherDistillationTargets.concat([])


_OPTIONAL_TARGET_GAPS = [
    pytest.param(
        {"with_action_params": False, "with_continue": False},
        "target_logits",
        "inconsistent action_params",
        id="action_params",
    ),
    pytest.param(
        {"with_winner": False},
        "winner_probabilities",
        "inconsistent winner_probabilities",
        id="winner_probabilities",
    ),
    pytest.param(
        {"with_continue": False},
        "continue_logits",
        "inconsistent optional tensors",
        id="continue_logits",
    ),
]


@pytest.mark.parametrize(("gap", "field", "message"), _OPTIONAL_TARGET_GAPS)
def test_concat_rejects_later_chunk_missing_a_target_the_first_carries(
    gap: dict[str, bool], field: str, message: str
) -> None:
    del field
    chunks = [_targets(seed=0, n=2), _targets(seed=1, n=3, **gap)]

    with pytest.raises(ValueError, match=message):
        CachedTeacherDistillationTargets.concat(chunks)


@pytest.mark.parametrize(("gap", "field", "message"), _OPTIONAL_TARGET_GAPS)
def test_concat_drops_target_carried_only_by_later_chunks(
    gap: dict[str, bool], field: str, message: str
) -> None:
    """Pin the inherited asymmetry: the first chunk's layout wins, silently.

    A populated target in a later chunk is discarded when the first chunk lacks
    it. This is Isaiah's behavior at 32b3ec9, preserved by the protocol refactor;
    rebuild Phase 4 decides whether ``concat`` should validate symmetrically.
    """
    del message
    first = _targets(seed=0, n=2, **gap)
    later = _targets(seed=1, n=3)
    assert _tensors(first)[field] is None
    assert _tensors(later)[field] is not None

    joined = CachedTeacherDistillationTargets.concat([first, later])

    assert _tensors(joined)[field] is None
    for name, tensor in _tensors(first).items():
        joined_tensor = _tensors(joined)[name]
        if tensor is None:
            assert joined_tensor is None, name
        else:
            later_tensor = _tensors(later)[name]
            assert later_tensor is not None, name
            assert joined_tensor is not None, name
            assert torch.equal(joined_tensor, torch.cat([tensor, later_tensor])), name
