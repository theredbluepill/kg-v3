"""Independent reward/config oracles for the Task 1.4 native ABI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from owl import rs
from owl.kaggriculture.rewards import (
    KaggricultureRewardConfig,
    economic_penalty,
    economic_rewards,
    terminal_rewards,
    transition_rewards,
)
from pydantic import ValidationError

from .test_env_reference import _recorder
from .test_native_env import buffers

if TYPE_CHECKING:
    from owl.rs import KaggricultureRewardDict

_COEFFICIENTS = (
    "econ_shaping",
    "econ_starvation_weight",
    "econ_drought_weight",
    "econ_cap",
    "econ_ineffective_weight",
    "econ_ineffective_cap",
)
_RECIPE = dict(zip(_COEFFICIENTS, (0.2, 4.0, 1.0, 0.25, 0.0, 0.1), strict=True))
# The ten paired cases are the shared ABI table, in exactly its published order.
_ADMISSION_CASES = (
    ((0.2, 4, 1, 0.25, 0, 0), True),
    ((0.2, 0, 1, 0.25, 0, 0), True),
    ((0.2, 0, 0, 0.25, 0, 0), False),
    ((0.2, 4, 1, 0, 0, 0), False),
    ((1e-300, 1e-300, 1e-300, 0.25, 0, 0), False),
    ((1e-300, 1e-300, 1, 0.25, 0, 0), True),
    ((0, 0, 0, 0, 0, 0), True),
    ((0, 4, 1, 0.25, 0, 0), True),
    ((0, 0, 0, 0, 0.001, 0), False),
    ((0, 0, 0, 0, 0.001, 0.1), True),
)
# Beyond the shared table: the one case separating the per-component predicate
# from the pinned reference's combined rule, which accepts inert death shaping
# (W > 0, both death products 0) whenever the ineffective component is active.
_STRENGTHENING_CASES = (((0.2, 0, 0, 0.25, 0.001, 0.1), False),)


def _config(**overrides: float) -> KaggricultureRewardConfig:
    return KaggricultureRewardConfig.model_validate(_RECIPE | overrides)


def test_reward_coefficients_required_and_bounded() -> None:
    with pytest.raises(ValidationError):
        KaggricultureRewardConfig.model_validate({})
    for name in _COEFFICIENTS:
        missing = _RECIPE.copy()
        del missing[name]
        with pytest.raises(ValidationError, match=name):
            KaggricultureRewardConfig.model_validate(missing)
        for bad in (-0.01, float("nan"), float("inf"), -float("inf"), True, "1"):
            with pytest.raises(ValidationError, match=name):
                KaggricultureRewardConfig.model_validate(_RECIPE | {name: bad})
    for extra in ({"extra": 0}, {"reward_mode": "win_loss"}):
        with pytest.raises(ValidationError, match="Extra inputs"):
            KaggricultureRewardConfig.model_validate(_RECIPE | extra)
    for cap in (0.4, 0.5):
        with pytest.raises(ValidationError, match="sum below one"):
            _config(econ_cap=0.6, econ_ineffective_weight=0.1, econ_ineffective_cap=cap)
    disabled = _config(econ_shaping=0, econ_cap=0, econ_ineffective_cap=0)
    assert disabled.terminal_scale == 1
    # Inactive caps have no upper bound beyond finite/nonnegative admission.
    assert (
        _config(econ_shaping=0, econ_cap=2, econ_ineffective_cap=3).terminal_scale == 1
    )
    assert _config().terminal_scale == 0.75
    both = _config(econ_ineffective_weight=0.001)
    assert both.terminal_scale == 1 - 0.25 - 0.1
    native = both.to_native_dict("win_loss")
    assert native == (
        _RECIPE | {"econ_ineffective_weight": 0.001, "reward_mode": "win_loss"}
    )
    with pytest.raises(ValueError, match="reward_mode"):
        both.to_native_dict("win_only")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("coefficients", "accepted"), _ADMISSION_CASES + _STRENGTHENING_CASES
)
def test_reward_admission_predicate_cases(
    coefficients: tuple[float, ...], accepted: bool
) -> None:
    case = dict(zip(_COEFFICIENTS, coefficients, strict=True))
    if accepted:
        assert KaggricultureRewardConfig.model_validate(case).terminal_scale > 0
    else:
        with pytest.raises(ValidationError):
            KaggricultureRewardConfig.model_validate(case)


@pytest.mark.parametrize(
    ("coefficients", "accepted"), _ADMISSION_CASES + _STRENGTHENING_CASES
)
def test_python_and_native_reward_admission_agree(
    coefficients: tuple[float, ...], accepted: bool
) -> None:
    w, starvation, drought, cap, ineffective, ineffective_cap = coefficients
    case: KaggricultureRewardDict = {
        "reward_mode": "win_loss",
        "econ_shaping": w,
        "econ_starvation_weight": starvation,
        "econ_drought_weight": drought,
        "econ_cap": cap,
        "econ_ineffective_weight": ineffective,
        "econ_ineffective_cap": ineffective_cap,
    }
    if accepted:
        config = KaggricultureRewardConfig.model_validate(
            dict(zip(_COEFFICIENTS, coefficients, strict=True))
        )
        assert config.to_native_dict("win_loss") == case
        rs.KaggricultureEnv(1, 0, 1, "{}", case, 1, hire_limit=241)
    else:
        with pytest.raises(ValidationError):
            KaggricultureRewardConfig.model_validate(
                dict(zip(_COEFFICIENTS, coefficients, strict=True))
            )
        with pytest.raises(ValueError, match=r"econ|shaping|weight|cap"):
            rs.KaggricultureEnv(1, 0, 1, "{}", case, 1, hire_limit=241)


@pytest.mark.parametrize("seat", [0, 1])
def test_hand_calculated_starvation_drought_and_ineffective_rewards(seat: int) -> None:
    config = _config(econ_shaping=0.01, econ_ineffective_weight=0.001)
    before = torch.zeros((1, 2, 32), dtype=torch.int64)
    after = before.clone()
    after[0, seat, :3] = torch.tensor([1, 2, 1000])
    expected = torch.zeros((1, 2), dtype=torch.float64)
    expected[0, seat] = 0.16  # .01 * (4 * 1 + 1 * 2) + min(.1, .001 * 1000)
    torch.testing.assert_close(economic_penalty(after, config), expected)
    expected[0, 1 - seat] = 0.16
    expected[0, seat] = -0.16
    torch.testing.assert_close(economic_rewards(before, after, config), expected)
    later = after.clone()
    later[0, seat, 1] += 1
    later[0, seat, 2] += 1000
    expected *= 0.01 / 0.16
    torch.testing.assert_close(economic_rewards(after, later, config), expected)


def test_other_counters_do_not_affect_rewards() -> None:
    before = torch.zeros((3, 2, 32), dtype=torch.int64)
    after = before.clone()
    after[..., 3:] = torch.arange(29) * 9_000_000
    config = _config(econ_ineffective_weight=0.01)
    assert torch.equal(
        economic_penalty(after, config), torch.zeros((3, 2), dtype=torch.float64)
    )
    assert not economic_rewards(before, after, config).any()


def test_reward_cap_crossings_are_cumulative_and_separate() -> None:
    config = _config(econ_shaping=0.01, econ_ineffective_weight=0.001)
    before = torch.zeros((2, 32), dtype=torch.int64)
    before[0, :3] = torch.tensor([5, 0, 99])
    after = before.clone()
    after[0, :3] = torch.tensor([6, 2, 101])
    torch.testing.assert_close(
        economic_penalty(before, config), torch.tensor([0.299, 0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        economic_penalty(after, config), torch.tensor([0.35, 0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        economic_rewards(before, after, config),
        torch.tensor([-0.051, 0.051], dtype=torch.float64),
    )
    saturated = after.clone()
    saturated[0, :3] += 100_000
    assert not economic_rewards(after, saturated, config).any()


def test_large_finite_coefficients_saturate_without_nan() -> None:
    counts = torch.zeros((2, 32), dtype=torch.int64)
    counts[0, :3] = torch.iinfo(torch.int64).max
    for config in (
        _config(
            econ_shaping=1e308,
            econ_starvation_weight=1e308,
            econ_ineffective_weight=1e308,
        ),
        _config(
            econ_shaping=0, econ_starvation_weight=1e308, econ_ineffective_weight=0
        ),
    ):
        penalty = economic_penalty(counts, config)
        assert torch.isfinite(penalty).all()
        assert penalty[1] == 0
        assert penalty[0].item() == pytest.approx(1 - config.terminal_scale)
    # Native evaluates the inner sum first: overflow saturates even for tiny W.
    counts[0, :3] = torch.tensor([10, 0, 0])
    config = _config(
        econ_shaping=1e-320, econ_starvation_weight=1e308, econ_drought_weight=0
    )
    assert economic_penalty(counts, config)[0].item() == 0.25


def test_terminal_sign_tie_and_finite_bank_extremes() -> None:
    config = _config(econ_ineffective_weight=0.01)
    banks = torch.tensor([[2, 1], [1, 2], [2, 2], [1e308, -1e308]], dtype=torch.float64)
    expected = torch.tensor(
        [[0.65, -0.65], [-0.65, 0.65], [0, 0], [0.65, -0.65]], dtype=torch.float64
    )
    torch.testing.assert_close(terminal_rewards(banks, config), expected)


def test_transition_uses_native_float32_rounding_schedule() -> None:
    config = _config(econ_shaping=0.0003, econ_ineffective_weight=0.000165)
    before = torch.zeros((2, 2, 32), dtype=torch.int64)
    after = before.clone()
    after[:, 0, :3] = torch.tensor([1, 0, 1])
    banks = torch.tensor([[2, 1], [1, 2]], dtype=torch.float64)
    dones = torch.tensor([[True, True], [False, False]])
    econ = economic_rewards(before, after, config)
    expected = (econ.float().double() + terminal_rewards(banks, config) * dones).float()
    actual = transition_rewards(before, after, banks, dones, config)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(actual[1], econ[1].float())
    # Exact discriminator: 4*.0003 + .000165 = .001365. Rounding this
    # economic term before adding .65 chooses the adjacent lower f32 value.
    assert actual[0, 0].item() == 0.6486349701881409
    single_round = (econ + terminal_rewards(banks, config) * dones).float()
    assert single_round[0, 0].item() == 0.6486350297927856
    assert not torch.equal(actual, single_round)


@pytest.mark.parametrize("winner", [-1, 0, 1])
@pytest.mark.parametrize("penalized_seat", [0, 1])
def test_complete_719_transition_return_telescopes_with_rounding_budget(
    winner: int, penalized_seat: int
) -> None:
    config = _config(econ_shaping=0.0003, econ_ineffective_weight=0.00017)
    counts = torch.zeros((720, 2, 32), dtype=torch.int64)
    time = torch.arange(720)
    counts[:, penalized_seat, 0] = time // 4
    counts[:, penalized_seat, 1] = time // 3
    counts[:, penalized_seat, 2] = time
    banks = torch.ones((719, 2), dtype=torch.float64)
    if winner != -1:
        banks[:, winner] += 1
    dones = torch.zeros((719, 2), dtype=torch.bool)
    dones[-1] = True
    econ64 = economic_rewards(counts[:-1], counts[1:], config)
    terminal64 = terminal_rewards(banks, config) * dones
    mathematical = econ64 + terminal64
    endpoint = economic_rewards(counts[0], counts[-1], config) + terminal64[-1]
    assert economic_penalty(counts[-1], config)[penalized_seat].item() == pytest.approx(
        0.35
    )
    torch.testing.assert_close(
        mathematical.sum(0), endpoint, atol=8 * torch.finfo(torch.float64).eps, rtol=0
    )
    assert endpoint.abs().max() <= 1
    actual = transition_rewards(counts[:-1], counts[1:], banks, dones, config)

    def half_ulp(values: torch.Tensor) -> torch.Tensor:
        positive = torch.full_like(values, float("inf"))
        negative = -positive
        up = torch.nextafter(values, positive).double() - values.double()
        down = values.double() - torch.nextafter(values, negative).double()
        return torch.maximum(up, down) / 2

    # Each transition rounds its economic difference once; only terminal rows
    # add a second f32 rounding. Include the explicit f64 summation bound.
    budget = half_ulp(econ64.float()).sum(0)
    budget += (half_ulp(actual) * dones).sum(0)
    budget += 719 * torch.finfo(torch.float64).eps * mathematical.abs().sum(0)
    assert torch.all((actual.double().sum(0) - endpoint).abs() <= budget)
    assert torch.all(actual.double().sum(0).abs() <= 1 + budget)


def test_reward_oracle_rejects_invalid_shape_dtype_finite_and_monotonicity() -> None:
    config = _config()
    counts = torch.zeros((1, 2, 32), dtype=torch.int64)
    banks = torch.zeros((1, 2), dtype=torch.float64)
    dones = torch.zeros((1, 2), dtype=torch.bool)
    for bad in (
        counts.float(),
        counts[..., :31],
        counts[:, :1],
        -torch.ones_like(counts),
    ):
        with pytest.raises((TypeError, ValueError)):
            economic_penalty(bad, config)
    for bad in (
        banks.float(),
        banks[:, :1],
        torch.full_like(banks, float("nan")),
        torch.full_like(banks, float("inf")),
    ):
        with pytest.raises((TypeError, ValueError)):
            terminal_rewards(bad, config)
    for bad in (counts.expand(2, 2, 32), counts[..., :31]):
        with pytest.raises(ValueError, match="shape"):
            economic_rewards(counts, bad, config)
    previous = counts.clone()
    previous[..., 31] = 1
    with pytest.raises(ValueError, match="monotonic"):
        economic_rewards(previous, counts, config)
    for bad in (dones.long(), dones[:, :1], dones.expand(2, 2)):
        with pytest.raises((TypeError, ValueError)):
            transition_rewards(counts, counts, banks, bad, config)
    with pytest.raises(ValueError, match="shape"):
        transition_rewards(counts, counts, banks.expand(2, 2), dones, config)


def test_native_fixture_rewards_match_independent_oracle() -> None:
    # Validate source, compressed/expanded hashes, array inventory and trajectory
    # custody before loading any recorded rewards into the independent oracle.
    manifest, fixture = _recorder().load_fixture()
    assert (manifest["games"], manifest["steps_per_game"]) == (16, 719)
    assert fixture["rewards"].shape == (16, 719, 2)
    for game in range(16):
        config = _config(
            econ_shaping=0.02 if game < 8 else 0.2,
            econ_ineffective_weight=0.001 if game < 8 else 0,
        )
        actual = transition_rewards(
            torch.from_numpy(fixture["econ_before"][game]),
            torch.from_numpy(fixture["econ_after"][game]),
            torch.from_numpy(fixture["banks_after"][game]),
            torch.from_numpy(fixture["dones"][game]),
            config,
        ).numpy()
        expected = fixture["rewards"][game]
        # Independent f64 arithmetic may move one f32 ULP; fixture vs live
        # native bit equality is qualified by Task 1.4's full native replay.
        tolerance = np.maximum(
            np.abs(np.nextafter(expected, np.float32(np.inf)) - expected),
            np.abs(expected - np.nextafter(expected, np.float32(-np.inf))),
        )
        assert np.all(np.abs(actual.astype(np.float64) - expected) <= tolerance)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "econ_shaping": 1e308,
            "econ_starvation_weight": 1e308,
            "econ_drought_weight": 1e308,
            "econ_ineffective_weight": 1e308,
            "econ_cap": 0.7,
        },
        {
            "econ_shaping": 0,
            "econ_starvation_weight": 1e308,
            "econ_drought_weight": 1e308,
            "econ_ineffective_weight": 0,
        },
        {
            "econ_shaping": 1e-320,
            "econ_starvation_weight": 1e308,
            "econ_drought_weight": 1e308,
            "econ_ineffective_weight": 0,
        },
    ],
    ids=["overflow-saturation", "disabled-extremes", "tiny-W-overflowing-inner-sum"],
)
def test_extreme_value_native_rewards_match_independent_oracle(overrides) -> None:
    config = _config(**overrides)
    recorder = _recorder()
    env = rs.KaggricultureEnv(
        1,
        17000,
        1,
        '{"episodeSteps":97}',
        config.to_native_dict("win_loss"),
        1,
        hire_limit=241,
    )
    out = buffers(1)
    env.observe(**out)
    for step in range(96):
        public = json.loads(env.state_snapshot(0))["public"]
        tokens = np.zeros((1, 2, 252, 12), dtype=np.int64)
        lengths = np.zeros((1, 2), dtype=np.int64)
        for seat in range(2):
            action = recorder.policy.action(public, seat)
            lengths[0, seat] = rs.kaggriculture_encode(
                json.dumps(action),
                int(out["actor_mask"][0, seat, :241].sum()),
                int(out["order_limits"][0, seat]),
                241,
                tokens[0, seat],
            )
        env.step(tokens, lengths, **out)
        expected = transition_rewards(
            torch.from_numpy(out["transition_econ_before"]),
            torch.from_numpy(out["transition_econ_after"]),
            torch.from_numpy(out["transition_banks_after"]),
            torch.from_numpy(out["dones"]),
            config,
        ).numpy()
        np.testing.assert_array_equal(out["rewards"], expected, err_msg=f"step={step}")
    assert out["dones"].all()
    assert out["transition_econ_after"][0, 0, 0] >= 1
    assert out["transition_econ_after"][0, 0, 1] >= 1
    assert out["transition_econ_after"][0, 0, 2] >= 1


def test_overflow_saturation_keeps_float64_caps() -> None:
    config = _config(
        econ_shaping=1e308,
        econ_starvation_weight=1e308,
        econ_drought_weight=0,
        econ_cap=0.7,
    )
    counts = torch.zeros((2, 32), dtype=torch.int64)
    counts[0, 0] = 2
    penalty = economic_penalty(counts, config)
    assert penalty.dtype == torch.float64
    assert penalty.tolist() == [0.7, 0.0]
