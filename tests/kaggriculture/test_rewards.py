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
    bank_rewards,
    bank_score,
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
    "econ_bank_weight",
    "econ_bank_scale",
    "econ_bank_cap",
)
_BANK_OFF = (0.0, 100_000.0, 0.0)
_RECIPE = dict(
    zip(_COEFFICIENTS, (0.2, 4.0, 1.0, 0.25, 0.0, 0.1, *_BANK_OFF), strict=True)
)
# Owner term A at the bank presets' values (w_b 1, S 100,000, cap_b .25).
_BANK_ON = {
    "econ_bank_weight": 1.0,
    "econ_bank_scale": 100_000.0,
    "econ_bank_cap": 0.25,
}
# The ten paired cases are the shared ABI table, in exactly its published order,
# run with the bank term off.
_ADMISSION_CASES = (
    ((0.2, 4, 1, 0.25, 0, 0, *_BANK_OFF), True),
    ((0.2, 0, 1, 0.25, 0, 0, *_BANK_OFF), True),
    ((0.2, 0, 0, 0.25, 0, 0, *_BANK_OFF), False),
    ((0.2, 4, 1, 0, 0, 0, *_BANK_OFF), False),
    ((1e-300, 1e-300, 1e-300, 0.25, 0, 0, *_BANK_OFF), False),
    ((1e-300, 1e-300, 1, 0.25, 0, 0, *_BANK_OFF), True),
    ((0, 0, 0, 0, 0, 0, *_BANK_OFF), True),
    ((0, 4, 1, 0.25, 0, 0, *_BANK_OFF), True),
    ((0, 0, 0, 0, 0.001, 0, *_BANK_OFF), False),
    ((0, 0, 0, 0, 0.001, 0.1, *_BANK_OFF), True),
)
# Beyond the shared table: the one case separating the per-component predicate
# from the pinned reference's combined rule, which accepts inert death shaping
# (W > 0, both death products 0) whenever the ineffective component is active.
_STRENGTHENING_CASES = (((0.2, 0, 0, 0.25, 0.001, 0.1, *_BANK_OFF), False),)
# Owner term A (2026-09-30), in the Rust and native tables' order: positive
# w_b needs positive S and cap_b, and its cap joins the below-one budget
# (.25 + .1 + .65 sums to exactly 1.0 in binary64). Inactive S and cap_b are
# unconstrained beyond finite/nonnegative.
_BANK_CASES = (
    ((0.2, 4, 1, 0.25, 0, 0.1, 1, 100_000, 0.25), True),
    ((0.2, 4, 1, 0.25, 0, 0.1, 1, 0, 0.25), False),
    ((0.2, 4, 1, 0.25, 0, 0.1, 1, 100_000, 0), False),
    ((0.2, 4, 1, 0.25, 0.001, 0.1, 1, 100_000, 0.65), False),
    ((0.2, 4, 1, 0.25, 0.001, 0.1, 1, 100_000, 0.64), True),
    ((0, 0, 0, 0, 0, 0, 1, 100_000, 1), False),
    ((0, 0, 0, 0, 0, 0, 0, 0, 2), True),
    ((0, 0, 0, 0, 0, 0, 1e-300, 1e300, 0.25), True),
)


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
    # With shaping inactive, only nonnegative admission rejects a negative cap.
    with pytest.raises(ValidationError, match="econ_cap"):
        _config(econ_shaping=0, econ_cap=-0.01)
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
    ("coefficients", "accepted"), _ADMISSION_CASES + _STRENGTHENING_CASES + _BANK_CASES
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
    ("coefficients", "accepted"), _ADMISSION_CASES + _STRENGTHENING_CASES + _BANK_CASES
)
def test_python_and_native_reward_admission_agree(
    coefficients: tuple[float, ...], accepted: bool
) -> None:
    (
        w,
        starvation,
        drought,
        cap,
        ineffective,
        ineffective_cap,
        bank_weight,
        bank_scale,
        bank_cap,
    ) = coefficients
    case: KaggricultureRewardDict = {
        "reward_mode": "win_loss",
        "econ_shaping": w,
        "econ_starvation_weight": starvation,
        "econ_drought_weight": drought,
        "econ_cap": cap,
        "econ_ineffective_weight": ineffective,
        "econ_ineffective_cap": ineffective_cap,
        "econ_bank_weight": bank_weight,
        "econ_bank_scale": bank_scale,
        "econ_bank_cap": bank_cap,
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
    actual = transition_rewards(before, after, banks, banks, dones, config)
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
    actual = transition_rewards(counts[:-1], counts[1:], banks, banks, dones, config)

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
            transition_rewards(counts, counts, banks, banks, bad, config)
    with pytest.raises(ValueError, match="shape"):
        transition_rewards(counts, counts, banks, banks.expand(2, 2), dones, config)
    with pytest.raises(ValueError, match="shape"):
        transition_rewards(counts, counts, banks.expand(2, 2), banks, dones, config)
    for bad_before in (banks.float(), torch.full_like(banks, float("nan"))):
        with pytest.raises((TypeError, ValueError), match="banks_before"):
            transition_rewards(counts, counts, bad_before, banks, dones, config)


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
            torch.from_numpy(fixture["banks_before"][game]),
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
        _BANK_ON,
        _BANK_ON | {"econ_bank_scale": 10_000.0, "econ_bank_cap": 0.29},
        _BANK_ON
        | {"econ_bank_weight": 1e308, "econ_bank_scale": 1e-300, "econ_bank_cap": 0.2},
    ],
    ids=[
        "overflow-saturation",
        "disabled-extremes",
        "tiny-W-overflowing-inner-sum",
        "bank-presets",
        "bank-cap-binds-near-the-start-bank",
        "bank-overflow-saturates",
    ],
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
            torch.from_numpy(out["transition_banks_before"]),
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


# --- owner term A: absolute own-bank shaping (2026-09-30) ---------------------


def _bank_only(**overrides: float) -> KaggricultureRewardConfig:
    return _config(econ_shaping=0, **(_BANK_ON | overrides))


def test_bank_score_is_own_bank_scaled_and_capped() -> None:
    config = _bank_only()
    banks = torch.tensor(
        [[3_000, 5_000], [25_000, 90_000], [-400, 0], [1e308, 12_345]],
        dtype=torch.float64,
    )
    assert bank_score(banks, config).tolist() == [
        [0.03, 0.05],
        [0.25, 0.25],
        [0.0, 0.0],
        [0.25, 0.12345],
    ]
    # Disabled, the score is zero whatever the scale and cap say.
    off = _config(econ_bank_scale=1.0, econ_bank_cap=0.9)
    assert not bank_score(banks, off).any()
    # An overflowing product saturates at the cap, never NaN.
    huge = _bank_only(econ_bank_weight=1e308, econ_bank_scale=1e-300)
    assert bank_score(banks[:1], huge).tolist() == [[0.25, 0.25]]


def test_bank_growth_pays_and_spending_costs() -> None:
    config = _bank_only()
    zero = torch.zeros((2, 2, 32), dtype=torch.int64)
    before = torch.tensor([[3_000, 3_000], [5_000, 3_000]], dtype=torch.float64)
    after = torch.tensor([[5_000, 3_000], [4_000, 3_000]], dtype=torch.float64)
    dones = torch.zeros((2, 2), dtype=torch.bool)
    rewards = transition_rewards(zero, zero, before, after, dones, config)
    assert rewards[0, 0].item() > 0
    assert rewards[1, 0].item() < 0
    assert rewards[:, 1].tolist() == [0.0, 0.0]
    torch.testing.assert_close(
        bank_rewards(before, after, config),
        torch.tensor([[0.02, 0.0], [-0.01, 0.0]], dtype=torch.float64),
    )
    # The cap binds: 20k -> 30k pays .05, 30k -> 90k pays nothing.
    capped = bank_rewards(
        torch.tensor([20_000.0, 30_000.0], dtype=torch.float64),
        torch.tensor([30_000.0, 90_000.0], dtype=torch.float64),
        config,
    )
    assert capped.tolist() == [0.25 - 0.2, 0.0]


def test_bank_term_is_own_seat_only() -> None:
    config = _config(**_BANK_ON)
    before = torch.full((4, 2), 3_000.0, dtype=torch.float64)
    after = torch.tensor(
        [[0, 7_000], [3_000, 7_000], [12_345, 7_000], [1e9, 7_000]],
        dtype=torch.float64,
    )
    counts = torch.zeros((4, 2, 32), dtype=torch.int64)
    dones = torch.zeros((4, 2), dtype=torch.bool)
    seat1 = transition_rewards(counts, counts, before, after, dones, config)[:, 1]
    assert torch.equal(seat1, seat1[:1].expand(4))
    assert seat1[0].item() == np.float32(0.07 - 0.03)


def test_bank_cap_reduces_the_terminal_scale() -> None:
    config = _config(**_BANK_ON)
    assert config.terminal_scale == 1 - 0.25 - 0.25 == 0.5
    both = _config(econ_ineffective_weight=0.001, **_BANK_ON)
    assert both.terminal_scale == 1 - 0.25 - 0.1 - 0.25
    counts = torch.zeros((1, 2, 32), dtype=torch.int64)
    banks = torch.tensor([[40_000.0, 10_000.0]], dtype=torch.float64)
    dones = torch.ones((1, 2), dtype=torch.bool)
    rewards = transition_rewards(counts, counts, banks, banks, dones, config)
    assert rewards.tolist() == [[0.5, -0.5]]


def test_bank_admission_rejects_bad_configs() -> None:
    for name in ("econ_bank_weight", "econ_bank_scale", "econ_bank_cap"):
        for bad in (-0.01, float("nan"), float("inf"), True, "1"):
            with pytest.raises(ValidationError, match=name):
                _config(**(_BANK_ON | {name: bad}))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="econ_bank_scale"):
        _config(**(_BANK_ON | {"econ_bank_scale": 0.0}))
    with pytest.raises(ValidationError, match="econ_bank_cap"):
        _config(**(_BANK_ON | {"econ_bank_cap": 0.0}))
    with pytest.raises(ValidationError, match="sum below one"):
        _config(econ_ineffective_weight=0.001, **(_BANK_ON | {"econ_bank_cap": 0.65}))
    # Each bank key is required.
    for name in ("econ_bank_weight", "econ_bank_scale", "econ_bank_cap"):
        missing = _RECIPE.copy()
        del missing[name]
        with pytest.raises(ValidationError, match=name):
            KaggricultureRewardConfig.model_validate(missing)


def test_disabled_bank_term_is_bit_identical_to_the_relative_reward() -> None:
    # w_b = 0: banks_before and inactive scale/cap never change a single bit.
    before = torch.zeros((6, 2, 32), dtype=torch.int64)
    after = before.clone()
    after[:, 0, :3] = torch.tensor([3, 1, 50])
    after[:, 1, 1] = 1
    banks_after = torch.tensor(
        [[10, 5], [5, 10], [7, 7], [-3, 1e300], [2, 1], [1, 2]], dtype=torch.float64
    )
    dones = torch.tensor([[True] * 2] * 3 + [[False] * 2] * 3)
    for scale, cap in ((100_000.0, 0.0), (0.0, 0.0), (1.0, 0.9)):
        config = _config(
            econ_shaping=0.0003,
            econ_ineffective_weight=0.000165,
            econ_bank_scale=scale,
            econ_bank_cap=cap,
        )
        legacy = (
            economic_rewards(before, after, config).float().double()
            + terminal_rewards(banks_after, config) * dones
        ).float()
        for banks_before in (banks_after, torch.zeros_like(banks_after), -banks_after):
            actual = transition_rewards(
                before, after, banks_before, banks_after, dones, config
            )
            assert torch.equal(actual.view(torch.int32), legacy.view(torch.int32))


def test_bank_episode_return_telescopes_to_final_minus_start_score() -> None:
    config = _config(**_BANK_ON)
    time = torch.arange(720, dtype=torch.float64)
    banks = torch.stack(
        (
            3_000 - 20 * time.clamp(max=100) + 60 * (time - 100).clamp(min=0),
            3_000 + 7.5 * time - 200 * torch.sin(time * 0.37),
        ),
        dim=-1,
    )
    counts = torch.zeros((720, 2, 32), dtype=torch.int64)
    counts[:, 1, 0] = time.long() // 150
    dones = torch.zeros((719, 2), dtype=torch.bool)
    dones[-1] = True
    actual = transition_rewards(
        counts[:-1], counts[1:], banks[:-1], banks[1:], dones, config
    )
    endpoint = (
        economic_rewards(counts[0], counts[-1], config)
        + bank_rewards(banks[0], banks[-1], config)
        + terminal_rewards(banks[-1], config)
    )
    economic64 = economic_rewards(counts[:-1], counts[1:], config) + bank_rewards(
        banks[:-1], banks[1:], config
    )
    # One f32 rounding per transition, a second on the terminal row, plus the
    # float64 summation bound.
    half_ulp = (
        torch.nextafter(economic64.float(), torch.tensor(float("inf"))).double()
        - economic64.float().double()
    ).abs()
    budget = half_ulp.sum(0) + (actual[-1].double().abs() * 2**-24)
    budget += 719 * torch.finfo(torch.float64).eps * economic64.abs().sum(0)
    assert torch.all((actual.double().sum(0) - endpoint).abs() <= budget)
    assert bank_score(banks[-1], config)[0].item() == 0.25
    assert bool((banks[100, 0] < banks[0, 0]).item())
    assert endpoint.abs().max() <= 1


def test_autoreset_never_spans_two_games_in_the_native_bank_term() -> None:
    config = _bank_only()
    env = rs.KaggricultureEnv(
        1, 17000, 1, '{"episodeSteps":4}', config.to_native_dict("win_loss"), 1,
        hire_limit=241,
    )  # fmt: skip
    recorder = _recorder()
    out = buffers(1)
    env.observe(**out)
    first_terminal = None
    for step in range(6):
        public = json.loads(env.state_snapshot(0))["public"]
        tokens = np.zeros((1, 2, 252, 12), dtype=np.int64)
        lengths = np.zeros((1, 2), dtype=np.int64)
        for seat in range(2):
            lengths[0, seat] = rs.kaggriculture_encode(
                json.dumps(recorder.policy.action(public, seat)),
                int(out["actor_mask"][0, seat, :241].sum()),
                int(out["order_limits"][0, seat]),
                241,
                tokens[0, seat],
            )
        env.step(tokens, lengths, **out)
        before = torch.from_numpy(out["transition_banks_before"].copy())
        after = torch.from_numpy(out["transition_banks_after"].copy())
        expected = transition_rewards(
            torch.from_numpy(out["transition_econ_before"]),
            torch.from_numpy(out["transition_econ_after"]),
            before,
            after,
            torch.from_numpy(out["dones"]),
            config,
        ).numpy()
        np.testing.assert_array_equal(out["rewards"], expected, err_msg=f"{step=}")
        if first_terminal is not None and step == first_terminal + 1:
            # The next game's first transition starts from its reset bank.
            assert before.tolist() == [[3_000.0, 3_000.0]]
        if bool(out["dones"].all()) and first_terminal is None:
            # Discriminating: the first game ended away from the reset bank.
            assert after[0, 0].item() != 3_000.0
            first_terminal = step
    assert first_terminal is not None
    assert first_terminal + 1 < 6
