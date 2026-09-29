"""Phase 4: Kaggriculture teacher distillation (brief ``briefs/4-teacher.md`` v2).

Tiny CPU models on the synthetic ``expected_grammar_tables``. The 4.1 block pins
the replay-conditioned per-slot KL computed inside ``policy_core``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model import kaggriculture_actor as ka
from owl.model import kaggriculture_teacher as kt_teacher
from owl.model.stateless_transformer_v1 import (
    CachedTeacherDistillationTargets,
    DiscreteTargetPolicyParams,
)
from owl.train import ppo

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import (
    HIRE,
    NONE,
    F,
    K,
    _base_case,
    _map_obs,
    _obs_double,
    _replay_case,
    _take_actions,
    _take_obs,
    _tiny,
)

CASES = ["dense", "hire_capacity", "stop_every_position", "mixed"]
UNIT_FRAMES = kt.MAX_ACTORS
POSITIONS = ka.MARKET_POSITIONS
IMPLICIT_SLOTS = [0, 2, 11]


# --- helpers --------------------------------------------------------------------------


def _sampled(model: Any, obs: kt.KaggricultureObsBatch) -> kt.KaggricultureActions:
    torch.manual_seed(11)
    with torch.no_grad():
        out = model(obs)
    return kt.KaggricultureActions(
        tokens=out.actions.tokens.clone(), lengths=out.actions.lengths.clone()
    )


def _collect(
    model: Any, obs: kt.KaggricultureObsBatch, actions: kt.KaggricultureActions
) -> ka.GrammarPolicyResult:
    encoded = model.encode_observations(obs)
    return model._policy(
        encoded,
        model._grammar_context(obs),
        actions,
        deterministic=False,
        collect_logits=True,
    )


def _with_teacher(
    student: Any,
    teacher_logits: dict[int, torch.Tensor],
    obs: kt.KaggricultureObsBatch,
    actions: kt.KaggricultureActions,
) -> ka.GrammarPolicyResult:
    encoded = student.encode_observations(obs)
    return student._policy(
        encoded,
        student._grammar_context(obs),
        actions,
        deterministic=False,
        teacher_logits=teacher_logits,
    )


def _teacher_kl(
    student: Any, teacher: Any, obs: kt.KaggricultureObsBatch, actions: Any
) -> torch.Tensor:
    with torch.no_grad():
        logits = _collect(teacher, obs, actions).slot_logits
    assert logits is not None
    result = _with_teacher(student, logits, obs, actions)
    assert result.kl is not None
    return result.kl


def _layout(
    obs: kt.KaggricultureObsBatch, actions: kt.KaggricultureActions
) -> dict[str, torch.Tensor]:
    """Row-layout liveness, stop and market frames of a replayed batch."""
    counts = obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1).reshape(-1)
    live = obs.still_playing.reshape(-1)
    lengths = actions.lengths.reshape(-1)
    limits = obs.order_limits.reshape(-1)
    frames = torch.arange(UNIT_FRAMES)
    positions = torch.arange(POSITIONS)
    stop = torch.where(live, lengths - counts - 1, -1)
    return {
        "counts": counts,
        "live": live,
        "limits": limits,
        "stop": stop,
        "unit_live": live[:, None] & (frames[None, :] < counts[:, None]),
        "market_live": live[:, None] & (positions[None, :] <= stop[:, None]),
        "market_frame": counts[:, None] + positions[None, :],
    }


def _market_view(event: torch.Tensor, market_frame: torch.Tensor) -> torch.Tensor:
    """``[R, 252, 12]`` -> ``[R, 11, 12]`` at the market frames."""
    return event.gather(1, market_frame[..., None].expand(-1, -1, K))


# --- T1 identity and degenerate zeros -------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_kl_is_zero_for_the_same_teacher_and_exactly_zero_off_support(
    case: str,
) -> None:
    model, obs = _replay_case(case)
    model.eval()
    actions = _sampled(model, obs)
    with torch.no_grad():
        same = _teacher_kl(model, model, obs, actions)
    assert same.shape == (obs.still_playing.numel(), F, K)
    assert torch.equal(same, torch.zeros_like(same))

    teacher = _tiny(seed=6).eval()
    with torch.no_grad():
        kl = _teacher_kl(model, teacher, obs, actions)
    layout = _layout(obs, actions)
    assert bool(torch.isfinite(kl).all())
    # Non-negative up to FP32 rounding of sum p_T (log p_T - log p_S): near-equal
    # distributions measured -5.6e-8 against terms of order 1e-5.
    assert float(kl.min()) >= -1e-6, float(kl.min())
    assert bool((kl[:, :, IMPLICIT_SLOTS] == 0).all())
    unit = kl[:, :UNIT_FRAMES, list(ka.UNIT_POLICY_SLOTS)]
    assert bool((unit[~layout["unit_live"]] == 0).all())
    assert bool((unit[layout["unit_live"]] > 0).any())
    # Frames after the market block hold nothing.
    beyond = torch.arange(F)[None, :] > (layout["counts"] + layout["stop"])[:, None]
    assert bool((kl[beyond] == 0).all())
    market = _market_view(kl, layout["market_frame"])[..., list(ka.MARKET_POLICY_SLOTS)]
    positions = torch.arange(POSITIONS)[None, :]
    # After STOP, at the forced sentinel, at unavailable positions: exactly 0.
    assert bool((market[~layout["market_live"]] == 0).all())
    sentinel = layout["live"][:, None] & (positions == layout["limits"][:, None])
    assert bool((market[sentinel] == 0).all())
    unavailable = positions >= layout["limits"][:, None]
    assert bool((market[unavailable] == 0).all())
    # At STOP the item and digit supports are {0}; the STOP decision is distilled.
    at_stop = layout["live"][:, None] & (positions == layout["stop"][:, None])
    assert bool((market[..., 1:][at_stop] == 0).all())
    available = layout["market_live"] & ~unavailable
    if bool(available.any()):
        assert bool((market[..., 0][available] > 0).any())
    assert bool((kl[~layout["live"]] == 0).all())
    if case == "mixed":
        assert int((~layout["live"]).sum()) == 2


# --- T2 collected logits reproduce the density ----------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("double", [False, True], ids=["fp32", "fp64"])
def test_collected_logits_reproduce_the_replay_density(case: str, double: bool) -> None:
    model, obs = _replay_case(case)
    model.eval()
    actions = _sampled(model, obs)
    if double:
        model = model.double()
        obs = _obs_double(obs)
    with torch.no_grad():
        result = _collect(model, obs, actions)
        event = model.evaluate_actions(obs, actions).log_probs.event.reshape(-1, F, K)
    assert result.slot_logits is not None
    assert set(result.slot_logits) == set(ka.POLICY_SLOTS)
    dtype = torch.float64 if double else torch.float32
    tokens = actions.tokens.reshape(-1, F, K)
    layout = _layout(obs, actions)
    for slot in ka.UNIT_POLICY_SLOTS:
        logits = result.slot_logits[slot]
        assert logits.dtype == dtype
        assert logits.shape == (tokens.shape[0], UNIT_FRAMES, kt.SLOT_WIDTHS[slot])
        chosen = (
            functional.log_softmax(logits, -1)
            .gather(-1, tokens[:, :UNIT_FRAMES, slot, None])
            .squeeze(-1)
        )
        mask = layout["unit_live"]
        assert torch.equal(chosen[mask], event[:, :UNIT_FRAMES, slot][mask]), slot
    market_tokens = _market_view(tokens, layout["market_frame"])
    market_event = _market_view(event, layout["market_frame"])
    for slot in ka.MARKET_POLICY_SLOTS:
        logits = result.slot_logits[slot]
        assert logits.dtype == dtype
        assert logits.shape == (tokens.shape[0], POSITIONS, kt.SLOT_WIDTHS[slot])
        assert bool(torch.isfinite(logits).all())
        chosen = (
            functional.log_softmax(logits, -1)
            .gather(-1, market_tokens[..., slot, None])
            .squeeze(-1)
        )
        mask = layout["market_live"]
        assert torch.equal(chosen[mask], market_event[..., slot][mask]), slot


def test_forward_and_evaluate_actions_leave_the_teacher_fields_unset() -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    with torch.no_grad():
        encoded = model.encode_observations(obs)
        plain = model._policy(
            encoded, model._grammar_context(obs), actions, deterministic=False
        )
    assert plain.slot_logits is None
    assert plain.kl is None


# --- T3 brute-force oracle ------------------------------------------------------------


def _admissible(
    tables: gg.GrammarTables,
    slot: int,
    row_tokens: torch.Tensor,
    frame: int,
    *,
    position: int | None,
    counts: int,
    limit: int,
    hire_limit: int,
) -> list[int]:
    """The admissible values at one slot, from the tables and the replayed prefix."""
    token = row_tokens[frame]
    if slot == 1:
        mask = tables.unit_kind
    elif slot == 3:
        mask = tables.unit_item[int(token[1])]
    elif slot == 4:
        mask = tables.unit_quantity_present[int(token[1])]
    elif slot == 5:
        mask = tables.unit_quantity_high[int(token[4])]
    elif slot == 6:
        mask = tables.unit_quantity_low[int(token[4]), int(int(token[5]) == 0)]
    elif slot == 7:
        assert position is not None
        if position >= min(limit, kt.MAX_ORDER_LIMIT):
            return [NONE]
        prior = int((row_tokens[counts : counts + position, 7] == HIRE).sum())
        mask = tables.market_kind.clone()
        if counts + prior >= hire_limit:
            mask[HIRE] = False
    elif slot == 8:
        mask = tables.market_item[int(token[7])]
    else:
        mask = tables.market_quantity[int(token[7])]
    return [int(v) for v in mask.nonzero().flatten()]


def _variant_log_probs(
    model: Any, obs: kt.KaggricultureObsBatch, rows: list[int], tokens: torch.Tensor
) -> torch.Tensor:
    """``policy_core`` log-probs with supplied variant tokens (``valid`` ignored)."""
    with torch.no_grad():
        encoded = model.encode_observations(obs)
        context = model._grammar_context(obs)
        unit_input, market_input = model._actor_inputs(encoded, slice(None))
        index = torch.tensor(rows)
        result = model.actor.policy_core(
            unit_input[index],
            market_input[index],
            context.actor_counts[index],
            context.order_limits[index],
            context.live[index],
            model.action_spec.hire_limit,
            tokens,
            torch.zeros(len(rows), dtype=torch.int64),
            False,
        )
    return result.log_probs


def _small_hire_case() -> tuple[Any, Any, kt.KaggricultureObsBatch]:
    """HIRE budgets 2, 3, 1 and 1 exhausted mid-queue under ``hire_limit=5``."""
    models = []
    for seed in (5, 6):
        model = _tiny(hire_limit=5, seed=seed).eval()
        with torch.no_grad():
            bias = model.actor.heads["market_kind"].out.bias
            bias[HIRE] = 4.0
            bias[NONE] = -6.0
        models.append(model)
    obs = make_obs(envs=2, own_actors=(3, 4), rival_actors=(2, 4), order_limit=6)
    return models[0], models[1], obs


@pytest.mark.parametrize("case", ["base", "hire"])
def test_kl_matches_a_brute_force_oracle_over_the_admissible_values(case: str) -> None:
    if case == "base":
        student, teacher = _tiny().eval(), _tiny(seed=6).eval()
        obs, actions = _base_case()
    else:
        student, teacher, obs = _small_hire_case()
        actions = _sampled(student, obs)
        hires = (actions.tokens[..., 7] == HIRE).sum(-1).reshape(-1)
        budget = 5 - obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1).reshape(-1)
        assert bool((hires == budget).all()), (hires, budget)
    with torch.no_grad():
        kl = _teacher_kl(student, teacher, obs, actions)
    tokens = actions.tokens.reshape(-1, F, K)
    layout = _layout(obs, actions)
    tables = student.actor.tables()
    hire_limit = student.action_spec.hire_limit
    sites: list[tuple[int, int, int, list[int]]] = []  # row, frame, slot, values
    for row in range(tokens.shape[0]):
        counts = int(layout["counts"][row])
        if not bool(layout["live"][row]):
            continue
        for frame in range(counts):
            for slot in ka.UNIT_POLICY_SLOTS:
                values = _admissible(
                    tables,
                    slot,
                    tokens[row],
                    frame,
                    position=None,
                    counts=counts,
                    limit=int(layout["limits"][row]),
                    hire_limit=hire_limit,
                )
                sites.append((row, frame, slot, values))
        for position in range(int(layout["stop"][row]) + 1):
            for slot in ka.MARKET_POLICY_SLOTS:
                values = _admissible(
                    tables,
                    slot,
                    tokens[row],
                    counts + position,
                    position=position,
                    counts=counts,
                    limit=int(layout["limits"][row]),
                    hire_limit=hire_limit,
                )
                sites.append((row, counts + position, slot, values))
    rows: list[int] = []
    variants: list[torch.Tensor] = []
    for row, frame, slot, values in sites:
        for value in values:
            variant = tokens[row].clone()
            variant[frame, slot] = value
            rows.append(row)
            variants.append(variant)
    stacked = torch.stack(variants)
    teacher_logp = _variant_log_probs(teacher, obs, rows, stacked)
    student_logp = _variant_log_probs(student, obs, rows, stacked)
    start = 0
    checked_hire_block = False
    for row, frame, slot, values in sites:
        stop = start + len(values)
        at = torch.arange(start, stop)
        log_t = teacher_logp[at, frame, slot].double()
        log_s = student_logp[at, frame, slot].double()
        # The Python admissible set is the model's whole support.
        assert float(log_t.exp().sum()) == pytest.approx(1.0, abs=1e-5)
        assert float(log_s.exp().sum()) == pytest.approx(1.0, abs=1e-5)
        oracle = float((log_t.exp() * (log_t - log_s)).sum())
        assert float(kl[row, frame, slot]) == pytest.approx(oracle, abs=1e-6), (
            row,
            frame,
            slot,
        )
        if slot == 7 and HIRE not in values and len(values) > 1:
            checked_hire_block = True
        start = stop
    assert start == len(rows)
    if case == "hire":
        assert checked_hire_block, "no capacity-blocked HIRE site was checked"


# --- T4 chunking ----------------------------------------------------------------------


def test_head_and_trunk_chunking_match_the_unchunked_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs = make_obs(envs=3, own_actors=(1, 4, 2), rival_actors=(3, 1, 5), order_limit=3)
    actions = _sampled(student, obs)
    with torch.no_grad():
        whole_teacher = _collect(teacher, obs, actions).slot_logits
        assert whole_teacher is not None
        whole = _with_teacher(student, whole_teacher, obs, actions).kl
    assert whole is not None

    head_calls: list[int] = []

    def head_spy(core: Any) -> Any:
        def run(*args: Any) -> ka.GrammarPolicyResult:
            head_calls.append(args[0].shape[0])
            return core(*args)

        return run

    monkeypatch.setattr(km, "head_rows_per_chunk", lambda _config: 2)
    for model in (student, teacher):
        model._compiled_actor_core = head_spy(model.actor.policy_core)
    with torch.no_grad():
        head_teacher = _collect(teacher, obs, actions).slot_logits
        assert head_teacher is not None
        head_kl = _with_teacher(student, head_teacher, obs, actions).kl
    assert head_calls == [2, 2, 2, 2, 2, 2]
    # Not bit-exact: CPU GEMM blocking varies with the row count (measured on
    # the market-kind head), as in test_model_heads' chunking test.
    for slot in ka.POLICY_SLOTS:
        torch.testing.assert_close(
            head_teacher[slot], whole_teacher[slot], msg=str(slot)
        )
    assert head_kl is not None
    torch.testing.assert_close(head_kl, whole)

    trunk_calls: list[int] = []
    original = student._forward_transformer_trunk
    teacher_original = teacher._forward_transformer_trunk
    x, _ = student._assemble_tokens(obs)
    tokens, width = x.shape[1], km.trunk_gemm_width(student.config)
    monkeypatch.setattr(km, "_GEMM_ELEMENT_LIMIT", 2 * tokens * width + 1)

    def trunk_spy(fn: Any) -> Any:
        def run(x: torch.Tensor, mask: Any, packed: Any) -> torch.Tensor:
            trunk_calls.append(x.shape[0])
            return fn(x, mask, packed)

        return run

    monkeypatch.setattr(student, "_forward_transformer_trunk", trunk_spy(original))
    monkeypatch.setattr(
        teacher, "_forward_transformer_trunk", trunk_spy(teacher_original)
    )
    with torch.no_grad():
        trunk_teacher = _collect(teacher, obs, actions).slot_logits
        assert trunk_teacher is not None
        trunk_kl = _with_teacher(student, trunk_teacher, obs, actions).kl
    assert trunk_calls == [2, 2, 2, 2, 2, 2]
    for slot in ka.POLICY_SLOTS:
        torch.testing.assert_close(trunk_teacher[slot], whole_teacher[slot])
    assert trunk_kl is not None
    torch.testing.assert_close(trunk_kl, whole)


# --- T5 compile capture ---------------------------------------------------------------


def test_fullgraph_captured_core_matches_eager_with_teacher_logits() -> None:
    student, teacher = _tiny(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    with torch.no_grad():
        teacher_logits = _collect(teacher, obs, actions).slot_logits
    assert teacher_logits is not None
    eager = _with_teacher(student, teacher_logits, obs, actions)
    assert eager.kl is not None
    eager.kl.sum().backward()
    eager_grads = {
        name: p.grad.clone()
        for name, p in student.named_parameters()
        if p.grad is not None
    }
    student.zero_grad(set_to_none=True)
    torch._dynamo.reset()
    for model in (student, teacher):
        model._compiled_actor_core = torch.compile(
            model.actor.policy_core, backend="eager", fullgraph=True, dynamic=True
        )
    with torch.no_grad():
        captured_logits = _collect(teacher, obs, actions).slot_logits
    assert captured_logits is not None
    for slot in ka.POLICY_SLOTS:
        torch.testing.assert_close(captured_logits[slot], teacher_logits[slot])
    captured = _with_teacher(student, teacher_logits, obs, actions)
    assert captured.kl is not None
    torch.testing.assert_close(captured.kl, eager.kl)
    captured.kl.sum().backward()
    for name, parameter in student.named_parameters():
        if name in eager_grads:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, eager_grads[name], msg=name)
    torch._dynamo.reset()


# --- T6 gradients ---------------------------------------------------------------------


def test_kl_gradients_reach_every_student_head_and_no_teacher_parameter() -> None:
    student, teacher = _tiny(), _tiny(seed=6).eval()
    teacher.requires_grad_(False)
    obs, actions = _base_case()
    with torch.no_grad():
        teacher_logits = _collect(teacher, obs, actions).slot_logits
    assert teacher_logits is not None
    kl = _with_teacher(student, teacher_logits, obs, actions).kl
    assert kl is not None
    kl.sum().backward()
    heads = {name: p for name, p in student.named_parameters() if ".heads." in name}
    assert len(heads) > 0
    for name, parameter in heads.items():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 0, name
    assert all(p.grad is None for p in teacher.parameters())


def test_kl_gradient_matches_finite_differences_in_fp64() -> None:
    student, teacher = _tiny().double(), _tiny(seed=6).double().eval()
    obs, actions = _base_case()
    obs = _obs_double(obs)
    with torch.no_grad():
        teacher_logits = _collect(teacher, obs, actions).slot_logits
    assert teacher_logits is not None
    assert all(t.dtype == torch.float64 for t in teacher_logits.values())
    weight = student.actor.heads["market_item"].out.weight
    item = 3  # seat 0 SELLs item 3 at frame 2

    def total() -> torch.Tensor:
        kl = _with_teacher(student, teacher_logits, obs, actions).kl
        assert kl is not None
        assert kl.dtype == torch.float64
        return kl.sum()

    total().backward()
    assert weight.grad is not None
    analytic = float(weight.grad[item, 0])
    assert abs(analytic) > 1e-8
    epsilon = 1e-6
    with torch.no_grad():
        weight[item, 0] += epsilon
        plus = float(total())
        weight[item, 0] -= 2 * epsilon
        minus = float(total())
        weight[item, 0] += epsilon
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-6)


# === 4.2 targets and cache ===========================================================

SEGMENTS, HORIZON = 3, 2


def _segment_major(
    model: Any,
) -> tuple[kt.KaggricultureObsBatch, kt.KaggricultureActions]:
    """A segment-major ``[N=3, T=2, 2]`` batch with actions sampled by ``model``."""
    flat = make_obs(
        envs=SEGMENTS * HORIZON,
        own_actors=(1, 4, 2, 241, 3, 5),
        rival_actors=(3, 1, 5, 2, 241, 4),
        order_limit=4,
    )
    flat.still_playing[4, 1] = False
    actions = _sampled(model, flat)

    def lead(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(SEGMENTS, HORIZON, *t.shape[1:])

    return _map_obs(flat, lead), kt.KaggricultureActions(
        tokens=lead(actions.tokens), lengths=lead(actions.lengths)
    )


def _unit_or_market(slot: int) -> int:
    return UNIT_FRAMES if slot in ka.UNIT_POLICY_SLOTS else POSITIONS


def _target_tensors(targets: kt_teacher.KaggricultureTeacherTargets) -> dict[str, Any]:
    tensors: dict[str, Any] = {"winner": targets.winner_probabilities}
    if targets.slot_logits is None:
        tensors["slots"] = None
    else:
        for slot, logits in targets.slot_logits.items():
            tensors[f"slot{slot}"] = logits
    return tensors


def _assert_targets_equal(actual: Any, expected: Any, *, exact: bool = True) -> None:
    assert type(actual) is type(expected)
    assert actual.grammar == expected.grammar
    got, want = _target_tensors(actual), _target_tensors(expected)
    assert set(got) == set(want)
    for name, tensor in want.items():
        if tensor is None:
            assert got[name] is None, name
        elif exact:
            assert torch.equal(got[name], tensor), name
        else:
            torch.testing.assert_close(got[name], tensor, msg=name)


def test_targets_have_the_lead_layout_dtypes_and_optional_fields() -> None:
    model = _tiny().eval()
    obs, actions = _segment_major(model)
    targets = model.compute_teacher_distillation_targets(obs, actions)
    assert isinstance(targets, kt_teacher.KaggricultureTeacherTargets)
    assert targets.slot_logits is not None
    assert set(targets.slot_logits) == set(ka.POLICY_SLOTS)
    for slot, logits in targets.slot_logits.items():
        width = kt.SLOT_WIDTHS[slot]
        assert logits.shape == (SEGMENTS, HORIZON, 2, _unit_or_market(slot), width)
        assert logits.dtype == torch.float32
        assert not logits.requires_grad
    assert targets.winner_probabilities is not None
    assert targets.winner_probabilities.shape == (SEGMENTS, HORIZON, 2, 2)
    assert targets.winner_probabilities.dtype == torch.float32
    assert not targets.winner_probabilities.requires_grad
    torch.testing.assert_close(
        targets.winner_probabilities.sum(-1), torch.ones(SEGMENTS, HORIZON, 2)
    )
    assert targets.grammar == model.grammar_signature()
    kl_only = model.compute_teacher_distillation_targets(
        obs, actions, compute_value=False
    )
    assert kl_only.winner_probabilities is None
    assert kl_only.slot_logits is not None
    value_only = model.compute_teacher_distillation_targets(
        obs, actions, compute_action_kl=False
    )
    assert value_only.slot_logits is None
    assert value_only.winner_probabilities is not None


def test_teacher_targets_reject_programs_the_teacher_grammar_does_not_admit() -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    bad = kt.KaggricultureActions(
        tokens=actions.tokens.clone(), lengths=actions.lengths.clone()
    )
    bad.lengths[0, 0] += 1
    with pytest.raises(ka.GrammarReplayError, match="length"):
        model.compute_teacher_distillation_targets(obs, bad)
    # A student (hire_limit 241) samples 3+ HIREs; a hire_limit-3 teacher with
    # 2 own actors admits one.
    student = _tiny().eval()
    with torch.no_grad():
        student.actor.heads["market_kind"].out.bias[HIRE] = 8.0
        student.actor.heads["market_kind"].out.bias[NONE] = -8.0
    hires_obs = make_obs(envs=1, own_actors=2, rival_actors=2, order_limit=6)
    sampled = _sampled(student, hires_obs)
    assert int((sampled.tokens[0, 0, :, 7] == HIRE).sum()) >= 3
    teacher = _tiny(hire_limit=3, seed=6).eval()
    with pytest.raises(ka.GrammarReplayError) as error:
        teacher.compute_teacher_distillation_targets(hires_obs, sampled)
    assert "support" in error.value.groups
    # Value-only targets never replay the program.
    teacher.compute_teacher_distillation_targets(
        hires_obs, sampled, compute_action_kl=False
    )


@pytest.mark.parametrize("layout", ["both", "kl", "value"])
@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_index_then_concat_restores_the_targets(layout: str, chunk: int) -> None:
    model = _tiny().eval()
    obs, actions = _segment_major(model)
    targets = model.compute_teacher_distillation_targets(
        obs,
        actions,
        compute_action_kl=layout != "value",
        compute_value=layout != "kl",
    )
    pieces = [
        targets.index(torch.arange(start, min(start + chunk, SEGMENTS)))
        for start in range(0, SEGMENTS, chunk)
    ]
    joined = kt_teacher.KaggricultureTeacherTargets.concat(pieces)
    _assert_targets_equal(joined, targets)
    if len(pieces) == 1:
        assert joined is pieces[0]
    picked = targets.index(torch.tensor([2, 0]))
    for name, tensor in _target_tensors(targets).items():
        if tensor is not None:
            assert torch.equal(
                _target_tensors(picked)[name], tensor[torch.tensor([2, 0])]
            ), name


def test_concat_validates_every_chunk_symmetrically() -> None:
    model = _tiny().eval()
    obs, actions = _segment_major(model)
    both = model.compute_teacher_distillation_targets(obs, actions)
    kl_only = model.compute_teacher_distillation_targets(
        obs, actions, compute_value=False
    )
    value_only = model.compute_teacher_distillation_targets(
        obs, actions, compute_action_kl=False
    )
    concat = kt_teacher.KaggricultureTeacherTargets.concat
    for first, second, field in (
        (both, kl_only, "winner_probabilities"),
        (kl_only, both, "winner_probabilities"),
        (both, value_only, "slot_logits"),
        (value_only, both, "slot_logits"),
    ):
        with pytest.raises(ValueError, match=field):
            concat([first, second])
    assert both.slot_logits is not None
    fewer = dataclasses.replace(
        both, slot_logits={k: v for k, v in both.slot_logits.items() if k != 7}
    )
    for pair in ([both, fewer], [fewer, both]):
        with pytest.raises(ValueError, match="slot_logits keys"):
            concat(pair)
    other = dataclasses.replace(
        both, grammar=kt_teacher.GrammarSignature(both.grammar.tables_sha256, 5)
    )
    for pair in ([both, other], [other, both]):
        with pytest.raises(ValueError, match="grammar"):
            concat(pair)
    with pytest.raises(ValueError, match="empty"):
        concat([])


def test_grammar_signature_tracks_tables_and_hire_limit() -> None:
    base = _tiny()
    assert _tiny(seed=6).grammar_signature() == base.grammar_signature()
    assert _tiny(hire_limit=5).grammar_signature() != base.grammar_signature()
    flipped = gg.expected_grammar_tables()
    flipped.market_kind[HIRE] = False
    no_hire = km.KaggricultureTransformer(
        base.config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(hire_limit=241),
        grammar_tables=flipped,
    )
    signature = no_hire.grammar_signature()
    assert signature.hire_limit == base.grammar_signature().hire_limit
    assert signature.tables_sha256 != base.grammar_signature().tables_sha256


def test_nbytes_counts_every_cached_tensor() -> None:
    model = _tiny().eval()
    obs, actions = _segment_major(model)
    targets = model.compute_teacher_distillation_targets(obs, actions)
    tensors = [t for t in _target_tensors(targets).values() if t is not None]
    assert targets.nbytes() == sum(t.nbytes for t in tensors)
    rows = SEGMENTS * HORIZON * 2
    assert targets.nbytes() == rows * kt_teacher.TEACHER_TARGET_BYTES_PER_ROW
    value_only = model.compute_teacher_distillation_targets(
        obs, actions, compute_action_kl=False
    )
    assert value_only.nbytes() == rows * 2 * 4
    isaiah = CachedTeacherDistillationTargets(
        action_params=DiscreteTargetPolicyParams(
            target_logits=torch.zeros(2, 3, 4, 7, 7),
            size_mix_logits=torch.zeros(2, 3, 4, 7, 7, 2),
            size_mu=torch.zeros(2, 3, 4, 7, 7, 2),
            size_scale=torch.zeros(2, 3, 4, 7, 7, 2),
            continue_logits=torch.zeros(2, 3, 4, 7),
        ),
        winner_probabilities=torch.zeros(2, 3, 4),
    )
    params = isaiah.action_params
    assert params is not None
    assert params.continue_logits is not None
    assert isaiah.nbytes() == sum(
        t.nbytes
        for t in (
            params.target_logits,
            params.size_mix_logits,
            params.size_mu,
            params.size_scale,
            params.continue_logits,
            torch.zeros(2, 3, 4),
        )
    )
    no_continue = dataclasses.replace(
        isaiah, action_params=dataclasses.replace(params, continue_logits=None)
    )
    assert no_continue.nbytes() == isaiah.nbytes() - params.continue_logits.nbytes


def test_cache_bytes_per_row_follow_the_contract_widths() -> None:
    unit = sum(kt.SLOT_WIDTHS[s] for s in ka.UNIT_POLICY_SLOTS)
    market = sum(kt.SLOT_WIDTHS[s] for s in ka.MARKET_POLICY_SLOTS)
    assert (unit, market) == (102, 88)
    assert kt_teacher.TEACHER_TARGET_BYTES_PER_ROW == 102_208
    two_rank_rows = 128 * 64 * 2
    four_rank_rows = 64 * 64 * 2
    assert two_rank_rows * kt_teacher.TEACHER_TARGET_BYTES_PER_ROW == 1_674_575_872
    assert four_rank_rows * kt_teacher.TEACHER_TARGET_BYTES_PER_ROW == 837_287_936


def test_chunked_precompute_equals_one_whole_batch_call() -> None:
    model = _tiny().eval()
    obs, actions = _segment_major(model)
    whole = model.compute_teacher_distillation_targets(obs, actions)
    chunks = []
    for start in range(SEGMENTS):
        index = torch.tensor([start])
        chunks.append(
            model.compute_teacher_distillation_targets(
                _take_obs(obs, index), _take_actions(actions, index)
            )
        )
    joined = kt_teacher.KaggricultureTeacherTargets.concat(chunks)
    # The trunk and head batch sizes differ, so GEMM blocking may differ.
    _assert_targets_equal(joined, whole, exact=False)


def test_isaiah_cached_teacher_rejects_foreign_targets_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from owl.model.stateless_transformer_v1 import (
        ActorDiscreteTargetsConfig,
        StatelessTransformerV1Config,
    )
    from owl.rl import ActionDiscreteTargetsConfig, EntityBasedConfig

    from tests.owl.model.test_stateless_transformer_v1 import (
        MAX_COMETS,
        MAX_PLANETS,
        _model,
        _obs_batch,
    )

    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(
        max_per_planet_launches=1, min_fleet_size=2
    )
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(n_action_mixtures=3, entropy_ship_quantiles=8),
        embed_dim=32,
        depth=2,
        n_heads=4,
    )
    student = _model(config, obs_spec=obs_spec, action_spec=action_spec).eval()
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    with torch.no_grad():
        actions = student(obs).actions
    calls: list[int] = []
    original = student.encode_observations

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(student, "encode_observations", spy)
    kaggriculture = _tiny().eval()
    k_obs, k_actions = _base_case()
    foreign = kaggriculture.compute_teacher_distillation_targets(k_obs, k_actions)
    with pytest.raises(TypeError, match="CachedTeacherDistillationTargets"):
        student.evaluate_actions_with_cached_teacher(obs, actions, foreign)
    assert calls == []


# === 4.3 model methods and trainer wiring ============================================


def _cached(student: Any, teacher: Any, obs: Any, actions: Any, **flags: bool) -> Any:
    targets = teacher.compute_teacher_distillation_targets(
        obs,
        actions,
        compute_action_kl=flags.get("compute_teacher_action_kl", True),
        compute_value=flags.get("compute_teacher_value", True),
    )
    return student.evaluate_actions_with_cached_teacher(obs, actions, targets, **flags)


def _value_ce(model: Any, evaluation: Any, obs: Any) -> torch.Tensor:
    assert evaluation.student_winner_log_probabilities is not None
    assert evaluation.teacher_winner_probabilities is not None
    return model.teacher_value_cross_entropy(
        evaluation.student_winner_log_probabilities,
        evaluation.teacher_winner_probabilities,
        value_mask=obs.still_playing,
    )


def _evaluation_tensors(evaluation: Any, model: Any, obs: Any) -> dict[str, Any]:
    kl = evaluation.action_kl
    tensors = {
        "kl_event": kl.event,
        "kl_launch": kl.launch,
        "kl_per_player_entity": kl.per_player_entity,
        "teacher_winner": evaluation.teacher_winner_probabilities,
        "student_winner_log": evaluation.student_winner_log_probabilities,
        "value_ce": _value_ce(model, evaluation, obs),
        "log_probs": evaluation.student.log_probs.event,
        "log_probs_entity": evaluation.student.log_probs.per_player_entity,
        "entropies": evaluation.student.entropies.event,
        "values": evaluation.student.values,
    }
    for name, component in kl.components.items():
        tensors[f"kl_{name}"] = component
    return tensors


def test_cached_path_is_bit_for_bit_the_combined_path() -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    teacher.requires_grad_(False)
    obs, actions = _segment_major(student)
    with torch.no_grad():
        combined = student.evaluate_actions_with_teacher(obs, actions, teacher)
        cached = _cached(student, teacher, obs, actions)
    assert combined.action_kl is not None
    assert cached.action_kl is not None
    assert set(cached.action_kl.components) == {
        kt.SLOT_NAMES[s] for s in ka.POLICY_SLOTS
    }
    assert cached.action_kl.target is None
    lead = (SEGMENTS, HORIZON, 2)
    assert cached.action_kl.event.shape == (*lead, F, K)
    assert cached.action_kl.per_player_entity.shape == (*lead, F)
    got = _evaluation_tensors(cached, student, obs)
    want = _evaluation_tensors(combined, student, obs)
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name
    assert float(cached.action_kl.per_player_entity.sum()) > 0
    # The student side is exactly evaluate_actions.
    plain = student.evaluate_actions(obs, actions)
    assert torch.equal(cached.student.log_probs.event, plain.log_probs.event)
    assert torch.equal(cached.student.values, plain.values)


def test_a_copied_teacher_gives_zero_kl_and_the_student_winner_entropy() -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    teacher.load_state_dict(student.state_dict())
    obs, actions = _segment_major(student)
    with torch.no_grad():
        evaluation = _cached(student, teacher, obs, actions)
    assert evaluation.action_kl is not None
    zeros = torch.zeros_like(evaluation.action_kl.event)
    assert torch.equal(evaluation.action_kl.event, zeros)
    log_q = evaluation.student_winner_log_probabilities
    entropy = -(log_q.exp() * log_q).sum(-1)
    live = obs.still_playing.to(entropy.dtype)
    expected = (entropy * live).sum(-1) / live.sum(-1).clamp_min(1.0)
    torch.testing.assert_close(
        _value_ce(student, evaluation, obs), expected, rtol=0, atol=1e-6
    )


def test_value_cross_entropy_is_the_live_seat_mean() -> None:
    model = _tiny()
    teacher = torch.tensor(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.5, 0.5]],
            [[0.9, 0.1], [0.1, 0.9]],
        ]
    )
    student_log = torch.log(
        torch.tensor(
            [
                [[0.5, 0.5], [0.4, 0.6]],
                [[0.7, 0.3], [0.5, 0.5]],
                [[0.2, 0.8], [0.6, 0.4]],
            ]
        )
    )
    live = torch.tensor([[True, True], [True, False], [False, False]])
    ce = model.teacher_value_cross_entropy(student_log, teacher, value_mask=live)
    per_seat = -(teacher * student_log).sum(-1)
    assert ce.shape == (3,)
    torch.testing.assert_close(ce[0], per_seat[0].mean())  # mean, not sum
    assert float(ce[0]) != pytest.approx(float(per_seat[0].sum()))
    torch.testing.assert_close(ce[1], per_seat[1, 0])  # the non-live seat is out
    assert float(ce[2]) == 0.0
    state_weight = ppo._value_state_weight(live.float(), dtype=torch.float32)
    assert state_weight.tolist() == [1.0, 1.0, 0.0]
    # No gradient flows to the teacher distribution.
    teacher_grad = teacher.clone().requires_grad_(True)
    student_grad = student_log.clone().requires_grad_(True)
    model.teacher_value_cross_entropy(
        student_grad, teacher_grad, value_mask=live
    ).sum().backward()
    assert teacher_grad.grad is None
    assert student_grad.grad is not None


def _encode_spy(monkeypatch: pytest.MonkeyPatch, model: Any) -> list[int]:
    calls: list[int] = []
    original = model.encode_observations

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "encode_observations", spy)
    return calls


def test_cached_admission_rejects_bad_targets_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _segment_major(student)
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    calls = _encode_spy(monkeypatch, student)
    evaluate = student.evaluate_actions_with_cached_teacher
    isaiah_targets = CachedTeacherDistillationTargets(
        action_params=None, winner_probabilities=None
    )
    with pytest.raises(TypeError, match="KaggricultureTeacherTargets"):
        evaluate(obs, actions, isaiah_targets)
    assert targets.slot_logits is not None
    slots = targets.slot_logits
    bad_cases: list[tuple[Any, str, dict[str, bool]]] = [
        (
            dataclasses.replace(targets, slot_logits=None),
            "cached teacher action targets are missing",
            {},
        ),
        (
            dataclasses.replace(targets, winner_probabilities=None),
            "cached teacher value targets are missing",
            {},
        ),
        (
            dataclasses.replace(
                targets, slot_logits={k: v for k, v in slots.items() if k != 9}
            ),
            "slot keys",
            {},
        ),
        (
            dataclasses.replace(targets, slot_logits={**slots, 3: slots[3][:, :1]}),
            "shape",
            {},
        ),
        (
            dataclasses.replace(
                targets, slot_logits={**slots, 8: slots[8].to(torch.bfloat16)}
            ),
            "dtype",
            {},
        ),
        (
            dataclasses.replace(
                targets, winner_probabilities=torch.zeros(SEGMENTS, HORIZON, 2, 3)
            ),
            "winner",
            {},
        ),
        (
            dataclasses.replace(
                targets,
                grammar=kt_teacher.GrammarSignature(targets.grammar.tables_sha256, 7),
            ),
            "grammar",
            {},
        ),
    ]
    for bad, match, flags in bad_cases:
        with pytest.raises(ValueError, match=match):
            evaluate(obs, actions, bad, **flags)
    with pytest.raises(ValueError, match="hidden_state"):
        evaluate(obs, actions, targets, hidden_state=object())
    with pytest.raises(ValueError, match="dones"):
        evaluate(obs, actions, targets, dones=torch.zeros(SEGMENTS, HORIZON, 2))
    assert calls == []
    # A disabled target may be absent, and a value-only call skips the grammar.
    evaluate(
        obs,
        actions,
        dataclasses.replace(
            targets, slot_logits=None, grammar=bad_cases[-1][0].grammar
        ),
        compute_teacher_action_kl=False,
    )
    assert calls == [1]


def test_combined_path_rejects_foreign_or_mismatched_teachers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from owl.model.stateless_transformer_v1 import (
        StatelessTransformerV1,
        StatelessTransformerV1Config,
    )
    from owl.rl import ActionPureConfig, EntityBasedConfig

    student = _tiny().eval()
    obs, actions = _base_case()
    calls = _encode_spy(monkeypatch, student)
    orbit = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=8, depth=1, n_heads=2),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    combined = student.evaluate_actions_with_teacher
    with pytest.raises(ValueError, match="KaggricultureTransformer"):
        combined(obs, actions, orbit)
    with pytest.raises(ValueError, match="action_spec"):
        combined(obs, actions, _tiny(hire_limit=5, seed=6))
    flipped = gg.expected_grammar_tables()
    flipped.market_kind[HIRE] = False
    no_hire = km.KaggricultureTransformer(
        student.config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
        grammar_tables=flipped,
    )
    with pytest.raises(ValueError, match="grammar"):
        combined(obs, actions, no_hire)
    edited = _tiny(seed=6)
    with torch.no_grad():
        edited.actor.table_market_kind[HIRE] = False  # after construction
    assert edited.grammar_signature() == student.grammar_signature()
    with pytest.raises(ValueError, match="table market_kind"):
        combined(obs, actions, edited)
    assert calls == []


def test_a_grammar_mismatch_that_replay_admits_is_rejected_by_the_signature() -> None:
    student = _tiny().eval()
    with torch.no_grad():
        student.actor.heads["market_kind"].out.bias[HIRE] = -30.0
    obs = make_obs(envs=2, own_actors=(2, 3), rival_actors=(4, 1), order_limit=6)
    actions = _sampled(student, obs)
    assert int((actions.tokens[..., 7] == HIRE).sum()) == 0
    matched = _tiny(seed=6).eval()
    flipped = gg.expected_grammar_tables()
    flipped.market_kind[HIRE] = False
    no_hire = km.KaggricultureTransformer(
        matched.config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
        grammar_tables=flipped,
    ).eval()
    no_hire.load_state_dict(matched.state_dict())
    # Replay admission passes under both grammars.
    foreign = no_hire.compute_teacher_distillation_targets(obs, actions)
    reference = matched.compute_teacher_distillation_targets(obs, actions)
    with pytest.raises(ValueError, match="grammar"):
        student.evaluate_actions_with_cached_teacher(obs, actions, foreign)
    # Non-vacuity: re-stamped, the foreign logits pass and change the KL.
    restamped = dataclasses.replace(foreign, grammar=student.grammar_signature())
    with torch.no_grad():
        wrong = student.evaluate_actions_with_cached_teacher(obs, actions, restamped)
        right = student.evaluate_actions_with_cached_teacher(obs, actions, reference)
    assert wrong.action_kl is not None
    assert right.action_kl is not None
    kind = wrong.action_kl.components["market_kind"]
    assert not torch.equal(kind, right.action_kl.components["market_kind"])


def test_ppo_teacher_wrappers_dispatch_statelessly() -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _segment_major(student)
    dones = torch.zeros(SEGMENTS, HORIZON, 2, dtype=torch.bool)
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    with torch.no_grad():
        cached = ppo._model_evaluate_actions_with_cached_teacher(
            student,
            obs,
            actions,
            targets,
            hidden_state=None,
            dones=dones,
            compute_teacher_action_kl=True,
            compute_teacher_value=True,
        )
        combined = ppo._model_evaluate_actions_with_teacher(
            student,
            obs,
            actions,
            teacher,
            hidden_state=None,
            dones=dones,
            compute_teacher_action_kl=True,
            compute_teacher_value=True,
        )
        direct = student.evaluate_actions_with_cached_teacher(obs, actions, targets)
    for got in (cached, combined):
        tensors = _evaluation_tensors(got, student, obs)
        for name, tensor in _evaluation_tensors(direct, student, obs).items():
            assert torch.equal(tensors[name], tensor), name


def test_ppo_teacher_wrappers_leave_orbit_results_unchanged() -> None:
    from owl.model.stateless_transformer_v1 import (
        ActorDiscreteTargetsConfig,
        StatelessTransformerV1,
        StatelessTransformerV1Config,
    )

    from tests.owl.train.test_ppo import TinyDiscreteTargetEnv

    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(n_action_mixtures=2, entropy_ship_quantiles=8),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    models = []
    for _ in range(2):
        model = StatelessTransformerV1(
            config, obs_spec=env.obs_spec, action_spec=env.action_spec
        )
        model.reset_parameters()
        models.append(model.eval())
    student, teacher = models
    teacher.requires_grad_(False)
    trainer = ppo.PPOTrainer(
        env=env,
        model=student,
        optimizer=torch.optim.AdamW(student.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(horizon=3, segments_per_minibatch=1),
        device=torch.device("cpu"),
        teacher_model=teacher,
        teacher_active=True,
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    targets = teacher.compute_teacher_distillation_targets(
        segments.obs, segments.actions
    )
    flags = {"compute_teacher_action_kl": True, "compute_teacher_value": True}
    with torch.no_grad():
        wrapped = ppo._model_evaluate_actions_with_cached_teacher(
            student,
            segments.obs,
            segments.actions,
            targets,
            hidden_state=None,
            dones=segments.dones,
            **flags,
        )
        direct = student.evaluate_actions_with_cached_teacher(
            segments.obs, segments.actions, targets, dones=segments.dones, **flags
        )
        wrapped_combined = ppo._model_evaluate_actions_with_teacher(
            student,
            segments.obs,
            segments.actions,
            teacher,
            hidden_state=None,
            dones=segments.dones,
            **flags,
        )
        direct_combined = student.evaluate_actions_with_teacher(
            segments.obs, segments.actions, teacher, dones=segments.dones, **flags
        )
    for got, want in ((wrapped, direct), (wrapped_combined, direct_combined)):
        assert got.action_kl is not None
        assert want.action_kl is not None
        assert torch.equal(got.action_kl.event, want.action_kl.event)
        assert torch.equal(got.student.log_probs.event, want.student.log_probs.event)
        assert torch.equal(
            got.teacher_winner_probabilities, want.teacher_winner_probabilities
        )
    # Isaiah's value CE, now a base-class method, is his formula exactly.
    student_log = wrapped.student_winner_log_probabilities
    teacher_probs = wrapped.teacher_winner_probabilities
    assert student_log is not None
    assert teacher_probs is not None
    assert torch.equal(
        student.teacher_value_cross_entropy(
            student_log, teacher_probs, value_mask=segments.obs.still_playing
        ),
        (-teacher_probs.detach() * student_log).sum(dim=-1),
    )


def test_targets_and_kl_are_seat_isolated_and_stateless() -> None:
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs = make_obs(envs=2, own_actors=(2, 3), rival_actors=(3, 2), order_limit=4)
    actions = _sampled(student, obs)
    changed = make_obs(envs=2, own_actors=(2, 3), rival_actors=(3, 2), order_limit=4)
    changed.tiles_float[:, 1] += 1.0
    changed.actors_float[:, 1] += 0.5
    with torch.no_grad():
        base = teacher.compute_teacher_distillation_targets(obs, actions)
        other = teacher.compute_teacher_distillation_targets(changed, actions)
        base_eval = student.evaluate_actions_with_cached_teacher(obs, actions, base)
        other_eval = student.evaluate_actions_with_cached_teacher(
            changed, actions, other
        )
        teacher.compute_teacher_distillation_targets(
            make_obs(envs=3, own_actors=7, order_limit=10),
            _sampled(teacher, make_obs(envs=3, own_actors=7, order_limit=10)),
        )
        again = teacher.compute_teacher_distillation_targets(obs, actions)
    assert base.slot_logits is not None
    assert other.slot_logits is not None
    for slot in ka.POLICY_SLOTS:
        assert torch.equal(other.slot_logits[slot][:, 0], base.slot_logits[slot][:, 0])
    assert other.winner_probabilities is not None
    assert base.winner_probabilities is not None
    assert torch.equal(
        other.winner_probabilities[:, 0], base.winner_probabilities[:, 0]
    )
    assert base_eval.action_kl is not None
    assert other_eval.action_kl is not None
    assert torch.equal(
        other_eval.action_kl.event[:, 0], base_eval.action_kl.event[:, 0]
    )
    assert not torch.equal(
        other_eval.action_kl.event[:, 1], base_eval.action_kl.event[:, 1]
    )
    _assert_targets_equal(again, base)


def test_the_model_supports_both_cached_distillation_paths() -> None:
    model = _tiny()
    assert model.supports_cached_teacher_distillation()
    assert model.supports_cached_value_distillation()
    # No teacher target or cache is ever a checkpoint entry.
    keys = set(model.state_dict())
    assert not any("table_" in key or "teacher" in key for key in keys)


class _FakeKaggricultureEnv:
    """Two-env Kaggriculture stand-in for the PPO trainer (T18)."""

    def __init__(self) -> None:
        self.n_envs = 2
        self.pin_memory_enabled = False
        self.obs_spec = kt.KaggricultureObsConfig()
        self.action_spec = kt.KaggricultureActionConfig()

    def reset(self) -> kt.KaggricultureObsBatch:
        return make_obs(envs=2, own_actors=(2, 3), rival_actors=(3, 1), order_limit=4)

    def step(self, actions: kt.KaggricultureActions) -> tuple[Any, ...]:
        assert actions.tokens.shape == (2, 2, F, K)
        # The native step's metric keys, empty without completed games.
        metrics: dict[str, list[float]] = {
            "total_games_played": [],
            "terminal_bank_0": [],
            "terminal_bank_1": [],
            "terminal_margin_0": [],
        }
        dones = torch.zeros(2, 2, dtype=torch.bool)
        return self.reset(), torch.zeros(2, 2), dones, metrics


def test_trainer_precomputes_once_and_logs_teacher_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student, teacher = _tiny(), _tiny(seed=6)
    horizon = 4
    trainer = ppo.PPOTrainer(
        env=_FakeKaggricultureEnv(),
        model=student,
        optimizer=torch.optim.AdamW(student.parameters(), lr=0.0),
        config=ppo.PPOConfig(
            horizon=horizon,
            segments_per_minibatch=1,
            teacher_mode="last_best",
            teacher_kl_coef=0.005,
            teacher_value_coef=0.005,
            teacher_segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher,
        teacher_active=True,
    )
    precomputes: list[int] = []
    teacher_grad_modes: list[bool] = []
    original = trainer._precompute_teacher_targets
    teacher_targets = teacher.compute_teacher_distillation_targets

    def spy(*args: Any, **kwargs: Any) -> Any:
        precomputes.append(1)
        return original(*args, **kwargs)

    def teacher_spy(*args: Any, **kwargs: Any) -> Any:
        teacher_grad_modes.append(torch.is_grad_enabled())
        return teacher_targets(*args, **kwargs)

    monkeypatch.setattr(trainer, "_precompute_teacher_targets", spy)
    monkeypatch.setattr(teacher, "compute_teacher_distillation_targets", teacher_spy)
    metrics = trainer.train_iteration()
    assert precomputes == [1]
    assert teacher_grad_modes == [False, False]  # two one-segment chunks
    assert metrics["teacher/kl"] > 0
    assert metrics["teacher/cache_bytes"] == (
        2 * horizon * 2 * kt_teacher.TEACHER_TARGET_BYTES_PER_ROW
    )
    assert metrics["teacher/kl_coef"] == 0.005
    assert metrics["teacher/value_coef"] == 0.005
    # A copied teacher gives exactly zero KL at zero learning rate.
    teacher.load_state_dict(student.state_dict())
    assert trainer.train_iteration()["teacher/kl"] == 0.0


def test_teacher_obs_spec_dispatch_covers_kaggriculture() -> None:
    from owl.rl import EntityBasedConfig

    import scripts.run_ppo as run_ppo

    path = Path("teacher/checkpoint.pt")
    spec = kt.KaggricultureObsConfig()
    assert (
        run_ppo._teacher_obs_spec_for_student(
            spec, student_obs_spec=kt.KaggricultureObsConfig(), checkpoint_path=path
        )
        == spec
    )
    future = kt.KaggricultureObsConfig.model_construct(schema_version=4)
    with pytest.raises(ValueError, match="obs_spec must match"):
        run_ppo._teacher_obs_spec_for_student(
            future, student_obs_spec=spec, checkpoint_path=path
        )
    for teacher_spec, student_spec in (
        (EntityBasedConfig(), spec),
        (spec, EntityBasedConfig()),
    ):
        with pytest.raises(TypeError, match="KaggricultureObsConfig"):
            run_ppo._teacher_obs_spec_for_student(
                teacher_spec, student_obs_spec=student_spec, checkpoint_path=path
            )


def test_last_best_refresh_keeps_the_tables_and_copies_the_student() -> None:
    import scripts.run_ppo as run_ppo

    student, last_best = _tiny(), _tiny(seed=6)
    obs, actions = _base_case()
    before = last_best.grammar_signature()
    run_ppo._refresh_eval_model_from_weights(last_best, student)
    assert not last_best.training
    assert last_best.grammar_signature() == before
    for name, table in gg.expected_grammar_tables().as_dict().items():
        assert torch.equal(last_best.actor.tables().as_dict()[name], table), name
    with torch.no_grad():
        evaluation = _cached(student, last_best, obs, actions)
    assert evaluation.action_kl is not None
    assert torch.equal(
        evaluation.action_kl.event, torch.zeros_like(evaluation.action_kl.event)
    )


# --- T19b: run_ppo launch and resume ---------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
# ``PPOTrainer.write_checkpoint``'s whole key set: run_ppo's loader rejects any other
# key, so a teacher cache can never ride along in a checkpoint.
CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "lr_scheduler",
    "env_steps",
    "optimizer_steps",
    "player_step_total",
    "total_games_played",
    "total_active_entities",
    "target_kl_exceeded_total",
    "wandb_run_id",
}
_TINY_MODEL = {
    "model_arch": km.KAGGRICULTURE_TRANSFORMER,
    "embed_dim": 16,
    "depth": 1,
    "n_heads": 2,
    "mlp_ratio": 2.0,
    "n_scratch_tokens": 1,
}


def _kaggriculture_run_config() -> Any:
    """``configs/kaggriculture.yaml`` with the tiny test model, last-best teacher."""
    from owl.train import FullConfig

    data = FullConfig.from_file(REPO_ROOT / "configs" / "kaggriculture.yaml")
    dumped = data.model_dump(mode="python")
    return FullConfig.model_validate(
        {
            **dumped,
            "model": _TINY_MODEL,
            "rl": {
                **dumped["rl"],
                "horizon": 4,
                "teacher_mode": "last_best",
                "model_compile": "none",
                "compile_mode": None,
                "dtype": "float32",
                "eval_replay_games": 0,
            },
        }
    )


def _write_run_checkpoint(
    path: Path, model: torch.nn.Module, *, env_steps: int, run_id: str
) -> None:
    """A checkpoint in ``PPOTrainer.write_checkpoint``'s layout (model state only)."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": env_steps,
        "optimizer_steps": 0,
        "player_step_total": 0,
        "total_games_played": 0,
        "total_active_entities": 0,
        "target_kl_exceeded_total": 0,
        "wandb_run_id": run_id,
    }
    assert set(checkpoint) == CHECKPOINT_KEYS
    torch.save(checkpoint, path)


def _assert_no_teacher_state(state: dict[str, torch.Tensor]) -> None:
    assert not any("teacher" in key or "table_" in key for key in state)


def _assert_same_weights(model: torch.nn.Module, source: torch.nn.Module) -> None:
    expected = source.state_dict()
    actual = model.state_dict()
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def _assert_expected_tables(model: Any) -> None:
    for name, table in gg.expected_grammar_tables().as_dict().items():
        assert torch.equal(model.actor.tables().as_dict()[name], table), name


def _write_attempt_receipt(run_dir: Path, *, run_id: str) -> None:
    """The ``attempts.jsonl`` a ``run_ppo`` resume needs: one online attempt."""
    from owl.train import logging as train_logging

    class _OnlineLogger(train_logging.DebugLogger):
        @property
        def run_id(self) -> str | None:
            return run_id

        def wandb_run_facts(self) -> train_logging.WandbRunFacts | None:
            return train_logging.WandbRunFacts(project="kg-v3", entity=None, url=None)

    identity = train_logging.plan_attempt(
        run_dir,
        job_type="ppo",
        resume=False,
        experiment_id=None,
        source_commit="v3-src-0",
        config_sha256="0" * 64,
        telemetry=train_logging.TelemetryMode.WANDB_ONLINE,
    )
    train_logging.record_attempt(run_dir, identity, _OnlineLogger(), start_env_steps=0)


def _run_ppo_main(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> tuple[Any, dict[str, Any]]:
    """Run ``run_ppo.main`` with a fake env and trainer; return both refs.

    Isaiah's launch-test pattern (``tests/scripts/test_run_ppo.py``): the fake
    trainer records ``set_teacher_model`` and loads model weights through
    ``run_ppo._checkpoint_metadata``; last-best construction and loading stay real.
    """
    import sys
    from contextlib import nullcontext

    from owl.train.distributed import DistributedContext

    import scripts.run_ppo as run_ppo

    refs: dict[str, Any] = {}
    session: dict[str, Any] = {}

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs["model"]
            self.teacher_model = kwargs["teacher_model"]
            self.teacher_active = kwargs["teacher_active"]
            self.teacher_updates: list[tuple[Any, bool]] = []
            refs["trainer"] = self

        def _load(self, path: Path) -> Any:
            checkpoint = torch.load(path, weights_only=False)
            metadata = run_ppo._checkpoint_metadata(checkpoint, path=path)
            self.model.load_state_dict(checkpoint["model"])
            return metadata

        def load_checkpoint(self, path: Path) -> Any:
            return self._load(path)

        def load_model_weights(
            self,
            path: Path,
            *,
            load_optimizer: bool,
            fresh_state_keys: frozenset[str] = frozenset(),
        ) -> Any:
            assert not load_optimizer
            assert fresh_state_keys == frozenset()
            return self._load(path)

        def set_teacher_model(self, teacher_model: Any, *, active: bool) -> None:
            self.teacher_updates.append((teacher_model, active))

    monkeypatch.setattr(sys, "argv", ["run_ppo.py", *argv])
    # Hermetic W&B credentials for the startup gate: never the host's netrc.
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    # The env patch point follows the Task 3.1 run_ppo game seam's constructor.
    monkeypatch.setattr(
        run_ppo, "create_env", lambda _env_config, **_kwargs: _FakeKaggricultureEnv()
    )
    monkeypatch.setattr(run_ppo, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(run_ppo, "_run_training_session", session.update)
    run_ppo.main()
    return refs["trainer"], session


def _assert_active_last_best_teacher(
    trainer: Any, session: dict[str, Any], *, source: Any
) -> Any:
    """One active ``set_teacher_model`` call with the session's last-best model."""
    assert trainer.teacher_model is None
    assert not trainer.teacher_active
    assert len(trainer.teacher_updates) == 1
    teacher, active = trainer.teacher_updates[0]
    assert active
    assert teacher is session["last_best_model"]
    assert teacher is not trainer.model
    assert not teacher.training
    _assert_same_weights(teacher, source)
    _assert_expected_tables(teacher)
    _assert_no_teacher_state(teacher.state_dict())
    obs, actions = _base_case()
    with torch.no_grad():
        _assert_targets_equal(
            teacher.compute_teacher_distillation_targets(obs, actions),
            source.eval().compute_teacher_distillation_targets(obs, actions),
        )
    return teacher


def test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T19b resume: the teacher is ``checkpoint_last_best.pt``, not the student."""
    cfg = _kaggriculture_run_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg.to_file(run_dir / "config.yaml")
    student, last_best = _tiny(), _tiny(seed=6)
    _write_run_checkpoint(
        run_dir / "checkpoint_final.pt", student, env_steps=128, run_id="run-123"
    )
    _write_run_checkpoint(
        run_dir / "checkpoint_last_best.pt", last_best, env_steps=64, run_id="run-123"
    )
    for name in ("checkpoint_final.pt", "checkpoint_last_best.pt"):
        saved = torch.load(run_dir / name, weights_only=False)
        assert set(saved) == CHECKPOINT_KEYS
        _assert_no_teacher_state(saved["model"])
    _write_attempt_receipt(run_dir, run_id="run-123")

    trainer, session = _run_ppo_main(monkeypatch, [str(run_dir), "--log-mode", "wandb"])

    assert session["start_env_steps"] == 128
    assert session["resume_run_id"] == "run-123"
    _assert_same_weights(trainer.model, student)
    _assert_active_last_best_teacher(trainer, session, source=last_best)


def test_run_ppo_fresh_launch_from_weights_activates_the_last_best_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T19b fresh launch: the loaded weights become the active last-best teacher."""
    cfg = _kaggriculture_run_config()
    config_path = tmp_path / "config.yaml"
    cfg.to_file(config_path)
    source = _tiny(seed=6)
    weights = tmp_path / "weights.pt"
    _write_run_checkpoint(weights, source, env_steps=64, run_id="run-123")

    trainer, session = _run_ppo_main(
        monkeypatch,
        [
            str(config_path),
            str(tmp_path / "runs"),
            "--load-model-weights",
            str(weights),
            "--log-mode",
            "debug",
        ],
    )

    assert session["start_env_steps"] == 64
    _assert_same_weights(trainer.model, source)
    teacher = _assert_active_last_best_teacher(trainer, session, source=source)
    obs, actions = _base_case()
    with torch.no_grad():
        evaluation = _cached(trainer.model.eval(), teacher, obs, actions)
    assert evaluation.action_kl is not None
    assert torch.equal(
        evaluation.action_kl.event, torch.zeros_like(evaluation.action_kl.event)
    )


def test_trainer_checkpoint_after_a_teacher_iteration_holds_no_teacher_cache(
    tmp_path: Path,
) -> None:
    """T19b: after an iteration fills the teacher cache, checkpoints omit it."""
    import scripts.run_ppo as run_ppo

    student, teacher = _tiny(), _tiny(seed=6)
    trainer = ppo.PPOTrainer(
        env=_FakeKaggricultureEnv(),
        model=student,
        optimizer=torch.optim.AdamW(student.parameters(), lr=0.0),
        config=ppo.PPOConfig(
            horizon=4,
            segments_per_minibatch=1,
            teacher_mode="last_best",
            teacher_segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher,
        teacher_active=True,
    )
    assert trainer.train_iteration()["teacher/cache_bytes"] > 0
    for name, model in (("checkpoint.pt", None), ("checkpoint_last_best.pt", teacher)):
        path = tmp_path / name
        trainer.write_checkpoint(path, env_steps=8, wandb_run_id="run-123", model=model)
        saved = torch.load(path, weights_only=False)
        assert set(saved) == CHECKPOINT_KEYS
        run_ppo._checkpoint_metadata(saved, path=path)
        _assert_no_teacher_state(saved["model"])
