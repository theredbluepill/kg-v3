"""Phase 4: Kaggriculture teacher distillation (brief ``briefs/4-teacher.md`` v2).

Tiny CPU models on the synthetic ``expected_grammar_tables``. The 4.1 block pins
the replay-conditioned per-slot KL computed inside ``policy_core``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model import kaggriculture_actor as ka

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import (
    HIRE,
    NONE,
    F,
    K,
    _base_case,
    _obs_double,
    _replay_case,
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
