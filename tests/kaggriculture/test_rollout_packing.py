"""Exact host-mask packing and observation-scoped rollout integration."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from owl.model import kaggriculture as km
from owl.model import stateless_transformer_v1 as st
from owl.model.attn import flash_attn_available, varlen_attention
from owl.train.ppo import forward_learner_rows

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.test_model_encoder import _tiny


@pytest.mark.parametrize("seed", [1, 17, 54])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_cpu_packing_matches_existing_indices_metadata_and_outputs(
    seed: int, noncontiguous: bool
) -> None:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((4, 13), generator=generator) > 0.6
    mask[:, 0] = True
    if noncontiguous:
        mask = mask.transpose(0, 1).contiguous().transpose(0, 1)
        assert not mask.is_contiguous()
    x = torch.randn((4, 13, 7), generator=generator)
    expected_x, expected = st.pack_sequence(x, mask, max_seqlen=13)
    actual_x, actual = st.pack_sequence_from_cpu_mask(x, mask)
    assert torch.equal(actual_x, expected_x)
    assert actual_x.shape[0] == int(mask.sum())
    assert torch.equal(actual.indices, expected.indices)
    assert torch.equal(actual.seqlens, expected.seqlens)
    assert torch.equal(actual.cu_seqlens, expected.cu_seqlens)
    assert actual.max_seqlen == expected.max_seqlen == 13
    assert actual.batch_size == expected.batch_size == 4
    assert actual.padded_seq_len == expected.padded_seq_len == 13
    assert torch.equal(
        st.unpack_sequence(actual_x, actual),
        st.unpack_sequence(expected_x, expected),
    )


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (torch.ones(2, 3), "2D boolean"),
        (torch.ones(6, dtype=torch.bool), "2D boolean"),
        (torch.ones(3, 2, dtype=torch.bool), "shape must match"),
        (torch.zeros(2, 3, dtype=torch.bool), "at least one unmasked"),
        (torch.ones(2, 3, dtype=torch.bool, device="meta"), "CPU token mask"),
    ],
)
def test_cpu_packing_rejects_invalid_host_masks(mask: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        st.pack_sequence_from_cpu_mask(torch.ones(2, 3, 4), mask)


def test_cpu_packing_debug_validation_checks_current_mask() -> None:
    mask = torch.tensor([[True, False, True]])
    x = torch.ones(1, 3, 2)
    expected, _ = st.pack_sequence_from_cpu_mask(x, mask, token_mask=mask.clone())
    assert expected.shape == (2, 2)
    with pytest.raises(ValueError, match="differs from the current"):
        st.pack_sequence_from_cpu_mask(x, mask, token_mask=~mask)
    with pytest.raises(ValueError, match="device token mask must match"):
        st.pack_sequence_from_cpu_mask(x, mask, token_mask=torch.ones(1, 3))
    with pytest.raises(ValueError, match="nonempty input"):
        st.pack_sequence_from_cpu_mask(torch.ones(0, 3, 2), mask[:0])


def test_host_metadata_path_does_not_read_accelerator_tensor_values() -> None:
    # Meta tensors cannot supply data-dependent nonzero sizes or host scalars.
    # Successful packing therefore catches accidental device-side sizing.
    mask = torch.tensor([[True, False, True], [False, True, False]])
    packed_x, packed = st.pack_sequence_from_cpu_mask(
        torch.empty(2, 3, 4, device="meta"), mask
    )
    assert packed_x.shape == (3, 4)
    assert packed.indices.device.type == "meta"
    assert packed.cu_seqlens.device.type == "meta"
    assert packed.seqlens.device.type == "meta"
    assert packed.max_seqlen == 3


def test_rollout_context_uses_current_masks_selected_in_flat_seat_order() -> None:
    model = _tiny()
    obs = make_obs(envs=2, own_actors=[1, 5], rival_actors=[3, 4], shops=[1, 5])
    obs.still_playing[0, 1] = False
    _, expected = model._assemble_tokens(obs)
    selected = torch.tensor([[False, True], [True, False]])
    with model.rollout_packing(obs, learner_mask=selected):
        assert torch.equal(model._rollout_token_mask_cpu, expected[[1, 2]])
        obs.actor_mask.zero_()
        # The native input buffers can be recycled only after the forward;
        # this fresh host mask is not a view into those mutable buffers.
        assert torch.equal(model._rollout_token_mask_cpu, expected[[1, 2]])
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            model.rollout_packing(obs),
        ):
            pass
    assert model._rollout_token_mask_cpu is None
    _, updated = model._assemble_tokens(obs)

    def interrupted_forward() -> None:
        with model.rollout_packing(obs):
            assert torch.equal(model._rollout_token_mask_cpu, updated)
            raise RuntimeError("sentinel")

    with pytest.raises(RuntimeError, match="sentinel"):
        interrupted_forward()
    assert model._rollout_token_mask_cpu is None


@pytest.mark.parametrize("chunked", [False, True])
def test_rollout_packed_trunk_keeps_order_bounds_and_unchanged_default(
    monkeypatch: pytest.MonkeyPatch, chunked: bool
) -> None:
    model = _tiny()
    obs = make_obs(envs=2, own_actors=[1, 5], rival_actors=[3, 4], shops=[1, 5])
    x, mask = model._assemble_tokens(obs)
    calls: list[int] = []

    def trunk(
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        packed: st.PackedSequence | None,
    ) -> torch.Tensor:
        assert token_mask is None
        assert packed is not None
        assert packed.max_seqlen == x.shape[1]
        calls.append(tokens.shape[0])
        return tokens + 1

    width = km.trunk_gemm_width(model.config)
    if chunked:
        monkeypatch.setattr(
            km, "_GEMM_ELEMENT_LIMIT", int(mask.sum(1).max()) * width + 1
        )
    expected = model._run_packed_trunk(trunk, x, mask, width)
    original_calls = calls.copy()
    calls.clear()
    with torch.no_grad(), model.rollout_packing(obs):
        actual = model._run_packed_trunk(trunk, x, mask, width)
    assert torch.equal(expected, actual)
    assert calls == original_calls
    assert sum(calls) == int(mask.sum())
    assert len(calls) == (4 if chunked else 1)
    # Outside the context the existing packing dispatch remains selected.
    monkeypatch.setattr(
        km,
        "pack_sequence_from_cpu_mask",
        lambda *_a, **_k: pytest.fail("default path used host packing"),
    )
    assert torch.equal(model._run_packed_trunk(trunk, x, mask, width), expected)
    with model.rollout_packing(obs), pytest.raises(RuntimeError, match="no-grad"):
        model._run_packed_trunk(trunk, x, mask, width)


def _cpu_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    outputs = []
    for start, stop in pairwise(cu_seqlens):
        assert int(stop - start) <= max_seqlen
        outputs.append(
            F.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1),
                k[start:stop].transpose(0, 1),
                v[start:stop].transpose(0, 1),
            ).transpose(0, 1)
        )
    return torch.cat(outputs)


@pytest.mark.parametrize("packed", [False, True])
def test_rollout_packing_preserves_model_sampling_density_value_and_rng(
    monkeypatch: pytest.MonkeyPatch, packed: bool
) -> None:
    model = _tiny(depth=1)
    obs = make_obs(envs=2, own_actors=[1, 5], rival_actors=[3, 4])
    monkeypatch.setattr(km, "use_flash_attn", lambda _: packed)
    monkeypatch.setattr(st, "varlen_attention", _cpu_varlen)
    with torch.no_grad():
        torch.manual_seed(55)
        expected = model(obs)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(55)
        with model.rollout_packing(obs):
            actual = model(obs)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        assert torch.equal(actual.actions.tokens, expected.actions.tokens)
        assert torch.equal(actual.actions.lengths, expected.actions.lengths)
        assert torch.equal(actual.log_probs.event, expected.log_probs.event)
        assert torch.equal(actual.entropies.event, expected.entropies.event)
        assert torch.equal(actual.values, expected.values)
        with model.rollout_packing(obs):
            assert torch.equal(model.compute_value(obs), expected.values)
        learner = torch.tensor([[False, True], [True, False]])
        torch.manual_seed(71)
        expected_rows = forward_learner_rows(model, obs, learner)
        torch.manual_seed(71)
        with model.rollout_packing(obs, learner_mask=learner):
            actual_rows = forward_learner_rows(model, obs, learner)
        assert torch.equal(actual_rows.actions.tokens, expected_rows.actions.tokens)
        assert torch.equal(actual_rows.logp, expected_rows.logp)
        assert torch.equal(actual_rows.values, expected_rows.values)


@pytest.mark.skipif(
    not flash_attn_available() or not torch.cuda.is_available(),
    reason="flash-attn CUDA backend is not available",
)
def test_cuda_host_packing_and_varlen_capacity_match_existing_path() -> None:
    # Capacity spans extra query blocks, exercising the kernel's early exits.
    mask = torch.zeros(2, 257, dtype=torch.bool)
    mask[0, [0, 2]] = True
    mask[1, :3] = True
    x = torch.randn(2, 257, 2, 32, device="cuda", dtype=torch.bfloat16)
    actual_x, actual = st.pack_sequence_from_cpu_mask(x, mask)
    expected_x, expected = st.pack_sequence(x, mask.cuda(), max_seqlen=257)
    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual.indices, expected.indices)
    assert torch.equal(actual.cu_seqlens, expected.cu_seqlens)
    args: dict[str, Any] = {"cu_seqlens": actual.cu_seqlens}
    bounded = varlen_attention(actual_x, actual_x, actual_x, max_seqlen=257, **args)
    tight = varlen_attention(expected_x, expected_x, expected_x, max_seqlen=3, **args)
    torch.testing.assert_close(bounded, tight, rtol=0, atol=0)
