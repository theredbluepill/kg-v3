"""Task 2.1: the Kaggriculture encoder on Isaiah's StatelessTransformerV1 topology."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model.stateless_transformer_v1 import (
    ObservationInputStem,
    StatelessTransformerV1,
    TransformerBlock,
)
from owl.train.optimizer import MuonConfig, create_optimizer
from torch import nn

from tests.kaggriculture.conftest import make_obs

ROOT = Path(__file__).resolve().parents[2]
SHARED_TRUNK_FIELDS = (
    "embed_dim",
    "depth",
    "n_heads",
    "mlp_ratio",
    "activation",
    "n_scratch_tokens",
    "force_flash_attn",
)


def _tiny(**overrides: Any) -> km.KaggricultureTransformer:
    torch.manual_seed(3)
    fields: dict[str, Any] = {
        "embed_dim": 16,
        "depth": 2,
        "n_heads": 2,
        "mlp_ratio": 2.0,
        "n_scratch_tokens": 2,
    }
    config = km.KaggricultureTransformerConfig(**(fields | overrides))
    return km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    )


def _isaiah_6m() -> StatelessTransformerV1:
    from owl.model import create_model
    from owl.train import FullConfig

    cfg = FullConfig.from_file(
        ROOT / "configs/scaling_6m.yaml", {"model.force_flash_attn": False}
    )
    model = create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    assert isinstance(model, StatelessTransformerV1)
    return model


def _preset() -> km.KaggricultureTransformerConfig:
    return km.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml"
    )


# --- topology -----------------------------------------------------------------


def test_stems_tokens_and_trunk_use_isaiah_classes_and_widths() -> None:
    model = _tiny()
    widths = {
        "tile_proj": 133,
        "actor_proj": 371,
        "shop_proj": 16,
        "market_proj": 11,
        "player_feature_proj": 44,
        "global_proj": 15,
    }
    for name, width in widths.items():
        stem = getattr(model, name)
        assert type(stem) is ObservationInputStem, name
        assert stem.in_features == width, name
        assert stem.input.out_features == int(16 * 2.0), name
    assert all(type(block) is TransformerBlock for block in model.blocks)
    assert len(model.blocks) == 2
    assert type(model.final_norm) is nn.LayerNorm
    assert model.player_tokens.shape == (2, 16)
    assert model.board_tokens.shape == (2, 16)
    assert model.actor_plan_tokens.shape == (1, 16)
    assert model.critic_value_tokens.shape == (2, 16)
    assert not any(isinstance(m, nn.Embedding) for m in model.modules())


def test_preset_shares_isaiah_6m_trunk_except_depth() -> None:
    ours = _preset()
    isaiah = _isaiah_6m().config
    for field in SHARED_TRUNK_FIELDS:
        if field in {"depth", "force_flash_attn"}:
            continue
        assert getattr(ours, field) == getattr(isaiah, field), field
    assert ours.depth == 8
    assert ours.force_flash_attn is True
    assert ours.embed_dim // ours.n_heads == 32


def test_isaiah_6m_uses_the_same_shared_classes() -> None:
    isaiah = _isaiah_6m()
    assert type(isaiah.global_proj) is ObservationInputStem
    assert all(type(block) is TransformerBlock for block in isaiah.blocks)
    assert type(isaiah.final_norm) is nn.LayerNorm


def test_swiglu_is_rejected() -> None:
    with pytest.raises(ValueError, match="activation"):
        km.KaggricultureTransformerConfig(activation="swiglu")  # type: ignore[arg-type]


def test_initialization_follows_isaiah_scheme() -> None:
    model = _tiny()
    for block in model.blocks:
        assert block.attn.out.bias is not None
        assert torch.count_nonzero(block.attn.out.bias) == 0
        # orthogonal with gain 1/sqrt(2*depth): singular values equal the gain
        gain = 1.0 / math.sqrt(2.0 * 2)
        singular = torch.linalg.svdvals(block.mlp.down.weight)
        torch.testing.assert_close(singular, torch.full_like(singular, gain))
    singular = torch.linalg.svdvals(model.tile_proj.input.weight)
    torch.testing.assert_close(
        singular[: min(model.tile_proj.input.weight.shape)], torch.ones_like(singular)
    )
    assert torch.count_nonzero(model.final_norm.bias) == 0


# --- encoded fields, masks, statelessness, seat isolation ---------------------


def test_encoded_fields_have_named_shapes_and_masked_zeros() -> None:
    model = _tiny().eval()
    obs = make_obs(envs=2)
    with torch.inference_mode():
        enc = model.encode_observations(obs)
    batch, tokens = 4, km.sequence_length(model.config)
    assert tokens == 482 + 200 + 8 + 9 + 2 + 1 + 2 + 1 + 2
    assert enc.hidden.shape == (batch, tokens, 16)
    assert enc.own_actor_hidden.shape == (batch, kt.MAX_ACTORS, 16)
    assert enc.player_hidden.shape == (batch, 2, 16)
    assert enc.global_hidden.shape == (batch, 1, 16)
    assert enc.board_hidden.shape == (batch, 2, 16)
    assert enc.actor_plan_hidden.shape == (batch, 1, 16)
    assert enc.critic_value_hidden.shape == (batch, 2, 16)
    masked = ~enc.token_mask
    assert torch.count_nonzero(enc.hidden[masked]) == 0
    assert int(model.count_non_masked_tokens(obs)) == int(enc.token_mask.sum())


def test_offsets_and_pre_trunk_player_addition(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny().eval()
    monkeypatch.setattr(model, "_forward_transformer_trunk", lambda x, *_: x)
    obs = make_obs()
    with torch.inference_mode():
        enc = model.encode_observations(obs)
        players = model.player_tokens + model.player_feature_proj(
            obs.player_features.reshape(-1, 2, kt.PLAYER_FEATURE_CHANNELS)
        )
        torch.testing.assert_close(enc.player_hidden, players)
        torch.testing.assert_close(
            enc.global_hidden[:, 0],
            model.global_proj(
                obs.global_features.reshape(-1, kt.GLOBAL_FEATURE_CHANNELS)
            ),
        )
        torch.testing.assert_close(
            enc.actor_plan_hidden[:, 0], model.actor_plan_tokens[0].expand(2, -1)
        )
        torch.testing.assert_close(
            enc.critic_value_hidden, model.critic_value_tokens.expand(2, -1, -1)
        )


def test_masked_actors_and_shops_do_not_affect_present_tokens() -> None:
    model = _tiny().eval()
    obs = make_obs()
    changed = make_obs()
    changed.actors_float[..., 300, :] += 5.0  # absent rival slot
    changed.shop_type[..., 7] = 3  # masked shop slot
    with torch.inference_mode():
        a, b = model.encode_observations(obs), model.encode_observations(changed)
    present = a.token_mask
    torch.testing.assert_close(a.hidden[present], b.hidden[present])


def test_hidden_state_is_rejected_and_heads_are_deferred() -> None:
    model = _tiny()
    obs = make_obs()
    for call in (
        lambda: model(obs, hidden_state=object()),
        lambda: model.compute_value(obs, hidden_state=object()),
    ):
        with pytest.raises(ValueError, match="stateless"):
            call()
    with pytest.raises(NotImplementedError, match=r"Task 2\.3"):
        model(obs)


def test_unexpected_recurrent_checkpoint_keys_are_rejected() -> None:
    model = _tiny()
    state = model.state_dict()
    state["recurrent_hidden_init"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="recurrent_hidden_init"):
        model.load_state_dict(state)


def test_outputs_depend_only_on_current_observation() -> None:
    model = _tiny().eval()
    obs = make_obs()
    with torch.inference_mode():
        first = model.encode_observations(obs).hidden
        model.encode_observations(make_obs(envs=3, own_actors=7))
        again = model.encode_observations(obs).hidden
    torch.testing.assert_close(first, again)


def test_seat_rows_are_encoded_independently() -> None:
    model = _tiny().eval()
    obs = make_obs()
    changed = make_obs()
    changed.tiles_float[:, 1] += 1.0
    with torch.inference_mode():
        a, b = (
            model.encode_observations(obs).hidden,
            model.encode_observations(changed).hidden,
        )
    torch.testing.assert_close(a[0], b[0])  # seat 0 row unchanged
    assert not torch.allclose(a[1], b[1])  # seat 1 row responds


# --- L6 guard and dispatch ----------------------------------------------------


def test_rows_per_chunk_boundary() -> None:
    kmax = 512
    tokens = 709
    rows = km.rows_per_chunk(tokens=tokens, kmax=kmax)
    assert rows * tokens * kmax < 2**31
    assert (rows + 1) * tokens * kmax >= 2**31
    assert km.rows_per_chunk(tokens=2**21, kmax=1024) == 0


def test_kmax_is_max_of_embed_and_mlp_hidden() -> None:
    assert km.gemm_kmax(_tiny().config) == 32
    assert km.gemm_kmax(_tiny(mlp_ratio=0.5).config) == 16
    assert km.gemm_kmax(_preset()) == 512


def test_padded_chunking_matches_unchunked(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny().eval()
    obs = make_obs(envs=3)
    calls: list[int] = []
    original = model._forward_transformer_trunk

    def spy(x: torch.Tensor, mask: torch.Tensor | None, packed: object) -> torch.Tensor:
        calls.append(x.shape[0])
        assert packed is None
        return original(x, mask, None)

    with torch.inference_mode():
        whole = model.encode_observations(obs).hidden
        monkeypatch.setattr(model, "_forward_transformer_trunk", spy)
        monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 2)
        chunked = model.encode_observations(obs).hidden
    assert calls == [2, 2, 2]
    torch.testing.assert_close(whole, chunked)


def test_zero_rows_per_chunk_rejects_before_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 0)
    monkeypatch.setattr(
        model, "_forward_transformer_trunk", lambda *_: pytest.fail("trunk ran")
    )
    with pytest.raises(ValueError, match="compiled-gemm-template-overflows"):
        model.encode_observations(make_obs())


def test_compiled_trunk_is_used_for_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny().eval()
    compiled_calls: list[int] = []

    def fake_compile(fn: Any, *, mode: str, dynamic: bool) -> Any:
        assert dynamic is True
        assert mode == "max-autotune-no-cudagraphs"

        def run(
            x: torch.Tensor, mask: torch.Tensor | None, packed: object
        ) -> torch.Tensor:
            compiled_calls.append(x.shape[0])
            return fn(x, mask, packed)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs") == 1
    monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 1)
    with torch.inference_mode():
        model.encode_observations(make_obs())
    assert compiled_calls == [1, 1]


def test_forced_flash_without_kernel_fails_before_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny(force_flash_attn=True)
    monkeypatch.setattr(km, "_requires_flash_attn", lambda *_, **__: True)
    monkeypatch.setattr(km, "use_flash_attn", lambda *_: False)
    monkeypatch.setattr(
        model, "_forward_transformer_trunk", lambda *_: pytest.fail("trunk ran")
    )
    with pytest.raises(RuntimeError, match="force_flash_attn"):
        model.encode_observations(make_obs())


# --- Muon grouping ------------------------------------------------------------


def test_muon_groups_follow_isaiah_rule() -> None:
    model = _tiny()
    optimizer = create_optimizer(model, MuonConfig())
    muon_ids = {
        id(p)
        for inner in optimizer.optimizers
        if isinstance(inner, torch.optim.Muon)
        for group in inner.param_groups
        for p in group["params"]
    }
    for name in (
        "tile_proj",
        "actor_proj",
        "shop_proj",
        "market_proj",
        "player_feature_proj",
        "global_proj",
    ):
        stem = getattr(model, name)
        assert id(stem.input.weight) not in muon_ids, name
        assert id(stem.output.weight) in muon_ids, name
        assert id(stem.output.bias) not in muon_ids, name
    for token in (
        model.player_tokens,
        model.board_tokens,
        model.actor_plan_tokens,
        model.critic_value_tokens,
    ):
        assert id(token) not in muon_ids
    assert id(model.final_norm.weight) not in muon_ids


# --- Task 2.2: critic ---------------------------------------------------------


def test_critic_is_isaiah_output_projection_shared_across_players() -> None:
    from owl.model.actor.common import OutputProjectionMLP

    model = _tiny()
    assert type(model.critic_head) is OutputProjectionMLP
    assert model.critic_head.out.out_features == 1
    assert model.get_output_layers() == (model.critic_head.out,)


def test_values_are_two_p_self_minus_one_from_a_winner_softmax() -> None:
    model = _tiny().eval()
    obs = make_obs(envs=3)
    with torch.inference_mode():
        values = model.compute_value(obs)
        log_probs = model.winner_log_probabilities(obs)
    assert values.shape == (3, 2)
    assert log_probs.shape == (3, 2, 2)
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(3, 2))
    torch.testing.assert_close(values, 2 * log_probs[..., 0].exp() - 1)
    assert bool(((values > -1) & (values < 1)).all())


def test_critic_logits_follow_player_token_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny().eval()
    obs = make_obs()
    with torch.inference_mode():
        enc = model.encode_observations(obs)
        swapped = KaggricultureEncodedSwap(enc)
        monkeypatch.setattr(model, "encode_observations", lambda _obs: swapped)
        flipped = model.winner_log_probabilities(obs)
        monkeypatch.undo()
        base = model.winner_log_probabilities(obs)
    torch.testing.assert_close(flipped, base.flip(-1))


def KaggricultureEncodedSwap(enc: km.KaggricultureEncoded) -> km.KaggricultureEncoded:
    from dataclasses import replace

    return replace(enc, critic_value_hidden=enc.critic_value_hidden.flip(1))


def test_critic_output_init_and_muon_group() -> None:
    model = _tiny()
    singular = torch.linalg.svdvals(model.critic_head.out.weight)
    torch.testing.assert_close(singular, torch.ones_like(singular))
    assert model.critic_head.out.bias is not None
    assert torch.count_nonzero(model.critic_head.out.bias) == 0
    optimizer = create_optimizer(model, MuonConfig())
    muon_ids = {
        id(p)
        for inner in optimizer.optimizers
        if isinstance(inner, torch.optim.Muon)
        for group in inner.param_groups
        for p in group["params"]
    }
    assert id(model.critic_head.out.weight) not in muon_ids
    assert id(model.critic_head.up.weight) in muon_ids


def test_values_are_seat_independent() -> None:
    model = _tiny().eval()
    obs = make_obs()
    changed = make_obs()
    changed.global_features[:, 1] += 2.0
    with torch.inference_mode():
        a, b = model.compute_value(obs), model.compute_value(changed)
    torch.testing.assert_close(a[:, 0], b[:, 0])
