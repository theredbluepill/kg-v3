"""Task 2.1: the Kaggriculture encoder on Isaiah's StatelessTransformerV1 topology."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model.stateless_transformer_v1 import (
    ObservationInputStem,
    PackedSequence,
    StatelessTransformerV1,
    TransformerBlock,
)
from owl.train.optimizer import MuonConfig, create_optimizer
from torch import nn

from tests.kaggriculture.conftest import make_obs

ROOT = Path(__file__).resolve().parents[2]
STEMS = (
    "tile_proj",
    "actor_proj",
    "shop_proj",
    "market_proj",
    "player_feature_proj",
    "global_proj",
)
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


def _assert_orthogonal_gain(linear: nn.Linear, gain: float, name: str) -> None:
    """Orthogonal init with ``gain``: every singular value equals the gain."""
    singular = torch.linalg.svdvals(linear.weight.detach())
    torch.testing.assert_close(
        singular, torch.full_like(singular, gain), rtol=1e-5, atol=1e-5, msg=name
    )
    assert linear.bias is not None, name
    assert int(torch.count_nonzero(linear.bias)) == 0, name


def test_initialization_follows_isaiah_scheme() -> None:
    model = _tiny(embed_dim=64, depth=2)
    residual = 1.0 / math.sqrt(2.0 * 2)
    hidden = math.sqrt(2.0)
    for index, block in enumerate(model.blocks):
        _assert_orthogonal_gain(block.attn.out, residual, f"blocks.{index}.attn.out")
        _assert_orthogonal_gain(block.mlp.down, residual, f"blocks.{index}.mlp.down")
        for linear, name in (
            (block.attn.q, "q"),
            (block.attn.k, "k"),
            (block.attn.v, "v"),
        ):
            _assert_orthogonal_gain(linear, hidden, f"blocks.{index}.attn.{name}")
        _assert_orthogonal_gain(block.mlp.up, hidden, f"blocks.{index}.mlp.up")
    for name in STEMS:
        stem = getattr(model, name)
        _assert_orthogonal_gain(stem.input, 1.0, f"{name}.input")
        _assert_orthogonal_gain(stem.output, hidden, f"{name}.output")
    _assert_orthogonal_gain(model.critic_head.up, hidden, "critic_head.up")
    _assert_orthogonal_gain(model.critic_head.out, 1.0, "critic_head.out")
    norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    assert len(norms) == 2 * 2 + 1  # two per block plus final_norm
    for norm in norms:
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
        assert int(torch.count_nonzero(norm.bias)) == 0
    tokens = torch.cat(
        [
            token.detach().flatten()
            for token in (
                model.player_tokens,
                model.board_tokens,
                model.actor_plan_tokens,
                model.critic_value_tokens,
            )
        ]
    )
    target = 64**-0.5
    assert abs(float(tokens.std()) - target) < 0.25 * target
    assert abs(float(tokens.mean())) < 4 * target / math.sqrt(tokens.numel())


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


def _marker_stem(base: float, width: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """A stem stub whose token ``i`` is the constant ``base + i``."""

    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return torch.full((x.shape[0], width), base)
        index = torch.arange(x.shape[1], dtype=torch.float32)
        return (base + index)[None, :, None].expand(x.shape[0], -1, width).clone()

    return forward


def _marked_model(monkeypatch: pytest.MonkeyPatch) -> km.KaggricultureTransformer:
    """Identity trunk, region-marker stems and distinct constant tokens."""
    model = _tiny().eval()
    width = model.config.embed_dim
    for name, base in (
        ("actor_proj", 1000.0),
        ("tile_proj", 2000.0),
        ("shop_proj", 3000.0),
        ("market_proj", 4000.0),
        ("player_feature_proj", 5000.0),
        ("global_proj", 6000.0),
    ):
        monkeypatch.setattr(getattr(model, name), "forward", _marker_stem(base, width))
    with torch.no_grad():
        model.player_tokens.copy_(torch.tensor([[10.0], [20.0]]).expand(-1, width))
        model.board_tokens.copy_(torch.tensor([[7000.0], [7001.0]]).expand(-1, width))
        model.actor_plan_tokens.fill_(8000.0)
        model.critic_value_tokens.copy_(
            torch.tensor([[9000.0], [9001.0]]).expand(-1, width)
        )
    monkeypatch.setattr(model, "_forward_transformer_trunk", lambda x, *_: x)
    return model


def test_every_region_and_named_readout_sits_at_its_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _marked_model(monkeypatch)
    obs = make_obs(envs=2, own_actors=(1, 3), rival_actors=(2, 1), shops=(2, 5))
    obs.still_playing[1, 0] = False  # flattened row 2
    with torch.inference_mode():
        enc = model.encode_observations(obs)
    rows, width = 4, model.config.embed_dim
    playing = obs.still_playing.reshape(rows, 1)
    # Contract token order, with region sizes written out independently.
    regions = (
        ("actors", torch.arange(482) + 1000.0, obs.actor_mask.reshape(rows, -1)),
        ("tiles", torch.arange(200) + 2000.0, torch.ones(rows, 200, dtype=torch.bool)),
        ("shops", torch.arange(8) + 3000.0, obs.shop_mask.reshape(rows, -1)),
        ("market", torch.arange(9) + 4000.0, torch.ones(rows, 9, dtype=torch.bool)),
        ("players", torch.tensor([5010.0, 5021.0]), playing.expand(rows, 2)),
        ("global", torch.tensor([6000.0]), torch.ones(rows, 1, dtype=torch.bool)),
        (
            "board",
            torch.tensor([7000.0, 7001.0]),
            torch.ones(rows, 2, dtype=torch.bool),
        ),
        ("plan", torch.tensor([8000.0]), playing),
        ("critic", torch.tensor([9000.0, 9001.0]), playing.expand(rows, 2)),
    )
    expected_mask = torch.cat([mask for _, _, mask in regions], dim=1)
    expected = torch.cat(
        [values.expand(rows, -1) * mask for _, values, mask in regions], dim=1
    )
    expected = expected.unsqueeze(-1).expand(-1, -1, width)
    assert torch.equal(enc.token_mask, expected_mask)
    start = 0
    spans: dict[str, slice] = {}
    for name, values, _ in regions:
        spans[name] = slice(start, start + values.numel())
        torch.testing.assert_close(
            enc.hidden[:, spans[name]], expected[:, spans[name]], msg=name
        )
        start += values.numel()
    assert enc.hidden.shape == (rows, start, width)
    readouts = {
        "own_actor_hidden": (enc.own_actor_hidden, slice(0, 241)),
        "player_hidden": (enc.player_hidden, spans["players"]),
        "global_hidden": (enc.global_hidden, spans["global"]),
        "board_hidden": (enc.board_hidden, spans["board"]),
        "actor_plan_hidden": (enc.actor_plan_hidden, spans["plan"]),
        "critic_value_hidden": (enc.critic_value_hidden, spans["critic"]),
    }
    for name, (readout, span) in readouts.items():
        torch.testing.assert_close(readout, expected[:, span], msg=name)
    # own actors are the first own_count tokens, each carrying its slot marker
    torch.testing.assert_close(enc.own_actor_hidden[0, 0], torch.full((width,), 1000.0))
    assert int(torch.count_nonzero(enc.own_actor_hidden[0, 1:])) == 0
    # still_playing=False zeroes that row's player, plan and critic tokens only
    for readout in (
        enc.player_hidden,
        enc.actor_plan_hidden,
        enc.critic_value_hidden,
    ):
        assert int(torch.count_nonzero(readout[2])) == 0
        assert bool((readout[[0, 1, 3]] != 0).all())
    assert bool((enc.global_hidden[2] == 6000.0).all())
    assert int(model.count_non_masked_tokens(obs)) == int(expected_mask.sum())


def test_pre_trunk_player_addition_and_global_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_every_api_rejects_hidden_state_and_heads_are_deferred() -> None:
    model = _tiny()
    obs = make_obs()
    actions = kt.KaggricultureActions(
        tokens=torch.zeros((1, 2, kt.MAX_FRAMES, kt.ACTION_SLOTS), dtype=torch.int64),
        lengths=torch.ones((1, 2), dtype=torch.int64),
    )
    state = object()
    for call in (
        lambda: model(obs, hidden_state=state),
        lambda: model.evaluate_actions(obs, actions, hidden_state=state),
        lambda: model.serve(obs, hidden_state=state),
        lambda: model.compute_value(obs, hidden_state=state),
    ):
        with pytest.raises(ValueError, match="stateless"):
            call()
    for call in (
        lambda: model(obs),
        lambda: model.evaluate_actions(obs, actions),
        lambda: model.serve(obs),
        lambda: model.serve(obs, deterministic=True),
    ):
        with pytest.raises(NotImplementedError, match=r"Task 2\.3"):
            call()


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


@pytest.mark.parametrize(("chunk", "sizes"), [(2, [2, 2, 2]), (4, [4, 2])])
def test_padded_chunks_dispatch_exact_row_slices_in_order(
    monkeypatch: pytest.MonkeyPatch, chunk: int, sizes: list[int]
) -> None:
    model = _tiny().eval()
    # distinct per-row actor and shop masks (seat 1 sees the counts swapped)
    obs = make_obs(
        envs=3, own_actors=(1, 2, 3), rival_actors=(4, 5, 6), shops=(1, 4, 8)
    )
    with torch.inference_mode():
        x_full, mask_full = model._assemble_tokens(obs)
        whole = model.encode_observations(obs).hidden
    assert len({tuple(row.tolist()) for row in mask_full}) == 6
    tokens, kmax = x_full.shape[1], km.gemm_kmax(model.config)
    monkeypatch.setattr(km, "_GEMM_ELEMENT_LIMIT", chunk * tokens * kmax + 1)
    assert km.rows_per_chunk(tokens=tokens, kmax=kmax) == chunk
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = model._forward_transformer_trunk

    def spy(x: torch.Tensor, mask: torch.Tensor | None, packed: object) -> torch.Tensor:
        assert packed is None
        assert mask is not None
        assert x.shape[0] * tokens * kmax < km._GEMM_ELEMENT_LIMIT  # strict bound
        seen.append((x.clone(), mask.clone()))
        return original(x, mask, None)

    monkeypatch.setattr(model, "_forward_transformer_trunk", spy)
    with torch.inference_mode():
        chunked = model.encode_observations(obs).hidden
    assert [x.shape[0] for x, _ in seen] == sizes
    start = 0
    for x, mask in seen:
        stop = start + x.shape[0]
        torch.testing.assert_close(x, x_full[start:stop])
        assert torch.equal(mask, mask_full[start:stop])
        start = stop
    assert start == x_full.shape[0]
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


# --- packed (flash) dispatch, CPU-mocked ------------------------------------------


@dataclass
class _PackedSpy:
    pack_calls: list[tuple[int | None, int, int]] = dataclasses.field(
        default_factory=list
    )
    unpack_calls: int = 0
    trunk_calls: list[tuple[int, bool, bool]] = dataclasses.field(default_factory=list)


def _mock_flash(
    monkeypatch: pytest.MonkeyPatch, model: km.KaggricultureTransformer
) -> _PackedSpy:
    """Force the packed path on CPU; stub the trunk (varlen attention needs CUDA)."""
    spy = _PackedSpy()
    real_pack, real_unpack = km.pack_sequence, km.unpack_sequence

    def pack(
        x: torch.Tensor, mask: torch.Tensor, *, max_seqlen: int | None = None
    ) -> tuple[torch.Tensor, PackedSequence]:
        packed_x, packed = real_pack(x, mask, max_seqlen=max_seqlen)
        spy.pack_calls.append((max_seqlen, packed_x.shape[0], int(mask.sum())))
        return packed_x, packed

    def unpack(x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
        spy.unpack_calls += 1
        return real_unpack(x, packed)

    def trunk(
        x: torch.Tensor, mask: torch.Tensor | None, packed: PackedSequence | None
    ) -> torch.Tensor:
        spy.trunk_calls.append((x.shape[0], mask is None, packed is not None))
        return x + 1.0

    monkeypatch.setattr(km, "use_flash_attn", lambda *_: True)
    monkeypatch.setattr(km, "pack_sequence", pack)
    monkeypatch.setattr(km, "unpack_sequence", unpack)
    monkeypatch.setattr(model, "_forward_transformer_trunk", trunk)
    return spy


def test_packed_dispatch_packs_once_and_unpacks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny().eval()
    obs = make_obs(envs=2, own_actors=(1, 3), rival_actors=(2, 1))
    with torch.inference_mode():
        x_full, mask_full = model._assemble_tokens(obs)
    spy = _mock_flash(monkeypatch, model)
    with torch.inference_mode():
        enc = model.encode_observations(obs)
    valid = int(mask_full.sum())
    assert spy.pack_calls == [(km.sequence_length(model.config), valid, valid)]
    assert spy.unpack_calls == 1
    assert spy.trunk_calls == [(valid, True, True)]
    expected = (x_full + 1.0).masked_fill(~mask_full.unsqueeze(-1), 0.0)
    torch.testing.assert_close(enc.hidden, expected)
    assert torch.equal(enc.token_mask, mask_full)


@pytest.mark.parametrize(("offset", "safe"), [(1, True), (0, False), (-1, False)])
def test_packed_overflow_boundary_rejects_before_trunk(
    monkeypatch: pytest.MonkeyPatch, offset: int, safe: bool
) -> None:
    model = _tiny().eval()
    obs = make_obs()
    with torch.inference_mode():
        _, mask_full = model._assemble_tokens(obs)
    elements = int(mask_full.sum()) * km.gemm_kmax(model.config)
    spy = _mock_flash(monkeypatch, model)
    monkeypatch.setattr(km, "_GEMM_ELEMENT_LIMIT", elements + offset)
    with torch.inference_mode():
        if safe:
            model.encode_observations(obs)
        else:
            with pytest.raises(ValueError, match="compiled-gemm-template-overflows"):
                model.encode_observations(obs)
    assert len(spy.pack_calls) == 1
    assert len(spy.trunk_calls) == (1 if safe else 0)
    assert spy.unpack_calls == (1 if safe else 0)


def test_compiled_trunk_serves_packed_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny().eval()
    spy = _mock_flash(monkeypatch, model)
    compiled: list[bool] = []

    def fake_compile(fn: Any, *, mode: str, dynamic: bool) -> Any:
        assert (mode, dynamic) == ("max-autotune-no-cudagraphs", True)

        def run(
            x: torch.Tensor, mask: torch.Tensor | None, packed: PackedSequence | None
        ) -> torch.Tensor:
            compiled.append(mask is None and packed is not None)
            return fn(x, mask, packed)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs")
    monkeypatch.setattr(
        model, "_forward_transformer_trunk", lambda *_: pytest.fail("eager trunk ran")
    )
    with torch.inference_mode():
        model.encode_observations(make_obs())
    assert compiled == [True]
    assert len(spy.trunk_calls) == 1  # the compiled callable wraps the stub
    assert spy.unpack_calls == 1


def test_compile_hook_leaves_state_dict_keys_unchanged() -> None:
    model = _tiny()
    before = model.state_dict()
    assert model.compile_transformer_trunk(mode="default") == 1  # lazy; not run
    after = model.state_dict()
    assert list(after) == list(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key
    _tiny().load_state_dict(after)


def test_silu_model_builds_and_encodes() -> None:
    gelu, silu = _tiny().eval(), _tiny(activation="silu").eval()
    assert all(getattr(silu, name).activation == "silu" for name in STEMS)
    assert all(block.mlp.activation == "silu" for block in silu.blocks)
    assert silu.critic_head.activation == "silu"
    obs = make_obs(envs=2)
    with torch.inference_mode():
        a, b = gelu.encode_observations(obs), silu.encode_observations(obs)
        values = silu.compute_value(obs)
    assert b.hidden.shape == a.hidden.shape
    assert bool(torch.isfinite(b.hidden).all())
    assert bool(torch.isfinite(values).all())
    assert not torch.allclose(a.hidden[a.token_mask], b.hidden[b.token_mask])


# --- Muon grouping ------------------------------------------------------------


def _optimizer_ids(
    model: km.KaggricultureTransformer,
) -> tuple[set[int], set[int]]:
    optimizer = create_optimizer(model, MuonConfig())
    groups: dict[type, set[int]] = {torch.optim.Muon: set(), torch.optim.AdamW: set()}
    for inner in optimizer.optimizers:
        ids = groups[type(inner)]
        ids.update(id(p) for group in inner.param_groups for p in group["params"])
    return groups[torch.optim.Muon], groups[torch.optim.AdamW]


def test_muon_groups_follow_isaiah_rule() -> None:
    model = _tiny()
    muon_ids, adamw_ids = _optimizer_ids(model)
    assert not muon_ids & adamw_ids
    assert muon_ids | adamw_ids == {id(p) for p in model.parameters()}
    for name in STEMS:
        stem = getattr(model, name)
        assert id(stem.input.weight) in adamw_ids, name
        assert id(stem.output.weight) in muon_ids, name
    for token in (
        model.player_tokens,
        model.board_tokens,
        model.actor_plan_tokens,
        model.critic_value_tokens,
    ):
        assert id(token) in adamw_ids
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            assert id(param) in adamw_ids, name
    norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    assert norms
    for norm in norms:
        assert id(norm.weight) in adamw_ids
        assert id(norm.bias) in adamw_ids
    assert id(model.critic_head.out.weight) in adamw_ids
    for block in model.blocks:
        for linear in (block.attn.q, block.attn.out, block.mlp.up, block.mlp.down):
            assert id(linear.weight) in muon_ids


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
