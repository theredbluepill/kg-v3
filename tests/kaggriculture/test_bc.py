"""BC trainer (plan Task 5.2) on tiny CPU models and synthetic Task 5.1 shards.

Recorded programs are sampled from a second tiny model, so the student's replay
validation admits them exactly as it admits the corpus.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from owl.kaggriculture import types as kt
from owl.kaggriculture.bc_data import (
    MANIFEST_NAME,
    OBS_FIELDS,
    SHARD_SCHEMA,
    BCEpisode,
    load_bc_dataset,
    write_bc_episode,
    write_bc_manifest,
)
from owl.model import create_model
from owl.train import FullConfig
from owl.train import bc as bc_module
from owl.train.bc import (
    BC_HISTORY,
    BC_RESULT,
    BC_STATE,
    CHECKPOINT_BC_BEST,
    CHECKPOINT_BC_BEST_RECORD,
    PPO_CONFIG_NAME,
    BCConfig,
    HeldOutSelection,
    RowMetrics,
    TrainSharding,
    bc_loss,
    bc_terms,
    build_bc_model,
    load_bc_configs,
    load_bc_state,
    restore_bc_state,
    train_bc,
    winner_targets,
)
from owl.train.distributed import DistributedContext
from owl.train.logging import DebugLogger
from owl.train.optimizer import AdamWConfig, create_lr_scheduler, create_optimizer
from owl.train.ppo import _checkpoint_metadata

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import _cat_obs, _tiny

_REPO = Path(__file__).resolve().parents[2]
_CPU_PPO_CONFIG = _REPO / "configs" / "kaggriculture.yaml"
_CONTEXT = DistributedContext.single_process_cpu()


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, _REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_ppo = _load_script("run_ppo")
train_bc_script = _load_script("train_bc")


def _episode(
    episode_id: str,
    split: str,
    rows: int,
    *,
    seed: int,
    terminal_banks: tuple[float, float] = (3000.0, 2500.0),
) -> BCEpisode:
    own = [1 + (seed + r) % 3 for r in range(rows)]
    rival = [1 + (seed + 2 * r) % 2 for r in range(rows)]
    # One make_obs env per row: the fixture's multi-env draws can reach a step
    # early enough that a synthetic lifespan falls below the -1 sentinel.
    obs = _cat_obs(
        [
            make_obs(envs=1, own_actors=o, rival_actors=r, order_limit=3)
            for o, r in zip(own, rival, strict=True)
        ]
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        actions = _tiny(seed=100 + seed)(obs).actions
    return BCEpisode(
        episode_id=episode_id,
        split=split,  # type: ignore[arg-type]
        day="2026-09-21",
        obs=obs,
        actions=actions,
        turn=torch.arange(rows, dtype=torch.int64) * 2 + 1,  # rejected gaps
        terminal_banks=terminal_banks,
    )


def _write(root: Path, episodes: list[BCEpisode]) -> Path:
    records = [write_bc_episode(root, episode) for episode in episodes]
    write_bc_manifest(root, records, run={"git": "test"})
    return root


def _dataset_root(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "bc-data",
        [
            _episode("ep-a", "train", 4, seed=1),
            _episode("ep-b", "train", 3, seed=2, terminal_banks=(100.0, 900.0)),
            _episode("ep-c", "train", 3, seed=3, terminal_banks=(50.0, 50.0)),
            _episode("ep-v", "validation", 3, seed=4),
        ],
    )


def _bc_config(**overrides: Any) -> BCConfig:
    fields: dict[str, Any] = {
        "ppo_config": _CPU_PPO_CONFIG,
        "optimizer": AdamWConfig(learning_rate=3e-3),
        "seed": 11,
        "rows_per_rank": 2,
        "max_grad_norm": 10.0,
        "eval_interval_steps": 2,
        "patience_evals": 3,
        "max_steps": 4,
        "validation_rows_per_forward": 2,
    }
    return BCConfig(**(fields | overrides))


def _ppo_config() -> FullConfig:
    return FullConfig.from_file(_CPU_PPO_CONFIG)


def _run(
    run_dir: Path,
    data: Path,
    config: BCConfig,
    *,
    resume_from: Path | None = None,
) -> tuple[bc_module.BCResult, torch.nn.Module]:
    ppo_config = _ppo_config()
    dataset = load_bc_dataset(data)
    model = build_bc_model(ppo_config, device=_CONTEXT.device, seed=config.seed)
    optimizer = create_optimizer(model, config.optimizer)
    scheduler = create_lr_scheduler(optimizer, config.optimizer.lr_schedule)
    resume = None
    if resume_from is not None:
        resume = load_bc_state(resume_from)
        restore_bc_state(
            resume, model=model, optimizer=optimizer, lr_scheduler=scheduler
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_config.to_file(run_dir / PPO_CONFIG_NAME)
    result = train_bc(
        config=config,
        ppo_config=ppo_config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        context=_CONTEXT,
        run_dir=run_dir,
        logger=DebugLogger(),
        provenance={"source_commit": "test"},
        resume=resume,
    )
    return result, model


def _rehash(root: Path, index: int, path: Path) -> None:
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    manifest["episodes"][index]["shard_sha256"] = bc_module.file_sha256(path)
    manifest["episodes"][index]["shard_bytes"] = path.stat().st_size
    (root / MANIFEST_NAME).write_text(json.dumps(manifest))


# --- shard format -----------------------------------------------------------------


def test_shards_round_trip_rows_at_contract_dtypes(tmp_path: Path) -> None:
    episodes = [
        _episode("ep-a", "train", 4, seed=1),
        _episode("ep-v", "validation", 2, seed=4),
    ]
    root = _write(tmp_path / "data", episodes)
    assert (root / "train" / "2026-09-21-ep-a.npz").is_file()
    dataset = load_bc_dataset(root)
    assert dataset.train.num_rows == 4
    assert dataset.validation.episode_ids == ("ep-v",)
    assert dataset.train.turn.tolist() == [1, 3, 5, 7]
    batch = dataset.train.gather(np.array([3, 0, 3]))
    batch.obs.check_contract()
    source = episodes[0]
    for name in OBS_FIELDS:
        assert torch.equal(
            getattr(batch.obs, name), getattr(source.obs, name)[[3, 0, 3]]
        ), name
    assert torch.equal(
        batch.obs.action_mask.can_act, source.obs.action_mask.can_act[[3, 0, 3]]
    )
    assert batch.actions.tokens.dtype == torch.int64
    assert torch.equal(batch.actions.tokens, source.actions.tokens[[3, 0, 3]])
    assert torch.equal(batch.actions.lengths, source.actions.lengths[[3, 0, 3]])
    assert batch.final_banks.tolist() == [[3000.0, 2500.0]] * 3
    assert json.loads((root / MANIFEST_NAME).read_text())["run"] == {"git": "test"}


def test_loader_rejects_tampered_shards_and_duplicate_episodes(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    shard = root / "train" / "2026-09-21-ep-a.npz"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        load_bc_dataset(root)
    shard.write_bytes(original)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    manifest["episodes"][1]["episode_id"] = "ep-a"
    (root / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="more than once"):
        load_bc_dataset(root)


def test_loader_rejects_old_or_foreign_shard_schemas(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    shard = root / "train" / "2026-09-21-ep-a.npz"
    shard.unlink()
    np.savez_compressed(
        shard, features=np.zeros((4, 2, 8), np.float32), context=np.zeros((4, 2, 4))
    )
    _rehash(root, 0, shard)
    with pytest.raises(ValueError, match=SHARD_SCHEMA):
        load_bc_dataset(root)


def test_loader_requires_both_splits_and_paired_rows(tmp_path: Path) -> None:
    root = _write(tmp_path / "train-only", [_episode("ep-a", "train", 2, seed=1)])
    with pytest.raises(ValueError, match="no admitted validation rows"):
        load_bc_dataset(root)
    unpaired = _episode("ep-u", "train", 2, seed=1)
    unpaired.obs.still_playing[0, 1] = False
    root = _write(
        tmp_path / "unpaired", [unpaired, _episode("ep-v", "validation", 2, seed=4)]
    )
    with pytest.raises(ValueError, match="still_playing"):
        load_bc_dataset(root)


def test_loader_range_checks_its_compact_integer_storage(tmp_path: Path) -> None:
    wide = _episode("ep-w", "train", 2, seed=1)
    wide.obs.globals_int[0, 0, 0] = 2**40  # contract-valid, not int32
    root = _write(tmp_path / "wide", [wide, _episode("ep-v", "validation", 2, seed=4)])
    with pytest.raises(ValueError, match="storage range"):
        load_bc_dataset(root)


def test_two_fake_ranks_hold_disjoint_rows_covering_the_split(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    ranks = [load_bc_dataset(root, rank=r, world_size=2) for r in range(2)]

    def identities(split: Any) -> set[tuple[str, int]]:
        return {
            (split.episode_ids[int(e)], int(t))
            for e, t in zip(split.episode, split.turn, strict=True)
        }

    everything = load_bc_dataset(root)
    for name in ("train", "validation"):
        a, b = (identities(getattr(d, name)) for d in ranks)
        assert not a & b
        assert a | b == identities(getattr(everything, name))
    assert ranks[0].train.rank_rows == ranks[1].train.rank_rows == (6, 4)
    shardings = [
        TrainSharding(
            seed=5,
            rows_per_rank=2,
            gradient_accumulation_steps=1,
            rank_rows=d.train.rank_rows,
        )
        for d in ranks
    ]
    assert shardings[0].steps_per_epoch == 2
    for rank, sharding in enumerate(shardings):
        epoch = np.concatenate(
            [sharding.rows(step=s, micro=0, rank=rank) for s in range(2)]
        )
        assert np.unique(epoch).size == 4
        assert np.array_equal(
            sharding.rows(step=1, micro=0, rank=rank),
            TrainSharding(
                seed=5,
                rows_per_rank=2,
                gradient_accumulation_steps=1,
                rank_rows=(6, 4),
            ).rows(step=1, micro=0, rank=rank),
        )


# --- objective --------------------------------------------------------------------


def test_winner_targets_follow_raw_final_banks() -> None:
    targets = winner_targets(
        torch.tensor([[10.0, 5.0], [1.0, 2.0], [7.0, 7.0]], dtype=torch.float64)
    )
    assert targets.tolist() == [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.5, 0.5], [0.5, 0.5]],
    ]


def test_loss_decreases_on_a_memorizable_batch(tmp_path: Path) -> None:
    dataset = load_bc_dataset(_dataset_root(tmp_path))
    batch = dataset.train.gather(np.array([0, 4]))
    model = _tiny(seed=3)
    optimizer = create_optimizer(model, AdamWConfig(learning_rate=1e-2))
    losses = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = bc_loss(
            bc_terms(model.evaluate_actions(batch.obs, batch.actions), batch),
            value_coef=1.0,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0], losses


def test_bc_loss_matches_a_hand_computed_objective() -> None:
    """Oracle: per-frame probabilities in, the length-normalized NLL + CE out."""
    frame_probs = torch.tensor(
        [[[0.5, 0.25, 1.0], [0.8, 1.0, 1.0]], [[0.1, 1.0, 1.0], [0.5, 0.5, 0.5]]]
    )
    lengths = torch.tensor([[2, 1], [1, 3]])
    # Seat rows: (self win, opponent win) probabilities.
    winner = torch.tensor([[[0.6, 0.4], [0.3, 0.7]], [[0.2, 0.8], [0.5, 0.5]]])
    evaluation = cast(
        Any,
        SimpleNamespace(
            log_probs=SimpleNamespace(per_player_entity=frame_probs.log()),
            winner_log_probabilities=winner.log(),
        ),
    )
    batch = cast(
        Any,
        SimpleNamespace(
            actions=SimpleNamespace(lengths=lengths),
            # Row 0: seat 0 wins; row 1: a draw.
            final_banks=torch.tensor([[9.0, 1.0], [4.0, 4.0]], dtype=torch.float64),
        ),
    )
    turn_nll = [
        [-math.log(0.5 * 0.25) / 2, -math.log(0.8) / 1],
        [-math.log(0.1) / 1, -math.log(0.125) / 3],
    ]
    value_ce = [
        [-math.log(0.6), -math.log(0.7)],
        [-0.5 * (math.log(0.2) + math.log(0.8)), -math.log(0.5)],
    ]
    terms = bc_terms(evaluation, batch)
    assert torch.allclose(terms.turn_nll, torch.tensor(turn_nll))
    assert torch.allclose(terms.value_ce, torch.tensor(value_ce))
    expected = float(np.mean(turn_nll)) + 0.25 * float(np.mean(value_ce))
    assert float(bc_loss(terms, value_coef=0.25)) == pytest.approx(expected)


def _train_objective(model: torch.nn.Module, data: Path) -> float:
    split = load_bc_dataset(data).train
    batch = split.gather(np.arange(split.num_rows, dtype=np.int64))
    with torch.no_grad():
        terms = bc_terms(
            cast(Any, model).evaluate_actions(batch.obs, batch.actions), batch
        )
    return float(bc_loss(terms, value_coef=1.0))


def test_the_real_training_loop_lowers_the_objective(tmp_path: Path) -> None:
    """train_bc's own update path moves every tensor and lowers the train loss."""
    data = _dataset_root(tmp_path)
    config = _bc_config(
        optimizer=AdamWConfig(learning_rate=1e-2), max_steps=12, eval_interval_steps=12
    )
    initial = build_bc_model(_ppo_config(), device=_CONTEXT.device, seed=config.seed)
    before = _train_objective(initial, data)
    _, trained = _run(tmp_path / "run", data, config)
    after = _train_objective(trained, data)
    assert 0.0 < after < 0.9 * before, (before, after)
    unchanged = [
        name
        for (name, a), (_, b) in zip(
            initial.state_dict().items(), trained.state_dict().items(), strict=True
        )
        if a.is_floating_point() and torch.equal(a, b)
    ]
    assert unchanged == []


def test_bc_loss_reaches_every_parameter(tmp_path: Path) -> None:
    """The critic is trained with the policy; DDP needs no unused-parameter scan."""
    dataset = load_bc_dataset(_dataset_root(tmp_path))
    batch = dataset.train.gather(np.array([0, 1]))
    model = _tiny(seed=3)
    bc_loss(
        bc_terms(model.evaluate_actions(batch.obs, batch.actions), batch),
        value_coef=1.0,
    ).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


# --- data order ---------------------------------------------------------------------


def test_train_sharding_is_deterministic_per_rank_and_epoch() -> None:
    sharding = TrainSharding(
        seed=5, rows_per_rank=3, gradient_accumulation_steps=2, rank_rows=(25, 26)
    )
    assert (sharding.world_size, sharding.rows_per_step) == (2, 12)
    assert sharding.steps_per_epoch == 4
    for rank in range(2):
        epoch = np.concatenate(
            [
                sharding.rows(step=step, micro=micro, rank=rank)
                for step in range(4)
                for micro in range(2)
            ]
        )
        assert epoch.size == 24 == np.unique(epoch).size
        assert int(epoch.max()) < sharding.rank_rows[rank]
    assert not np.array_equal(
        sharding.rows(step=4, micro=0, rank=0), sharding.rows(step=0, micro=0, rank=0)
    )
    assert not np.array_equal(
        sharding.rows(step=0, micro=0, rank=0), sharding.rows(step=0, micro=0, rank=1)
    )
    with pytest.raises(ValueError, match="smallest rank partition"):
        TrainSharding(
            seed=5, rows_per_rank=3, gradient_accumulation_steps=2, rank_rows=(25, 5)
        )


# --- selection and the stop rule ----------------------------------------------------


def test_held_out_selection_keeps_the_lowest_nll_and_stops_after_patience() -> None:
    selection = HeldOutSelection(patience_evals=2, min_delta=0.05)
    assert selection.observe(3.0, step=0)
    assert selection.observe(2.0, step=2)
    # A new minimum inside min_delta is still the best checkpoint; it only fails
    # to reset the patience count.
    assert selection.observe(1.97, step=4)
    assert (selection.best_nll, selection.best_step) == (1.97, 4)
    assert selection.evals_since_improvement == 1
    assert not selection.should_stop
    assert not selection.observe(2.4, step=6)
    assert selection.should_stop
    assert (selection.best_nll, selection.best_step) == (1.97, 4)
    with pytest.raises(ValueError, match="advance"):
        selection.observe(1.0, step=6)
    with pytest.raises(ValueError, match="finite"):
        selection.observe(float("nan"), step=8)


def test_held_out_patience_ignores_a_trickle_below_min_delta() -> None:
    """Sub-min_delta gains keep the best current but do not slide the reference."""
    selection = HeldOutSelection(patience_evals=2, min_delta=0.05)
    assert selection.observe(2.0, step=0)
    assert selection.observe(1.97, step=1)
    assert selection.observe(1.94, step=2)  # 0.06 below 2.0, only 0.03 below 1.97
    assert selection.evals_since_improvement == 0
    assert selection.observe(1.91, step=3)
    assert selection.observe(1.90, step=4)
    assert selection.should_stop
    assert (selection.best_nll, selection.best_step) == (1.90, 4)


def test_best_checkpoint_is_the_minimum_even_inside_min_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier's probe: NLLs 2.0, 1.97, 2.4 with min_delta 0.05 keep step 1."""
    data = _dataset_root(tmp_path)
    scripted = iter([2.0, 1.97, 2.4])
    _script_validation(monkeypatch, scripted, snapshots={})
    run_dir = tmp_path / "run"
    config = _bc_config(
        eval_interval_steps=1, patience_evals=2, min_delta=0.05, max_steps=50
    )
    result, _ = _run(run_dir, data, config)
    assert result.stop_reason == "no_held_out_improvement"
    assert (result.steps, result.best_step, result.best_validation_nll) == (2, 1, 1.97)
    record = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())
    assert (record["bc_step"], record["validation_nll"]) == (1, 1.97)
    assert (
        torch.load(run_dir / CHECKPOINT_BC_BEST, weights_only=False)["optimizer_steps"]
        == 1
    )


def _script_validation(
    monkeypatch: pytest.MonkeyPatch,
    scripted: Iterator[float],
    *,
    snapshots: dict[int, dict[str, torch.Tensor]],
) -> None:
    real = bc_module.evaluate_rows

    def scripted_eval(model: Any, split: Any, rows: Any, **kwargs: Any) -> RowMetrics:
        metrics = real(model, split, rows, **kwargs)
        if split.split == "train":
            return metrics
        snapshots[len(snapshots)] = copy.deepcopy(model.state_dict())
        return RowMetrics(
            turn_nll=next(scripted),
            frame_nll=metrics.frame_nll,
            value_ce=metrics.value_ce,
            seat_rows=metrics.seat_rows,
        )

    monkeypatch.setattr(bc_module, "evaluate_rows", scripted_eval)


def test_training_keeps_the_best_checkpoint_and_stops_on_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset_root(tmp_path)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    _script_validation(
        monkeypatch, iter([3.0, 2.0, 2.5, 2.6, 2.7]), snapshots=snapshots
    )
    run_dir = tmp_path / "run"
    config = _bc_config(eval_interval_steps=1, patience_evals=2, max_steps=50)
    result, _ = _run(run_dir, data, config)
    assert result.stop_reason == "no_held_out_improvement"
    assert (result.steps, result.best_step) == (3, 1)
    assert (result.best_validation_nll, result.final_validation_nll) == (2.0, 2.6)
    best = torch.load(run_dir / CHECKPOINT_BC_BEST, weights_only=False)
    assert best["optimizer_steps"] == 1
    for name, tensor in snapshots[1].items():
        assert torch.equal(best["model"][name], tensor), name
    record = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())
    assert record["bc_step"] == 1
    assert record["validation_nll"] == 2.0
    assert record["sha256"] == bc_module.file_sha256(run_dir / CHECKPOINT_BC_BEST)
    history = [
        json.loads(line) for line in (run_dir / BC_HISTORY).read_text().splitlines()
    ]
    assert [h["bc/validation_nll"] for h in history] == [3.0, 2.0, 2.5, 2.6]
    assert json.loads((run_dir / BC_RESULT).read_text())["stop_reason"] == (
        "no_held_out_improvement"
    )


# --- restart ------------------------------------------------------------------------


def test_restart_from_state_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    data = _dataset_root(tmp_path)
    straight, straight_model = _run(
        tmp_path / "straight", data, _bc_config(max_steps=4)
    )
    assert straight.stop_reason == "max_steps"
    interrupted = tmp_path / "interrupted"
    first, _ = _run(interrupted, data, _bc_config(max_steps=2))
    assert first.steps == 2
    resumed, resumed_model = _run(
        interrupted, data, _bc_config(max_steps=4), resume_from=interrupted / BC_STATE
    )
    assert resumed.steps == 4
    for (name, a), (_, b) in zip(
        straight_model.state_dict().items(),
        resumed_model.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(a, b), name
    assert resumed.final_validation_nll == straight.final_validation_nll
    lines = (interrupted / BC_HISTORY).read_text().splitlines()
    assert [json.loads(line)["bc/step"] for line in lines] == [0.0, 2.0, 4.0]


def test_resume_rejects_another_dataset_or_world_size(tmp_path: Path) -> None:
    data = _dataset_root(tmp_path)
    _run(tmp_path / "run", data, _bc_config(max_steps=2))
    state = load_bc_state(tmp_path / "run" / BC_STATE)
    other = _write(
        tmp_path / "other",
        [_episode("x", "train", 4, seed=7), _episode("y", "validation", 2, seed=8)],
    )
    config, ppo_config = _bc_config(max_steps=2), _ppo_config()
    with pytest.raises(ValueError, match="manifest"):
        bc_module.check_resume_compatible(
            state,
            dataset=load_bc_dataset(other),
            world_size=1,
            config=config,
            ppo_config=ppo_config,
        )
    with pytest.raises(ValueError, match="world size"):
        bc_module.check_resume_compatible(
            state,
            dataset=load_bc_dataset(data),
            world_size=2,
            config=config,
            ppo_config=ppo_config,
        )


def test_resume_rejects_changed_trajectory_settings(tmp_path: Path) -> None:
    """Only max_steps may change on resume; the rest set the continued trajectory."""
    data = _dataset_root(tmp_path)
    config = _bc_config(max_steps=2)
    _run(tmp_path / "run", data, config)
    state = load_bc_state(tmp_path / "run" / BC_STATE)
    dataset = load_bc_dataset(data)
    ppo_config = _ppo_config()

    def check(bc: BCConfig, ppo: FullConfig = ppo_config) -> None:
        bc_module.check_resume_compatible(
            state, dataset=dataset, world_size=1, config=bc, ppo_config=ppo
        )

    check(config.model_copy(update={"max_steps": 40}))
    for field, value in (
        ("seed", 12),
        ("rows_per_rank", 1),
        ("gradient_accumulation_steps", 2),
        ("min_delta", 0.1),
        ("optimizer", AdamWConfig(learning_rate=1e-4)),
    ):
        with pytest.raises(ValueError, match=rf"bc\.{field}\b"):
            check(config.model_copy(update={field: value}))
    changed_ppo = ppo_config.model_copy(
        update={"rl": ppo_config.rl.model_copy(update={"dtype": "bfloat16"})}
    )
    with pytest.raises(ValueError, match=r"ppo\.rl"):
        check(config, changed_ppo)


def _script_argv(
    monkeypatch: pytest.MonkeyPatch, target: Path, data: Path, *extra: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bc.py",
            str(target),
            "--data",
            str(data),
            "--log-mode",
            "debug",
            *extra,
        ],
    )


def test_script_resume_records_a_new_attempt_with_its_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset_root(tmp_path)
    config_path = tmp_path / "bc.yaml"
    _bc_config(max_steps=2, eval_interval_steps=1).to_file(config_path)
    out = tmp_path / "runs"
    _script_argv(
        monkeypatch, config_path, data, "--output-dir", str(out), "--source-commit", "a"
    )
    train_bc_script.main()
    (run_dir,) = out.iterdir()
    saved = BCConfig.from_file(run_dir / "bc_config.yaml")
    saved.model_copy(update={"max_steps": 4}).to_file(run_dir / "bc_config.yaml")
    parent_sha = bc_module.file_sha256(run_dir / BC_STATE)
    monkeypatch.setattr(train_bc_script, "_git_head", lambda: "b")
    _script_argv(monkeypatch, run_dir, data)
    train_bc_script.main()

    attempts = [
        json.loads(line)
        for line in (run_dir / train_bc_script.ATTEMPTS).read_text().splitlines()
    ]
    assert [a["source_commit"] for a in attempts] == ["a", "b"]
    assert [a["attempt"] for a in attempts] == [0, 1]
    assert [a["start_step"] for a in attempts] == [0, 2]
    assert [a["parent_state_sha256"] for a in attempts] == [None, parent_sha]
    result = json.loads((run_dir / BC_RESULT).read_text())
    assert result["steps"] == 4
    assert result["source_commit"] == "b"
    assert result["attempt"] == 1
    assert result["attempt_source_commits"] == ["a", "b"]
    assert result["parent_state_sha256"] == parent_sha
    best = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())
    # The best checkpoint's sidecar names the attempt that wrote it.
    assert best["source_commit"] == ["a", "b"][best["attempt"]]
    assert best["attempt_source_commits"] == ["a", "b"][: best["attempt"] + 1]


def test_script_resume_rejects_an_edited_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset_root(tmp_path)
    config_path = tmp_path / "bc.yaml"
    _bc_config(max_steps=2).to_file(config_path)
    out = tmp_path / "runs"
    _script_argv(
        monkeypatch, config_path, data, "--output-dir", str(out), "--source-commit", "a"
    )
    train_bc_script.main()
    (run_dir,) = out.iterdir()
    saved = BCConfig.from_file(run_dir / "bc_config.yaml")
    saved.model_copy(update={"seed": saved.seed + 1, "max_steps": 4}).to_file(
        run_dir / "bc_config.yaml"
    )
    _script_argv(monkeypatch, run_dir, data, "--source-commit", "a")
    with pytest.raises(ValueError, match=r"bc\.seed"):
        train_bc_script.main()
    assert load_bc_state(run_dir / BC_STATE).step == 2


# --- PPO handoff --------------------------------------------------------------------


def test_best_checkpoint_loads_through_run_ppo(tmp_path: Path) -> None:
    data = _dataset_root(tmp_path)
    run_dir = tmp_path / "run"
    _run(run_dir, data, _bc_config(max_steps=2, eval_interval_steps=1))
    best_path = run_dir / CHECKPOINT_BC_BEST
    checkpoint = torch.load(best_path, weights_only=False)
    metadata = run_ppo._checkpoint_metadata(checkpoint, path=best_path)
    assert metadata.env_steps == 0
    assert _checkpoint_metadata(checkpoint).env_steps == 0
    ppo_config = FullConfig.from_file(run_ppo._checkpoint_config_path(best_path))
    fresh = create_model(
        ppo_config.model,
        obs_spec=ppo_config.env.obs_spec,
        action_spec=ppo_config.env.action_spec,
    )
    run_ppo._load_model_from_checkpoint(
        fresh, path=best_path, device=torch.device("cpu")
    )
    teacher = create_model(
        ppo_config.model,
        obs_spec=ppo_config.env.obs_spec,
        action_spec=ppo_config.env.action_spec,
    )
    run_ppo._load_model_weights(teacher, path=best_path, device=torch.device("cpu"))
    best_step = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())["bc_step"]
    assert checkpoint["optimizer_steps"] == best_step
    batch = load_bc_dataset(data).validation.gather(np.array([0, 1]))
    with torch.no_grad():
        a = fresh.evaluate_actions(batch.obs, batch.actions)
        b = teacher.evaluate_actions(batch.obs, batch.actions)
    assert torch.equal(a.log_probs.event, b.log_probs.event)
    assert torch.equal(a.values, b.values)


# --- configs and the script ---------------------------------------------------------


def test_two_rank_bc_config_targets_the_two_rank_ppo_config() -> None:
    bc_config, ppo_config = load_bc_configs(
        _REPO / "configs" / "bc" / "kaggriculture_2rank.yaml"
    )
    assert ppo_config == FullConfig.from_file(
        _REPO / "configs" / "kaggriculture_2rank.yaml"
    )
    assert ppo_config.rl.dtype == "bfloat16"
    assert ppo_config.rl.model_compile == "trunk"
    assert bc_config.optimizer == ppo_config.optimizer.model_copy(
        update={"lr_schedule": bc_config.optimizer.lr_schedule}
    )
    reports = bc_module.check_bc_workload(bc_config, ppo_config)
    assert [r.name for r in reports] == ["bc_microbatch", "bc_validation"]
    assert all(r.head_headroom >= 1 and r.min_trunk_headroom >= 1 for r in reports)


def test_script_runs_fresh_and_resumes_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset_root(tmp_path)
    config_path = tmp_path / "bc.yaml"
    _bc_config(max_steps=2).to_file(config_path)
    out = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bc.py",
            str(config_path),
            "--data",
            str(data),
            "--output-dir",
            str(out),
            "--log-mode",
            "debug",
            "--source-commit",
            "test-commit",
        ],
    )
    train_bc_script.main()
    (run_dir,) = out.iterdir()
    (attempt,) = (run_dir / train_bc_script.ATTEMPTS).read_text().splitlines()
    assert json.loads(attempt)["source_commit"] == "test-commit"
    assert load_bc_state(run_dir / BC_STATE).step == 2
    saved = BCConfig.from_file(run_dir / "bc_config.yaml")
    assert saved.ppo_config == Path(PPO_CONFIG_NAME)
    # Raise the step budget in place, then resume from the run directory.
    saved.model_copy(update={"max_steps": 4}).to_file(run_dir / "bc_config.yaml")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bc.py",
            str(run_dir),
            "--data",
            str(data),
            "--log-mode",
            "debug",
            "--source-commit",
            "test-commit",
        ],
    )
    train_bc_script.main()
    assert load_bc_state(run_dir / BC_STATE).step == 4
    assert json.loads((run_dir / BC_RESULT).read_text())["steps"] == 4


def test_contract_shapes_cover_every_observation_field() -> None:
    assert set(OBS_FIELDS) | {"action_mask"} == set(
        kt.KaggricultureObsBatch.model_fields
    )
