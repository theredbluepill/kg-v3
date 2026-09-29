"""BC trainer (plan Task 5.2) on tiny CPU models and synthetic Task 5.1 shards.

Recorded programs are sampled from a second tiny model, so the student's replay
validation admits them exactly as it admits the corpus.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import wandb as _real_wandb
import wandb.errors  # the gate imports it beside a fake wandb
from owl.kaggriculture import bc_data as bc_data_module
from owl.kaggriculture import types as kt
from owl.kaggriculture.bc_data import (
    MANIFEST_NAME,
    OBS_FIELDS,
    SHARD_SCHEMA,
    BCEpisode,
    BCManifest,
    load_bc_dataset,
    write_bc_episode,
    write_bc_manifest,
)
from owl.model import create_model
from owl.train import FullConfig
from owl.train import bc as bc_module
from owl.train import logging as train_logging
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
from owl.train.ppo import PPOTrainer, _checkpoint_metadata

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
    policy_seat: tuple[bool, bool] = (True, True),
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
        policy_seat=torch.tensor([policy_seat] * rows, dtype=torch.bool),
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
    assert torch.equal(batch.policy_seat, source.policy_seat[[3, 0, 3]])
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


def test_loader_requires_a_policy_seat_on_every_row(tmp_path: Path) -> None:
    bare = _episode("ep-n", "train", 2, seed=1, policy_seat=(False, False))
    root = _write(tmp_path / "bare", [bare, _episode("ep-v", "validation", 2, seed=4)])
    with pytest.raises(ValueError, match="policy_seat"):
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
            policy_seat=torch.ones(2, 2, dtype=torch.bool),
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
    # Winner-only imitation: row 0 imitates only seat 0 (the winner), the draw
    # row both seats; the critic still averages over every seat row.
    batch.policy_seat = torch.tensor([[True, False], [True, True]])
    terms = bc_terms(evaluation, batch)
    policy_nll = (turn_nll[0][0] + turn_nll[1][0] + turn_nll[1][1]) / 3
    expected = policy_nll + 0.25 * float(np.mean(value_ce))
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


def test_equal_rank_partitions_draw_different_permutations() -> None:
    """The rank enters the permutation seed, not only the partition length."""
    sharding = TrainSharding(
        seed=5, rows_per_rank=4, gradient_accumulation_steps=1, rank_rows=(24, 24)
    )
    for step in (0, sharding.steps_per_epoch):
        by_rank = [
            np.concatenate(
                [
                    sharding.rows(step=step + s, micro=0, rank=rank)
                    for s in range(sharding.steps_per_epoch)
                ]
            )
            for rank in range(2)
        ]
        assert not np.array_equal(by_rank[0], by_rank[1]), step


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


def test_an_unscheduled_evaluation_selects_but_leaves_patience_alone() -> None:
    """An off-cadence evaluation may keep a lower checkpoint but not add patience.

    A budget or runtime stop evaluates between scheduled steps; a restart must
    keep the scheduled cadence.
    """
    selection = HeldOutSelection(patience_evals=2, min_delta=0.0)
    assert selection.observe(3.0, step=0)
    assert not selection.observe(4.0, step=1, scheduled=False)
    assert selection.evals_since_improvement == 0
    assert selection.observe(2.5, step=3, scheduled=False)
    assert (selection.best_nll, selection.best_step) == (2.5, 3)
    assert (selection.improvement_nll, selection.evals_since_improvement) == (3.0, 0)
    assert (selection.last_nll, selection.last_step) == (2.5, 3)
    assert not selection.observe(2.8, step=4)
    assert selection.evals_since_improvement == 0  # 2.8 beat the scheduled 3.0
    assert (selection.best_nll, selection.best_step) == (2.5, 3)
    with pytest.raises(ValueError, match="advance"):
        selection.observe(1.0, step=4, scheduled=False)


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
            policy_seat_rows=metrics.policy_seat_rows,
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


@pytest.mark.parametrize(
    ("interrupted_nlls", "best_step"),
    [
        # The verifier's probe: the off-cadence NLL is no better.
        ([3.0, 4.0, 4.0, 4.0], 0),
        # A lower off-cadence NLL is kept, but patience still runs on cadence.
        ([3.0, 2.5, 4.0, 4.0], 1),
    ],
)
def test_an_off_cadence_stop_restarts_on_the_uninterrupted_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_nlls: list[float],
    best_step: int,
) -> None:
    """A budget stop between evaluations must not shorten the resumed patience."""
    data = _dataset_root(tmp_path)
    config = _bc_config(eval_interval_steps=2, patience_evals=2, max_steps=8)
    # Uninterrupted: evaluations at steps 0, 2, 4; then the interrupted run's.
    nlls = iter([3.0, 4.0, 4.0, *interrupted_nlls])
    _script_validation(monkeypatch, nlls, snapshots={})
    straight, straight_model = _run(tmp_path / "straight", data, config)
    assert (straight.stop_reason, straight.steps) == ("no_held_out_improvement", 4)

    interrupted = tmp_path / "interrupted"
    first, _ = _run(interrupted, data, config.model_copy(update={"max_steps": 1}))
    assert (first.stop_reason, first.steps) == ("max_steps", 1)
    resumed, resumed_model = _run(
        interrupted, data, config, resume_from=interrupted / BC_STATE
    )
    assert next(nlls, None) is None
    assert (resumed.stop_reason, resumed.steps) == ("no_held_out_improvement", 4)
    assert resumed.best_step == best_step
    assert resumed.best_validation_nll <= straight.best_validation_nll
    for (name, a), (_, b) in zip(
        straight_model.state_dict().items(),
        resumed_model.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(a, b), name
    lines = (interrupted / BC_HISTORY).read_text().splitlines()
    assert [json.loads(line)["bc/step"] for line in lines] == [0.0, 1.0, 2.0, 4.0]
    record = json.loads((interrupted / CHECKPOINT_BC_BEST_RECORD).read_text())
    assert record["bc_step"] == best_step


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
        sys, "argv", ["train_bc.py", str(target), "--data", str(data), *extra]
    )


class _FakeWandb:
    """``sys.modules['wandb']`` for the real ``WandbLogger``.

    wandb's own ``Settings`` validator and errors stay real for the gate.
    """

    def __init__(self) -> None:
        self.inits: list[dict[str, Any]] = []
        self.logs: list[tuple[dict[str, float], int]] = []
        self.runs: list[SimpleNamespace] = []
        self.Settings = _real_wandb.Settings
        self.errors = _real_wandb.errors
        self.run: SimpleNamespace | None = None

    def init(self, **kwargs: Any) -> SimpleNamespace:
        self.inits.append(kwargs)
        offline = kwargs["mode"] == "offline"
        finished: list[int] = []
        run = SimpleNamespace(
            id=kwargs.get("id", f"bc-run-{len(self.inits)}"),
            project=kwargs["project"],
            entity="team",
            url=None if offline else "https://wandb.ai/team/kg-v3/runs/x",
            offline=offline,
            disabled=False,
            summary={},
            finished=finished,
            finish=lambda *, exit_code=0: finished.append(exit_code),
        )
        self.run = run
        self.runs.append(run)
        return run

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logs.append((metrics, step))


@pytest.fixture
def bc_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    """Online W&B credentials (a key wandb's validator accepts) and a fake run.

    ``git_source_commit`` answers ``None`` so ``--source-commit`` names the
    attempt's source, as on a checkout without git metadata.
    """
    for name in ("NETRC", "WANDB_BASE_URL", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(train_logging, "git_source_commit", lambda _cwd: None)
    return fake


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_script_resume_records_a_new_attempt_with_its_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bc_wandb: _FakeWandb
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
    monkeypatch.setattr(train_logging, "git_source_commit", lambda _cwd: "b")
    _script_argv(monkeypatch, run_dir, data)
    train_bc_script.main()
    assert [init.get("id") for init in bc_wandb.inits] == [None, "bc-run-1"]

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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bc_wandb: _FakeWandb,  # noqa: ARG001
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


def _bc_identity(
    *, attempt: int, sources: tuple[str, ...]
) -> train_logging.RunIdentity:
    return train_logging.RunIdentity(
        experiment_id="bc-exp",
        job_type="bc",
        attempt=attempt,
        source_commit=sources[-1],
        attempt_source_commits=sources,
        config_sha256="0" * 64,
        telemetry=train_logging.TelemetryMode.WANDB_ONLINE,
    )


def test_attempt_records_reject_malformed_lineage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / train_bc_script.ATTEMPTS
    resume = cast(Any, SimpleNamespace(step=2))

    def start(
        resume: Any, *, attempt: int = 1, sources: tuple[str, ...] = ("a", "b")
    ) -> dict[str, object]:
        return cast(
            dict[str, object],
            train_bc_script._plan_bc_attempt(
                run_dir,
                identity=_bc_identity(attempt=attempt, sources=sources),
                data=tmp_path,
                dataset_manifest_sha256="0" * 64,
                world_size=1,
                resume=resume,
            ),
        )

    with pytest.raises(FileNotFoundError, match="attempt records missing"):
        start(resume)
    path.write_text("")
    with pytest.raises(ValueError, match="needs the run's attempt records"):
        start(resume)
    with pytest.raises(FileExistsError, match="already has attempts"):
        start(None, attempt=0, sources=("b",))
    for lines, match in (
        (['{"attempt": 1, "source_commit": "a"}'], "is not attempt 0"),
        (['["attempt", 0]'], "is not attempt 0"),
        (['{"attempt": 0, "source_commit": 7}'], "has no source_commit"),
        (['{"attempt": 0}'], "has no source_commit"),
    ):
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=match):
            start(resume)
    assert path.read_text() == '{"attempt": 0}\n'


def test_bc_receipt_must_agree_with_the_shared_attempt_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / BC_STATE).write_bytes(b"state")
    (run_dir / train_bc_script.ATTEMPTS).write_text(
        '{"attempt": 0, "source_commit": "a"}\n'
    )
    resume = cast(Any, SimpleNamespace(step=2))

    def plan(attempt: int, sources: tuple[str, ...]) -> dict[str, object]:
        return cast(
            dict[str, object],
            train_bc_script._plan_bc_attempt(
                run_dir,
                identity=_bc_identity(attempt=attempt, sources=sources),
                data=tmp_path,
                dataset_manifest_sha256="0" * 64,
                world_size=1,
                resume=resume,
            ),
        )

    record = plan(1, ("a", "b"))
    assert record["attempt"] == 1
    assert record["experiment_id"] == "bc-exp"
    assert record["telemetry_mode"] == "wandb-online"
    assert record["config_sha256"] == "0" * 64
    assert record["attempt_source_commits"] == ["a", "b"]
    assert record["parent_state_sha256"] == bc_module.file_sha256(run_dir / BC_STATE)
    for attempt, sources in ((2, ("a", "x", "b")), (1, ("z", "b"))):
        with pytest.raises(ValueError, match="receipts are inconsistent"):
            plan(attempt, sources)


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


# The BC run's PPO config and the ranked configs Phase 6.2 starts from; the
# model section must match exactly for the BC best to load without surprises.
_BC_RUN_PPO_CONFIG = _REPO / "configs" / "kaggriculture_1gpu_eager.yaml"
_RANKED_PPO_CONFIGS = sorted((_REPO / "configs").glob("kaggriculture_*rank.yaml"))


def _cpu_model_config(cfg: FullConfig) -> FullConfig:
    """``force_flash_attn`` needs CUDA; it selects a kernel, not parameters."""
    return cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"force_flash_attn": False})}
    )


def _ppo_model(cfg: FullConfig, *, seed: int) -> Any:
    """The model a fresh ``run_ppo`` launch builds before loading weights."""
    torch.manual_seed(seed)
    model = run_ppo._create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    model.reset_parameters()
    return model


def _ppo_trainer(model: Any, cfg: FullConfig) -> PPOTrainer:
    """The trainer state ``PPOTrainer.load_model_weights`` touches."""
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = create_optimizer(model, cfg.optimizer)
    trainer.device = torch.device("cpu")
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 0
    return trainer


@pytest.fixture(scope="module")
def bc_best(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any, Path]:
    """A BC best checkpoint of the BC run's model written by the BC trainer's saver.

    One optimizer step gives the optimizer real moment/momentum state. Returns
    the checkpoint path, the source model (eval mode) and the dataset root.
    """
    tmp_path = tmp_path_factory.mktemp("bc-best")
    cfg = _cpu_model_config(FullConfig.from_file(_BC_RUN_PPO_CONFIG))
    data = _dataset_root(tmp_path)
    dataset = load_bc_dataset(data)
    model = _ppo_model(cfg, seed=3)
    optimizer = create_optimizer(model, cfg.optimizer)
    batch = dataset.train.gather(np.array([0, 1, 2]))
    terms = bc_terms(model.evaluate_actions(batch.obs, batch.actions), batch)
    bc_loss(terms, value_coef=1.0).backward()
    optimizer.step()
    optimizer.zero_grad()
    run_dir = tmp_path / "bc-run"
    run_dir.mkdir()
    bc_module._save_best(
        run_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        nll=1.0,
        wandb_run_id=None,
        dataset=dataset,
        provenance={"source_commit": "test"},
    )
    return run_dir / CHECKPOINT_BC_BEST, model.eval(), data


def test_ranked_ppo_configs_share_the_bc_run_model() -> None:
    bc_run = FullConfig.from_file(_BC_RUN_PPO_CONFIG)
    assert [p.name for p in _RANKED_PPO_CONFIGS][:2] == [
        "kaggriculture_2rank.yaml",
        "kaggriculture_4rank.yaml",
    ]
    for path in _RANKED_PPO_CONFIGS:
        ranked = FullConfig.from_file(path)
        assert ranked.model == bc_run.model, path
        assert ranked.env.obs_spec == bc_run.env.obs_spec, path
        assert ranked.env.action_spec == bc_run.env.action_spec, path
    flash = run_ppo._create_model(
        bc_run.model,
        obs_spec=bc_run.env.obs_spec,
        action_spec=bc_run.env.action_spec,
    )
    cpu = _cpu_model_config(bc_run)
    no_flash = run_ppo._create_model(
        cpu.model, obs_spec=cpu.env.obs_spec, action_spec=cpu.env.action_spec
    )
    assert {k: v.shape for k, v in flash.state_dict().items()} == {
        k: v.shape for k, v in no_flash.state_dict().items()
    }


def test_one_gpu_ppo_config_is_the_two_rank_config_on_one_rank() -> None:
    """The 1-GPU config differs from the 2-rank one only as its header says.

    It undoes Isaiah's multi-GPU division (same global batch on one rank) and
    runs eager; everything else, including the reward schema and the replay
    count ``run_ppo`` accepts before Task 7.3, is the 2-rank config's.
    """
    one = FullConfig.from_file(_BC_RUN_PPO_CONFIG)
    two = FullConfig.from_file(_REPO / "configs" / "kaggriculture_2rank.yaml")
    assert one.env.n_envs == 2 * two.env.n_envs
    assert one.rl.segments_per_minibatch == 2 * two.rl.segments_per_minibatch
    assert one.rl.model_compile == "none"
    assert one.rl.eval_replay_games == 0
    assert (
        one.model_copy(
            update={
                "env": one.env.model_copy(update={"n_envs": two.env.n_envs}),
                "rl": one.rl.model_copy(
                    update={
                        "segments_per_minibatch": two.rl.segments_per_minibatch,
                        "model_compile": two.rl.model_compile,
                    }
                ),
            }
        )
        == two
    )


@pytest.mark.parametrize(
    "config_path",
    [_BC_RUN_PPO_CONFIG, *_RANKED_PPO_CONFIGS],
    ids=lambda p: p.stem,
)
@pytest.mark.parametrize(
    "mode", ["model_only", "model_and_optimizer", "model_fresh_critic_head"]
)
def test_bc_best_loads_through_ppo_load_model_weights(
    bc_best: tuple[Path, Any, Path], config_path: Path, mode: str
) -> None:
    best_path, source, data = bc_best
    cfg = _cpu_model_config(FullConfig.from_file(config_path))
    model = _ppo_model(cfg, seed=5)
    fresh_critic = {k: v.clone() for k, v in model.critic_head.state_dict().items()}
    trainer = _ppo_trainer(model, cfg)
    metadata = trainer.load_model_weights(
        best_path,
        load_optimizer=mode == "model_and_optimizer",
        fresh_state_keys=run_ppo._fresh_state_keys_for_mode(model, mode),
    )
    assert metadata.env_steps == 0
    model.eval()
    batch = load_bc_dataset(data).validation.gather(np.array([0, 1, 2]))
    with torch.no_grad():
        want = source.evaluate_actions(batch.obs, batch.actions)
        got = model.evaluate_actions(batch.obs, batch.actions)
    assert torch.equal(
        got.log_probs.per_player_entity, want.log_probs.per_player_entity
    )
    assert torch.equal(got.log_probs.event, want.log_probs.event)
    for name, value in model.state_dict().items():
        expected = (
            fresh_critic[name.removeprefix("critic_head.")]
            if mode == "model_fresh_critic_head" and name.startswith("critic_head.")
            else source.state_dict()[name]
        )
        assert torch.equal(value, expected), name
    if mode == "model_fresh_critic_head":
        assert not torch.equal(got.values, want.values)
    else:
        assert torch.equal(got.values, want.values)
        assert torch.equal(got.winner_probabilities, want.winner_probabilities)
    loaded = _optimizer_state_tensors(trainer.optimizer.state_dict())
    saved = _optimizer_state_tensors(
        torch.load(best_path, weights_only=False)["optimizer"]
    )
    assert saved
    if mode == "model_and_optimizer":
        assert len(loaded) == len(saved)
        assert all(torch.equal(a, b) for a, b in zip(loaded, saved, strict=True))
    else:
        assert not loaded


def _optimizer_state_tensors(state_dict: dict[str, Any]) -> list[torch.Tensor]:
    """Every moment/momentum tensor of a (composite) optimizer state dict."""
    return [
        value
        for optimizer in state_dict["optimizers"]
        for _, param_state in sorted(optimizer["state"].items())
        for _, value in sorted(param_state.items())
        if isinstance(value, torch.Tensor) and value.numel() > 1
    ]


def test_ppo_load_rejects_prohibited_checkpoint_state(
    tmp_path: Path, bc_best: tuple[Path, Any, Path]
) -> None:
    cfg = _cpu_model_config(FullConfig.from_file(_BC_RUN_PPO_CONFIG))
    best_path = bc_best[0]
    clean = torch.load(best_path, weights_only=False)

    def load(checkpoint: dict[str, Any], name: str) -> None:
        path = tmp_path / name
        torch.save(checkpoint, path)
        model = _ppo_model(cfg, seed=5)
        _ppo_trainer(model, cfg).load_model_weights(path)

    # Identity-bearing or carried state beside the weights is not ignored.
    for extra in ("opponent_id", "hidden_state"):
        with pytest.raises(ValueError, match=f"unexpected keys \\['{extra}'\\]"):
            load({**clean, extra: torch.zeros(1)}, f"{extra}.pt")
        with pytest.raises(ValueError, match=extra):
            run_ppo._checkpoint_metadata({**clean, extra: 0}, path=best_path)
        # The teacher_init / initial last-best loader, including for a minimal
        # model-only checkpoint.
        for name, checkpoint in (
            (f"{extra}-teacher.pt", {**clean, extra: 0}),
            (f"{extra}-minimal.pt", {"model": clean["model"], extra: 0}),
        ):
            torch.save(checkpoint, tmp_path / name)
            with pytest.raises(ValueError, match=f"unexpected keys \\['{extra}'\\]"):
                run_ppo._load_model_weights(
                    _ppo_model(cfg, seed=5),
                    path=tmp_path / name,
                    device=torch.device("cpu"),
                )
    torch.save({"model": clean["model"]}, tmp_path / "minimal.pt")
    run_ppo._load_model_weights(
        _ppo_model(cfg, seed=5),
        path=tmp_path / "minimal.pt",
        device=torch.device("cpu"),
    )
    # --load-model-weights needs the run metadata; a minimal checkpoint fails
    # with a named error rather than a bare KeyError.
    with pytest.raises(
        ValueError,
        match=(
            r"checkpoint is missing keys \['env_steps', 'player_step_total', "
            r"'total_games_played', 'wandb_run_id'\]"
        ),
    ):
        load({"model": clean["model"]}, "minimal-weights.pt")
    # Nor inside the model state: an opponent embedding has no place to load.
    tainted = {
        **clean,
        "model": {**clean["model"], "opponent_embedding.weight": torch.zeros(3, 4)},
    }
    with pytest.raises(RuntimeError, match=r"opponent_embedding\.weight"):
        load(tainted, "tainted.pt")
    torch.save(tainted, tmp_path / "tainted-teacher.pt")
    with pytest.raises(RuntimeError, match=r"opponent_embedding\.weight"):
        run_ppo._load_model_weights(
            _ppo_model(cfg, seed=5),
            path=tmp_path / "tainted-teacher.pt",
            device=torch.device("cpu"),
        )
    # A fresh critic head still requires the checkpoint's critic tensors.
    no_critic = {
        **clean,
        "model": {
            k: v for k, v in clean["model"].items() if not k.startswith("critic_head.")
        },
    }
    torch.save(no_critic, tmp_path / "no-critic.pt")
    model = _ppo_model(cfg, seed=5)
    with pytest.raises(RuntimeError, match="missing non-LoRA model state_dict keys"):
        _ppo_trainer(model, cfg).load_model_weights(
            tmp_path / "no-critic.pt",
            fresh_state_keys=run_ppo._fresh_state_keys_for_mode(
                model, "model_fresh_critic_head"
            ),
        )


def test_fresh_critic_head_mode_is_explicit(bc_best: tuple[Path, Any, Path]) -> None:
    cfg = _cpu_model_config(FullConfig.from_file(_BC_RUN_PPO_CONFIG))
    model = _ppo_model(cfg, seed=5)
    assert run_ppo._fresh_state_keys_for_mode(model, "model_only") == frozenset()
    assert run_ppo._fresh_state_keys_for_mode(
        model, "model_fresh_critic_head"
    ) == frozenset(
        {
            "critic_head.up.weight",
            "critic_head.up.bias",
            "critic_head.out.weight",
            "critic_head.out.bias",
        }
    )
    assert "model_fresh_critic_head" in run_ppo.LOAD_MODEL_WEIGHTS_MODES
    with pytest.raises(ValueError, match="KaggricultureTransformer only"):
        run_ppo._fresh_state_keys_for_mode(
            torch.nn.Linear(2, 1), "model_fresh_critic_head"
        )
    with pytest.raises(ValueError, match="not model state keys"):
        _ppo_trainer(model, cfg).load_model_weights(
            bc_best[0], fresh_state_keys=frozenset({"critic_head.missing"})
        )


_REAL_BC_BEST = "KG_V3_BC_BEST"
_REAL_BC_SHARDS = "KG_V3_BC_SHARDS"
_REAL_BC_BEST_SHA256 = (
    "fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51"
)


@pytest.mark.skipif(
    not (os.environ.get(_REAL_BC_BEST) and os.environ.get(_REAL_BC_SHARDS)),
    reason=f"set {_REAL_BC_BEST} and {_REAL_BC_SHARDS} to check the real BC best",
)
def test_real_bc_best_loads_and_forwards_on_a_real_shard() -> None:
    """Custody check of the A100 BC best (bc-20260929-142216, step 3200).

    ``KG_V3_BC_SHARDS`` is a directory holding the run's ``manifest.json`` and at
    least one of its validation shards; the shard's SHA-256 is checked.
    """
    best_path = Path(os.environ[_REAL_BC_BEST])
    shards = Path(os.environ[_REAL_BC_SHARDS])
    assert bc_module.file_sha256(best_path) == _REAL_BC_BEST_SHA256
    manifest = BCManifest.model_validate_json((shards / MANIFEST_NAME).read_bytes())
    present = tuple(
        e
        for e in manifest.episodes
        if e.split == "validation" and (shards / e.shard_path).is_file()
    )
    assert present
    split = bc_data_module._load_split(
        shards,
        "validation",
        manifest.model_copy(update={"episodes": present[:1]}),
        rank=0,
        world_size=1,
    )
    batch = split.gather(np.arange(0, split.rank_rows[0], 10, dtype=np.int64))
    outputs = {}
    for mode in ("model_only", "model_fresh_critic_head"):
        cfg = _cpu_model_config(
            FullConfig.from_file(_REPO / "configs" / "kaggriculture_2rank.yaml")
        )
        model = _ppo_model(cfg, seed=0)
        _ppo_trainer(model, cfg).load_model_weights(
            best_path,
            fresh_state_keys=run_ppo._fresh_state_keys_for_mode(model, mode),
        )
        model.eval()
        with torch.no_grad():
            outputs[mode] = model.evaluate_actions(batch.obs, batch.actions)
        for tensor in (
            outputs[mode].values,
            outputs[mode].winner_probabilities,
            outputs[mode].log_probs.per_player_entity,
        ):
            assert bool(torch.isfinite(tensor).all()), mode
    kept, fresh = outputs["model_only"], outputs["model_fresh_critic_head"]
    assert torch.equal(
        kept.log_probs.per_player_entity, fresh.log_probs.per_player_entity
    )
    nll = -kept.log_probs.per_player_entity.float().sum(-1) / batch.actions.lengths
    assert float(nll[batch.policy_seat].mean()) < 1.0


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bc_wandb: _FakeWandb,  # noqa: ARG001
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


# --- W&B: the shared v3 path, its credential gate and outage receipts ----------


def _fresh_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> tuple[Path, Path]:
    data = _dataset_root(tmp_path)
    config_path = tmp_path / "bc.yaml"
    _bc_config(max_steps=2).to_file(config_path)
    out = tmp_path / "runs"
    _script_argv(
        monkeypatch,
        config_path,
        data,
        "--output-dir",
        str(out),
        "--source-commit",
        "src-0",
        *extra,
    )
    return out, data


def test_script_logs_online_to_the_v3_project_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bc_wandb: _FakeWandb
) -> None:
    out, _ = _fresh_argv(monkeypatch, tmp_path, "--experiment-id", "bc-top1")
    train_bc_script.main()

    (run_dir,) = out.iterdir()
    (init,) = bc_wandb.inits
    assert init["project"] == train_logging.WANDB_PROJECT == "kg-v3"
    assert init["mode"] == "online"
    assert init["job_type"] == "bc"
    assert init["group"] == "bc-top1"
    assert init["name"] == f"bc-{run_dir.name}"
    assert init["tags"] == ["kaggriculture-v3", "bc", "kaggriculture"]
    assert init["config"]["v3"] == {"experiment_id": "bc-top1", "job_type": "bc"}
    assert init["config"]["provenance"]["telemetry_mode"] == "wandb-online"
    (run,) = bc_wandb.runs
    assert run.summary["v3/telemetry_mode"] == "wandb-online"
    assert run.summary["v3/source_commit"] == "src-0"
    assert run.finished == [0]
    assert bc_wandb.logs

    (shared,) = _jsonl(run_dir / train_logging.ATTEMPTS_FILE)
    (bc_attempt,) = _jsonl(run_dir / train_bc_script.ATTEMPTS)
    assert shared["job_type"] == "bc"
    assert shared["telemetry_mode"] == "wandb-online"
    assert shared["wandb_project"] == "kg-v3"
    assert shared["wandb_run_id"] == run.id
    bc_config, ppo_config = bc_module.load_bc_configs(run_dir / "bc_config.yaml")
    assert shared["config_sha256"] == bc_module.bc_config_sha256(bc_config, ppo_config)
    for key in ("experiment_id", "config_sha256", "telemetry_mode", "source_commit"):
        assert bc_attempt[key] == shared[key]
    result = json.loads((run_dir / BC_RESULT).read_text())
    best = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())
    for record in (result, best):
        assert record["telemetry_mode"] == "wandb-online"
        assert record["experiment_id"] == "bc-top1"
        assert record["wandb_run_id"] == run.id


@pytest.mark.parametrize(
    ("env", "error", "match"),
    [
        ({}, train_logging.MissingWandbCredentialsError, "--wandb-mode offline"),
        (
            {"WANDB_API_KEY": "test-key-not-real", "WANDB_MODE": "offline"},
            ValueError,
            "WANDB_MODE='offline' disagrees",
        ),
    ],
)
def test_script_fails_fast_before_config_or_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bc_wandb: _FakeWandb,
    env: dict[str, str],
    error: type[Exception],
    match: str,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    out, _ = _fresh_argv(monkeypatch, tmp_path)

    def reached(name: str) -> Any:
        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(f"{name} ran before the telemetry gate")

        return fail

    for name in ("load_bc_configs", "load_bc_dataset", "resolve_source_commit"):
        monkeypatch.setattr(train_bc_script, name, reached(name))

    with pytest.raises(error, match=match) as info:
        train_bc_script.main()

    if error is train_logging.MissingWandbCredentialsError:
        message = str(info.value)
        assert "WANDB_API_KEY" in message
        assert "install-the-wandb-credential-before-any-pod-launch" in message
    assert not out.exists()
    assert bc_wandb.inits == []


@pytest.mark.parametrize(
    ("extra", "telemetry", "wandb_mode"),
    [
        (("--wandb-mode", "offline"), "wandb-offline", "offline"),
        (("--log-mode", "debug"), "disabled", None),
    ],
)
def test_script_outage_needs_a_flag_and_is_announced_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bc_wandb: _FakeWandb,
    extra: tuple[str, ...],
    telemetry: str,
    wandb_mode: str | None,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY")  # no credentials: the flag alone runs
    out, _ = _fresh_argv(monkeypatch, tmp_path, *extra)
    train_bc_script.main()

    (run_dir,) = out.iterdir()
    assert [init["mode"] for init in bc_wandb.inits] == (
        [] if wandb_mode is None else [wandb_mode]
    )
    captured = capsys.readouterr()
    assert f"W&B TELEMETRY OUTAGE: telemetry_mode={telemetry}" in captured.err
    assert f"W&B TELEMETRY OUTAGE recorded: telemetry_mode={telemetry}" in captured.err
    assert ("until `wandb sync`" in captured.err) is (wandb_mode == "offline")
    assert json.loads(captured.out.splitlines()[-1])["telemetry_mode"] == telemetry
    (shared,) = _jsonl(run_dir / train_logging.ATTEMPTS_FILE)
    (bc_attempt,) = _jsonl(run_dir / train_bc_script.ATTEMPTS)
    result = json.loads((run_dir / BC_RESULT).read_text())
    best = json.loads((run_dir / CHECKPOINT_BC_BEST_RECORD).read_text())
    for record in (shared, bc_attempt, result, best):
        assert record["telemetry_mode"] == telemetry
    assert result["wandb_run_id"] == shared["wandb_run_id"]
    assert (result["wandb_run_id"] is None) is (wandb_mode is None)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("--log-mode", "debug"), "resume launches require wandb logging"),
        (("--wandb-mode", "offline"), "do not support --wandb-mode offline"),
        (("--experiment-id", "other"), "keep the recorded --experiment-id"),
    ],
)
def test_script_resume_rejects_telemetry_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bc_wandb: _FakeWandb,  # noqa: ARG001
    extra: tuple[str, ...],
    match: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _script_argv(monkeypatch, run_dir, tmp_path, *extra)
    with pytest.raises(ValueError, match=match):
        train_bc_script.main()


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("--log-mode", "debug", "--wandb-mode", "offline"), "requires --log-mode"),
        (("--experiment-id", "bad id"), "experiment id must match"),
    ],
)
def test_script_rejects_contradictory_fresh_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bc_wandb: _FakeWandb,
    extra: tuple[str, ...],
    match: str,
) -> None:
    out, _ = _fresh_argv(monkeypatch, tmp_path, *extra)
    with pytest.raises(ValueError, match=match):
        train_bc_script.main()
    assert not out.exists()
    assert bc_wandb.inits == []


def test_script_resume_continues_the_saved_wandb_run_and_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bc_wandb: _FakeWandb
) -> None:
    out, data = _fresh_argv(monkeypatch, tmp_path, "--experiment-id", "bc-top1")
    train_bc_script.main()
    (run_dir,) = out.iterdir()
    saved = BCConfig.from_file(run_dir / "bc_config.yaml")
    saved.model_copy(update={"max_steps": 4}).to_file(run_dir / "bc_config.yaml")
    _script_argv(monkeypatch, run_dir, data, "--source-commit", "src-1")
    train_bc_script.main()

    first, second = bc_wandb.inits
    assert "id" not in first
    assert second["id"] == bc_wandb.runs[0].id
    assert second["resume"] == "must"
    assert second["group"] == "bc-top1"
    shared = _jsonl(run_dir / train_logging.ATTEMPTS_FILE)
    bc_attempts = _jsonl(run_dir / train_bc_script.ATTEMPTS)
    assert (
        [a["attempt"] for a in shared] == [a["attempt"] for a in bc_attempts] == [0, 1]
    )
    assert [a["experiment_id"] for a in bc_attempts] == ["bc-top1", "bc-top1"]
    assert bc_attempts[1]["attempt_source_commits"] == ["src-0", "src-1"]
    # The raised budget is part of the settings hash, so the attempts differ.
    assert shared[0]["config_sha256"] != shared[1]["config_sha256"]
    assert shared[1]["config_sha256"] == bc_attempts[1]["config_sha256"]
    assert json.loads((run_dir / BC_RESULT).read_text())["attempt"] == 1


def test_script_resume_of_a_debug_run_is_rejected_before_any_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bc_wandb: _FakeWandb
) -> None:
    out, data = _fresh_argv(monkeypatch, tmp_path, "--log-mode", "debug")
    train_bc_script.main()
    (run_dir,) = out.iterdir()
    receipts = {
        name: (run_dir / name).read_text()
        for name in (train_logging.ATTEMPTS_FILE, train_bc_script.ATTEMPTS)
    }
    saved = BCConfig.from_file(run_dir / "bc_config.yaml")
    saved.model_copy(update={"max_steps": 4}).to_file(run_dir / "bc_config.yaml")
    _script_argv(monkeypatch, run_dir, data, "--source-commit", "src-1")
    with pytest.raises(ValueError, match="no wandb_run_id"):
        train_bc_script.main()
    assert bc_wandb.inits == []
    for name, text in receipts.items():
        assert (run_dir / name).read_text() == text


def test_script_rejects_a_source_commit_that_disagrees_with_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bc_wandb: _FakeWandb
) -> None:
    monkeypatch.setattr(train_logging, "git_source_commit", lambda _cwd: "abc123")
    out, _ = _fresh_argv(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="disagrees with git's 'abc123'"):
        train_bc_script.main()
    assert not out.exists()
    assert bc_wandb.inits == []


def test_non_main_ranks_skip_the_credential_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("WANDB_API_KEY", "NETRC", "WANDB_BASE_URL", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    args = SimpleNamespace(
        log_mode=train_logging.LogMode.WANDB,
        wandb_mode=train_logging.WandbMode.ONLINE,
    )
    assert (
        train_bc_script._check_launch_telemetry(cast(Any, args), is_main_process=False)
        is train_logging.TelemetryMode.WANDB_ONLINE
    )
    with pytest.raises(train_logging.MissingWandbCredentialsError):
        train_bc_script._check_launch_telemetry(cast(Any, args), is_main_process=True)
    assert train_bc_script._NoopLogger("r").wandb_run_facts() is None


def test_bc_config_sha256_hashes_the_ppo_config_content_not_its_path(
    tmp_path: Path,
) -> None:
    ppo_config = _ppo_config()
    config = _bc_config()
    digest = bc_module.bc_config_sha256(config, ppo_config)
    copy_path = tmp_path / "config.yaml"
    ppo_config.to_file(copy_path)
    moved = config.model_copy(update={"ppo_config": copy_path})
    assert bc_module.bc_config_sha256(moved, ppo_config) == digest
    assert (
        bc_module.bc_config_sha256(
            config.model_copy(update={"max_steps": 5}), ppo_config
        )
        != digest
    )
    changed_ppo = ppo_config.model_copy(
        update={"rl": ppo_config.rl.model_copy(update={"dtype": "bfloat16"})}
    )
    assert bc_module.bc_config_sha256(config, changed_ppo) != digest
