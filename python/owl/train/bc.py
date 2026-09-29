"""Offline behavior cloning of the Kaggriculture policy (rebuild plan Task 5.2).

BC is a warm start for ``scripts/run_ppo.py``, not a second PPO loop. It trains
the registered ``KaggricultureTransformer`` from the PPO config it will start
(``ppo_config``) on the Task 5.1 shards (``owl.kaggriculture.bc_data``) and
reuses Isaiah's seams unchanged: ``create_model``/``reset_parameters``,
``create_optimizer``/``create_lr_scheduler`` (Muon/AdamW), ``autocast_context``
(BF16 from ``rl.dtype``), ``configure_model_compile`` (the Kaggriculture
cuBLAS-only claim), ``wrap_model_for_distributed`` (DDP under torchrun) and
``MetricLogger``.

Objective. One ``evaluate_actions`` call per microbatch replays each recorded
program teacher-forced (the model's replay validation admits it) and returns
log-probabilities and the winner distribution from one encode (Isaiah's I0):

- policy: the turn's negative log-likelihood divided by its program length,
  averaged over seat rows, so every learner turn weighs the same;
- critic: Isaiah's winner cross-entropy against the episode's raw final banks
  (lesson L1: 1 for the seat with the larger bank, 0.5 each on a draw), scaled
  by ``value_coef``.

The critic is trained, not frozen. Isaiah trains every parameter of the shared
trunk and both heads together, and distills a teacher's policy and winner
distribution with equal coefficients (``teacher_kl_coef = teacher_value_coef``).
BC is distillation from the recorded player, so ``value_coef`` defaults to that
ratio (1.0). The reference branch froze the critic head, but its shared trunk
still moved, so its critic arrived at PPO miscalibrated without a target.

Selection. ``HeldOutSelection`` keeps the checkpoint with the lowest held-out
policy NLL on the fixed validation episodes and stops after ``patience_evals``
evaluations without an improvement (lesson L9: held-out NLL reached its minimum
and then degraded while training loss kept falling).

Data. Each rank holds rows ``[rank::world_size]`` of every episode
(``load_bc_dataset``); training order is a pure function of ``(seed, step,
micro, rank)`` over that partition (``TrainSharding``), so ranks never share a
row and a restart from ``bc_state.pt`` replays exactly. Every evaluation covers
all validation rows (each rank its partition, reduced across ranks).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from pydantic import Field

from owl.config import BaseConfig
from owl.kaggriculture import types as kt
from owl.kaggriculture.bc_data import BCBatch, BCDataset, BCSplit
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.model import BaseModelAPI, KaggricultureTransformer, create_model
from owl.model.base import ModelEvaluation
from owl.model.kaggriculture import KaggricultureTransformerConfig
from owl.model.kaggriculture_workload import (
    ForwardWorkload,
    WorkloadHeadroom,
    check_workload_headroom,
)
from owl.train.config import FullConfig
from owl.train.distributed import (
    DistributedContext,
    all_reduce_any,
    all_reduce_sum,
    model_no_sync_context,
    unwrap_model,
    wrap_model_for_distributed,
)
from owl.train.logging import DebugLogger, LogMode, MetricLogger
from owl.train.optimizer import (
    LRScheduler,
    Optimizer,
    OptimizerConfig,
)
from owl.train.utils import autocast_context

BC_WANDB_PROJECT = "kg-v3"
CHECKPOINT_BC_BEST = "checkpoint_bc_best.pt"
CHECKPOINT_BC_BEST_RECORD = "checkpoint_bc_best.json"
BC_STATE = "bc_state.pt"
BC_HISTORY = "bc_history.jsonl"
BC_RESULT = "bc_result.json"
PPO_CONFIG_NAME = "config.yaml"
BC_CONFIG_NAME = "bc_config.yaml"
StopReason = Literal["no_held_out_improvement", "max_steps", "max_runtime"]


class BCConfig(BaseConfig):
    """BC settings; the model, dtype and compile settings come from ``ppo_config``.

    ``ppo_config`` is the PPO ``FullConfig`` the BC checkpoint will start, relative
    to this file. ``rows_per_rank`` counts turn rows (two seat rows each) per
    microbatch per rank; the global optimizer batch is ``rows_per_rank x
    world_size x gradient_accumulation_steps`` turn rows.
    """

    ppo_config: Path
    optimizer: OptimizerConfig
    seed: int = Field(ge=0)
    rows_per_rank: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(gt=0.0)
    value_coef: float = Field(default=1.0, ge=0.0)
    eval_interval_steps: int = Field(ge=1)
    patience_evals: int = Field(ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    max_steps: int = Field(ge=1)
    validation_rows_per_forward: int = Field(ge=1)
    preflight_train_replay: bool = True


def load_bc_configs(
    path: Path, overrides: dict[str, Any] | None = None
) -> tuple[BCConfig, FullConfig]:
    """The BC config and the Kaggriculture PPO config it names."""
    bc_config = BCConfig.from_file(path, overrides=overrides)
    ppo_path = bc_config.ppo_config
    if not ppo_path.is_absolute():
        ppo_path = path.resolve().parent / ppo_path
    ppo_config = FullConfig.from_file(ppo_path)
    require_kaggriculture(ppo_config)
    return bc_config, ppo_config


def require_kaggriculture(
    cfg: FullConfig,
) -> tuple[KaggricultureTransformerConfig, KaggricultureEnvConfig]:
    if not isinstance(cfg.model, KaggricultureTransformerConfig) or not isinstance(
        cfg.env, KaggricultureEnvConfig
    ):
        raise ValueError(
            "BC trains the Kaggriculture policy; ppo_config must be a Kaggriculture "
            f"config, got model_arch={cfg.model.model_arch!r}"
        )
    return cfg.model, cfg.env


def bc_forward_workloads(config: BCConfig) -> tuple[ForwardWorkload, ...]:
    """Per-rank seat rows of the BC training and validation forwards."""
    return (
        ForwardWorkload("bc_microbatch", config.rows_per_rank * kt.PLAYERS),
        ForwardWorkload(
            "bc_validation", config.validation_rows_per_forward * kt.PLAYERS
        ),
    )


def check_bc_workload(
    config: BCConfig, ppo_config: FullConfig
) -> tuple[WorkloadHeadroom, ...]:
    model_config, _ = require_kaggriculture(ppo_config)
    return check_workload_headroom(model_config, bc_forward_workloads(config))


def build_bc_model(
    ppo_config: FullConfig, *, device: torch.device, seed: int
) -> KaggricultureTransformer:
    """The PPO config's model, freshly initialized from ``seed`` on every rank."""
    model_config, env_config = require_kaggriculture(ppo_config)
    torch.manual_seed(seed)
    model = create_model(
        model_config,
        obs_spec=env_config.obs_spec,
        action_spec=env_config.action_spec,
    ).to(device)
    model.reset_parameters()
    return model


def wrap_bc_model_for_distributed(
    model: KaggricultureTransformer, context: DistributedContext
) -> BaseModelAPI[Any, Any, Any]:
    """Isaiah's DDP adapter around the Kaggriculture model.

    The adapter's annotations name Orbit's batch types, but its dispatch passes
    observations and actions through unchanged; the Kaggriculture trainer seam
    (plan Task 3.1) has not generalized them yet, so the model is cast here.
    """
    return wrap_model_for_distributed(cast(BaseModelAPI, model), context)


# --- data order -----------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _epoch_permutation(
    seed: int, epoch: int, rank: int, num_rows: int
) -> npt.NDArray[np.int64]:
    permutation = np.random.default_rng([seed, epoch, rank]).permutation(num_rows)
    permutation.setflags(write=False)
    return permutation


@dataclass(frozen=True)
class TrainSharding:
    """Deterministic training order over each rank's own resident rows.

    ``rank_rows`` holds every rank's training row count (from the manifest).
    Epoch ``e`` of rank ``r`` draws one permutation of that rank's rows from
    ``(seed, e, r)``; micro ``m`` of step ``s`` takes the next ``rows_per_rank``
    of it. Every rank runs the same ``steps_per_epoch`` (set by the smallest
    partition) and leaves its remainder unused that epoch.
    """

    seed: int
    rows_per_rank: int
    gradient_accumulation_steps: int
    rank_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 1:
            raise ValueError(
                f"the smallest rank partition ({min(self.rank_rows)} rows) cannot "
                f"fill one optimizer step of {self.rows_per_rank} x "
                f"{self.gradient_accumulation_steps} rows"
            )

    @property
    def world_size(self) -> int:
        return len(self.rank_rows)

    @property
    def rows_per_step(self) -> int:
        """Global turn rows per optimizer step."""
        return self.rows_per_rank * self.world_size * self.gradient_accumulation_steps

    @property
    def steps_per_epoch(self) -> int:
        per_rank_step = self.rows_per_rank * self.gradient_accumulation_steps
        return min(self.rank_rows) // per_rank_step

    def rows(self, *, step: int, micro: int, rank: int) -> npt.NDArray[np.int64]:
        """Indices into ``rank``'s resident training rows."""
        if step < 0 or not 0 <= micro < self.gradient_accumulation_steps:
            raise ValueError(f"invalid step {step} / micro {micro}")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} outside world size {self.world_size}")
        epoch, step_in_epoch = divmod(step, self.steps_per_epoch)
        start = (
            step_in_epoch * self.gradient_accumulation_steps + micro
        ) * self.rows_per_rank
        permutation = _epoch_permutation(self.seed, epoch, rank, self.rank_rows[rank])
        return permutation[start : start + self.rows_per_rank]


# --- objective ------------------------------------------------------------------


def winner_targets(final_banks: torch.Tensor) -> torch.Tensor:
    """Per-seat winner distribution ``[rows, seat, (self, opponent)]`` (L1).

    ``final_banks`` is ``[rows, 2]`` in seat order; the larger raw bank wins and
    equal banks are a draw (0.5 each).
    """
    if final_banks.ndim != 2 or final_banks.shape[1] != kt.PLAYERS:
        raise ValueError(f"final_banks must be [rows, 2], got {final_banks.shape}")
    seat0 = torch.where(
        final_banks[:, 0] > final_banks[:, 1],
        1.0,
        torch.where(final_banks[:, 0] < final_banks[:, 1], 0.0, 0.5),
    ).to(torch.float32)
    self_wins = torch.stack((seat0, 1.0 - seat0), dim=1)
    return torch.stack((self_wins, 1.0 - self_wins), dim=-1)


@dataclass(frozen=True)
class BCTerms:
    """Per seat row ``[rows, 2]``, float32."""

    turn_nll: torch.Tensor
    program_nll: torch.Tensor
    frames: torch.Tensor
    value_ce: torch.Tensor


def bc_terms(evaluation: ModelEvaluation, batch: BCBatch) -> BCTerms:
    if evaluation.winner_log_probabilities is None:
        raise ValueError("BC needs winner log-probabilities from evaluate_actions")
    program_nll = -evaluation.log_probs.per_player_entity.float().sum(dim=-1)
    frames = batch.actions.lengths.to(program_nll.dtype)
    targets = winner_targets(batch.final_banks).to(program_nll.device)
    value_ce = -(targets * evaluation.winner_log_probabilities.float()).sum(dim=-1)
    return BCTerms(
        turn_nll=program_nll / frames,
        program_nll=program_nll,
        frames=frames,
        value_ce=value_ce,
    )


def bc_loss(terms: BCTerms, *, value_coef: float) -> torch.Tensor:
    return terms.turn_nll.mean() + value_coef * terms.value_ce.mean()


def evaluate_bc_batch(
    model: BaseModelAPI[Any, Any, Any],
    batch: BCBatch,
    *,
    ppo_config: FullConfig,
    device: torch.device,
) -> BCTerms:
    with autocast_context(ppo_config.rl, device):
        evaluation = model.evaluate_actions(batch.obs, batch.actions)
    return bc_terms(evaluation, batch)


@dataclass(frozen=True)
class RowMetrics:
    """Means over seat rows, reduced across ranks."""

    turn_nll: float
    frame_nll: float
    value_ce: float
    seat_rows: int


def evaluate_rows(
    model: BaseModelAPI[Any, Any, Any],
    split: BCSplit,
    rows: npt.NDArray[np.int64],
    *,
    rows_per_forward: int,
    ppo_config: FullConfig,
    context: DistributedContext,
) -> RowMetrics:
    """Teacher-forced NLL over ``rows`` (this rank's share), all ranks reduced.

    Runs the unwrapped model without gradients; each call also admits every
    recorded program through the model's replay validation.
    """
    inner = unwrap_model(model)
    was_training = inner.training
    inner.eval()
    sums = torch.zeros(5, dtype=torch.float64, device=context.device)
    with torch.no_grad():
        for start in range(0, rows.size, rows_per_forward):
            batch = split.gather(rows[start : start + rows_per_forward]).to(
                context.device
            )
            terms = evaluate_bc_batch(
                inner, batch, ppo_config=ppo_config, device=context.device
            )
            sums += torch.stack(
                (
                    terms.turn_nll.sum(),
                    terms.program_nll.sum(),
                    terms.frames.sum(),
                    terms.value_ce.sum(),
                    torch.tensor(float(terms.turn_nll.numel()), device=sums.device),
                )
            ).to(sums)
    inner.train(was_training)
    total = all_reduce_sum(sums, context).tolist()
    seat_rows = int(total[4])
    if seat_rows == 0:
        raise ValueError("no rows to evaluate on any rank")
    return RowMetrics(
        turn_nll=total[0] / seat_rows,
        frame_nll=total[1] / total[2],
        value_ce=total[3] / seat_rows,
        seat_rows=seat_rows,
    )


# --- selection and stopping (L9) --------------------------------------------------


@dataclass
class HeldOutSelection:
    """Best-by-held-out-NLL selection and the sustained-non-improvement stop."""

    patience_evals: int
    min_delta: float
    best_nll: float = math.inf
    best_step: int = -1
    evals_since_best: int = 0
    last_nll: float = math.nan
    last_step: int = -1

    def observe(self, nll: float, *, step: int) -> bool:
        """Record one evaluation; ``True`` when it is the new best."""
        if not math.isfinite(nll):
            raise ValueError(f"held-out NLL must be finite, got {nll} at step {step}")
        if step <= self.last_step:
            raise ValueError(
                f"evaluations must advance: step {step} after {self.last_step}"
            )
        self.last_nll = nll
        self.last_step = step
        if nll < self.best_nll - self.min_delta:
            self.best_nll = nll
            self.best_step = step
            self.evals_since_best = 0
            return True
        self.evals_since_best += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.evals_since_best >= self.patience_evals


# --- checkpoints ----------------------------------------------------------------


def ppo_warm_start_checkpoint(
    model: BaseModelAPI[Any, Any, Any],
    optimizer: Optimizer,
    *,
    bc_steps: int,
    wandb_run_id: str | None,
) -> dict[str, object]:
    """Exactly the keys ``scripts/run_ppo.py`` requires of a checkpoint.

    PPO starts from it with ``--load-model-weights`` (``env_steps`` 0 keeps the
    PPO schedule fresh) or as ``rl.teacher_init``; BC facts live in the sidecar.
    """
    return {
        "model": _cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": None,
        "env_steps": 0,
        "optimizer_steps": bc_steps,
        "player_step_total": 0,
        "total_games_played": 0,
        "total_active_entities": 0,
        "target_kl_exceeded_total": 0,
        "wandb_run_id": wandb_run_id,
    }


def _cpu_state_dict(model: BaseModelAPI[Any, Any, Any]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}


def write_atomic(obj: object, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BCResumeState:
    step: int
    selection: HeldOutSelection
    wandb_run_id: str | None
    dataset_manifest_sha256: str
    world_size: int
    model: Mapping[str, torch.Tensor]
    optimizer: Mapping[str, Any]
    lr_scheduler: Mapping[str, Any] | None


_RESUME_KEYS = {
    "step",
    "selection",
    "wandb_run_id",
    "dataset_manifest_sha256",
    "world_size",
    "model",
    "optimizer",
    "lr_scheduler",
}


def load_bc_state(path: Path) -> BCResumeState:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != _RESUME_KEYS:
        raise ValueError(f"BC state must have exactly {sorted(_RESUME_KEYS)}: {path}")
    selection = state["selection"]
    if not isinstance(selection, dict):
        raise ValueError(f"BC state selection must be a mapping: {path}")
    return BCResumeState(
        step=int(state["step"]),
        selection=HeldOutSelection(**selection),
        wandb_run_id=state["wandb_run_id"],
        dataset_manifest_sha256=str(state["dataset_manifest_sha256"]),
        world_size=int(state["world_size"]),
        model=state["model"],
        optimizer=state["optimizer"],
        lr_scheduler=state["lr_scheduler"],
    )


def check_resume_compatible(
    state: BCResumeState, *, dataset: BCDataset, world_size: int
) -> None:
    if state.dataset_manifest_sha256 != dataset.manifest_sha256:
        raise ValueError(
            "resume dataset manifest differs from the run's: "
            f"{dataset.manifest_sha256} != {state.dataset_manifest_sha256}"
        )
    if state.world_size != world_size:
        raise ValueError(
            f"resume needs the run's world size {state.world_size} for the same "
            f"data order, got {world_size}"
        )


def restore_bc_state(
    state: BCResumeState,
    *,
    model: BaseModelAPI[Any, Any, Any],
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
) -> None:
    unwrap_model(model).load_state_dict(dict(state.model))
    optimizer.load_state_dict(dict(state.optimizer))
    if (lr_scheduler is None) != (state.lr_scheduler is None):
        raise ValueError("BC state and optimizer config disagree on an LR schedule")
    if lr_scheduler is not None and state.lr_scheduler is not None:
        lr_scheduler.load_state_dict(dict(state.lr_scheduler))


# --- logging --------------------------------------------------------------------


class BCWandbLogger(MetricLogger):
    """W&B run under the v3 project, ``job_type='bc'``; offline mode supported."""

    def __init__(
        self,
        run_dir: Path,
        *,
        config: Mapping[str, Any],
        mode: Literal["online", "offline"],
        resume_run_id: str | None,
    ) -> None:
        import wandb

        init_kwargs: dict[str, Any] = {}
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._wandb = wandb
        self._run = wandb.init(
            project=BC_WANDB_PROJECT,
            job_type="bc",
            group="bc",
            tags=["kaggriculture-v3", "bc"],
            name=f"bc-{run_dir.name}",
            dir=run_dir,
            config=dict(config),
            mode=mode,
            **init_kwargs,
        )

    @property
    def run_id(self) -> str | None:
        return str(self._run.id)

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._run.log(metrics, step=step)

    def set_summary(self, key: str, value: int | float | str) -> None:
        self._run.summary[key] = value

    def close(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def create_bc_logger(
    log_mode: LogMode,
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    wandb_mode: Literal["online", "offline"],
    resume_run_id: str | None,
) -> MetricLogger:
    if log_mode == LogMode.DEBUG:
        return DebugLogger()
    return BCWandbLogger(
        run_dir, config=config, mode=wandb_mode, resume_run_id=resume_run_id
    )


# --- training loop --------------------------------------------------------------


@dataclass(frozen=True)
class BCResult:
    stop_reason: StopReason
    steps: int
    best_step: int
    best_validation_nll: float
    final_validation_nll: float


@dataclass
class _Interval:
    losses: torch.Tensor
    steps: int = 0
    seat_rows: int = 0
    train_seconds: float = 0.0


def train_bc(
    *,
    config: BCConfig,
    ppo_config: FullConfig,
    dataset: BCDataset,
    model: BaseModelAPI[Any, Any, Any],
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    context: DistributedContext,
    run_dir: Path,
    logger: MetricLogger,
    provenance: Mapping[str, str | int],
    resume: BCResumeState | None = None,
    max_runtime_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BCResult:
    """Train until held-out NLL stops improving, ``max_steps`` or the runtime cap.

    Rank 0 writes the best checkpoint (run_ppo schema), its sidecar record, the
    resume state and the NLL history; every rank takes the same decisions from
    all-reduced metrics.
    """
    device = context.device
    if (dataset.rank, dataset.world_size) != (context.rank, context.world_size):
        raise ValueError(
            f"dataset holds rank {dataset.rank}/{dataset.world_size} rows, but this "
            f"is rank {context.rank}/{context.world_size}"
        )
    sharding = TrainSharding(
        seed=config.seed,
        rows_per_rank=config.rows_per_rank,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        rank_rows=dataset.train.rank_rows,
    )
    held_out = np.arange(dataset.validation.num_rows, dtype=np.int64)
    history_path = run_dir / BC_HISTORY
    started = clock()
    if resume is None:
        step = 0
        selection = HeldOutSelection(
            patience_evals=config.patience_evals, min_delta=config.min_delta
        )
        if context.is_main_process:
            _write_text_atomic("", history_path)
    else:
        step = resume.step
        selection = resume.selection
        if context.is_main_process:
            _truncate_history(history_path, last_step=step)
    wandb_run_id = logger.run_id

    def evaluate_and_record(interval: _Interval | None) -> None:
        eval_started = clock()
        metrics = evaluate_rows(
            model,
            dataset.validation,
            held_out,
            rows_per_forward=config.validation_rows_per_forward,
            ppo_config=ppo_config,
            context=context,
        )
        improved = selection.observe(metrics.turn_nll, step=step)
        record: dict[str, float] = {
            "bc/step": float(step),
            "bc/validation_nll": metrics.turn_nll,
            "bc/validation_frame_nll": metrics.frame_nll,
            "bc/validation_value_ce": metrics.value_ce,
            "bc/validation_seat_rows": float(metrics.seat_rows),
            "bc/best_validation_nll": selection.best_nll,
            "bc/best_step": float(selection.best_step),
            "bc/evals_since_best": float(selection.evals_since_best),
            "bc/learner_seat_rows": float(step * sharding.rows_per_step * kt.PLAYERS),
            "bc/epoch": step / sharding.steps_per_epoch,
            "time/eval_seconds": clock() - eval_started,
            "time/elapsed_seconds": clock() - started,
        }
        if interval is not None and interval.steps > 0:
            reduced = all_reduce_sum(interval.losses, context).tolist()
            record |= {
                "bc/train_nll": reduced[0] / reduced[3],
                "bc/train_value_ce": reduced[1] / reduced[3],
                "bc/grad_norm": reduced[2] / (interval.steps * context.world_size),
                "perf/learner_seat_rows_per_second": (
                    interval.seat_rows * context.world_size / interval.train_seconds
                ),
            }
        if lr_scheduler is not None:
            record["bc/learning_rate"] = float(lr_scheduler.get_last_lr()[0])
        if context.is_main_process:
            if improved:
                _save_best(
                    run_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    nll=metrics.turn_nll,
                    wandb_run_id=wandb_run_id,
                    dataset=dataset,
                    provenance=provenance,
                )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            write_atomic(
                {
                    "step": step,
                    "selection": asdict(selection),
                    "wandb_run_id": wandb_run_id,
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "world_size": context.world_size,
                    "model": _cpu_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": (
                        None if lr_scheduler is None else lr_scheduler.state_dict()
                    ),
                },
                run_dir / BC_STATE,
            )
            logger.log(record, step=step)
        context.barrier()

    if resume is None:
        if config.preflight_train_replay:
            initial_train = evaluate_rows(
                model,
                dataset.train,
                np.arange(dataset.train.num_rows, dtype=np.int64),
                rows_per_forward=config.validation_rows_per_forward,
                ppo_config=ppo_config,
                context=context,
            )
            if context.is_main_process:
                logger.set_summary("bc/initial_train_nll", initial_train.turn_nll)
        evaluate_and_record(None)
    interval = _new_interval(device)
    stop_reason: StopReason | None = (
        "no_held_out_improvement" if selection.should_stop else None
    )
    while stop_reason is None:
        if step >= config.max_steps:
            stop_reason = "max_steps"
            break
        if max_runtime_seconds is not None and all_reduce_any(
            clock() - started >= max_runtime_seconds, context
        ):
            stop_reason = "max_runtime"
            break
        step_started = clock()
        _train_step(
            config=config,
            ppo_config=ppo_config,
            dataset=dataset,
            sharding=sharding,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            context=context,
            step=step,
            losses=interval.losses,
        )
        step += 1
        interval.steps += 1
        interval.seat_rows += (
            config.rows_per_rank * config.gradient_accumulation_steps * kt.PLAYERS
        )
        interval.train_seconds += clock() - step_started
        if step % config.eval_interval_steps == 0:
            evaluate_and_record(interval)
            interval = _new_interval(device)
            if selection.should_stop:
                stop_reason = "no_held_out_improvement"
    if selection.last_step != step:
        evaluate_and_record(interval)
    result = BCResult(
        stop_reason=stop_reason,
        steps=step,
        best_step=selection.best_step,
        best_validation_nll=selection.best_nll,
        final_validation_nll=selection.last_nll,
    )
    if context.is_main_process:
        _write_text_atomic(
            json.dumps(
                {
                    **asdict(result),
                    "selection": "lowest held-out policy NLL; not PPO promotion",
                    "best_checkpoint": CHECKPOINT_BC_BEST,
                    "best_checkpoint_sha256": _sha256(run_dir / CHECKPOINT_BC_BEST),
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "world_size": context.world_size,
                    "rows_per_step": sharding.rows_per_step,
                    "steps_per_epoch": sharding.steps_per_epoch,
                    **provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            run_dir / BC_RESULT,
        )
        for key, value in asdict(result).items():
            logger.set_summary(f"bc/{key}", value)
    context.barrier()
    return result


def _new_interval(device: torch.device) -> _Interval:
    return _Interval(losses=torch.zeros(4, dtype=torch.float64, device=device))


def _train_step(
    *,
    config: BCConfig,
    ppo_config: FullConfig,
    dataset: BCDataset,
    sharding: TrainSharding,
    model: BaseModelAPI[Any, Any, Any],
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    context: DistributedContext,
    step: int,
    losses: torch.Tensor,
) -> None:
    """One optimizer step; adds [nll sum, value CE sum, grad norm, seat rows]."""
    optimizer.zero_grad(set_to_none=True)
    accumulation = config.gradient_accumulation_steps
    for micro in range(accumulation):
        rows = sharding.rows(step=step, micro=micro, rank=context.rank)
        batch = dataset.train.gather(rows).to(context.device)
        with model_no_sync_context(model, enabled=micro < accumulation - 1):
            terms = evaluate_bc_batch(
                model, batch, ppo_config=ppo_config, device=context.device
            )
            loss = bc_loss(terms, value_coef=config.value_coef)
            (loss / accumulation).backward()
        losses[0] += terms.turn_nll.detach().sum().to(losses)
        losses[1] += terms.value_ce.detach().sum().to(losses)
        losses[3] += terms.turn_nll.numel()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        unwrap_model(model).parameters(),
        config.max_grad_norm,
        error_if_nonfinite=True,
    )
    losses[2] += grad_norm.detach().to(losses)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()


def _save_best(
    run_dir: Path,
    *,
    model: BaseModelAPI[Any, Any, Any],
    optimizer: Optimizer,
    step: int,
    nll: float,
    wandb_run_id: str | None,
    dataset: BCDataset,
    provenance: Mapping[str, str | int],
) -> None:
    path = run_dir / CHECKPOINT_BC_BEST
    write_atomic(
        ppo_warm_start_checkpoint(
            model, optimizer, bc_steps=step, wandb_run_id=wandb_run_id
        ),
        path,
    )
    _write_text_atomic(
        json.dumps(
            {
                "checkpoint": CHECKPOINT_BC_BEST,
                "sha256": _sha256(path),
                "bc_step": step,
                "validation_nll": nll,
                "dataset_manifest_sha256": dataset.manifest_sha256,
                "wandb_run_id": wandb_run_id,
                **provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        run_dir / CHECKPOINT_BC_BEST_RECORD,
    )


def _truncate_history(path: Path, *, last_step: int) -> None:
    """Drop history lines written after the resume state's evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"BC history missing for resume: {path}")
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["bc/step"] <= last_step
    ]
    _write_text_atomic("".join(f"{line}\n" for line in kept), path)
