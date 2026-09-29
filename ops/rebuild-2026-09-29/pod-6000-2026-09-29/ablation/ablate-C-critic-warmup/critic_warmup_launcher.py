"""DIAGNOSTIC-ONLY critic warm-up hook for PPO collapse ablation C.

Pre-landing pod diagnostic (branch kg/pod-ppo-prelanding, pending Codex review).
This is NOT a training feature: it lives outside scripts/run_ppo.py and
python/owl, and nothing in the repository imports it.

It loads the shared early-smoke launcher (KG_BASE_LAUNCHER, byte-identical to
the one used by the 6.2 control and ablations A and B) with all of its receipt
instrumentation, W&B project/online check and nonfinite stop, and then adds
two hooks on owl.train.ppo.PPOTrainer through the same monkeypatch seam:

1. KG_DIAG_CRITIC_WARMUP_ITERS=N (required, N >= 0). For launcher iterations
   1..N every call of the trainer's PPO loss (``self._ppo_loss``) receives
   - ``advantages`` replaced by zeros, so the clipped policy-gradient term is
     exactly 0 and has zero gradient;
   - ``teacher_kl_coef`` = 0 (teacher action-KL anchor off);
   - ``config.ent_coef`` = 0 (entropy bonus off; control 1e-6).
   The value loss (vf_coef * value loss) and the teacher value distillation
   term are unchanged, so only the critic side trains. Entropy, approx KL,
   clip fraction and teacher KL metrics are still computed from the real
   tensors (loss/policy_loss, loss/teacher_kl_loss and loss/entropy_loss log
   0 during warm-up). The shared trunk still moves under the value loss, and
   Muon's decoupled weight decay (0.01 * lr) still applies to 2-D actor
   matrices; the audit below measures both. From iteration N+1 the loss call
   is passed through untouched (the control's PPO).
2. KG_DIAG_GRAD_AUDIT=first|all|off. Before the first (``first``) or every
   (``all``) optimizer step of an iteration, after backward and the DDP
   all-reduce, emit a ``grad_audit`` record with per-group gradient L2 norm,
   max |grad| and None-grad counts for actor-only parameters
   (``actor_input_proj.*``, ``actor.*``: reached only by policy terms),
   critic-only parameters (``critic_head.*``, ``critic_value_tokens``) and the
   shared remainder, plus each group's relative parameter change from the
   loaded weights (||theta - theta0|| / ||theta0||).

Every record is one "[kg-probe] {json}" line via the base launcher's emit().
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def _load_base_launcher() -> Any:
    path = Path(os.environ["KG_BASE_LAUNCHER"]).resolve()
    spec = importlib.util.spec_from_file_location("launch_kg_run_ppo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_kg_run_ppo"] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_launcher()
WARMUP_ITERS = int(os.environ["KG_DIAG_CRITIC_WARMUP_ITERS"])
if WARMUP_ITERS < 0:
    raise ValueError("KG_DIAG_CRITIC_WARMUP_ITERS must be >= 0")
GRAD_AUDIT = os.environ.get("KG_DIAG_GRAD_AUDIT", "first")
if GRAD_AUDIT not in ("first", "all", "off"):
    raise ValueError(f"KG_DIAG_GRAD_AUDIT must be first|all|off, got {GRAD_AUDIT!r}")

_CALLS = {"masked": 0, "passthrough": 0}
_AUDIT = {"last_iteration": 0}
_THETA0: dict[str, torch.Tensor] = {}


def _iteration() -> int:
    return int(BASE._ITER["n"])


def _in_warmup() -> bool:
    return 1 <= _iteration() <= WARMUP_ITERS


def _warmup_loss(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("critic warm-up hook expects keyword-only loss calls")
        if not _in_warmup():
            _CALLS["passthrough"] += 1
            return fn(**kwargs)
        for key in ("advantages", "teacher_kl_coef", "config"):
            if key not in kwargs:
                raise RuntimeError(f"critic warm-up hook: loss call lacks {key!r}")
        kwargs["advantages"] = torch.zeros_like(kwargs["advantages"])
        kwargs["teacher_kl_coef"] = 0.0
        kwargs["config"] = kwargs["config"].model_copy(update={"ent_coef": 0.0})
        _CALLS["masked"] += 1
        return fn(**kwargs)

    return wrapped


_WRAPPER_PARTS = ("_ddp", "module", "_orig_mod", "model")


def _group(name: str) -> str:
    # The trainer's model is wrapped (seen on the pod: "_ddp.module.model.<name>");
    # drop leading wrapper components. No KaggricultureTransformer top-level
    # submodule is named like a wrapper part.
    parts = name.split(".")
    while parts and parts[0] in _WRAPPER_PARTS:
        parts.pop(0)
    name = ".".join(parts)
    if name.startswith(("actor.", "actor_input_proj.")):
        return "actor_only"
    if name.startswith("critic_head.") or name == "critic_value_tokens":
        return "critic_only"
    return "shared"


@torch.no_grad()
def _audit(trainer: Any) -> None:
    named = list(trainer.model.named_parameters())
    if not _THETA0:
        for name, param in named:
            _THETA0[name] = param.detach().float().clone()
    stats: dict[str, dict[str, float]] = {}
    for name, param in named:
        group = stats.setdefault(
            _group(name),
            {"params": 0, "none_grads": 0, "grad_sq": 0.0, "grad_max_abs": 0.0,
             "delta_sq": 0.0, "theta0_sq": 0.0},
        )
        group["params"] += param.numel()
        theta = param.detach().float()
        group["delta_sq"] += float((theta - _THETA0[name]).pow(2).sum())
        group["theta0_sq"] += float(_THETA0[name].pow(2).sum())
        if param.grad is None:
            group["none_grads"] += 1
            continue
        grad = param.grad.detach().float()
        group["grad_sq"] += float(grad.pow(2).sum())
        group["grad_max_abs"] = max(group["grad_max_abs"], float(grad.abs().max()))
    for required in ("actor_only", "critic_only", "shared"):
        if required not in stats:
            sample = [name for name, _ in named[:8]]
            raise RuntimeError(
                f"grad audit found no {required} parameters; first names {sample}"
            )
    record = {
        name: {
            "params": int(g["params"]),
            "none_grads": int(g["none_grads"]),
            "grad_norm": g["grad_sq"] ** 0.5,
            "grad_max_abs": g["grad_max_abs"],
            "param_rel_change": (g["delta_sq"] / max(g["theta0_sq"], 1e-30)) ** 0.5,
        }
        for name, g in stats.items()
    }
    BASE.emit(
        "grad_audit",
        iteration=_iteration(),
        optimizer_steps_before=int(trainer.optimizer_steps),
        warmup=_in_warmup(),
        loss_calls=dict(_CALLS),
        groups=record,
    )


def _install() -> None:
    from owl.train import ppo

    trainer_cls = ppo.PPOTrainer
    original_init = trainer_cls.__init__
    original_step = trainer_cls._step_optimizer

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._ppo_loss = _warmup_loss(self._ppo_loss)

    def step_optimizer(self: Any) -> Any:
        if GRAD_AUDIT == "all" or (
            GRAD_AUDIT == "first" and _AUDIT["last_iteration"] != _iteration()
        ):
            _AUDIT["last_iteration"] = _iteration()
            _audit(self)
        return original_step(self)

    trainer_cls.__init__ = init
    trainer_cls._step_optimizer = step_optimizer
    BASE.emit(
        "diagnostic_hook",
        hook="critic_warmup",
        diagnostic_only=True,
        warmup_iterations=WARMUP_ITERS,
        grad_audit=GRAD_AUDIT,
        base_launcher=str(Path(os.environ["KG_BASE_LAUNCHER"]).resolve()),
        masked_terms=["policy_gradient(advantages=0)", "teacher_kl_coef=0", "ent_coef=0"],
    )


if __name__ == "__main__":
    _install()
    BASE.main()
