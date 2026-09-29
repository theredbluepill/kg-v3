"""DIAGNOSTIC-ONLY trunk audit hook for PPO collapse ablation D (telemetry only).

Pre-landing pod diagnostic (branch kg/pod-ppo-prelanding, pending Codex review).
This is NOT a training feature: it lives outside scripts/run_ppo.py and
python/owl, nothing in the repository imports it, and it changes no loss,
gradient, optimizer step or parameter. It only reads parameters and gradients.

It loads the shared early-smoke launcher (KG_BASE_LAUNCHER, byte-identical to
the one used by the 6.2 control and ablations A, B and C) with all of its
receipt instrumentation, W&B project/online check and nonfinite stop, and adds
two read-only hooks on owl.train.ppo.PPOTrainer through the same monkeypatch
seam as ablation C's hook:

1. KG_DIAG_TRUNK_AUDIT_ITERS="12,23,34,46" (required, comma-separated launcher
   iterations). The loaded weights theta0 are snapshotted at the start of
   iteration 1, before any optimizer step. After each listed iteration's
   train_iteration returns (so after that iteration's last optimizer step),
   emit a ``trunk_audit`` record with, per parameter group, the relative
   parameter change ||theta - theta0|| / ||theta0||, ||theta0|| and the
   parameter count. A ``trunk_audit_theta0`` record at the snapshot gives
   ||theta0|| per group (identity check for which critic head was loaded).
2. KG_DIAG_GRAD_AUDIT=first|off. Before the first optimizer step of every
   iteration, after backward and the DDP all-reduce, emit a ``grad_audit``
   record with per-group gradient L2 norm, max |grad|, None-grad counts and
   relative parameter change (the same record as ablation C's hook).

Groups (as in ablation C): actor-only (``actor_input_proj.*``, ``actor.*``;
reached only by policy terms), critic-only (``critic_head.*``,
``critic_value_tokens``) and the shared remainder (the trunk).

Every record is one "[kg-probe] {json}" line via the base launcher's emit().
"""

from __future__ import annotations

import importlib.util
import os
import sys
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
AUDIT_ITERS = frozenset(
    int(item) for item in os.environ["KG_DIAG_TRUNK_AUDIT_ITERS"].split(",") if item.strip()
)
if not AUDIT_ITERS or min(AUDIT_ITERS) < 1:
    raise ValueError("KG_DIAG_TRUNK_AUDIT_ITERS must list iterations >= 1")
GRAD_AUDIT = os.environ.get("KG_DIAG_GRAD_AUDIT", "first")
if GRAD_AUDIT not in ("first", "off"):
    raise ValueError(f"KG_DIAG_GRAD_AUDIT must be first|off, got {GRAD_AUDIT!r}")

_GROUPS = ("actor_only", "critic_only", "shared")
_WRAPPER_PARTS = ("_ddp", "module", "_orig_mod", "model")
_THETA0: dict[str, torch.Tensor] = {}
_AUDIT = {"last_grad_iteration": 0}


def _iteration() -> int:
    return int(BASE._ITER["n"])


def _group(name: str) -> str:
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
def _snapshot(trainer: Any) -> None:
    named = list(trainer.model.named_parameters())
    norms = {g: {"params": 0, "theta0_sq": 0.0, "critic_head_sq": 0.0} for g in _GROUPS}
    for name, param in named:
        theta0 = param.detach().float().clone()
        _THETA0[name] = theta0
        g = norms[_group(name)]
        g["params"] += param.numel()
        g["theta0_sq"] += float(theta0.pow(2).sum())
        if "critic_head." in name:
            g["critic_head_sq"] += float(theta0.pow(2).sum())
    for required in _GROUPS:
        if norms[required]["params"] == 0:
            raise RuntimeError(
                f"trunk audit found no {required} parameters; first names "
                f"{[name for name, _ in named[:8]]}"
            )
    BASE.emit(
        "trunk_audit_theta0",
        iteration=_iteration(),
        optimizer_steps=int(trainer.optimizer_steps),
        groups={
            g: {
                "params": int(v["params"]),
                "theta0_norm": v["theta0_sq"] ** 0.5,
                "critic_head_norm": v["critic_head_sq"] ** 0.5,
            }
            for g, v in norms.items()
        },
    )


@torch.no_grad()
def _param_change(trainer: Any, with_grads: bool) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, param in trainer.model.named_parameters():
        g = stats.setdefault(
            _group(name),
            {"params": 0, "none_grads": 0, "grad_sq": 0.0, "grad_max_abs": 0.0,
             "delta_sq": 0.0, "theta0_sq": 0.0},
        )
        g["params"] += param.numel()
        theta = param.detach().float()
        g["delta_sq"] += float((theta - _THETA0[name]).pow(2).sum())
        g["theta0_sq"] += float(_THETA0[name].pow(2).sum())
        if not with_grads:
            continue
        if param.grad is None:
            g["none_grads"] += 1
            continue
        grad = param.grad.detach().float()
        g["grad_sq"] += float(grad.pow(2).sum())
        g["grad_max_abs"] = max(g["grad_max_abs"], float(grad.abs().max()))
    out: dict[str, dict[str, float]] = {}
    for name, g in stats.items():
        row: dict[str, float] = {
            "params": int(g["params"]),
            "param_rel_change": (g["delta_sq"] / max(g["theta0_sq"], 1e-30)) ** 0.5,
            "param_delta_norm": g["delta_sq"] ** 0.5,
        }
        if with_grads:
            row.update(
                none_grads=int(g["none_grads"]),
                grad_norm=g["grad_sq"] ** 0.5,
                grad_max_abs=g["grad_max_abs"],
            )
        out[name] = row
    return out


def _install() -> None:
    from owl.train import ppo

    trainer_cls = ppo.PPOTrainer
    original_iteration = trainer_cls.train_iteration
    original_step = trainer_cls._step_optimizer

    # BASE.main() wraps train_iteration again after this, so BASE's wrapper
    # increments _ITER before this one runs and emits the iteration record after.
    def train_iteration(self: Any) -> Any:
        if not _THETA0:
            if int(self.optimizer_steps) != 0:
                raise RuntimeError("trunk audit snapshot must precede the first optimizer step")
            _snapshot(self)
        metrics = original_iteration(self)
        if _iteration() in AUDIT_ITERS:
            BASE.emit(
                "trunk_audit",
                iteration=_iteration(),
                optimizer_steps=int(self.optimizer_steps),
                groups=_param_change(self, with_grads=False),
            )
        return metrics

    def step_optimizer(self: Any) -> Any:
        if GRAD_AUDIT == "first" and _AUDIT["last_grad_iteration"] != _iteration():
            _AUDIT["last_grad_iteration"] = _iteration()
            BASE.emit(
                "grad_audit",
                iteration=_iteration(),
                optimizer_steps_before=int(self.optimizer_steps),
                groups=_param_change(self, with_grads=True),
            )
        return original_step(self)

    trainer_cls.train_iteration = train_iteration
    trainer_cls._step_optimizer = step_optimizer
    BASE.emit(
        "diagnostic_hook",
        hook="trunk_audit",
        diagnostic_only=True,
        telemetry_only=True,
        trunk_audit_iterations=sorted(AUDIT_ITERS),
        grad_audit=GRAD_AUDIT,
        base_launcher=str(Path(os.environ["KG_BASE_LAUNCHER"]).resolve()),
    )


if __name__ == "__main__":
    _install()
    BASE.main()
