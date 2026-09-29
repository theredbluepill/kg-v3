"""DIAGNOSTIC-ONLY critic stop-gradient hook for PPO collapse ablation G.

Pre-landing pod diagnostic (branch kg/pod-ppo-prelanding, pending Codex review).
This is NOT a training feature and NOT a recipe: it lives outside
scripts/run_ppo.py and python/owl, nothing in the repository imports it, and
it exists only to attribute the round-1 economy collapse. It CHANGES TRAINING
(unlike ablation D's telemetry-only hook): the critic's gradient no longer
reaches the shared trunk.

It loads ablation D's unchanged telemetry-only trunk audit hook
(KG_TRUNK_AUDIT_HOOK, which itself loads the unchanged shared early-smoke
launcher KG_BASE_LAUNCHER with its receipts, W&B online check and nonfinite
stop) and adds, through the same monkeypatch seam:

1. KG_DIAG_CRITIC_STOPGRAD=on|off (required). ``on`` replaces
   ``KaggricultureTransformer._winner_log_probabilities`` (the only function
   that turns encoder output into critic output: rollout values, the PPO
   value loss and the student side of the teacher value distillation all go
   through it) by a copy that applies ``critic_head`` to
   ``encoded.critic_value_hidden.detach()``. Forward values are bit-identical;
   in backward, every critic-output loss (vf_coef * value loss and
   teacher_value_coef * teacher value CE) reaches ``critic_head.*`` only. The
   trunk blocks, stems, final norm and the ``critic_value_tokens`` input
   embeddings (trunk inputs upstream of the detach) receive no gradient from
   the critic; the policy terms still reach all of them, including
   ``critic_value_tokens`` through attention. ``off`` leaves the model
   untouched (control for the check below).
2. KG_DIAG_STOPGRAD_CHECK_ITERS="1,2" (optional, may be empty). On the first
   PPO-loss call of each listed launcher iteration, before the real loss, the
   eager module-level ``owl.train.ppo._ppo_loss`` is evaluated on the same
   inputs with the policy-side tensors (``new_logp``, ``entropy``,
   ``teacher_kl``) detached and ``context=None`` (local reduction, no
   collective), so its loss is exactly the critic side: vf_coef * value loss
   + teacher_value_coef * teacher value CE. ``torch.autograd.grad`` of that
   loss w.r.t. every model parameter (retain_graph, allow_unused; ``.grad`` is
   not touched, so training is unaffected) gives per-group gradient norms for
   actor-only, critic_head, critic_value_tokens and shared-trunk parameters.
   With ``on`` the check REQUIRES exactly zero gradient (None or all-zero) on
   shared, actor-only and critic_value_tokens parameters and a non-zero
   gradient on critic_head, and raises otherwise.
   KG_DIAG_DISABLE_DONATED_BUFFER=1 (control dry run only) sets
   torch._functorch.config.donated_buffer=False: the "off" check backpropagates
   through the compiled trunk with retain_graph, which donated buffers forbid.
   It changes compiled-backward buffer reuse, not math; the "on" check never
   reaches the compiled trunk, so the main run leaves it unset.
3. At the trunk audit iterations, a ``critic_split_audit`` record splits D's
   critic-only group into critic_head and critic_value_tokens relative change.

Every record is one "[kg-probe] {json}" line via the base launcher's emit().
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def _load_audit_hook() -> Any:
    path = Path(os.environ["KG_TRUNK_AUDIT_HOOK"]).resolve()
    spec = importlib.util.spec_from_file_location("trunk_audit_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trunk audit hook {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trunk_audit_launcher"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_hook()
BASE = AUDIT.BASE
STOPGRAD = os.environ["KG_DIAG_CRITIC_STOPGRAD"]
if STOPGRAD not in ("on", "off"):
    raise ValueError(f"KG_DIAG_CRITIC_STOPGRAD must be on|off, got {STOPGRAD!r}")
CHECK_ITERS = frozenset(
    int(item)
    for item in os.environ.get("KG_DIAG_STOPGRAD_CHECK_ITERS", "").split(",")
    if item.strip()
)
DISABLE_DONATED_BUFFER = os.environ.get("KG_DIAG_DISABLE_DONATED_BUFFER", "0") == "1"
if DISABLE_DONATED_BUFFER:
    import torch._functorch.config as functorch_config

    functorch_config.donated_buffer = False
_CHECKED: set[int] = set()
_CALLS = {"detached_critic_calls": 0}
_CHECK_GROUPS = ("actor_only", "critic_head", "critic_value_tokens", "shared")
_WRAPPER_PARTS = ("_ddp", "module", "_orig_mod", "model")


def _iteration() -> int:
    return int(BASE._ITER["n"])


def _bare(name: str) -> str:
    parts = name.split(".")
    while parts and parts[0] in _WRAPPER_PARTS:
        parts.pop(0)
    return ".".join(parts)


def _check_group(name: str) -> str:
    bare = _bare(name)
    if bare.startswith(("actor.", "actor_input_proj.")):
        return "actor_only"
    if bare.startswith("critic_head."):
        return "critic_head"
    if bare == "critic_value_tokens":
        return "critic_value_tokens"
    return "shared"


def _install_stopgrad() -> None:
    from owl.model import kaggriculture

    cls = kaggriculture.KaggricultureTransformer
    original = cls._winner_log_probabilities

    def winner_log_probabilities(self: Any, encoded: Any) -> torch.Tensor:
        _CALLS["detached_critic_calls"] += 1
        detached = dataclasses.replace(
            encoded, critic_value_hidden=encoded.critic_value_hidden.detach()
        )
        return original(self, detached)

    cls._winner_log_probabilities = winner_log_probabilities


def _grad_check(trainer: Any, kwargs: dict[str, Any]) -> None:
    from owl.train import ppo

    check = dict(kwargs)
    detached = []
    for key in ("new_logp", "entropy", "teacher_kl"):
        value = check.get(key)
        if isinstance(value, torch.Tensor):
            check[key] = value.detach()
            detached.append(key)
    check["context"] = None
    check["loss_components"] = ppo._ppo_loss_components
    metrics, critic_loss = ppo._ppo_loss(**check)
    if not critic_loss.requires_grad:
        raise RuntimeError("stop-grad check: critic-side loss has no autograd graph")
    named = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        critic_loss, [p for _, p in named], retain_graph=True, allow_unused=True
    )
    stats = {
        g: {"tensors": 0, "params": 0, "none_grads": 0, "all_zero_grads": 0,
            "grad_sq": 0.0, "grad_max_abs": 0.0}
        for g in _CHECK_GROUPS
    }
    for (name, param), grad in zip(named, grads, strict=True):
        s = stats[_check_group(name)]
        s["tensors"] += 1
        s["params"] += param.numel()
        if grad is None:
            s["none_grads"] += 1
            continue
        g = grad.detach().float()
        if not bool(torch.any(g != 0)):
            s["all_zero_grads"] += 1
        s["grad_sq"] += float(g.pow(2).sum())
        s["grad_max_abs"] = max(s["grad_max_abs"], float(g.abs().max()))
    groups = {
        g: {**{k: v for k, v in s.items() if k != "grad_sq"},
            "grad_norm": s["grad_sq"] ** 0.5}
        for g, s in stats.items()
    }
    for required in _CHECK_GROUPS:
        if groups[required]["tensors"] == 0:
            raise RuntimeError(f"stop-grad check found no {required} parameters")
    zero_expected = ("shared", "actor_only", "critic_value_tokens")
    exact_zero = {g: groups[g]["grad_max_abs"] == 0.0 for g in _CHECK_GROUPS}
    passed = (
        all(exact_zero[g] for g in zero_expected) and groups["critic_head"]["grad_norm"] > 0.0
        if STOPGRAD == "on"
        else groups["shared"]["grad_norm"] > 0.0 and groups["critic_head"]["grad_norm"] > 0.0
    )
    BASE.emit(
        "stopgrad_check",
        iteration=_iteration(),
        optimizer_steps=int(trainer.optimizer_steps),
        stopgrad=STOPGRAD,
        detached_inputs=detached,
        critic_side_loss=float(critic_loss.detach()),
        value_loss=float(metrics.value_loss.detach()),
        teacher_value_loss=float(metrics.teacher_value_loss.detach()),
        policy_loss_has_grad=bool(metrics.policy_loss.requires_grad),
        exact_zero=exact_zero,
        passed=passed,
        expectation=(
            "on: zero grad on shared/actor_only/critic_value_tokens, nonzero on critic_head"
            if STOPGRAD == "on"
            else "off (control): nonzero grad on shared and critic_head"
        ),
        groups=groups,
    )
    if not passed:
        raise RuntimeError(f"stop-grad check failed (stopgrad={STOPGRAD}): {groups}")


def _checked_loss(trainer: Any, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("stop-grad hook expects keyword-only loss calls")
        iteration = _iteration()
        if iteration in CHECK_ITERS and iteration not in _CHECKED:
            _CHECKED.add(iteration)
            _grad_check(trainer, kwargs)
        return fn(**kwargs)

    return wrapped


@torch.no_grad()
def _critic_split(trainer: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, param in trainer.model.named_parameters():
        group = _check_group(name)
        if group not in ("critic_head", "critic_value_tokens"):
            continue
        s = out.setdefault(group, {"params": 0, "delta_sq": 0.0, "theta0_sq": 0.0})
        theta0 = AUDIT._THETA0[name]
        s["params"] += param.numel()
        s["delta_sq"] += float((param.detach().float() - theta0).pow(2).sum())
        s["theta0_sq"] += float(theta0.pow(2).sum())
    return {
        g: {"params": int(s["params"]),
            "param_rel_change": (s["delta_sq"] / max(s["theta0_sq"], 1e-30)) ** 0.5,
            "theta0_norm": s["theta0_sq"] ** 0.5}
        for g, s in out.items()
    }


def _install() -> None:
    from owl.train import ppo

    if STOPGRAD == "on":
        _install_stopgrad()
    trainer_cls = ppo.PPOTrainer
    original_init = trainer_cls.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._ppo_loss = _checked_loss(self, self._ppo_loss)

    trainer_cls.__init__ = init
    # D's audit wraps train_iteration first (inner); this wraps it (middle);
    # BASE.main() wraps it last (outer, increments the iteration counter).
    AUDIT._install()
    audited = trainer_cls.train_iteration

    def train_iteration(self: Any) -> Any:
        metrics = audited(self)
        if _iteration() in AUDIT.AUDIT_ITERS:
            BASE.emit(
                "critic_split_audit",
                iteration=_iteration(),
                detached_critic_calls=_CALLS["detached_critic_calls"],
                groups=_critic_split(self),
            )
        return metrics

    trainer_cls.train_iteration = train_iteration
    BASE.emit(
        "diagnostic_hook",
        hook="critic_stopgrad",
        diagnostic_only=True,
        telemetry_only=False,
        changes_training=STOPGRAD == "on",
        critic_stopgrad=STOPGRAD,
        detach_point="KaggricultureTransformer._winner_log_probabilities: "
        "critic_head(encoded.critic_value_hidden.detach())",
        stopgrad_check_iterations=sorted(CHECK_ITERS),
        donated_buffer_disabled=DISABLE_DONATED_BUFFER,
        trunk_audit_hook=str(Path(os.environ["KG_TRUNK_AUDIT_HOOK"]).resolve()),
        base_launcher=str(Path(os.environ["KG_BASE_LAUNCHER"]).resolve()),
    )


if __name__ == "__main__":
    _install()
    BASE.main()
