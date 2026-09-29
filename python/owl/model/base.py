from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeAlias

import torch
from torch import nn
from typing_extensions import TypeVar

from owl.model.teacher_targets import TeacherTargets
from owl.rl import ActionBundle, ActionConfig, ObsBatch

InputLayer = nn.Module | nn.Parameter
ModelActions: TypeAlias = ActionBundle
ModelHiddenState: TypeAlias = object

# PEP 696 defaults: a bare ``BaseModelAPI`` / ``ModelOutput`` keeps meaning the
# Orbit types, so Isaiah's existing annotations are unchanged. Other games
# parameterize them, e.g. ``BaseModelAPI[KaggricultureObsBatch, ...]``.
ObsT = TypeVar("ObsT", default=ObsBatch)
ActT = TypeVar("ActT", default=ActionBundle)
ActSpecT = TypeVar("ActSpecT", default=ActionConfig)


@dataclass
class ModelActionLogProbs:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    target: torch.Tensor | None = None


@dataclass
class ModelActionEntropies:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    components: dict[str, torch.Tensor]
    target: torch.Tensor | None = None


@dataclass
class ModelActionKLDivergences:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    components: dict[str, torch.Tensor]
    target: torch.Tensor | None = None


@dataclass
class ModelOutput(Generic[ActT]):
    actions: ActT
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    next_hidden_state: ModelHiddenState | None = None


@dataclass
class ModelEvaluation:
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    winner_log_probabilities: torch.Tensor | None = None
    next_hidden_state: ModelHiddenState | None = None


@dataclass
class ModelTeacherEvaluation:
    student: ModelEvaluation
    action_kl: ModelActionKLDivergences | None
    teacher_winner_probabilities: torch.Tensor | None
    student_winner_log_probabilities: torch.Tensor | None


@dataclass
class ModelServingOutput(Generic[ActT]):
    actions: ActT
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    next_hidden_state: ModelHiddenState | None = None


class TrunkCompileAPI(ABC):
    """A model that supports ``rl.model_compile='trunk'``.

    ``compile_transformer_trunk`` compiles only the self-attention trunk
    (blocks plus final norm). Stems, token assembly, packing, heads and critic
    stay eager, and the model calls the compiled callable only from its own
    trunk dispatch, so any guard in that dispatch runs before it. Before
    ``torch.compile`` it claims the process-global GEMM backends for its game
    (``owl.model.compile_gemm.claim_gemm_backends``), because it is a public
    entry point that callers may use without ``configure_model_compile``.
    Returns the number of compiled callables.
    """

    @abstractmethod
    def compile_transformer_trunk(self, *, mode: str) -> int: ...


class BaseModelAPI(nn.Module, ABC, Generic[ObsT, ActT, ActSpecT]):
    @property
    def action_spec(self) -> ActSpecT:
        return self._action_spec

    @action_spec.setter
    def action_spec(self, value: ActSpecT) -> None:
        self._action_spec = value

    @abstractmethod
    def forward(
        self,
        obs: ObsT,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput[ActT]: ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs: ObsT,
        actions: ActT,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation: ...

    @abstractmethod
    def compute_value(
        self,
        obs: ObsT,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor: ...

    def evaluate_action_kl(
        self,
        obs: ObsT,
        teacher: BaseModelAPI[ObsT, ActT, ActSpecT],
        actions: ActT,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelActionKLDivergences:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement action KL evaluation"
        )

    def evaluate_actions_with_teacher(
        self,
        obs: ObsT,
        actions: ActT,
        teacher: BaseModelAPI[ObsT, ActT, ActSpecT],
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
        compute_teacher_action_kl: bool = True,
        compute_teacher_value: bool = True,
    ) -> ModelTeacherEvaluation:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement teacher evaluation"
        )

    def compute_teacher_distillation_targets(
        self,
        obs: ObsT,
        actions: ActT,
        *,
        compute_action_kl: bool = True,
        compute_value: bool = True,
    ) -> TeacherTargets:
        """Frozen-teacher targets for ``obs``/``actions`` (``TeacherTargets``).

        Implementations may return their concrete target type.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement teacher distillation targets"
        )

    def evaluate_actions_with_cached_teacher(
        self,
        obs: ObsT,
        actions: ActT,
        teacher_targets: TeacherTargets,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
        compute_teacher_action_kl: bool = True,
        compute_teacher_value: bool = True,
    ) -> ModelTeacherEvaluation:
        """Student evaluation plus the distillation terms against cached targets.

        Implementations narrow ``teacher_targets`` to the type their
        ``compute_teacher_distillation_targets`` returns with ``isinstance``
        and raise ``TypeError`` for any other type, before any kernel.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement cached teacher evaluation"
        )

    def supports_cached_teacher_distillation(self) -> bool:
        """Whether this model supports the cached teacher action-KL path.

        PPO precomputes teacher distillation targets once per iteration and
        consumes them per minibatch (``compute_teacher_distillation_targets`` /
        ``evaluate_actions_with_cached_teacher``). A model returns ``True`` only
        when it implements both for its action heads (for ``StatelessTransformerV1``
        the discrete_targets actor without player-count adapters). The trainer
        rejects active teachers (with ``teacher_kl_coef`` > 0) whose
        student/teacher models return ``False`` instead of failing mid-training.
        """
        return False

    def supports_cached_value_distillation(self) -> bool:
        """Whether this model supports the cached teacher value-distillation path.

        Value distillation only uses the critic's winner distribution, so it is
        actor-agnostic; this is a looser condition than
        ``supports_cached_teacher_distillation``.
        """
        return False

    def teacher_value_cross_entropy(
        self,
        student_winner_log_probabilities: torch.Tensor,
        teacher_winner_probabilities: torch.Tensor,
        *,
        value_mask: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Per-state teacher value cross-entropy, before PPO's state weighting.

        The default is Isaiah's joint winner distribution over player slots:
        one categorical per state whose inactive slots already have zero
        probability, so ``value_mask`` is unused. Games whose winner layout
        differs (e.g. one distribution per seat) override the reduction.
        """
        return (
            -teacher_winner_probabilities.detach() * student_winner_log_probabilities
        ).sum(dim=-1)

    def count_non_masked_tokens(self, obs: ObsT) -> torch.Tensor:
        """Return the number of unmasked model tokens represented by ``obs``.

        Models with learned or architecture-specific tokens should override this.
        The base implementation counts only unmasked observation entities.
        """
        if not isinstance(obs, ObsBatch):
            raise NotImplementedError(
                f"{type(self).__name__} must override count_non_masked_tokens "
                f"for {type(obs).__name__}"
            )
        return obs.entity_mask.sum(dtype=torch.int64)

    def serve(
        self,
        obs: ObsT,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelServingOutput[ActT]:
        output = self.forward(
            obs,
            deterministic=deterministic,
            hidden_state=hidden_state,
        )
        return ModelServingOutput[ActT](
            actions=output.actions,
            values=output.values,
            winner_probabilities=output.winner_probabilities,
            next_hidden_state=output.next_hidden_state,
        )

    @abstractmethod
    def reset_parameters(self) -> None: ...

    @abstractmethod
    def get_input_layers(self) -> tuple[InputLayer, ...]: ...

    @abstractmethod
    def get_output_layers(self) -> tuple[nn.Module, ...]: ...

    def initial_hidden_state(
        self,
        batch_size: int,  # noqa: ARG002
        *,
        device: torch.device,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return None

    def detach_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
    ) -> ModelHiddenState | None:
        return hidden_state

    def index_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        indices: torch.Tensor,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return hidden_state

    def reset_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        dones: torch.Tensor,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return hidden_state
