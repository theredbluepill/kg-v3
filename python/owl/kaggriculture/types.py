"""Game tensors; leading batch dimensions may include rollout time.

Each feature row is a separate player's legal perspective. The actor mask counts
the 241 observable own actors, while the action mask spans 252 possible command
frames (241 actors, ten market orders, one STOP). No private state may cross the
perspective axis in a shared-attention operation.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, Field

from owl.config import BaseConfig

PLAYERS = 2
MAX_ACTORS = 241
MAX_FRAMES = 252
ACTION_SLOTS = 12
SLOT_WIDTHS = (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)


class KaggricultureObsConfig(BaseConfig):
    obs_spec: Literal["kaggriculture"] = "kaggriculture"
    observation_version: Literal[1, 2] = 2

    @property
    def feature_count(self) -> int:
        return 8176 if self.observation_version == 2 else 8165


class KaggricultureActionConfig(BaseConfig):
    action_spec: Literal["kaggriculture"] = "kaggriculture"
    max_frames: Literal[252] = 252
    hire_limit: int = Field(default=241, ge=1, le=241)


@dataclass(frozen=True)
class KaggricultureActionMask:
    # Potential command positions, not vocabulary/feasibility masks. The decoder's
    # native grammar supplies prefix-dependent vocabulary masks during sampling.
    can_act: torch.Tensor


class KaggricultureObsBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    features: torch.Tensor
    context: torch.Tensor
    entity_mask: torch.Tensor
    still_playing: torch.Tensor
    action_mask: KaggricultureActionMask


@dataclass
class KaggricultureActions:
    tokens: torch.Tensor
    lengths: torch.Tensor
