# ruff: noqa: E402
"""Kaggle entrypoint for the Kaggriculture agent (``main.py`` in the tarball).

Kaggle ``exec``s this file lazily inside the first timed call, so every import,
the weight load and the warm-up bill to turn 0's overage bank. All bundled
``owl`` modules are imported here at top level: the agent directory is on
``sys.path`` only during that ``exec``. ``agent`` must stay the last callable
defined in this module (Kaggle's loader takes the last callable).

Two optional stateless endgame rules are read from the environment; for each,
unset or ``0`` keeps it off, ``1`` turns it on and any other value raises:

* ``KAGGRICULTURE_FINAL_TURN_LIQUIDATION`` (rule 1, ``owl.kaggriculture.final_turn``);
* ``KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS`` (rule 2,
  ``owl.kaggriculture.late_invest``).

Kaggle sets no such variable, so a shipped package runs with both rules off.
"""

import os

THREADS = 1
DETERMINISTIC = True
MIN_OVERAGE_TIME = 2.0
os.environ.setdefault("OMP_NUM_THREADS", f"{THREADS}")
os.environ.setdefault("MKL_NUM_THREADS", f"{THREADS}")

import torch

torch.set_num_threads(THREADS)
# A second set_num_interop_threads call in one process raises; a local
# harness may load two copies of this file into one interpreter.
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

from typing import Any

from owl import OWL_ROOT
from owl.kaggriculture import final_turn, late_invest
from owl.kaggriculture.kaggle_agent import KaggricultureAgent
from owl.rs import assert_release_build

# Tests set this to load the unoptimized development extension; Kaggle never does.
if os.environ.get("KAGGRICULTURE_AGENT_ALLOW_DEBUG_BUILD") != "1":
    assert_release_build()

AGENT = KaggricultureAgent(
    OWL_ROOT.parent / "models" / "primary",
    deterministic=DETERMINISTIC,
    strict=os.environ.get("KAGGRICULTURE_AGENT_STRICT") == "1",
    min_overage_time=MIN_OVERAGE_TIME,
    final_turn_liquidation=final_turn.enabled_from_env(),
    block_late_investments=late_invest.enabled_from_env(),
)
AGENT.warm_up()


def agent(obs: Any, config: Any) -> Any:
    return AGENT.act(obs, config)
