"""Task 7.4: the packaged Kaggle agent, its error boundary and its entrypoint."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from kaggle_environments import make
from owl.kaggriculture.kaggle_agent import (
    ActionValidationError,
    KaggricultureAgent,
    pass_action,
    validate_action,
)
from owl.kaggriculture.kaggle_view import SeatArrays
from owl.kaggriculture.types import ACTION_SLOTS, MAX_FRAMES

from tests.kaggriculture.kaggle_fixtures import (
    TRAINING_CONFIG,
    tiny_state,
    write_agent_dir,
    write_model_root,
)


def _agent(root: Path, *, strict: bool = True, **kwargs: Any) -> KaggricultureAgent:
    return KaggricultureAgent(
        root,
        deterministic=kwargs.pop("deterministic", True),
        strict=strict,
        min_overage_time=kwargs.pop("min_overage_time", 2.0),
    )


def _observations(count: int, seed: int = 4) -> list[tuple[dict[str, Any], Any]]:
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.reset()
    result = []
    for _ in range(count):
        observation = json.loads(json.dumps(env.state[0].observation))
        result.append((observation, dict(env.configuration)))
        env.step([{"farmer": ["WEST"], "hands": [], "market": [["HIRE"]]}] * 2)
    return result


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    return write_model_root(tmp_path / "primary")


def test_agent_loads_strictly_and_overrides_only_flash_attention(
    model_root: Path,
) -> None:
    agent = _agent(model_root)
    assert agent.model.config.force_flash_attn is False
    expected = dict(TRAINING_CONFIG["model"]) | {"force_flash_attn": False}
    assert agent.model.config.model_dump() == expected
    assert not agent.model.training
    for name, tensor in agent.model.state_dict().items():
        assert torch.equal(tensor, tiny_state()[name]), name


def test_hire_limit_comes_from_the_packaged_action_spec(tmp_path: Path) -> None:
    config = copy.deepcopy(TRAINING_CONFIG)
    config["env"]["action_spec"]["hire_limit"] = 5
    agent = _agent(write_model_root(tmp_path / "m", config=config))
    assert agent.hire_limit == 5 == agent.model.action_spec.hire_limit
    del config["env"]["action_spec"]
    with pytest.raises(ValueError, match=r"env\.action_spec"):
        _agent(write_model_root(tmp_path / "n", config=config))


@pytest.mark.parametrize(
    ("checkpoint", "match"),
    [
        ({"model": tiny_state(), "optimizer": {}}, "only 'model'"),
        ({"model": tiny_state() | {"hidden_state": torch.zeros(1)}}, "prohibited"),
        ({"model": tiny_state() | {"opponent_embed": torch.zeros(1)}}, "prohibited"),
        (
            {"model": {k: v.to(torch.bfloat16) for k, v in tiny_state().items()}},
            "fp32",
        ),
        ({"model": tiny_state() | {"extra.weight": torch.zeros(1)}}, "Unexpected"),
    ],
)
def test_checkpoint_admission_rejects_non_policy_state(
    tmp_path: Path, checkpoint: dict[str, Any], match: str
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        _agent(write_model_root(tmp_path / "m", checkpoint=checkpoint))


def test_act_returns_a_validated_action_and_drops_the_loader_path(
    model_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _agent(model_root)
    for observation, configuration in _observations(4):
        configuration["__raw_path__"] = "/kaggle_simulations/agent/main.py"
        action = agent.act(observation, configuration)
        seat = observation["player"]
        validate_action(
            action,
            actors=len(observation["farms"][seat]["hands"]) + 1,
            order_limit=configuration["maxMarketOrdersPerTurn"],
            hire_limit=241,
            scratch=np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64),
        )
    assert (agent.calls, agent.caught_errors, agent.budget_passes) == (4, 0, 0)
    assert (
        capsys.readouterr().out.count("dropped configuration key '__raw_path__'") == 1
    )


def test_actions_depend_only_on_the_current_observation(model_root: Path) -> None:
    agent = _agent(model_root)
    observations = _observations(3)
    first = [agent.act(obs, config) for obs, config in observations]
    # Poison every reusable buffer, replay out of order: identical actions.
    for name in SeatArrays.__dataclass_fields__:
        array = getattr(agent.arrays, name)
        array[...] = np.ones_like(array)
    again = [agent.act(obs, config) for obs, config in reversed(observations)]
    assert again[::-1] == first
    fresh = _agent(model_root)
    assert fresh.act(*observations[2]) == first[2]


def test_error_boundary_returns_pass_or_reraises_when_strict(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation, configuration = _observations(1)[0]

    def broken(*_: Any) -> Any:
        raise RuntimeError("injected")

    guarded = _agent(model_root, strict=False)
    monkeypatch.setattr(guarded, "policy_action", broken)
    assert guarded.act(observation, configuration) == pass_action()
    assert guarded.caught_errors == 1
    strict = _agent(model_root, strict=True)
    monkeypatch.setattr(strict, "policy_action", broken)
    with pytest.raises(RuntimeError, match="injected"):
        strict.act(observation, configuration)


def test_invalid_decoded_actions_are_caught_by_the_validator(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation, configuration = _observations(1)[0]
    for strict in (False, True):
        agent = _agent(model_root, strict=strict)
        monkeypatch.setattr(
            "owl.kaggriculture.kaggle_agent.decode_action",
            lambda *_, **__: {"farmer": "PASS", "hands": [], "market": []},
        )
        if strict:
            with pytest.raises(ActionValidationError):
                agent.act(observation, configuration)
        else:
            assert agent.act(observation, configuration) == pass_action()
            assert agent.caught_errors == 1
        monkeypatch.undo()


def test_low_overage_bank_returns_pass_without_the_model(model_root: Path) -> None:
    agent = _agent(model_root, min_overage_time=2.0)
    observation, configuration = _observations(1)[0]
    observation["remainingOverageTime"] = 1.5
    assert agent.act(observation, configuration) == pass_action()
    assert (agent.budget_passes, agent.caught_errors) == (1, 0)


def _validate(action: object, *, actors: int = 3, order_limit: int = 10) -> None:
    validate_action(
        action,
        actors=actors,
        order_limit=order_limit,
        hire_limit=241,
        scratch=np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("action", "context"),
    [
        (pass_action(), {"actors": 1}),
        (
            {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []},
            {"actors": 3},
        ),
        (
            {
                "farmer": ["PICKUP", "WHEAT", 3],
                "hands": [["NORTH"], ["PLANT", "CARROT"]],
                "market": [
                    ["BUY_SEED", "WHEAT", 2],
                    ["SELL", "MILK", 1],
                    ["HIRE"],
                    ["BUY_LAND"],
                ],
            },
            {"actors": 3},
        ),
    ],
)
def test_validator_accepts_canonical_programs(
    action: dict[str, Any], context: dict[str, int]
) -> None:
    _validate(action, **context)


@pytest.mark.parametrize(
    ("action", "context"),
    [
        (None, {}),
        (["PASS"], {}),
        ({"farmer": ["PASS"], "hands": []}, {}),
        ({"farmer": ["PASS"], "hands": ["NORTH"], "market": []}, {}),
        ({"farmer": ["PICKUP", "WHEAT", "3"], "hands": [], "market": []}, {}),
        (
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 3},
            {"order_limit": 2},
        ),
        ({"farmer": ["PASS"], "hands": [["NORTH"]], "market": []}, {"actors": 1}),
        ({"farmer": [], "hands": [], "market": []}, {}),
        # The schema default omits hand commands; canonical programs list each hand.
        (pass_action(), {"actors": 3}),
    ],
)
def test_validator_rejects_malformed_or_out_of_context_programs(
    action: object, context: dict[str, int]
) -> None:
    with pytest.raises(ActionValidationError):
        _validate(action, **context)


_LOADER = textwrap.dedent(
    """
    import json, sys
    agent_dir = sys.argv[1]
    sys.path.insert(0, agent_dir)
    from kaggle_environments import make
    from kaggle_environments.agent import get_last_callable
    main = agent_dir + "/main.py"
    agent = get_last_callable(open(main).read(), path=main)
    import owl, owl.rs
    env = make("kaggriculture", configuration={"seed": 9, "episodeSteps": 12},
               debug=False)
    env.run([main, "starter"])
    print(json.dumps({
        "name": agent.__name__,
        "owl": owl.__file__,
        "rs": owl.rs.__file__,
        "statuses": [[s["status"] for s in step] for step in env.steps],
        "logs": [log[0]["stdout"] for log in env.logs if log],
    }))
    """
)


def test_packaged_main_loads_through_kaggles_loader_and_plays(tmp_path: Path) -> None:
    agent_dir = write_agent_dir(tmp_path / "agent")
    environment = os.environ | {
        "KAGGRICULTURE_AGENT_STRICT": "1",
        "KAGGRICULTURE_AGENT_ALLOW_DEBUG_BUILD": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", _LOADER, str(agent_dir)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["name"] == "agent"
    for key in ("owl", "rs"):
        assert Path(result[key]).resolve().is_relative_to(agent_dir.resolve())
    assert len(result["statuses"]) == 12
    assert all(
        status in ("ACTIVE", "DONE") for step in result["statuses"] for status in step
    )
    assert result["statuses"][-1] == ["DONE", "DONE"]
    assert sum("kg step=" in log for log in result["logs"]) == 11


def test_packaged_main_import_failure_propagates(tmp_path: Path) -> None:
    agent_dir = write_agent_dir(tmp_path / "agent")
    (agent_dir / "models" / "primary" / "checkpoint.pt").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
            "runpy.run_path(sys.argv[1] + '/main.py')",
            str(agent_dir),
        ],
        cwd=tmp_path,
        env=os.environ | {"KAGGRICULTURE_AGENT_ALLOW_DEBUG_BUILD": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode != 0
    assert "checkpoint.pt" in completed.stderr
