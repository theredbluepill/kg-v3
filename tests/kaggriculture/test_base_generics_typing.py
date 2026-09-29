"""Regression probe for the PEP 696 generic ``BaseModelAPI`` (Task 2.1).

mypy (``--strict`` plus the project config) checks two in-test probe modules:
the positive probe must type-check cleanly, and every line of the negative probe
marked ``# E`` (and no other line) must be a type error.
"""

from __future__ import annotations

import re
from pathlib import Path

from mypy import api

ROOT = Path(__file__).resolve().parents[2]

POSITIVE = """\
from typing import assert_type

import torch
from owl.kaggriculture import types as kt
from owl.model.base import BaseModelAPI, ModelOutput, ModelServingOutput
from owl.model.kaggriculture import KaggricultureTransformer
from owl.rl import ActionBundle, ActionConfig, ObsBatch

KaggBase = BaseModelAPI[
    kt.KaggricultureObsBatch, kt.KaggricultureActions, kt.KaggricultureActionConfig
]


def bare_base_is_orbit(model: BaseModelAPI, obs: ObsBatch) -> None:
    assert_type(model.forward(obs), ModelOutput[ActionBundle])
    assert_type(model.serve(obs), ModelServingOutput[ActionBundle])
    assert_type(model.action_spec, ActionConfig)
    assert_type(model.count_non_masked_tokens(obs), torch.Tensor)


def bare_output_is_orbit(output: ModelOutput) -> ModelOutput[ActionBundle]:
    return output


def kaggriculture(
    model: KaggricultureTransformer, obs: kt.KaggricultureObsBatch
) -> None:
    assert_type(model.forward(obs), ModelOutput[kt.KaggricultureActions])
    assert_type(model.serve(obs), ModelServingOutput[kt.KaggricultureActions])
    assert_type(model.action_spec, kt.KaggricultureActionConfig)
    assert_type(model.count_non_masked_tokens(obs), torch.Tensor)
    assert_type(model.compute_value(obs), torch.Tensor)
    base: KaggBase = model
    assert_type(base.forward(obs), ModelOutput[kt.KaggricultureActions])
    assert_type(base.serve(obs), ModelServingOutput[kt.KaggricultureActions])
    assert_type(base.count_non_masked_tokens(obs), torch.Tensor)
"""

NEGATIVE = """\
from owl.kaggriculture import types as kt
from owl.model.base import BaseModelAPI, ModelOutput
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)
from owl.rl import ActionBundle, ActionPureConfig, EntityBasedConfig, ObsBatch


def mixed_games(
    kagg: KaggricultureTransformer,
    orbit: BaseModelAPI,
    kobs: kt.KaggricultureObsBatch,
    oobs: ObsBatch,
    bundle: ActionBundle,
) -> None:
    kagg.forward(oobs)  # E
    kagg.serve(oobs)  # E
    kagg.compute_value(oobs)  # E
    kagg.count_non_masked_tokens(oobs)  # E
    kagg.evaluate_actions(kobs, bundle)  # E
    orbit.forward(kobs)  # E
    orbit.count_non_masked_tokens(kobs)  # E
    as_orbit: BaseModelAPI = kagg  # E
    orbit_output: ModelOutput[ActionBundle] = kagg.forward(kobs)  # E
    del as_orbit, orbit_output


def mixed_specs(config: KaggricultureTransformerConfig) -> None:
    KaggricultureTransformer(
        config,
        obs_spec=EntityBasedConfig(),  # E
        action_spec=kt.KaggricultureActionConfig(),
    )
    KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=ActionPureConfig(),  # E
    )
"""

_ERROR = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: (?P<message>.*)$")


def _expected_error_lines(source: str) -> set[int]:
    return {
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if line.rstrip().endswith("# E")
    }


def test_generic_base_typing_probes(tmp_path: Path) -> None:
    positive, negative = tmp_path / "probe_positive.py", tmp_path / "probe_negative.py"
    positive.write_text(POSITIVE)
    negative.write_text(NEGATIVE)
    stdout, stderr, _status = api.run(
        [
            "--strict",
            "--config-file",
            str(ROOT / "pyproject.toml"),
            # A fresh cache: mypy replays a cached module's errors, with the old
            # path, when an identically named probe's content is unchanged.
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--no-error-summary",
            "--show-absolute-path",
            str(positive),
            str(negative),
        ]
    )
    assert not stderr, stderr
    errors: dict[str, set[int]] = {str(positive): set(), str(negative): set()}
    for line in stdout.splitlines():
        match = _ERROR.match(line)
        if match is None:
            continue
        assert match["file"] in errors, f"error outside the probes: {line}"
        assert match["message"].endswith(("[arg-type]", "[assignment]")), line
        errors[match["file"]].add(int(match["line"]))
    assert errors[str(positive)] == set(), stdout
    expected = _expected_error_lines(NEGATIVE)
    assert len(expected) == 11
    assert errors[str(negative)] == expected, stdout
