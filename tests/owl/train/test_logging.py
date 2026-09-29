from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from owl.train import FullConfig
from owl.train.logging import (
    ATTEMPTS_FILE,
    WANDB_PROJECT,
    DebugLogger,
    LogMode,
    MissingWandbCredentialsError,
    RunIdentity,
    TelemetryMode,
    WandbLogger,
    WandbMode,
    check_telemetry,
    config_sha256,
    create_logger,
    create_metric_logger,
    plan_attempt,
    read_attempts,
    record_attempt,
    require_wandb_credentials,
    telemetry_mode,
    wandb_credential_source,
)

_CONFIGS = Path(__file__).parents[3] / "configs"
_SECRET = "0123456789abcdef0123456789abcdef01234567"


def test_wandb_logger_reports_exit_code() -> None:
    finished: list[int] = []

    class _Run:
        def finish(self, *, exit_code: int = 0) -> None:
            finished.append(exit_code)

    logger = WandbLogger.__new__(WandbLogger)
    logger._run = _Run()
    logger.close(exit_code=1)
    logger.close()

    assert finished == [1, 0]


def test_debug_logger_accepts_exit_code() -> None:
    logger = DebugLogger()

    logger.close(exit_code=1)
    logger.close()


# --- credentials -------------------------------------------------------------


def _netrc(home: Path, text: str) -> Path:
    path = home / ".netrc"
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_no_key_and_no_netrc_fails_with_the_fix(tmp_path: Path) -> None:
    assert wandb_credential_source({}, home=tmp_path) is None

    with pytest.raises(MissingWandbCredentialsError) as info:
        require_wandb_credentials({}, home=tmp_path)

    message = str(info.value)
    assert "WANDB_API_KEY is unset" in message
    assert f"{tmp_path / '.netrc'} has no 'api.wandb.ai' entry" in message
    assert "install-the-wandb-credential-before-any-pod-launch.md" in message
    assert "--wandb-mode offline" in message
    assert "--log-mode debug" in message


def test_environment_key_counts_but_blank_does_not(tmp_path: Path) -> None:
    assert (
        require_wandb_credentials({"WANDB_API_KEY": _SECRET}, home=tmp_path)
        == "WANDB_API_KEY"
    )
    with pytest.raises(MissingWandbCredentialsError):
        require_wandb_credentials({"WANDB_API_KEY": "  "}, home=tmp_path)


def test_netrc_needs_the_api_wandb_ai_entry_with_a_password(tmp_path: Path) -> None:
    path = _netrc(tmp_path, "machine github.com login x password y\n")
    assert wandb_credential_source({}, home=tmp_path) is None

    path.write_text('machine api.wandb.ai login user password ""\n')
    assert wandb_credential_source({}, home=tmp_path) is None

    path.write_text(f"machine api.wandb.ai login user password {_SECRET}\n")
    assert require_wandb_credentials({}, home=tmp_path) == str(path)


def test_netrc_and_host_follow_wandbs_environment(tmp_path: Path) -> None:
    other = tmp_path / "other.netrc"
    other.write_text(f"machine wandb.example.org login user password {_SECRET}\n")
    environ = {"NETRC": str(other), "WANDB_BASE_URL": "https://wandb.example.org"}

    assert wandb_credential_source(environ, home=tmp_path) == str(other)
    assert wandb_credential_source({"NETRC": str(other)}, home=tmp_path) is None


def test_malformed_netrc_error_never_quotes_the_file(tmp_path: Path) -> None:
    _netrc(tmp_path, f"{_SECRET} machine api.wandb.ai\n")

    with pytest.raises(MissingWandbCredentialsError, match="cannot parse") as info:
        wandb_credential_source({}, home=tmp_path)

    assert _SECRET not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


# --- telemetry gate ----------------------------------------------------------


def test_telemetry_modes_and_the_offline_debug_conflict() -> None:
    assert telemetry_mode(LogMode.WANDB, WandbMode.ONLINE) is TelemetryMode.WANDB_ONLINE
    assert (
        telemetry_mode(LogMode.WANDB, WandbMode.OFFLINE) is TelemetryMode.WANDB_OFFLINE
    )
    assert telemetry_mode(LogMode.DEBUG, WandbMode.ONLINE) is TelemetryMode.DISABLED
    with pytest.raises(ValueError, match="applies only to --log-mode wandb"):
        telemetry_mode(LogMode.DEBUG, WandbMode.OFFLINE)


def test_online_gate_requires_credentials_and_stays_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(MissingWandbCredentialsError):
        check_telemetry(LogMode.WANDB, WandbMode.ONLINE, environ={}, home=tmp_path)

    mode = check_telemetry(
        LogMode.WANDB,
        WandbMode.ONLINE,
        environ={"WANDB_API_KEY": _SECRET},
        home=tmp_path,
    )

    assert mode is TelemetryMode.WANDB_ONLINE
    captured = capsys.readouterr()
    assert "OUTAGE" not in captured.err
    assert _SECRET not in captured.out + captured.err


@pytest.mark.parametrize(
    ("log_mode", "wandb_mode", "expected"),
    [
        (LogMode.WANDB, WandbMode.OFFLINE, TelemetryMode.WANDB_OFFLINE),
        (LogMode.DEBUG, WandbMode.ONLINE, TelemetryMode.DISABLED),
    ],
)
def test_outage_modes_skip_credentials_and_warn_loudly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_mode: LogMode,
    wandb_mode: WandbMode,
    expected: TelemetryMode,
) -> None:
    mode = check_telemetry(log_mode, wandb_mode, environ={}, home=tmp_path)

    assert mode is expected
    err = capsys.readouterr().err
    assert f"W&B TELEMETRY OUTAGE: telemetry_mode={expected}" in err
    assert f"telemetry_mode={expected} in the run's {ATTEMPTS_FILE}" in err


def test_wandb_mode_environment_cannot_override_the_flag(tmp_path: Path) -> None:
    environ = {"WANDB_API_KEY": _SECRET, "WANDB_MODE": "offline"}
    with pytest.raises(ValueError, match="WANDB_MODE=offline disagrees"):
        check_telemetry(LogMode.WANDB, WandbMode.ONLINE, environ=environ, home=tmp_path)
    with pytest.raises(ValueError, match="WANDB_MODE=disabled disagrees"):
        check_telemetry(
            LogMode.WANDB,
            WandbMode.OFFLINE,
            environ={"WANDB_MODE": "disabled"},
            home=tmp_path,
        )


# --- identity and attempt receipts --------------------------------------------


def _plan(run_dir: Path, **overrides: Any) -> RunIdentity:
    kwargs: dict[str, Any] = {
        "job_type": "ppo",
        "resume": False,
        "experiment_id": None,
        "source_commit": "c0",
        "config_sha256": "f" * 64,
        "telemetry": TelemetryMode.WANDB_ONLINE,
    }
    kwargs.update(overrides)
    return plan_attempt(run_dir, **kwargs)


class _Logger(DebugLogger):
    def __init__(self, run_id: str | None) -> None:
        self._id = run_id

    @property
    def run_id(self) -> str | None:
        return self._id


def test_fresh_and_resumed_attempts_keep_the_experiment_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "20260929-120000"
    run_dir.mkdir()
    fresh = _plan(run_dir, telemetry=TelemetryMode.DISABLED)
    assert fresh.experiment_id == "20260929-120000"
    assert fresh.attempt == 0
    record_attempt(run_dir, fresh, _Logger(None), start_env_steps=0)

    with pytest.raises(FileExistsError, match="already has attempt records"):
        _plan(run_dir)

    resumed = _plan(
        run_dir, resume=True, source_commit="c1", telemetry=TelemetryMode.DISABLED
    )
    assert resumed.experiment_id == "20260929-120000"
    assert resumed.attempt == 1
    assert resumed.attempt_source_commits == ("c0", "c1")
    record_attempt(run_dir, resumed, _Logger(None), start_env_steps=64)

    records = read_attempts(run_dir / ATTEMPTS_FILE)
    assert [r["attempt"] for r in records] == [0, 1]
    assert [r["telemetry_mode"] for r in records] == ["disabled", "disabled"]
    assert records[1]["attempt_source_commits"] == ["c0", "c1"]
    with pytest.raises(ValueError, match="resume keeps experiment id"):
        _plan(run_dir, resume=True, experiment_id="renamed")


def test_explicit_experiment_ids_are_validated(tmp_path: Path) -> None:
    assert _plan(tmp_path, experiment_id="ppo-2rank.a_1").experiment_id == (
        "ppo-2rank.a_1"
    )
    with pytest.raises(ValueError, match="experiment id must match"):
        _plan(tmp_path, experiment_id="has space")


def test_resume_needs_well_formed_attempt_records(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="attempt records"):
        _plan(tmp_path, resume=True)

    (tmp_path / ATTEMPTS_FILE).write_text(json.dumps({"attempt": 0}) + "\n")
    with pytest.raises(ValueError, match="must have exactly"):
        _plan(tmp_path, resume=True)


def test_record_rejects_a_logger_that_contradicts_the_telemetry_mode(
    tmp_path: Path,
) -> None:
    identity = _plan(tmp_path, telemetry=TelemetryMode.WANDB_OFFLINE)
    with pytest.raises(RuntimeError, match="does not match telemetry_mode"):
        record_attempt(tmp_path, identity, _Logger(None), start_env_steps=0)
    assert not (tmp_path / ATTEMPTS_FILE).exists()


def test_config_hash_is_canonical_and_content_sensitive() -> None:
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    same = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    changed = cfg.model_copy(update={"rl": cfg.rl.model_copy(update={"horizon": 7})})

    assert config_sha256(cfg) == config_sha256(same)
    assert len(config_sha256(cfg)) == 64
    assert config_sha256(changed) != config_sha256(cfg)


# --- W&B init ----------------------------------------------------------------


class _FakeRun:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.id = "run-1"
        self.project = kwargs["project"]
        self.entity = "team"
        self.url = None if kwargs["mode"] == "offline" else "https://wandb.ai/x"
        self.summary: dict[str, Any] = {}


@pytest.mark.parametrize(
    ("telemetry", "mode"),
    [
        (TelemetryMode.WANDB_ONLINE, "online"),
        (TelemetryMode.WANDB_OFFLINE, "offline"),
    ],
)
def test_wandb_logger_opens_the_v3_project_with_v3_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry: TelemetryMode,
    mode: str,
) -> None:
    runs: list[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        runs.append(_FakeRun(kwargs))
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, run=None))
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    identity = _plan(tmp_path, experiment_id="exp-1", telemetry=telemetry)

    logger = create_logger(LogMode.WANDB, tmp_path, cfg, identity=identity)

    (run,) = runs
    assert run.kwargs["project"] == WANDB_PROJECT == "kg-v3"
    assert run.kwargs["mode"] == mode
    assert run.kwargs["group"] == "exp-1"
    assert run.kwargs["job_type"] == "ppo"
    assert run.kwargs["tags"] == ["kaggriculture-v3", "ppo", "kaggriculture"]
    assert run.kwargs["config"]["v3"] == {"experiment_id": "exp-1", "job_type": "ppo"}
    assert run.summary == {
        "v3/attempt": 0,
        "v3/source_commit": "c0",
        "v3/config_sha256": "f" * 64,
        "v3/telemetry_mode": str(telemetry),
    }
    facts = logger.wandb_run_facts()
    assert facts is not None
    assert facts.project == "kg-v3"
    assert facts.entity == "team"


def test_create_logger_rejects_a_mode_that_contradicts_the_identity(
    tmp_path: Path,
) -> None:
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    with pytest.raises(ValueError, match="debug logging needs telemetry_mode"):
        create_logger(LogMode.DEBUG, tmp_path, cfg, identity=_plan(tmp_path))
    disabled = _plan(tmp_path, telemetry=TelemetryMode.DISABLED)
    with pytest.raises(ValueError, match="cannot use telemetry_mode=disabled"):
        create_logger(LogMode.WANDB, tmp_path, cfg, identity=disabled)


def test_other_launchers_reuse_the_logger_with_their_own_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        runs.append(_FakeRun(kwargs))
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, run=None))
    identity = _plan(tmp_path, job_type="bc", telemetry=TelemetryMode.WANDB_OFFLINE)

    create_metric_logger(
        tmp_path, config={"bc": {"seed": 1}}, game="kaggriculture", identity=identity
    )
    disabled = create_metric_logger(
        tmp_path,
        config={},
        game="kaggriculture",
        identity=_plan(tmp_path, telemetry=TelemetryMode.DISABLED),
    )

    (run,) = runs
    assert run.kwargs["project"] == "kg-v3"
    assert run.kwargs["mode"] == "offline"
    assert run.kwargs["job_type"] == "bc"
    assert run.kwargs["tags"] == ["kaggriculture-v3", "bc", "kaggriculture"]
    assert run.kwargs["config"] == {
        "bc": {"seed": 1},
        "v3": {"experiment_id": tmp_path.name, "job_type": "bc"},
    }
    assert isinstance(disabled, DebugLogger)
