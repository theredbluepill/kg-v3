from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
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
    WandbRunFacts,
    check_telemetry,
    config_sha256,
    create_logger,
    create_metric_logger,
    git_source_commit,
    plan_attempt,
    read_attempts,
    record_attempt,
    require_wandb_credentials,
    resolve_source_commit,
    telemetry_mode,
    wandb_credential_source,
    wandb_host,
    wandb_netrc_entry,
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
    # wandb reads a set key before netrc and rejects a blank or padded one, so
    # a valid netrc must not hide it.
    _netrc(tmp_path, f"machine api.wandb.ai login user password {_SECRET}\n")
    for bad in ("", "  ", f" {_SECRET}", f"{_SECRET}\n"):
        with pytest.raises(
            MissingWandbCredentialsError, match="blank or padded"
        ) as info:
            require_wandb_credentials({"WANDB_API_KEY": bad}, home=tmp_path)
        assert _SECRET not in str(info.value)


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
    with pytest.raises(ValueError, match="WANDB_MODE='offline' disagrees"):
        check_telemetry(LogMode.WANDB, WandbMode.ONLINE, environ=environ, home=tmp_path)
    with pytest.raises(ValueError, match="WANDB_MODE='disabled' disagrees"):
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
        self.offline = kwargs["mode"] == "offline"
        self.disabled = False
        self.summary: dict[str, Any] = {}
        self.finished: list[int] = []

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished.append(exit_code)


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


def test_base_url_is_validated_first_and_never_quoted(tmp_path: Path) -> None:
    assert wandb_host({}) == "api.wandb.ai"
    assert wandb_host({"WANDB_BASE_URL": "https://w.example:8443/"}) == "w.example:8443"
    for url, message in (
        (f"https://user:{_SECRET}@api.wandb.ai", "must not embed credentials"),
        ("api.wandb.ai", "http\\(s\\) URL with a host"),
        (f"ftp://{_SECRET}.example", "http\\(s\\) URL with a host"),
        (f"https://{_SECRET}.example:99999", "not a valid URL"),
        ("https://", "http\\(s\\) URL with a host"),
        # wandb's own validator quotes the value; the gate must not.
        (f"https://{_SECRET}.wandb.ai", "not a server address wandb accepts"),
        (f"https://{_SECRET} x.example", "not a server address wandb accepts"),
    ):
        environ = {"WANDB_BASE_URL": url, "WANDB_API_KEY": _SECRET}
        with pytest.raises(ValueError, match=message) as info:
            require_wandb_credentials(environ, home=tmp_path)
        assert _SECRET not in str(info.value)
        assert info.value.__cause__ is None


def test_an_empty_wandb_mode_is_a_disagreement(tmp_path: Path) -> None:
    environ = {"WANDB_API_KEY": _SECRET, "WANDB_MODE": ""}
    with pytest.raises(ValueError, match="WANDB_MODE='' disagrees"):
        check_telemetry(LogMode.WANDB, WandbMode.ONLINE, environ=environ, home=tmp_path)
    assert (
        check_telemetry(
            LogMode.WANDB,
            WandbMode.ONLINE,
            environ={"WANDB_API_KEY": _SECRET, "WANDB_MODE": "online"},
            home=tmp_path,
        )
        is TelemetryMode.WANDB_ONLINE
    )


# --- copying only the W&B netrc entry -----------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # packed: another machine and a default on the W&B entry's own line
        f"machine api.wandb.ai login user password {_SECRET} machine github.com "
        "login gh password GHSECRET default login a password DEFSECRET\n",
        # split: every token on its own line
        f"machine\napi.wandb.ai\nlogin\nuser\npassword\n{_SECRET}\n"
        "machine github.com login gh password GHSECRET\n",
        # conventional multi-line layout after other entries
        "default login a password DEFSECRET\nmachine github.com\n  login gh\n"
        f"  password GHSECRET\nmachine api.wandb.ai\n  login user\n"
        f"  password {_SECRET}\n",
    ],
)
def test_netrc_entry_export_keeps_only_the_wandb_host(
    tmp_path: Path, text: str
) -> None:
    _netrc(tmp_path, text)

    entry = wandb_netrc_entry({}, home=tmp_path)

    assert entry == f"machine api.wandb.ai login user password {_SECRET}\n"
    copied = tmp_path / "copied"
    copied.write_text(entry)
    assert wandb_credential_source({"NETRC": str(copied)}, home=tmp_path) == str(copied)


def test_netrc_entry_export_refuses_default_only_or_unsafe_entries(
    tmp_path: Path,
) -> None:
    path = _netrc(tmp_path, f"default login a password {_SECRET}\n")
    # wandb itself would use the default entry, but it is not copied.
    assert wandb_credential_source({}, home=tmp_path) == str(path)
    with pytest.raises(
        MissingWandbCredentialsError, match=re.escape("no 'machine api.wandb.ai'")
    ):
        wandb_netrc_entry({}, home=tmp_path)

    path.write_text(f'machine api.wandb.ai login user password "{_SECRET}#x"\n')
    with pytest.raises(MissingWandbCredentialsError, match="cannot hold") as info:
        wandb_netrc_entry({}, home=tmp_path)
    assert _SECRET not in str(info.value)

    (tmp_path / ".netrc").unlink()
    with pytest.raises(
        MissingWandbCredentialsError, match=re.escape("no 'machine api.wandb.ai'")
    ):
        wandb_netrc_entry({}, home=tmp_path)


# --- strict receipts -----------------------------------------------------------


def _receipt_lines(tmp_path: Path) -> list[dict[str, Any]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fresh = _plan(run_dir, experiment_id="exp", telemetry=TelemetryMode.DISABLED)
    record_attempt(run_dir, fresh, _Logger(None), start_env_steps=0)
    resumed = _plan(
        run_dir, resume=True, source_commit="c1", telemetry=TelemetryMode.DISABLED
    )
    record_attempt(run_dir, resumed, _Logger(None), start_env_steps=8)
    return [
        json.loads(line) for line in (run_dir / ATTEMPTS_FILE).read_text().splitlines()
    ]


def _write_receipts(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.mark.parametrize(
    ("line", "field", "value", "message"),
    [
        (0, "attempt", False, "malformed fields: attempt"),
        (0, "experiment_id", ["exp"], "malformed fields: experiment_id"),
        (0, "telemetry_mode", "sometimes", "malformed fields: telemetry_mode"),
        (0, "config_sha256", None, "malformed fields: config_sha256"),
        (1, "start_env_steps", -1, "malformed fields: start_env_steps"),
        (1, "wandb_run_id", 7, "malformed fields: wandb_run_id"),
        (1, "attempt", 0, "is not attempt 1"),
        (1, "experiment_id", "other", "changes experiment_id"),
        (1, "job_type", "bc", "changes job_type"),
        (1, "attempt_source_commits", ["zz", "c1"], "breaks the history"),
    ],
)
def test_receipts_are_validated_field_by_field(
    tmp_path: Path, line: int, field: str, value: object, message: str
) -> None:
    records = _receipt_lines(tmp_path)
    records[line][field] = value
    path = tmp_path / "attempts.jsonl"
    _write_receipts(path, records)

    with pytest.raises(ValueError, match=message):
        read_attempts(path)


def test_attempts_out_of_order_are_rejected(tmp_path: Path) -> None:
    records = _receipt_lines(tmp_path)
    path = tmp_path / "attempts.jsonl"
    _write_receipts(path, records[::-1])

    with pytest.raises(ValueError, match="line 1 is not attempt 0"):
        read_attempts(path)


def test_resume_keeps_the_recorded_job_type(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fresh = _plan(run_dir, telemetry=TelemetryMode.DISABLED)
    record_attempt(run_dir, fresh, _Logger(None), start_env_steps=0)

    with pytest.raises(ValueError, match="records job type 'ppo', not 'bc'"):
        _plan(run_dir, resume=True, job_type="bc")


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_resume_forwards_the_saved_run_id_with_resume_must(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runs: list[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        runs.append(_FakeRun(kwargs))
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, run=None))
    telemetry = (
        TelemetryMode.WANDB_ONLINE if mode == "online" else TelemetryMode.WANDB_OFFLINE
    )
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")

    create_logger(
        LogMode.WANDB,
        tmp_path,
        cfg,
        identity=_plan(tmp_path, telemetry=telemetry),
        resume_run_id="abc123",
    )

    (run,) = runs
    assert run.kwargs["id"] == "abc123"
    assert run.kwargs["resume"] == "must"
    assert run.kwargs["mode"] == mode


# --- claude-verify-wandb-r1 ----------------------------------------------------


@pytest.mark.parametrize("wandb_mode", [WandbMode.ONLINE, WandbMode.OFFLINE])
@pytest.mark.parametrize(
    ("environ", "error", "message"),
    [
        # wandb 0.26.1 rejects each inside wandb.init, even in offline mode.
        ({"WANDB_BASE_URL": ""}, ValueError, "WANDB_BASE_URL is set but empty"),
        (
            {"WANDB_BASE_URL": "api.wandb.ai"},
            ValueError,
            "http\\(s\\) URL with a host",
        ),
        # wandb 0.26.1's own Settings validator rejects these server addresses.
        (
            {"WANDB_BASE_URL": "https://wandb.ai"},
            ValueError,
            "not a server address wandb accepts",
        ),
        (
            {"WANDB_BASE_URL": "http://api.wandb.ai"},
            ValueError,
            "not a server address wandb accepts",
        ),
        (
            {"WANDB_BASE_URL": "https://app.wandb.ai"},
            ValueError,
            "not a server address wandb accepts",
        ),
        (
            {"WANDB_BASE_URL": "https://ho st.example"},
            ValueError,
            "not a server address wandb accepts",
        ),
        ({"WANDB_BASE_URL": "https://"}, ValueError, "http\\(s\\) URL with a host"),
        ({"WANDB_API_KEY": "  "}, MissingWandbCredentialsError, "blank or padded"),
        ({"WANDB_API_KEY": ""}, MissingWandbCredentialsError, "blank or padded"),
        (
            {"WANDB_API_KEY": f" {_SECRET}"},
            MissingWandbCredentialsError,
            "blank or padded",
        ),
    ],
)
def test_gate_rejects_settings_wandb_init_would_reject_in_both_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    wandb_mode: WandbMode,
    environ: dict[str, str],
    error: type[Exception],
    message: str,
) -> None:
    # A valid netrc so only the rejected setting can fail the online gate.
    _netrc(tmp_path, f"machine api.wandb.ai login user password {_SECRET}\n")

    with pytest.raises(error, match=message) as info:
        check_telemetry(LogMode.WANDB, wandb_mode, environ=environ, home=tmp_path)

    assert _SECRET not in str(info.value)
    assert "OUTAGE" not in capsys.readouterr().err


def test_debug_logging_ignores_wandb_settings_it_never_uses(tmp_path: Path) -> None:
    environ = {"WANDB_BASE_URL": "", "WANDB_API_KEY": "  "}
    assert (
        check_telemetry(LogMode.DEBUG, WandbMode.ONLINE, environ=environ, home=tmp_path)
        is TelemetryMode.DISABLED
    )


def test_gate_applies_wandbs_own_key_validation_without_quoting_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wandb
    from wandb.errors import UsageError

    real_settings = wandb.Settings

    def settings(**kwargs: str) -> object:
        if "api_key" in kwargs:
            raise UsageError(f"bad key {kwargs['api_key']}")
        return real_settings(**kwargs)

    monkeypatch.setattr(wandb, "Settings", settings)
    for wandb_mode in (WandbMode.ONLINE, WandbMode.OFFLINE):
        with pytest.raises(
            MissingWandbCredentialsError, match="not a key wandb accepts"
        ) as info:
            check_telemetry(
                LogMode.WANDB,
                wandb_mode,
                environ={"WANDB_API_KEY": _SECRET},
                home=tmp_path,
            )
        assert _SECRET not in str(info.value)
        assert info.value.__cause__ is None


@pytest.mark.parametrize(
    "url", ["https://api.wandb.ai", "https://w.example:8443/", "http://localhost:8080"]
)
def test_gate_accepts_server_addresses_wandb_accepts(tmp_path: Path, url: str) -> None:
    environ = {"WANDB_BASE_URL": url, "WANDB_API_KEY": _SECRET}
    for wandb_mode in (WandbMode.ONLINE, WandbMode.OFFLINE):
        check_telemetry(LogMode.WANDB, wandb_mode, environ=environ, home=tmp_path)


def test_offline_gate_accepts_a_valid_key_and_url_without_netrc(tmp_path: Path) -> None:
    environ = {"WANDB_BASE_URL": "https://w.example:8443", "WANDB_API_KEY": _SECRET}
    assert (
        check_telemetry(
            LogMode.WANDB, WandbMode.OFFLINE, environ=environ, home=tmp_path
        )
        is TelemetryMode.WANDB_OFFLINE
    )


def test_record_rejects_a_wandb_logger_under_disabled_telemetry(
    tmp_path: Path,
) -> None:
    class _WandbFactsLogger(_Logger):
        def wandb_run_facts(self) -> WandbRunFacts | None:
            return WandbRunFacts(project="kg-v3", entity=None, url=None)

    identity = _plan(tmp_path, telemetry=TelemetryMode.DISABLED)
    with pytest.raises(RuntimeError, match="does not match telemetry_mode=disabled"):
        record_attempt(tmp_path, identity, _WandbFactsLogger("r1"), start_env_steps=0)
    assert not (tmp_path / ATTEMPTS_FILE).exists()


def _git_repo(path: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=path,
        ).stdout.strip()

    git("init", "-q")
    (path / "tracked.txt").write_text("a\n")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "init")
    return git("rev-parse", "HEAD")


def test_git_source_commit_marks_tracked_edits_dirty(tmp_path: Path) -> None:
    head = _git_repo(tmp_path)
    assert git_source_commit(tmp_path) == head

    (tmp_path / "untracked.txt").write_text("ignored by the dirty check\n")
    assert git_source_commit(tmp_path) == head

    (tmp_path / "tracked.txt").write_text("b\n")
    assert git_source_commit(tmp_path) == f"{head}-dirty"


def test_git_source_commit_is_none_without_git_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_source_commit(tmp_path) is None


def test_source_commit_flag_is_only_for_checkouts_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git_repo(tmp_path)
    assert resolve_source_commit(tmp_path, override=None) == head
    # Matching the checkout is harmless; a different value is rejected.
    assert resolve_source_commit(tmp_path, override=head) == head
    with pytest.raises(ValueError, match="disagrees with git"):
        resolve_source_commit(tmp_path, override="0" * 40)
    (tmp_path / "tracked.txt").write_text("b\n")
    with pytest.raises(ValueError, match="disagrees with git"):
        resolve_source_commit(tmp_path, override=head)

    no_git = tmp_path / "no-git"
    no_git.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert resolve_source_commit(no_git, override="v3-src-abc") == "v3-src-abc"
    with pytest.raises(RuntimeError, match="pass --source-commit"):
        resolve_source_commit(no_git, override=None)
    for bad in ("", " abc", "a b"):
        with pytest.raises(ValueError, match="--source-commit must be"):
            resolve_source_commit(no_git, override=bad)


@pytest.mark.parametrize("started_at", ["", "yesterday", "2026-09-29T12:00:00", 7])
def test_receipts_need_a_timezone_aware_started_at(
    tmp_path: Path, started_at: object
) -> None:
    records = _receipt_lines(tmp_path)
    assert datetime.fromisoformat(records[0]["started_at"]).tzinfo is not None
    records[1]["started_at"] = started_at
    path = tmp_path / "attempts.jsonl"
    _write_receipts(path, records)

    with pytest.raises(ValueError, match="malformed fields: started_at"):
        read_attempts(path)


def test_netrc_entry_export_fills_a_missing_login(tmp_path: Path) -> None:
    _netrc(tmp_path, f"machine api.wandb.ai password {_SECRET}\n")

    assert wandb_netrc_entry({}, home=tmp_path) == (
        f"machine api.wandb.ai login user password {_SECRET}\n"
    )


@pytest.mark.parametrize(
    ("requested", "offline", "disabled", "observed"),
    [
        (TelemetryMode.WANDB_ONLINE, True, False, "offline"),
        (TelemetryMode.WANDB_ONLINE, False, True, "disabled"),
        (TelemetryMode.WANDB_OFFLINE, False, False, "online"),
        (TelemetryMode.WANDB_OFFLINE, True, True, "disabled"),
    ],
)
def test_wandb_logger_rejects_a_run_the_sdk_started_in_another_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: TelemetryMode,
    offline: bool,
    disabled: bool,
    observed: str,
) -> None:
    runs: list[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        runs.append(_FakeRun(kwargs))
        runs[-1].offline = offline
        runs[-1].disabled = disabled
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, run=None))
    identity = _plan(tmp_path, telemetry=requested)

    with pytest.raises(RuntimeError, match=f"started the run {observed}"):
        create_metric_logger(tmp_path, config={}, game="orbit", identity=identity)

    (run,) = runs
    assert run.finished == [1]
