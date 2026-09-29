from __future__ import annotations

import hashlib
import json
import netrc
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, assert_never
from urllib.parse import urlsplit

from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.train import FullConfig

WANDB_PROJECT = "kg-v3"
"""Every run launched from this repository logs to the v3 project, any game."""
V3_TAG = "kaggriculture-v3"
ATTEMPTS_FILE = "attempts.jsonl"
"""One JSON record per launch or resume of a run directory: its receipt."""
DEFAULT_WANDB_BASE_URL = "https://api.wandb.ai"
WANDB_CREDENTIAL_WORKFLOW = (
    "cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md"
)


class LogMode(StrEnum):
    DEBUG = auto()
    WANDB = auto()


class WandbMode(StrEnum):
    """W&B transport for ``LogMode.WANDB``; offline is an explicit outage."""

    ONLINE = auto()
    OFFLINE = auto()


class TelemetryMode(StrEnum):
    """What reaches W&B live; recorded in every attempt receipt."""

    WANDB_ONLINE = "wandb-online"
    WANDB_OFFLINE = "wandb-offline"
    DISABLED = "disabled"

    @property
    def is_outage(self) -> bool:
        return self is not TelemetryMode.WANDB_ONLINE


class MissingWandbCredentialsError(RuntimeError):
    """Online W&B was requested but no API key is installed."""


def telemetry_mode(log_mode: LogMode, wandb_mode: WandbMode) -> TelemetryMode:
    match log_mode:
        case LogMode.DEBUG:
            if wandb_mode != WandbMode.ONLINE:
                raise ValueError(
                    "--wandb-mode offline applies only to --log-mode wandb; "
                    "--log-mode debug already disables W&B"
                )
            return TelemetryMode.DISABLED
        case LogMode.WANDB:
            match wandb_mode:
                case WandbMode.ONLINE:
                    return TelemetryMode.WANDB_ONLINE
                case WandbMode.OFFLINE:
                    return TelemetryMode.WANDB_OFFLINE
                case _:
                    assert_never(wandb_mode)
        case _:
            assert_never(log_mode)


def wandb_credential_source(environ: Mapping[str, str], *, home: Path) -> str | None:
    """Where online W&B would read its API key, or ``None`` when it has none.

    Mirrors wandb's lookup: ``WANDB_API_KEY`` when set, else the ``NETRC``
    file (default ``~/.netrc``) entry for the ``WANDB_BASE_URL`` host (default
    ``api.wandb.ai``, or netrc's ``default``) with a non-empty password.
    Settings wandb would reject later (a malformed base URL, a blank or padded
    environment key) fail here instead. Never returns or prints the key.
    """
    host = wandb_host(environ)
    if _check_environment_key(environ):
        return "WANDB_API_KEY"
    netrc_path = _netrc_path(environ, home=home)
    parsed = _read_netrc(netrc_path)
    if parsed is None:
        return None
    entry = parsed.authenticators(host)
    if entry is None or not entry[2]:
        return None
    return str(netrc_path)


def _check_environment_key(environ: Mapping[str, str]) -> bool:
    """Whether ``WANDB_API_KEY`` is set; a blank or padded one is rejected.

    wandb reads a set key before netrc and rejects a blank one in every mode,
    offline included, so it can neither be ignored nor hidden by netrc.
    """
    if "WANDB_API_KEY" not in environ:
        return False
    key = environ["WANDB_API_KEY"]
    if not key or key != key.strip():
        raise MissingWandbCredentialsError(
            "WANDB_API_KEY is set but blank or padded with whitespace; unset "
            "it or set the exact key"
        )
    if (reason := _wandb_settings_rejection(api_key=key)) is not None:
        raise MissingWandbCredentialsError(
            f"WANDB_API_KEY is not a key wandb accepts ({reason}); unset it or "
            "set the exact key"
        )
    return True


def _read_netrc(path: Path) -> netrc.netrc | None:
    try:
        return netrc.netrc(str(path))
    except FileNotFoundError:
        return None
    except netrc.NetrcParseError as error:
        # The parser's message can quote a token from the file; keep it out.
        raise MissingWandbCredentialsError(
            f"cannot parse {path} (line {error.lineno}); repair it with "
            f"{WANDB_CREDENTIAL_WORKFLOW}"
        ) from None
    except OSError as error:
        raise MissingWandbCredentialsError(
            f"cannot read {path}: {error.strerror}"
        ) from None


_NETRC_UNSAFE = frozenset(" \t\n\r\"'#\\")


def wandb_netrc_entry(environ: Mapping[str, str], *, home: Path) -> str:
    """The netrc line for the W&B host alone, for copying to a pod via stdin.

    Only an explicit ``machine <host>`` entry qualifies (never ``default``), and
    every token must serialize unquoted. Errors never quote the file.
    """
    host = wandb_host(environ)
    netrc_path = _netrc_path(environ, home=home)
    parsed = _read_netrc(netrc_path)
    entry = None if parsed is None else parsed.hosts.get(host)
    if entry is None or not entry[2]:
        raise MissingWandbCredentialsError(
            f"{netrc_path} has no 'machine {host}' entry with a password"
        )
    login, _account, password = entry
    login = login or "user"
    if any(char in _NETRC_UNSAFE for char in login + password):
        raise MissingWandbCredentialsError(
            f"the 'machine {host}' entry in {netrc_path} has a login or password "
            "that netrc cannot hold unquoted; repair it before copying"
        )
    return f"machine {host} login {login} password {password}\n"


def require_wandb_credentials(environ: Mapping[str, str], *, home: Path) -> str:
    source = wandb_credential_source(environ, home=home)
    if source is None:
        raise MissingWandbCredentialsError(
            "W&B online logging has no credentials: WANDB_API_KEY is unset and "
            f"{_netrc_path(environ, home=home)} has no '{wandb_host(environ)}' "
            "entry. Fix: install the operator's api.wandb.ai netrc entry "
            f"({WANDB_CREDENTIAL_WORKFLOW}: stdin copy, chmod 600, check "
            "wandb.Api()) or export WANDB_API_KEY. To launch without live "
            "telemetry, pass --wandb-mode offline or --log-mode debug explicitly; "
            f"the outage is then recorded in the run's {ATTEMPTS_FILE}."
        )
    return source


def check_telemetry(
    log_mode: LogMode,
    wandb_mode: WandbMode,
    *,
    environ: Mapping[str, str],
    home: Path,
) -> TelemetryMode:
    """Startup gate: online needs credentials; an outage is announced loudly.

    Either W&B mode also rejects, before any config, env or model, a
    ``WANDB_MODE`` that contradicts the flag, and a ``WANDB_BASE_URL`` or
    ``WANDB_API_KEY`` that is empty, blank, padded, embeds credentials or fails
    the installed wandb's own ``Settings`` validation, which ``wandb.init``
    applies in every mode. Debug logging never starts wandb.
    """
    mode = telemetry_mode(log_mode, wandb_mode)
    if mode is not TelemetryMode.DISABLED:
        if "WANDB_MODE" in environ:
            env_mode = environ["WANDB_MODE"]
            if env_mode != wandb_mode:
                raise ValueError(
                    f"WANDB_MODE={env_mode!r} disagrees with --wandb-mode "
                    f"{wandb_mode}; unset WANDB_MODE and choose the mode with the flag"
                )
        wandb_host(environ)
        _check_environment_key(environ)
    if mode is TelemetryMode.WANDB_ONLINE:
        require_wandb_credentials(environ, home=home)
    else:
        print(telemetry_outage_banner(mode), file=sys.stderr, flush=True)
    return mode


def telemetry_outage_banner(mode: TelemetryMode) -> str:
    if not mode.is_outage:
        raise ValueError(f"{mode} is not a telemetry outage")
    detail = (
        "Metrics stay in the run directory's wandb/ folder and reach W&B only "
        "after `wandb sync`."
        if mode is TelemetryMode.WANDB_OFFLINE
        else "No metrics go to W&B; they print to stdout only."
    )
    rule = "=" * 78
    return "\n".join(
        (
            rule,
            f"W&B TELEMETRY OUTAGE: telemetry_mode={mode} (explicit flag)",
            detail,
            f"Recorded as telemetry_mode={mode} in the run's {ATTEMPTS_FILE}.",
            rule,
        )
    )


def _netrc_path(environ: Mapping[str, str], *, home: Path) -> Path:
    configured = environ.get("NETRC")
    return Path(configured).expanduser() if configured else home / ".netrc"


def wandb_host(environ: Mapping[str, str]) -> str:
    """The netrc machine name wandb uses: the base URL's host and port.

    Rejects, without quoting it, a base URL that is set but empty, is not
    http(s) with a host, embeds credentials or fails wandb's own validation.
    """
    if "WANDB_BASE_URL" not in environ:
        base_url = DEFAULT_WANDB_BASE_URL
    elif not (base_url := environ["WANDB_BASE_URL"]):
        raise ValueError(
            "WANDB_BASE_URL is set but empty; unset it or set the full URL"
        )
    try:
        parts = urlsplit(base_url)
        parts.port  # noqa: B018 - raises for a malformed port
    except ValueError:
        raise ValueError("WANDB_BASE_URL is not a valid URL") from None
    if parts.username is not None or parts.password is not None:
        raise ValueError("WANDB_BASE_URL must not embed credentials; use netrc")
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("WANDB_BASE_URL must be an http(s) URL with a host")
    if (reason := _wandb_settings_rejection(base_url=base_url)) is not None:
        raise ValueError(
            f"WANDB_BASE_URL is not a server address wandb accepts ({reason}); "
            f"for W&B cloud use {DEFAULT_WANDB_BASE_URL} or unset it"
        )
    return parts.netloc


def _wandb_settings_rejection(**settings: str) -> str | None:
    """Why the installed wandb's own ``Settings`` rejects ``settings``, if it does.

    ``wandb.init`` runs the same validation in every mode, offline included.
    The reason is wandb's error type only: its messages quote the value.
    """
    import pydantic
    import wandb
    from wandb.errors import UsageError

    try:
        wandb.Settings(**settings)
    except pydantic.ValidationError as error:
        return ", ".join(sorted({str(detail["type"]) for detail in error.errors()}))
    except UsageError:
        return "UsageError"
    return None


# --- v3 run identity ---------------------------------------------------------


@dataclass(frozen=True)
class RunIdentity:
    """Stable experiment identity plus this attempt's source and settings."""

    experiment_id: str
    job_type: str
    attempt: int
    source_commit: str
    attempt_source_commits: tuple[str, ...]
    config_sha256: str
    telemetry: TelemetryMode

    def wandb_config(self) -> dict[str, str]:
        """Constant across attempts, so a resumed W&B run keeps its config."""
        return {"experiment_id": self.experiment_id, "job_type": self.job_type}

    def wandb_summary(self) -> dict[str, int | str]:
        return {
            "v3/attempt": self.attempt,
            "v3/source_commit": self.source_commit,
            "v3/config_sha256": self.config_sha256,
            "v3/telemetry_mode": str(self.telemetry),
        }


def config_sha256(cfg: FullConfig) -> str:
    canonical = json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_source_commit(cwd: Path) -> str | None:
    """``HEAD`` of the checkout at ``cwd``, suffixed ``-dirty`` for tracked edits.

    ``None`` when git or the checkout's git metadata is unavailable.
    """
    try:
        commit = _git(("rev-parse", "HEAD"), cwd=cwd)
        status = _git(("status", "--porcelain", "--untracked-files=no"), cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit


_SOURCE_COMMIT_RE = re.compile(r"\S+")


def resolve_source_commit(cwd: Path, *, override: str | None) -> str:
    """This attempt's source identity: git's, or ``override`` without git.

    ``override`` (``--source-commit``) exists for checkouts without git
    metadata. Where git answers, a different override is rejected rather than
    recorded, so a stale or mistyped value cannot become the receipt's source.
    """
    if override is not None and not _SOURCE_COMMIT_RE.fullmatch(override):
        raise ValueError("--source-commit must be a non-empty value without whitespace")
    commit = git_source_commit(cwd)
    if commit is None:
        if override is None:
            raise RuntimeError(
                f"cannot read the source commit with git in {cwd}; pass --source-commit"
            )
        return override
    if override is not None and override != commit:
        raise ValueError(
            f"--source-commit {override!r} disagrees with git's {commit!r} in "
            f"{cwd}; drop the flag, which is only for checkouts without git"
        )
    return commit


def _git(args: tuple[str, ...], *, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, cwd=cwd
    ).stdout.strip()


_EXPERIMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_experiment_id(experiment_id: str) -> None:
    if not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(
            f"experiment id must match {_EXPERIMENT_ID_RE.pattern}, got "
            f"{experiment_id!r}"
        )


def plan_attempt(
    run_dir: Path,
    *,
    job_type: str,
    resume: bool,
    experiment_id: str | None,
    source_commit: str,
    config_sha256: str,
    telemetry: TelemetryMode,
) -> RunIdentity:
    """This launch's identity from the run's earlier attempt receipts.

    A fresh run starts attempt 0 under ``experiment_id`` (default: the run
    directory's name). A resume continues the recorded experiment id and
    carries every earlier attempt's source commit.
    """
    if experiment_id is not None:
        validate_experiment_id(experiment_id)
    path = run_dir / ATTEMPTS_FILE
    if not resume:
        if path.exists():
            raise FileExistsError(f"fresh run already has attempt records: {path}")
        return RunIdentity(
            experiment_id=experiment_id or run_dir.name,
            job_type=job_type,
            attempt=0,
            source_commit=source_commit,
            attempt_source_commits=(source_commit,),
            config_sha256=config_sha256,
            telemetry=telemetry,
        )
    earlier = read_attempts(path)
    recorded_id: str = earlier[0]["experiment_id"]
    if experiment_id is not None and experiment_id != recorded_id:
        raise ValueError(
            f"resume keeps experiment id {recorded_id!r}, got {experiment_id!r}"
        )
    if earlier[0]["job_type"] != job_type:
        raise ValueError(
            f"{path} records job type {earlier[0]['job_type']!r}, not {job_type!r}"
        )
    return RunIdentity(
        experiment_id=recorded_id,
        job_type=job_type,
        attempt=len(earlier),
        source_commit=source_commit,
        attempt_source_commits=(*earlier[-1]["attempt_source_commits"], source_commit),
        config_sha256=config_sha256,
        telemetry=telemetry,
    )


_ATTEMPT_KEYS = frozenset(
    {
        "attempt",
        "experiment_id",
        "job_type",
        "source_commit",
        "attempt_source_commits",
        "config_sha256",
        "telemetry_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_run_id",
        "wandb_url",
        "start_env_steps",
        "started_at",
    }
)


def read_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"resume needs the run's attempt records: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        raise ValueError(f"{path} has no attempt records")
    commits: list[str] = []
    for index, record in enumerate(records):
        where = f"{path} line {index + 1}"
        if not isinstance(record, dict) or set(record) != _ATTEMPT_KEYS:
            raise ValueError(f"{where} must have exactly {sorted(_ATTEMPT_KEYS)}")
        _check_attempt_types(record, where=where)
        if record["attempt"] != index:
            raise ValueError(f"{where} is not attempt {index}")
        for key in ("experiment_id", "job_type"):
            if record[key] != records[0][key]:
                raise ValueError(f"{where} changes {key} from line 1")
        commits.append(record["source_commit"])
        if record["attempt_source_commits"] != commits:
            raise ValueError(f"{where} attempt_source_commits breaks the history")
    return records


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _check_attempt_types(record: dict[str, Any], *, where: str) -> None:
    def is_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def is_text(value: object) -> bool:
        return isinstance(value, str) and bool(value)

    commits = record["attempt_source_commits"]
    checks = {
        "attempt": is_int(record["attempt"]),
        "experiment_id": isinstance(record["experiment_id"], str)
        and _EXPERIMENT_ID_RE.fullmatch(record["experiment_id"]) is not None,
        "job_type": is_text(record["job_type"]),
        "source_commit": is_text(record["source_commit"]),
        "attempt_source_commits": isinstance(commits, list)
        and all(is_text(commit) for commit in commits),
        "config_sha256": isinstance(record["config_sha256"], str)
        and _SHA256_RE.fullmatch(record["config_sha256"]) is not None,
        "telemetry_mode": record["telemetry_mode"] in set(TelemetryMode),
        "start_env_steps": is_int(record["start_env_steps"])
        and record["start_env_steps"] >= 0,
        "started_at": _is_aware_timestamp(record["started_at"]),
        **{
            key: record[key] is None or is_text(record[key])
            for key in ("wandb_project", "wandb_entity", "wandb_run_id", "wandb_url")
        },
    }
    bad = sorted(key for key, ok in checks.items() if not ok)
    if bad:
        raise ValueError(f"{where} has malformed fields: {', '.join(bad)}")


def _is_aware_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def record_attempt(
    run_dir: Path,
    identity: RunIdentity,
    logger: MetricLogger,
    *,
    start_env_steps: int,
) -> Path:
    """Append this attempt's receipt, including its telemetry mode."""
    run = logger.wandb_run_facts()
    if (run is None) != (identity.telemetry is TelemetryMode.DISABLED):
        raise RuntimeError(f"logger does not match telemetry_mode={identity.telemetry}")
    record: dict[str, Any] = {
        "attempt": identity.attempt,
        "experiment_id": identity.experiment_id,
        "job_type": identity.job_type,
        "source_commit": identity.source_commit,
        "attempt_source_commits": list(identity.attempt_source_commits),
        "config_sha256": identity.config_sha256,
        "telemetry_mode": str(identity.telemetry),
        "wandb_project": None if run is None else run.project,
        "wandb_entity": None if run is None else run.entity,
        "wandb_run_id": logger.run_id,
        "wandb_url": None if run is None else run.url,
        "start_env_steps": start_env_steps,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path = run_dir / ATTEMPTS_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


# --- metric loggers ----------------------------------------------------------


@dataclass(frozen=True)
class WandbRunFacts:
    project: str
    entity: str | None
    url: str | None


class MetricLogger:
    @property
    def run_id(self) -> str | None:
        raise NotImplementedError

    def wandb_run_facts(self) -> WandbRunFacts | None:
        """The W&B run this logger writes, or ``None`` when W&B is disabled."""
        raise NotImplementedError

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        raise NotImplementedError

    def set_summary(self, key: str, value: int | float | str) -> None:
        raise NotImplementedError

    def close(self, *, exit_code: int = 0) -> None:
        raise NotImplementedError


class DebugLogger(MetricLogger):
    @property
    def run_id(self) -> str | None:
        return None

    def wandb_run_facts(self) -> WandbRunFacts | None:
        return None

    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        print(metrics)

    def set_summary(
        self,
        key: str,  # noqa: ARG002
        value: int | float | str,  # noqa: ARG002
    ) -> None:
        return None

    def close(self, *, exit_code: int = 0) -> None:  # noqa: ARG002
        return None


class WandbLogger(MetricLogger):
    """A W&B run in project ``kg-v3``, grouped by the v3 experiment id.

    ``config`` is the launcher's resolved settings (``FullConfig`` JSON for
    PPO); ``game`` joins the ``kaggriculture-v3`` and job-type tags.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        config: Mapping[str, Any],
        game: str,
        identity: RunIdentity,
        mode: WandbMode,
        resume_run_id: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        init_kwargs: dict[str, Any] = {}
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._run = wandb.init(
            project=WANDB_PROJECT,
            dir=run_dir,
            name=run_dir.name,
            group=identity.experiment_id,
            job_type=identity.job_type,
            tags=[V3_TAG, identity.job_type, game],
            config={**config, "v3": identity.wandb_config()},
            mode=_wandb_init_mode(mode),
            **init_kwargs,
        )
        observed = _observed_wandb_mode(self._run)
        if observed != _wandb_init_mode(mode):
            # The receipt records the requested mode, so it must be the real one.
            self._run.finish(exit_code=1)
            raise RuntimeError(
                f"W&B started the run {observed}, not {mode} as requested; "
                "check WANDB_* settings and wandb's login state"
            )
        for key, value in identity.wandb_summary().items():
            self._run.summary[key] = value

    @property
    def run_id(self) -> str | None:
        return self._run.id

    def wandb_run_facts(self) -> WandbRunFacts | None:
        return WandbRunFacts(
            project=self._run.project,
            entity=self._run.entity or None,
            url=self._run.url,
        )

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def set_summary(self, key: str, value: int | float | str) -> None:
        run = self._wandb.run
        if run is None:
            raise RuntimeError("wandb run is not initialized")
        run.summary[key] = value

    def close(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def _wandb_init_mode(mode: WandbMode) -> Literal["online", "offline"]:
    match mode:
        case WandbMode.ONLINE:
            return "online"
        case WandbMode.OFFLINE:
            return "offline"
        case _:
            assert_never(mode)


def _observed_wandb_mode(run: Any) -> str:
    if run.disabled:
        return "disabled"
    return "offline" if run.offline else "online"


def _game(cfg: FullConfig) -> str:
    return "kaggriculture" if isinstance(cfg.env, KaggricultureEnvConfig) else "orbit"


def create_logger(
    log_mode: LogMode,
    run_dir: Path,
    cfg: FullConfig,
    *,
    identity: RunIdentity,
    resume_run_id: str | None = None,
) -> MetricLogger:
    """``run_ppo``'s logger; ``log_mode`` must agree with the identity."""
    match log_mode:
        case LogMode.DEBUG:
            if identity.telemetry is not TelemetryMode.DISABLED:
                raise ValueError("debug logging needs telemetry_mode=disabled")
        case LogMode.WANDB:
            if identity.telemetry is TelemetryMode.DISABLED:
                raise ValueError("wandb logging cannot use telemetry_mode=disabled")
        case _:
            assert_never(log_mode)
    return create_metric_logger(
        run_dir,
        config=cfg.model_dump(mode="json"),
        game=_game(cfg),
        identity=identity,
        resume_run_id=resume_run_id,
    )


def create_metric_logger(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    game: str,
    identity: RunIdentity,
    resume_run_id: str | None = None,
) -> MetricLogger:
    """Any v3 launcher's logger, chosen by the checked telemetry mode.

    The launcher contract: ``check_telemetry`` at startup, ``plan_attempt`` once
    the run directory exists, this, then ``record_attempt``.
    """
    match identity.telemetry:
        case TelemetryMode.DISABLED:
            return DebugLogger()
        case TelemetryMode.WANDB_ONLINE:
            mode = WandbMode.ONLINE
        case TelemetryMode.WANDB_OFFLINE:
            mode = WandbMode.OFFLINE
        case _:
            assert_never(identity.telemetry)
    return WandbLogger(
        run_dir,
        config=config,
        game=game,
        identity=identity,
        mode=mode,
        resume_run_id=resume_run_id,
    )
