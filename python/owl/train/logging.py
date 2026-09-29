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

    Mirrors wandb's lookup: a non-empty ``WANDB_API_KEY``, else the ``NETRC``
    file (default ``~/.netrc``) entry for the ``WANDB_BASE_URL`` host (default
    ``api.wandb.ai``) with a non-empty password. Never returns or prints the key.
    """
    if environ.get("WANDB_API_KEY", "").strip():
        return "WANDB_API_KEY"
    netrc_path = _netrc_path(environ, home=home)
    try:
        parsed = netrc.netrc(str(netrc_path))
    except FileNotFoundError:
        return None
    except netrc.NetrcParseError as error:
        # The parser's message can quote a token from the file; keep it out.
        raise MissingWandbCredentialsError(
            f"cannot parse {netrc_path} (line {error.lineno}); repair it with "
            f"{WANDB_CREDENTIAL_WORKFLOW}"
        ) from None
    except OSError as error:
        raise MissingWandbCredentialsError(
            f"cannot read {netrc_path}: {error.strerror}"
        ) from None
    entry = parsed.authenticators(_wandb_host(environ))
    if entry is None or not entry[2]:
        return None
    return str(netrc_path)


def require_wandb_credentials(environ: Mapping[str, str], *, home: Path) -> str:
    source = wandb_credential_source(environ, home=home)
    if source is None:
        raise MissingWandbCredentialsError(
            "W&B online logging has no credentials: WANDB_API_KEY is unset and "
            f"{_netrc_path(environ, home=home)} has no '{_wandb_host(environ)}' "
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
    """Startup gate: online needs credentials; an outage is announced loudly."""
    mode = telemetry_mode(log_mode, wandb_mode)
    env_mode = environ.get("WANDB_MODE")
    if mode is not TelemetryMode.DISABLED and env_mode and env_mode != wandb_mode:
        raise ValueError(
            f"WANDB_MODE={env_mode} disagrees with --wandb-mode {wandb_mode}; "
            "unset WANDB_MODE and choose the mode with the flag"
        )
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


def _wandb_host(environ: Mapping[str, str]) -> str:
    base_url = environ.get("WANDB_BASE_URL") or DEFAULT_WANDB_BASE_URL
    host = urlsplit(base_url).netloc
    if not host:
        raise ValueError(f"WANDB_BASE_URL has no host: {base_url!r}")
    return host


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


def git_source_commit(cwd: Path) -> str:
    """``HEAD`` of the checkout at ``cwd``, suffixed ``-dirty`` for tracked edits."""
    try:
        commit = _git(("rev-parse", "HEAD"), cwd=cwd)
        status = _git(("status", "--porcelain", "--untracked-files=no"), cwd=cwd)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot read the source commit with git in {cwd}; pass --source-commit"
        ) from error
    return f"{commit}-dirty" if status else commit


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
    recorded_id = str(earlier[0]["experiment_id"])
    if experiment_id is not None and experiment_id != recorded_id:
        raise ValueError(
            f"resume keeps experiment id {recorded_id!r}, got {experiment_id!r}"
        )
    return RunIdentity(
        experiment_id=recorded_id,
        job_type=job_type,
        attempt=len(earlier),
        source_commit=source_commit,
        attempt_source_commits=(
            *(str(record["source_commit"]) for record in earlier),
            source_commit,
        ),
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
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _ATTEMPT_KEYS:
            raise ValueError(
                f"{path} line {index + 1} must have exactly {sorted(_ATTEMPT_KEYS)}"
            )
        if record["attempt"] != index:
            raise ValueError(f"{path} line {index + 1} is not attempt {index}")
    return records


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
