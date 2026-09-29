from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "export_wandb_netrc_entry.py"
_spec = importlib.util.spec_from_file_location("export_wandb_netrc_entry", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

_SECRET = "0123456789abcdef0123456789abcdef01234567"


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".netrc").write_text(
        "machine github.com login gh password GHSECRET\n"
        f"machine api.wandb.ai login user password {_SECRET}\n"
    )
    for name in ("NETRC", "WANDB_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def test_writes_only_the_wandb_entry_to_a_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _home(tmp_path, monkeypatch)

    assert export.main() == 0

    captured = capsys.readouterr()
    assert captured.out == f"machine api.wandb.ai login user password {_SECRET}\n"
    assert captured.err == ""


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_refuses_to_print_the_key_to_a_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _home(tmp_path, monkeypatch)
    terminal = _Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)

    assert export.main() == 2

    assert terminal.getvalue() == ""
    assert "refusing to print" in capsys.readouterr().err


def test_a_missing_entry_exports_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _home(tmp_path, monkeypatch)
    (tmp_path / ".netrc").write_text("machine github.com login gh password GHSECRET\n")

    assert export.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nothing exported" in captured.err
    assert "GHSECRET" not in captured.err
