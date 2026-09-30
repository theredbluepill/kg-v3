"""Assert the loaded native extension inside the actual pytest process."""

import sys
from pathlib import Path

import pytest
from owl import rs


def pytest_sessionstart(session: pytest.Session) -> None:
    assert session.config.rootpath.resolve() == Path.cwd().resolve()
    expected = Path.cwd() / "python/owl/rs.abi3.so"
    assert Path(rs.__file__).resolve() == expected.resolve(), (rs.__file__, expected)
    assert Path(sys.prefix).resolve() == (Path.cwd() / ".venv").resolve(), sys.prefix
    print(
        f"SPS runtime verified: interpreter={sys.executable}; "
        f"prefix={sys.prefix}; extension={rs.__file__}",
        flush=True,
    )
