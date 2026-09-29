"""Verifier-only GC for in-process mypy AST and model temporary cycles."""
import gc
import pytest

def pytest_configure(config):
    gc.set_threshold(100, 5, 5)

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    yield
    if nextitem is None or item.path != nextitem.path:
        gc.collect()
