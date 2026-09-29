"""Verifier-only allocator cleanup between tests to stay within owner RSS budget."""
import ctypes
import gc
import pytest
_libc = ctypes.CDLL(None)
_relief = _libc.malloc_zone_pressure_relief
_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_relief.restype = ctypes.c_size_t

count = 0


def pytest_collection_finish(session):
    gc.collect()
    _relief(None, 0)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    global count
    yield
    count += 1
    if count == 1 or count % 16 == 0 or nextitem is None or item.path != nextitem.path:
        gc.collect()
        _relief(None, 0)
