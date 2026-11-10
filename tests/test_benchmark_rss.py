# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Portable RSS contracts; Windows APIs are mocked, not measured."""

import builtins
import ctypes
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks import _common


def test_import_without_resource_or_eager_ctypes():
    original_import = builtins.__import__
    blocked = []

    def restricted_import(name, *args, **kwargs):
        if name == "resource" or name.startswith("ctypes"):
            blocked.append(name)
            raise ImportError(f"blocked import: {name}")
        return original_import(name, *args, **kwargs)

    module = ModuleType("isolated_benchmark_common")
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=restricted_import)
    source = Path(_common.__file__).read_text(encoding="utf-8")
    exec(compile(source, _common.__file__, "exec"), module.__dict__)

    assert module.resource is None
    assert callable(module._peak_rss_mib)
    assert blocked == ["resource"]


@pytest.mark.parametrize(
    "platform, raw", [("linux", 128 * 1024), ("darwin", 128 * 1024 * 1024)]
)
def test_resource_units_and_monkeypatch_hook(monkeypatch, platform, raw):
    sentinel = object()
    calls = []

    def getrusage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=raw)

    monkeypatch.setattr(_common, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        _common, "resource", SimpleNamespace(RUSAGE_SELF=sentinel, getrusage=getrusage)
    )
    assert _common._peak_rss_mib() == 128.0
    assert calls == [sentinel]


class _FakeFunction:
    """A callable supporting the ctypes function signature attributes."""

    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


@pytest.fixture
def windows_api(monkeypatch):
    monkeypatch.setattr(_common, "resource", None)
    monkeypatch.setattr(_common, "sys", SimpleNamespace(platform="win32"))
    state = SimpleNamespace(handle=0x123456789, success=True, peak=173 * 1024**2 + 512)
    calls = []

    def read_memory(handle, counters_pointer, size):
        counters = counters_pointer._obj
        assert handle == state.handle
        assert counters.cb == size == ctypes.sizeof(counters)
        # DWORD fields are 32-bit; SIZE_T fields follow pointer width.
        assert type(counters)._fields_[:2] == [
            ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)
        ]
        assert [name for name, _ in type(counters)._fields_[2:]] == [
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
        ]
        assert all(kind is ctypes.c_size_t for _, kind in type(counters)._fields_[2:])
        counters.PeakWorkingSetSize = state.peak
        counters.WorkingSetSize = 1024  # Must read peak, not current working set.
        calls.append((handle, size))
        return state.success

    current = _FakeFunction(lambda: state.handle)
    memory = _FakeFunction(read_memory)
    libraries = {
        "kernel32": SimpleNamespace(GetCurrentProcess=current),
        "psapi": SimpleNamespace(GetProcessMemoryInfo=memory),
    }
    loads = []

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return libraries[name]

    error = OSError(5, "mock Windows access denied")
    error_codes = []

    def win_error(code):
        error_codes.append(code)
        return error

    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", win_error, raising=False)
    return SimpleNamespace(
        state=state, calls=calls, current=current, memory=memory,
        libraries=libraries, loads=loads, error=error, error_codes=error_codes,
    )


def test_windows_peak_bytes_and_native_signatures(windows_api):
    api = windows_api
    assert _common._peak_rss_mib() == api.state.peak / 1024**2
    assert api.loads == [
        ("kernel32", {"use_last_error": True}), ("psapi", {"use_last_error": True})
    ]
    assert api.current.argtypes == []
    assert api.current.restype is ctypes.c_void_p
    assert api.memory.argtypes[0] is ctypes.c_void_p
    assert api.memory.argtypes[1]._type_.__name__ == "PROCESS_MEMORY_COUNTERS"
    assert api.memory.argtypes[2] is ctypes.c_uint32
    assert api.memory.restype is ctypes.c_int
    assert len(api.calls) == 1


@pytest.mark.parametrize("failure", ["memory", "process"])
def test_windows_failed_calls_raise_last_error(windows_api, failure):
    api = windows_api
    if failure == "memory":
        api.state.success = False
    else:
        api.state.handle = None
    with pytest.raises(OSError) as caught:
        _common._peak_rss_mib()
    assert caught.value is api.error
    assert api.error_codes == [5]
    if failure == "process":
        assert api.calls == []


@pytest.mark.parametrize("library, symbol", [
    ("kernel32", "GetCurrentProcess"), ("psapi", "GetProcessMemoryInfo")
])
def test_windows_missing_api_raises_oserror(windows_api, library, symbol):
    delattr(windows_api.libraries[library], symbol)
    with pytest.raises(OSError, match="Windows peak RSS API"):
        _common._peak_rss_mib()


def test_windows_library_load_error_propagates(monkeypatch, windows_api):
    def fail_load(*args, **kwargs):
        raise windows_api.error

    monkeypatch.setattr(ctypes, "WinDLL", fail_load)
    with pytest.raises(OSError) as caught:
        _common._peak_rss_mib()
    assert caught.value is windows_api.error


def test_windows_missing_loader_raises_oserror(monkeypatch, windows_api):
    monkeypatch.delattr(ctypes, "WinDLL")
    with pytest.raises(OSError, match="Windows peak RSS API"):
        _common._peak_rss_mib()


def test_unknown_platform_without_resource(monkeypatch):
    monkeypatch.setattr(_common, "resource", None)
    monkeypatch.setattr(_common, "sys", SimpleNamespace(platform="unsupported-os"))
    with pytest.raises(RuntimeError, match="unsupported-os"):
        _common._peak_rss_mib()