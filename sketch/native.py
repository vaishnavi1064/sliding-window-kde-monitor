"""Loader for the optional C++17 core (Phase 5).

The extension is optional by design: the package installs and every validation
tier passes without a compiler present, and `AVAILABLE` says whether the native
path can be used. Import errors are captured rather than raised so that asking
for the native core produces one clear message at the point of use, instead of
an ImportError from somewhere deep in a sketch constructor.
"""

from types import ModuleType

try:
    from sketch import _native as _module

    IMPORT_ERROR: Exception | None = None
except ImportError as error:  # pragma: no cover - depends on whether it was built
    _module = None
    IMPORT_ERROR = error

AVAILABLE = _module is not None


def module() -> ModuleType:
    """The `sketch._native` extension module, or raise explaining why not."""
    if _module is None:
        raise RuntimeError(
            f"the native core is not built: {IMPORT_ERROR}. Build it with `make native`."
        )
    return _module


def compiler() -> str | None:
    """What compiled the extension, for the benchmark report."""
    return None if _module is None else _module.__compiler__
