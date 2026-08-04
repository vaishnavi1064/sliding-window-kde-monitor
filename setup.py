"""Build the optional native core (Phase 5).

Package metadata lives in pyproject.toml; this file exists only to describe the
C++ extension, which setuptools still reads from `ext_modules` here.

The extension is *optional* by default: a machine with no C++ compiler installs
a working pure-Python package, and `sketch.native.AVAILABLE` reports False.

Two environment variables override that default, and CI uses both:
  SWAKDE_REQUIRE_NATIVE=1  a build failure is fatal, so a broken native core
                           cannot quietly degrade into "not built"
  SWAKDE_SKIP_NATIVE=1     do not build it at all, which is the only way to
                           exercise the pure-Python path on a machine that does
                           have a compiler (every GitHub Linux runner does)
"""

import os

from setuptools import setup


def _flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0")


REQUIRE_NATIVE = _flag("SWAKDE_REQUIRE_NATIVE")
SKIP_NATIVE = _flag("SWAKDE_SKIP_NATIVE")

if SKIP_NATIVE and REQUIRE_NATIVE:
    raise ValueError("SWAKDE_SKIP_NATIVE and SWAKDE_REQUIRE_NATIVE are contradictory")


def _native_extension():
    """The extension and its build command, or None to build a pure-Python package."""
    if SKIP_NATIVE:
        return None
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        if REQUIRE_NATIVE:
            raise
        # No pybind11 available (for instance an install with build isolation
        # disabled and only the runtime dependencies present).
        return None

    class BuildExt(build_ext):
        """Set optimisation flags explicitly.

        Without this the extension inherits whatever CFLAGS the host CPython was
        built with, which differs between python.org builds and distro packages
        -- so the benchmark numbers would depend on the interpreter's build
        rather than on our code. Deliberately no `-march=native`: it would make
        the build non-portable and the measurements non-reproducible across
        machines, for a gain the profile does not point at.
        """

        FLAGS = {"msvc": ["/O2"], "unix": ["-O3"]}

        def build_extensions(self):
            extra = self.FLAGS.get(self.compiler.compiler_type, [])
            for extension in self.extensions:
                extension.extra_compile_args = list(extension.extra_compile_args or []) + extra
            super().build_extensions()

    extension = Pybind11Extension(
        "sketch._native",
        ["src/bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,
        # A failed compile is a warning rather than an error unless the caller
        # asked for the native core specifically.
        optional=not REQUIRE_NATIVE,
    )
    return extension, BuildExt


_native = _native_extension()

if _native is None:
    setup()
else:
    _extension, _build_ext = _native
    setup(ext_modules=[_extension], cmdclass={"build_ext": _build_ext})
