"""Compile the native core in place (`make native`).

One entry point for both platforms so the Makefile and CI invoke the build
identically. On Linux and macOS this is a thin wrapper around
`setup.py build_ext --inplace`. On Windows it first reconstructs the MSVC
environment, because setuptools' own compiler detection can fail on an
otherwise working toolchain:

setuptools runs `vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64`,
and `vswhere` hides installations it considers incomplete unless passed `-all`.
A Build Tools install whose component registration is partial therefore reports
"Microsoft Visual C++ 14.0 or greater is required" even though cl.exe, the CRT
and the Windows SDK are all present and vcvars64.bat works. Locating vcvars64
with `-all` and exporting its environment (plus DISTUTILS_USE_SDK, which tells
setuptools to trust the ambient environment rather than re-detect) builds fine.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_vcvars() -> Path | None:
    """Locate vcvars64.bat via vswhere, including installs flagged incomplete."""
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    relative = r"VC\Auxiliary\Build\vcvars64.bat"
    found = subprocess.run(
        [str(vswhere), "-all", "-prerelease", "-products", "*", "-find", relative],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in found.stdout.splitlines():
        candidate = Path(line.strip())
        if candidate.is_file():
            return candidate
    return None


def _msvc_environment() -> dict[str, str] | None:
    """The environment vcvars64.bat produces, or None if it could not be run."""
    vcvars = _find_vcvars()
    if vcvars is None:
        return None

    # `set` after vcvars dumps the fully initialised environment; parse it back.
    #
    # Passed as a string with shell=True rather than as an argument list: the
    # list form goes through subprocess.list2cmdline, which re-quotes the whole
    # `"...vcvars64.bat" && set` argument and leaves cmd unable to parse it
    # (verified -- it exits 1 with no output). The path comes from vswhere, not
    # from user input.
    dump = subprocess.run(
        f'"{vcvars}" >nul 2>&1 && set',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if dump.returncode != 0:
        return None

    environment = dict(os.environ)
    for line in dump.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            environment[name] = value
    # Skip setuptools' vswhere probe and use the environment we just built.
    environment["DISTUTILS_USE_SDK"] = "1"
    environment["MSSdk"] = "1"
    return environment


def _build_environment() -> dict[str, str]:
    if os.name != "nt" or shutil.which("cl") is not None:
        return dict(os.environ)

    environment = _msvc_environment()
    if environment is None:
        print(
            "warning: no MSVC environment found. Install the Visual Studio C++ "
            "Build Tools, or build from a Developer Command Prompt.",
            file=sys.stderr,
        )
        return dict(os.environ)
    return environment


def main() -> int:
    for module in ("setuptools", "pybind11"):
        try:
            __import__(module)
        except ImportError:
            print(
                f"error: {module} is required to build the native core.\n"
                f'       pip install "pybind11>=2.13" "setuptools>=68"',
                file=sys.stderr,
            )
            return 1

    completed = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=REPO_ROOT,
        env=_build_environment(),
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    # `optional=True` in setup.py means a failed compile is only a warning, so
    # confirm the module actually imports rather than trusting the exit code.
    check = subprocess.run(
        [sys.executable, "-c", "import sketch._native as n; print(n.__compiler__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        print("error: build reported success but sketch._native does not import:", file=sys.stderr)
        print(check.stderr, file=sys.stderr)
        return 1

    print(f"native core built ({check.stdout.strip()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
