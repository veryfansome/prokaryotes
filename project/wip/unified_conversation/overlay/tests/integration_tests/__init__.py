"""Overlay's integration_tests package — falls through to the real repo for
unchanged sibling modules (env_bootstrap, fakes, judges, stream_utils, conftest)."""

from __future__ import annotations

import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_REAL = pathlib.Path(__file__).resolve().parents[6] / "tests" / "integration_tests"
if _REAL != _HERE and _REAL.is_dir() and str(_REAL) not in __path__:
    __path__.append(str(_REAL))
