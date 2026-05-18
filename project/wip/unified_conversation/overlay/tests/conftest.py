"""Overlay bootstrap.

Prepends `project/wip/unified_conversation/overlay/prokaryotes` to
`prokaryotes.__path__` so overlay modules take precedence while unchanged
modules fall back to the real repo package.
"""

from __future__ import annotations

import pathlib

import prokaryotes

OVERLAY_PKG = pathlib.Path(__file__).resolve().parent.parent / "prokaryotes"

if str(OVERLAY_PKG) not in prokaryotes.__path__:
    prokaryotes.__path__.insert(0, str(OVERLAY_PKG))
