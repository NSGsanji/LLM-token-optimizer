"""Smoke tests that every example script runs to completion.

Executing the examples in-process (via ``runpy``) keeps them working as the
library evolves, and contributes their code paths to coverage.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_SCRIPTS = sorted(_EXAMPLES_DIR.glob("*.py"))


def _script_id(path: Path) -> str:
    return path.name


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=_script_id)
def test_example_runs(script: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Each example runs as ``__main__`` without raising and prints output."""
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert out.strip(), f"{script.name} produced no output"


def test_examples_exist() -> None:
    """Guard against the examples directory going missing or empty."""
    assert _EXAMPLE_SCRIPTS, "no example scripts found"
