"""Interactive browser for the bundled ``examples/`` scripts.

Discovers the example scripts, presents them in a numbered menu with a short
description pulled from each file's module docstring, and runs the chosen one in
a subprocess, streaming its output. Discovery is best-effort across a few
candidate locations so it works from a source checkout; when no examples are
found (e.g. an installed wheel without the source tree) it reports that clearly
instead of failing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .interactive import Menu, MenuItem
from .style import Ansi


@dataclass(frozen=True, slots=True)
class Example:
    """A discovered example script.

    Attributes:
        path: Absolute path to the ``.py`` file.
        title: Human-readable title from the module docstring's first line.
        description: A short description (docstring first line, minus the
            ``Example:`` prefix).
    """

    path: Path
    title: str
    description: str


def _candidate_dirs() -> list[Path]:
    """Return directories that might contain the examples, best first."""
    here = Path(__file__).resolve()
    # src/smarttokenoptimizer/cli/examples.py -> repo root is parents[3].
    roots = [here.parents[3], Path.cwd()]
    return [root / "examples" for root in roots]


def find_examples_dir() -> Path | None:
    """Return the first existing examples directory, or ``None``."""
    for candidate in _candidate_dirs():
        if candidate.is_dir() and any(candidate.glob("*.py")):
            return candidate
    return None


def _extract_summary(path: Path) -> str:
    """Return the first docstring line of ``path`` (empty string if none)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def discover_examples(directory: Path | None = None) -> list[Example]:
    """Discover example scripts in ``directory`` (auto-detected if ``None``).

    Args:
        directory: Where to look. When ``None``, :func:`find_examples_dir` is
            used.

    Returns:
        Examples sorted by filename. Empty when no directory/scripts are found.
    """
    directory = directory or find_examples_dir()
    if directory is None:
        return []
    examples: list[Example] = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        summary = _extract_summary(path)
        description = summary
        for prefix in ("Example:", "Example -"):
            if description.startswith(prefix):
                description = description[len(prefix) :].strip()
                break
        title = path.stem.replace("_", " ")
        examples.append(
            Example(path=path, title=title, description=description or title)
        )
    return examples


def run_example(
    example: Example,
    *,
    output: IO[str],
    runner: Sequence[str] | None = None,
) -> int:
    """Run one example script and stream its output to ``output``.

    Args:
        example: The example to run.
        output: Stream to write headers and the script's output to.
        runner: Command prefix used to launch the script. Defaults to
            ``[sys.executable]``. Injectable for testing.

    Returns:
        The script's process exit code.
    """
    cmd = list(runner) if runner is not None else [sys.executable]
    output.write(f"\n$ python {example.path.name}\n")
    output.flush()
    completed = subprocess.run(
        [*cmd, str(example.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        output.write(completed.stdout)
    if completed.stderr:
        output.write(completed.stderr)
    output.write(f"\n[exit code {completed.returncode}]\n")
    output.flush()
    return completed.returncode


def browse_examples(
    *,
    input: IO[str] | None = None,
    output: IO[str] | None = None,
    ansi: Ansi | None = None,
    directory: Path | None = None,
    runner: Sequence[str] | None = None,
) -> int:
    """Run the interactive examples browser.

    Lists discovered examples, lets the user pick one to run, and repeats until
    they quit. All streams are injectable so the flow is fully testable.

    Returns:
        A process exit code: ``0`` normally, ``1`` when no examples are found.
    """
    out = output if output is not None else sys.stdout
    styler = ansi if ansi is not None else Ansi(stream=out)
    examples = discover_examples(directory)
    if not examples:
        out.write(
            styler.yellow("No example scripts found.")
            + "\nExpected a top-level 'examples/' directory.\n"
        )
        return 1

    items = [MenuItem(ex.title, ex.description, value=ex) for ex in examples]
    while True:
        menu = Menu(
            "SmartTokenOptimizer — Examples",
            items,
            ansi=styler,
            input=input,
            output=out,
        )
        result = menu.run()
        if result.cancelled or result.item is None:
            out.write(styler.dim("Bye.\n"))
            return 0
        example = result.item.value
        assert isinstance(example, Example)
        run_example(example, output=out, runner=runner)
