"""Tests for the examples browser and the ui/examples CLI commands."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from smarttokenoptimizer.cli.examples import (
    Example,
    browse_examples,
    discover_examples,
    find_examples_dir,
    run_example,
)
from smarttokenoptimizer.cli.style import Ansi


def _make_example(dir_: Path, name: str, docstring: str, body: str = "") -> Path:
    path = dir_ / name
    content = f'"""{docstring}"""\n{body}'
    path.write_text(content, encoding="utf-8")
    return path


class TestDiscovery:
    def test_finds_repo_examples(self) -> None:
        # The real repo ships an examples/ directory with scripts.
        examples = discover_examples()
        assert examples, "expected bundled examples to be discovered"
        assert all(isinstance(e, Example) for e in examples)

    def test_titles_and_descriptions(self, tmp_path: Path) -> None:
        _make_example(
            tmp_path,
            "01_demo.py",
            "Example: a short demo of things.\n\nMore text.",
        )
        examples = discover_examples(tmp_path)
        assert len(examples) == 1
        assert examples[0].title == "01 demo"
        assert examples[0].description == "a short demo of things."

    def test_skips_dunder_files(self, tmp_path: Path) -> None:
        _make_example(tmp_path, "__init__.py", "not an example")
        _make_example(tmp_path, "real.py", "Example: real one.")
        names = [e.path.name for e in discover_examples(tmp_path)]
        assert names == ["real.py"]

    def test_missing_docstring(self, tmp_path: Path) -> None:
        (tmp_path / "bare.py").write_text("x = 1\n", encoding="utf-8")
        examples = discover_examples(tmp_path)
        # Falls back to the title when there is no docstring.
        assert examples[0].description == "bare"

    def test_syntax_error_tolerated(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def (:\n", encoding="utf-8")
        examples = discover_examples(tmp_path)
        assert examples[0].description == "broken"

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_examples(tmp_path) == []

    def test_find_examples_dir_returns_path_or_none(self) -> None:
        result = find_examples_dir()
        assert result is None or result.is_dir()

    def test_find_returns_none_when_no_candidates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Point discovery at empty directories only.
        monkeypatch.setattr(
            "smarttokenoptimizer.cli.examples._candidate_dirs",
            lambda: [tmp_path / "nope"],
        )
        assert find_examples_dir() is None

    def test_discover_auto_none_when_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "smarttokenoptimizer.cli.examples.find_examples_dir",
            lambda: None,
        )
        assert discover_examples() == []


class TestRunExample:
    def test_runs_and_reports_exit_code(self, tmp_path: Path) -> None:
        path = _make_example(
            tmp_path, "ok.py", "Example: prints hi.", body="print('hi there')\n"
        )
        example = Example(path=path, title="ok", description="prints hi")
        out = io.StringIO()
        code = run_example(example, output=out)
        text = out.getvalue()
        assert code == 0
        assert "hi there" in text
        assert "exit code 0" in text

    def test_captures_stderr_and_nonzero_exit(self, tmp_path: Path) -> None:
        path = _make_example(
            tmp_path,
            "boom.py",
            "Example: fails.",
            body="import sys\nsys.stderr.write('bad\\n')\nsys.exit(3)\n",
        )
        example = Example(path=path, title="boom", description="fails")
        out = io.StringIO()
        code = run_example(example, output=out)
        assert code == 3
        assert "bad" in out.getvalue()
        assert "exit code 3" in out.getvalue()


class TestBrowse:
    def _dir_with_two(self, tmp_path: Path) -> Path:
        _make_example(tmp_path, "01_a.py", "Example: first.", body="print('AAA')\n")
        _make_example(tmp_path, "02_b.py", "Example: second.", body="print('BBB')\n")
        return tmp_path

    def test_run_then_quit(self, tmp_path: Path) -> None:
        directory = self._dir_with_two(tmp_path)
        out = io.StringIO()
        rc = browse_examples(
            input=io.StringIO("1\nq\n"),
            output=out,
            ansi=Ansi(enabled=False),
            directory=directory,
        )
        assert rc == 0
        text = out.getvalue()
        assert "AAA" in text  # first example actually ran
        assert "Bye." in text

    def test_quit_immediately(self, tmp_path: Path) -> None:
        directory = self._dir_with_two(tmp_path)
        out = io.StringIO()
        rc = browse_examples(
            input=io.StringIO("q\n"),
            output=out,
            ansi=Ansi(enabled=False),
            directory=directory,
        )
        assert rc == 0
        assert "AAA" not in out.getvalue()

    def test_no_examples_found(self, tmp_path: Path) -> None:
        out = io.StringIO()
        rc = browse_examples(output=out, ansi=Ansi(enabled=False), directory=tmp_path)
        assert rc == 1
        assert "No example scripts found" in out.getvalue()


class TestCliCommands:
    def test_examples_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from smarttokenoptimizer.cli.main import main

        monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
        assert main(["examples"]) in (0, 1)

    def test_ui_dashboard_then_quit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from smarttokenoptimizer.cli.main import main

        monkeypatch.setattr("sys.stdin", io.StringIO("2\n3\n"))
        assert main(["ui"]) == 0
        assert "SmartTokenOptimizer" in capsys.readouterr().out

    def test_ui_quit_immediately(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from smarttokenoptimizer.cli.main import main

        monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
        assert main(["ui"]) == 0

    def test_ui_examples_then_quit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from smarttokenoptimizer.cli.main import main

        # 1 = Browse examples -> immediately quit browser -> 3 = Quit home.
        monkeypatch.setattr("sys.stdin", io.StringIO("1\nq\n3\n"))
        assert main(["ui"]) == 0
