"""Tests for the sto CLI and the terminal dashboard renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smarttokenoptimizer.cli import build_parser, main, render_dashboard
from smarttokenoptimizer.cost import UsageTracker
from smarttokenoptimizer.credentials import CredentialPool

SAMPLE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the capital of France?"},
]


def _write(tmp_path: Path, data: object, name: str = "chat.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestRenderDashboard:
    def _snapshot(self) -> object:
        tracker = UsageTracker()
        tracker.record(
            model="gpt-4o",
            input_tokens=12400,
            output_tokens=3100,
            cost=0.0431,
            tokens_saved=5000,
            cost_saved=0.0189,
            cache_hit=True,
        )
        tracker.record(cache_hit=False)
        return tracker.snapshot()

    def test_contains_core_sections(self) -> None:
        out = render_dashboard(self._snapshot())
        assert "SmartTokenOptimizer" in out
        assert "Tokens" in out
        assert "Cost" in out
        assert "Cache" in out
        assert "Success" in out

    def test_is_boxed(self) -> None:
        out = render_dashboard(self._snapshot())
        lines = out.splitlines()
        assert lines[0].startswith("┌") and lines[0].endswith("┐")
        assert lines[-1].startswith("└") and lines[-1].endswith("┘")
        assert all(line.startswith(("┌", "├", "└", "│")) for line in lines)

    def test_lines_share_width(self) -> None:
        out = render_dashboard(self._snapshot(), width=48)
        widths = {len(line) for line in out.splitlines()}
        assert len(widths) == 1  # every line is the same visual width

    def test_compact_number_formatting(self) -> None:
        out = render_dashboard(self._snapshot())
        assert "12.4k" in out  # 12400 tokens rendered compactly

    def test_providers_section(self) -> None:
        pool = CredentialPool()
        pool.add_key("sk-openai-1", id="k1", provider="openai")
        rate_limited = pool.add_key("sk-anth-1", id="k2", provider="anthropic")
        pool.record_rate_limited(rate_limited, retry_after=30)
        out = render_dashboard(self._snapshot(), pool.health())
        assert "Providers" in out
        assert "openai" in out
        assert "anthropic" in out
        assert "rate-limited" in out
        assert "●" in out  # a healthy provider marker
        assert "○" in out  # an unavailable provider marker

    def test_width_is_clamped(self) -> None:
        # Absurdly small widths must not crash or produce ragged output.
        out = render_dashboard(self._snapshot(), width=1)
        assert out.splitlines()

    def test_millions_formatting(self) -> None:
        tracker = UsageTracker()
        tracker.record(model="gpt-4o", input_tokens=2_500_000)
        out = render_dashboard(tracker.snapshot())
        assert "2.5M" in out

    def test_non_usd_currency(self) -> None:
        tracker = UsageTracker(currency="EUR")
        tracker.record(model="gpt-4o", cost=1.5)
        out = render_dashboard(tracker.snapshot())
        # Non-USD amounts render without the '$' symbol.
        assert "1.5000" in out
        assert "$" not in out

    @pytest.mark.parametrize(
        ("health_kwargs", "expected"),
        [
            ({"enabled": False}, "disabled"),
            ({"circuit_open": True}, "circuit-open"),
            ({"available": False}, "down"),
            ({"available": True}, "up"),
        ],
    )
    def test_provider_status_labels(
        self, health_kwargs: dict[str, object], expected: str
    ) -> None:
        from smarttokenoptimizer.credentials.pool import CredentialHealth

        base = {
            "id": "k1",
            "provider": "openai",
            "enabled": True,
            "available": True,
            "uses": 3,
            "successes": 3,
            "failures": 0,
            "consecutive_failures": 0,
            "rate_limited": False,
            "circuit_open": False,
            "last_error": None,
        }
        base.update(health_kwargs)
        health = CredentialHealth(**base)  # type: ignore[arg-type]
        out = render_dashboard(self._snapshot(), [health])
        assert expected in out


class TestParser:
    def test_build_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["count", "chat.json"])
        assert args.command == "count"
        assert args.model == "gpt-4o"

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "sto" in capsys.readouterr().out


class TestCount:
    def test_counts_from_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write(tmp_path, SAMPLE)
        assert main(["count", "-m", "gpt-4o", "--fast", path]) == 0
        assert "tokens" in capsys.readouterr().out

    def test_counts_from_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(SAMPLE)))
        assert main(["count", "--fast", "-"]) == 0
        assert "tokens" in capsys.readouterr().out

    def test_messages_key_object(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write(tmp_path, {"messages": SAMPLE})
        assert main(["count", "--fast", path]) == 0
        assert "tokens" in capsys.readouterr().out


class TestOptimize:
    def test_optimizes_and_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        convo = [{"role": "system", "content": "sys"}]
        convo += [{"role": "user", "content": f"message number {i}"} for i in range(30)]
        path = _write(tmp_path, convo)
        assert main(["optimize", "--fast", "--max", "40", path]) == 0
        out = capsys.readouterr().out
        assert "→" in out and "smaller" in out

    def test_writes_output_file(self, tmp_path: Path) -> None:
        convo = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        path = _write(tmp_path, convo)
        out_path = tmp_path / "out.json"
        main(["optimize", "--fast", "--max", "10", "-o", str(out_path), path])
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(written, list)
        assert len(written) < len(convo)

    def test_unsatisfiable_budget_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        convo = [{"role": "system", "content": "a b c d e f g h"}]
        path = _write(tmp_path, convo)
        main(["optimize", "--fast", "--max", "1", path])
        assert "warning" in capsys.readouterr().out

    def test_show_savings_known_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        convo = [{"role": "system", "content": "sys"}]
        convo += [{"role": "user", "content": f"message number {i}"} for i in range(30)]
        path = _write(tmp_path, convo)
        main(
            [
                "optimize",
                "--fast",
                "--max",
                "40",
                "--show-savings",
                "-m",
                "gpt-4o",
                path,
            ]
        )
        out = capsys.readouterr().out
        assert "saved" in out
        assert "$" in out
        assert "per 1k requests" in out

    def test_show_savings_unknown_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        convo = [{"role": "system", "content": "sys"}]
        convo += [{"role": "user", "content": f"message number {i}"} for i in range(30)]
        path = _write(tmp_path, convo)
        main(
            [
                "optimize",
                "--fast",
                "--max",
                "40",
                "--show-savings",
                "-m",
                "mystery",
                path,
            ]
        )
        out = capsys.readouterr().out
        assert "no pricing" in out
        assert "$" not in out


class TestCost:
    def test_from_token_counts(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert (
            main(
                [
                    "cost",
                    "-m",
                    "gpt-4o",
                    "--input-tokens",
                    "1000",
                    "--output-tokens",
                    "500",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "$0.0075" in out

    def test_from_messages_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write(tmp_path, SAMPLE)
        assert main(["cost", "-m", "gpt-4o", path]) == 0
        assert "cost" in capsys.readouterr().out

    def test_unknown_model_errors(self) -> None:
        with pytest.raises(SystemExit):
            main(["cost", "-m", "made-up-model", "--input-tokens", "10"])


class TestDashboardCommand:
    def test_prints_box(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["dashboard"]) == 0
        out = capsys.readouterr().out
        assert "SmartTokenOptimizer" in out
        assert "┌" in out


class TestErrorHandling:
    def test_bad_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["count", "--fast", str(path)])

    def test_non_list_json(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"unexpected": "shape"})
        with pytest.raises(SystemExit):
            main(["count", "--fast", str(path)])

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["count", "--fast", str(tmp_path / "nope.json")])
