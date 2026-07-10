"""The ``sto`` command-line interface.

A small, dependency-free CLI over the core library. It reads chat messages as
JSON (a list of ``{"role", "content"}`` objects, or an object with a
``"messages"`` key) from a file or stdin, and exposes the most common
operations:

- ``sto count``     — count prompt tokens for a conversation.
- ``sto optimize``  — fit a conversation into a token budget.
- ``sto cost``      — estimate the cost of a request.
- ``sto dashboard`` — print the analytics dashboard (demo data by default).

Run ``sto --help`` or ``sto <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .. import __version__
from ..budgeting import DropOldestStrategy, SmartTokenOptimizer
from ..cost import CostEstimator, UsageTracker
from ..cost.errors import UnknownPricingError
from ..tokenization.registry import get_counter
from ..tokenization.types import Message
from .dashboard import render_dashboard


def _load_messages(path: str) -> list[Message]:
    """Load chat messages from ``path`` (or stdin when ``path`` is ``"-"``)."""
    raw = sys.stdin.read() if path == "-" else _read_file(path)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: could not parse JSON from {path!r}: {exc}") from exc
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise SystemExit(
            "error: expected a JSON list of messages or an object with a "
            "'messages' key"
        )
    return [dict(m) for m in data]  # type: ignore[misc]


def _read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise SystemExit(f"error: cannot read {path!r}: {exc}") from exc


def _prefer_exact(args: argparse.Namespace) -> bool:
    return not getattr(args, "fast", False)


def _cmd_count(args: argparse.Namespace) -> int:
    messages = _load_messages(args.file)
    counter = get_counter(args.model, prefer_exact=_prefer_exact(args))
    print(f"{counter.count_messages(messages)} tokens")
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    messages = _load_messages(args.file)
    counter = get_counter(args.model, prefer_exact=_prefer_exact(args))
    optimizer = SmartTokenOptimizer(
        args.max,
        counter=counter,
        strategy=DropOldestStrategy(keep_last=args.keep_last),
    )
    result = optimizer.optimize_detailed(messages)
    if args.output:
        _write_json(args.output, result.messages)
    print(
        f"{len(messages)} → {len(result.messages)} msgs   "
        f"{result.original_tokens} → {result.optimized_tokens} tok   "
        f"({result.compression_ratio:.0%} smaller)"
    )
    if getattr(args, "show_savings", False):
        _print_savings(args.model, result.tokens_saved)
    if not result.within_budget:
        print("warning: could not fit within budget (protected messages exceed it)")
    return 0


def _print_savings(model: str, tokens_saved: int) -> None:
    """Print the estimated per-request input-token cost saved for ``model``.

    Uses the input-token price for the model; unknown models simply report the
    token count with no dollar figure. This is the recurring saving per request
    — a busy app makes the same call many times a day.
    """
    from .. import CostEstimator
    from ..cost.errors import UnknownPricingError

    try:
        pricing = CostEstimator().pricing_for(model)
    except UnknownPricingError:
        print(f"saved {tokens_saved} input tokens/request (no pricing for {model!r})")
        return
    per_request = pricing.input_cost(tokens_saved)
    print(
        f"saved {tokens_saved} input tokens/request "
        f"≈ ${per_request:.4f}/request  "
        f"(${per_request * 1000:.2f} per 1k requests)"
    )


def _cmd_cost(args: argparse.Namespace) -> int:
    estimator = CostEstimator()
    try:
        if args.file:
            messages = _load_messages(args.file)
            estimate = estimator.estimate_messages(
                messages,
                model=args.model,
                expected_output_tokens=args.output_tokens,
            )
        else:
            estimate = estimator.estimate(
                args.model,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
            )
    except UnknownPricingError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        f"input {estimate.input_tokens} tok  output {estimate.output_tokens} tok\n"
        f"cost  ${estimate.total_cost:.4f}  "
        f"(in ${estimate.input_cost:.4f} + out ${estimate.output_cost:.4f})"
    )
    return 0


def _demo_tracker() -> UsageTracker:
    """Build a tracker with demo data so the dashboard is useful out of the box.

    Real applications pass their own tracker/snapshot to ``render_dashboard``.
    """
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
    for _ in range(47):
        tracker.record(cache_hit=True)
    for _ in range(29):
        tracker.record(cache_hit=False)
    return tracker


def _cmd_dashboard(args: argparse.Namespace) -> int:
    tracker = _demo_tracker()
    if getattr(args, "watch", False):
        from .live import run_live
        from .screen import Screen
        from .style import Ansi

        screen = Screen(stream=sys.stdout, enabled=True)
        return run_live(
            lambda: (tracker.snapshot(), None),
            screen=screen,
            ansi=Ansi(stream=sys.stdout),
            width=args.width,
            interval=args.interval,
            iterations=args.iterations,
        )
    print(render_dashboard(tracker.snapshot(), width=args.width))
    return 0


def _cmd_examples(args: argparse.Namespace) -> int:
    from .examples import browse_examples

    return browse_examples()


def _cmd_ui(args: argparse.Namespace) -> int:
    from .examples import browse_examples
    from .interactive import Menu, MenuItem
    from .style import Ansi

    ansi = Ansi(stream=sys.stdout)
    items = [
        MenuItem("Browse examples", "run a bundled example script", value="examples"),
        MenuItem("Dashboard", "show the analytics dashboard", value="dashboard"),
        MenuItem("Quit", value="quit"),
    ]
    while True:
        result = Menu("SmartTokenOptimizer", items, ansi=ansi).run()
        if result.cancelled or result.item is None:
            print(ansi.dim("Bye."))
            return 0
        choice = result.item.value
        if choice == "quit":
            print(ansi.dim("Bye."))
            return 0
        if choice == "examples":
            browse_examples()
        elif choice == "dashboard":
            _cmd_dashboard(argparse.Namespace(width=42))


def _write_json(path: str, messages: list[Message]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(messages, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``sto`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="sto",
        description="SmartTokenOptimizer command-line interface.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-m", "--model", default="gpt-4o", help="model id (default: gpt-4o)"
        )
        p.add_argument(
            "--fast",
            action="store_true",
            help="use the fast heuristic counter instead of an exact tokenizer",
        )

    p_count = sub.add_parser("count", help="count prompt tokens for a conversation")
    add_model(p_count)
    p_count.add_argument("file", help="JSON messages file, or '-' for stdin")
    p_count.set_defaults(func=_cmd_count)

    p_opt = sub.add_parser("optimize", help="fit a conversation into a token budget")
    add_model(p_opt)
    p_opt.add_argument("--max", type=int, required=True, help="maximum prompt tokens")
    p_opt.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="always keep at least this many recent messages",
    )
    p_opt.add_argument(
        "-o", "--output", help="write optimized messages as JSON to this path"
    )
    p_opt.add_argument(
        "--show-savings",
        action="store_true",
        help="also print the estimated per-request cost saved for the model",
    )
    p_opt.add_argument("file", help="JSON messages file, or '-' for stdin")
    p_opt.set_defaults(func=_cmd_optimize)

    p_cost = sub.add_parser("cost", help="estimate the cost of a request")
    add_model(p_cost)
    p_cost.add_argument(
        "file", nargs="?", help="JSON messages file for input tokens (optional)"
    )
    p_cost.add_argument(
        "--input-tokens", type=int, default=0, help="input tokens (if no file)"
    )
    p_cost.add_argument(
        "--output-tokens", type=int, default=0, help="expected output tokens"
    )
    p_cost.set_defaults(func=_cmd_cost)

    p_dash = sub.add_parser("dashboard", help="print the analytics dashboard")
    p_dash.add_argument("--width", type=int, default=42, help="box width (default: 42)")
    p_dash.add_argument(
        "--watch",
        action="store_true",
        help="live-refresh the dashboard until Ctrl-C",
    )
    p_dash.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between refreshes with --watch (default: 1.0)",
    )
    p_dash.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="stop after N refreshes (default: run until Ctrl-C)",
    )
    p_dash.set_defaults(func=_cmd_dashboard)

    p_examples = sub.add_parser(
        "examples", help="interactively browse and run the example scripts"
    )
    p_examples.set_defaults(func=_cmd_examples)

    p_ui = sub.add_parser("ui", help="open the interactive home menu")
    p_ui.set_defaults(func=_cmd_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
