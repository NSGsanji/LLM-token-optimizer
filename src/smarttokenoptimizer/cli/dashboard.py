"""Render a compact, dependency-free terminal dashboard.

The renderer is a pure function of its inputs: it takes an
:class:`~smarttokenoptimizer.cost.AnalyticsSnapshot` and an optional sequence of
credential-health rows and returns a boxed, aligned string. Keeping it pure (no
direct printing, no terminal control codes) makes it trivial to unit-test and
lets callers embed it wherever they like — a CLI, a logging line, a status
endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..cost.analytics import AnalyticsSnapshot
from ..credentials.pool import CredentialHealth

_TITLE = "SmartTokenOptimizer"


def _fmt_count(value: int) -> str:
    """Format an integer compactly (e.g. ``12400`` -> ``12.4k``)."""
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def _fmt_money(value: float, currency: str) -> str:
    symbol = "$" if currency == "USD" else ""
    return f"{symbol}{value:.4f}"


def _provider_line(row: CredentialHealth) -> str:
    """Render one credential/provider health row."""
    if not row.enabled:
        marker, status = "○", "disabled"
    elif row.rate_limited:
        marker, status = "○", "rate-limited"
    elif row.circuit_open:
        marker, status = "○", "circuit-open"
    elif row.available:
        marker, status = "●", "up"
    else:
        marker, status = "○", "down"
    name = row.provider or row.id
    return f"{marker} {name:<12} {status:<13} used {row.uses}"


def render_dashboard(
    snapshot: AnalyticsSnapshot,
    providers: Sequence[CredentialHealth] | None = None,
    *,
    width: int = 42,
) -> str:
    """Return the dashboard as a boxed, aligned multi-line string.

    Args:
        snapshot: The analytics snapshot to display.
        providers: Optional credential-health rows for the providers section.
            When ``None`` or empty, the providers section is omitted.
        width: Total box width in characters (including borders). Clamped to a
            sensible minimum.

    Returns:
        A ready-to-print string using box-drawing characters.
    """
    inner = max(width, 24) - 2
    hit = snapshot.cache_hit_rate
    lookups = snapshot.cache_hits + snapshot.cache_misses
    saved = _fmt_money(snapshot.cost_saved, snapshot.currency)
    savings_pct = f"{snapshot.savings_ratio:.0%}"

    rows = [
        f"Tokens   in {_fmt_count(snapshot.input_tokens)}"
        f"  out {_fmt_count(snapshot.output_tokens)}",
        f"Cost     {_fmt_money(snapshot.cost, snapshot.currency)}"
        f"   saved {saved} ({savings_pct})",
        f"Cache    hit {hit:.0%}   ({snapshot.cache_hits}/{lookups})",
        f"Success  {snapshot.success_rate:.0%}" f"   requests {snapshot.requests}",
    ]

    top = "┌─ " + _TITLE + " " + "─" * (inner - len(_TITLE) - 3) + "┐"
    lines = [top]
    for row in rows:
        lines.append("│ " + row[: inner - 2].ljust(inner - 2) + " │")

    if providers:
        label = "Providers"
        sep = "├─ " + label + " " + "─" * (inner - len(label) - 3) + "┤"
        lines.append(sep)
        for health in providers:
            text = _provider_line(health)
            lines.append("│ " + text[: inner - 2].ljust(inner - 2) + " │")

    lines.append("└" + "─" * inner + "┘")
    return "\n".join(lines)
