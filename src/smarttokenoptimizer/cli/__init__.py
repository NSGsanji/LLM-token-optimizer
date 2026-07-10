"""Command-line interface and terminal dashboard for SmartTokenOptimizer.

Exposes the ``sto`` console command (see :func:`main`) and a reusable
:func:`render_dashboard` you can call directly to embed the analytics view
anywhere.
"""

from __future__ import annotations

from .dashboard import render_dashboard
from .main import build_parser, main

__all__ = ["build_parser", "main", "render_dashboard"]
