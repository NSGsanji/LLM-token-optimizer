# Contributing to LLM-Token-Optimizer

Thanks for your interest in improving SmartTokenOptimizer! This project aims to
be a widely-adopted, production-grade framework, so we value small, well-tested,
incremental contributions.

## Getting started

```bash
git clone https://github.com/NSGsanji/LLM-token-optimizer.git
cd ai-cred-management
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tiktoken]"
```

## Development workflow

1. Create a feature branch (`feat/...`, `fix/...`, `docs/...`).
2. Make a **single logical change** per pull request.
3. Run the full quality gate locally before pushing:

   ```bash
   ruff check .
   black --check .
   mypy
   pytest
   ```

4. Update `CHANGELOG.md` under `## [Unreleased]`.
5. Open a pull request with a concise summary.

## Code standards

- **Python 3.11+.**
- **Zero required runtime dependencies** in the core. Anything heavier belongs
  behind an optional extra and must degrade gracefully when absent.
- **Type everything.** The codebase passes `mypy --strict`.
- **Docstrings** on every public module, class and function.
- **Tests** for every feature, including edge cases and regressions.
- **Conventional Commits** for commit messages, e.g.
  `feat(counting): add tiktoken backend`.

## No placeholders

Everything committed should be production-ready. Avoid TODO-only commits, fake
implementations, or code paths that raise `NotImplementedError` on the happy
path.
