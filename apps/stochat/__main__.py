"""stochat — an interactive chat REPL that optimizes prompts before sending.

Talk to any OpenAI-compatible coding agent (Claude, GPT, Groq, local Ollama, …)
with SmartTokenOptimizer trimming redundant history, duplicates and whitespace
out of every request. Mention a file or directory and it is attached (and
pinned so it's never dropped). Up-arrow recalls history; Tab completes paths.

Run `python -m stochat providers` for the list, or see the README for setup.
"""

from __future__ import annotations

import argparse
import atexit
import glob
import os
import readline
import sys
from pathlib import Path

from openai import OpenAI

from smarttokenoptimizer import SmartTokenOptimizer
from smarttokenoptimizer.budgeting import CompositeStrategy, DropOldestStrategy
from smarttokenoptimizer.compression import CompressionStrategy, WhitespaceCompressor
from smarttokenoptimizer.context import DeduplicateStrategy
from smarttokenoptimizer.tokenization import get_counter

from .providers import PROVIDERS, resolve

DIM, CYAN, BOLD, YELLOW, RESET = (
    "\033[2m",
    "\033[36m",
    "\033[1m",
    "\033[33m",
    "\033[0m",
)
MAX_FILE_BYTES = 200_000
MAX_DIR_ENTRIES = 300
# Safe compressor: collapse blank-line runs only, so attached CODE keeps its
# indentation (collapsing spaces would corrupt Python).
_SAFE = WhitespaceCompressor(collapse_spaces=False, strip_lines=False)


def _setup_readline() -> None:
    hist = os.path.expanduser("~/.stochat_history")
    try:
        readline.read_history_file(hist)
    except OSError:
        pass
    atexit.register(lambda: _safe_write_history(hist))
    readline.set_completer_delims(" \t\n")

    def complete(text: str, state: int) -> str | None:
        stub = os.path.expanduser(text)
        matches = [m + ("/" if os.path.isdir(m) else "") for m in glob.glob(stub + "*")]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def _safe_write_history(path: str) -> None:
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def _attach_paths(text: str) -> tuple[str, list[str]]:
    """Read any existing files/dirs mentioned in ``text`` and append them."""
    attached: list[str] = []
    blocks: list[str] = []
    for raw in text.split():
        cand = os.path.expanduser(raw.strip("\"'`,;:()[]"))
        if not cand or not os.path.exists(cand):
            continue
        p = Path(cand)
        if p.is_file():
            try:
                body = p.read_bytes()[:MAX_FILE_BYTES].decode("utf-8", "replace")
            except OSError:
                continue
            blocks.append(f"\n\n--- file: {cand} ---\n{body}")
            attached.append(cand)
        elif p.is_dir():
            files = sorted(str(f) for f in p.rglob("*") if f.is_file())
            listing = "\n".join(files[:MAX_DIR_ENTRIES])
            more = (
                f"\n... (+{len(files) - MAX_DIR_ENTRIES} more)"
                if len(files) > MAX_DIR_ENTRIES
                else ""
            )
            blocks.append(f"\n\n--- dir listing: {cand}/ ---\n{listing}{more}")
            attached.append(cand + "/")
    return text + "".join(blocks), attached


def _optimize(history: list[dict], counter, budget: int, keep_last: int):
    """Trim ``history`` to ``budget`` tokens, pinning file-carrying messages."""
    optimizer = SmartTokenOptimizer(
        max_tokens=budget,
        counter=counter,
        strategy=CompositeStrategy(
            CompressionStrategy(compressor=_SAFE),
            DeduplicateStrategy(),
            DropOldestStrategy(keep_last=keep_last, protected_roles={"pinned"}),
        ),
    )
    opt_input = [
        {"role": "pinned" if m.get("pin") else m["role"], "content": m["content"]}
        for m in history
    ]
    result = optimizer.optimize_detailed(opt_input)
    outgoing = [
        {
            "role": "user" if m["role"] == "pinned" else m["role"],
            "content": m["content"],
        }
        for m in result.messages
    ]
    return result, outgoing


def run_chat(args: argparse.Namespace) -> int:
    prov = resolve(args.provider)
    model = args.model or prov.default_model
    count_model = args.count_model or prov.count_model
    key = os.environ.get(prov.key_env, "sk-none")

    _setup_readline()
    client = OpenAI(api_key=key, base_url=prov.base_url)
    counter = get_counter(count_model, prefer_exact=False)
    history: list[dict] = []
    budget = args.budget
    keep_last = args.keep_last

    print(
        f"{BOLD}stochat{RESET}  provider={prov.name}  model={model}  "
        f"budget={budget} tok"
    )
    print(
        f"{DIM}mention a file/dir to attach (pinned) · Tab completes · ↑ recalls · "
        f"/budget N · /model M · /tokens · /clear · /exit{RESET}\n"
    )

    while True:
        try:
            line = input(f"{CYAN}you› {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            print("bye")
            return 0
        if line == "/clear":
            history.clear()
            print(f"{DIM}(cleared){RESET}")
            continue
        if line == "/tokens":
            clean = [{"role": m["role"], "content": m["content"]} for m in history]
            print(f"{DIM}conversation: {counter.count_messages(clean)} tok{RESET}")
            continue
        if line.startswith("/budget "):
            budget = int(line.split()[1])
            print(f"{DIM}(budget = {budget} tok){RESET}")
            continue
        if line.startswith("/model "):
            model = line.split(maxsplit=1)[1].strip()
            print(f"{DIM}(model = {model}){RESET}")
            continue

        content, attached = _attach_paths(line)
        if attached:
            print(f"{DIM}(attached: {', '.join(attached)} — pinned){RESET}")
        history.append({"role": "user", "content": content, "pin": bool(attached)})

        result, outgoing = _optimize(history, counter, budget, keep_last)
        note = ""
        if not result.within_budget:
            note = f"  {YELLOW}⚠ pinned content exceeds budget — kept it{RESET}{DIM}"
        print(
            f"{DIM}[sto] {result.original_tokens}→{result.optimized_tokens} tok "
            f"(saved {result.tokens_saved}; budget {budget}){note}{RESET}"
        )

        print(f"{BOLD}ai›{RESET} ", end="", flush=True)
        parts: list[str] = []
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=args.reply_tokens,
                messages=outgoing,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    print(delta, end="", flush=True)
        except Exception as exc:  # surface any provider error to the user
            print(f"\n{DIM}[error] {exc}{RESET}")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": "".join(parts), "pin": False})


def list_providers(_: argparse.Namespace) -> int:
    print(f"{BOLD}Available providers{RESET} (use --provider NAME):\n")
    for p in PROVIDERS.values():
        keyset = "set" if os.environ.get(p.key_env) else "unset"
        print(f"  {BOLD}{p.name:11}{RESET} {p.note}")
        print(
            f"  {DIM}{'':11} key: {p.key_env} ({keyset}) · default: {p.default_model}{RESET}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stochat", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    chat = sub.add_parser("chat", help="start the interactive chat (default)")
    _add_chat_args(chat)
    chat.set_defaults(func=run_chat)

    provs = sub.add_parser("providers", help="list available providers")
    provs.set_defaults(func=list_providers)

    _add_chat_args(parser)  # allow bare `stochat` with chat flags
    parser.set_defaults(func=run_chat)
    return parser


def _add_chat_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider", help="ollama | openai | groq | openrouter | anthropic | together"
    )
    p.add_argument("--model", help="override the provider's default model")
    p.add_argument("--count-model", help="model id used for token counting only")
    p.add_argument("--budget", type=int, default=8000, help="prompt token budget")
    p.add_argument(
        "--keep-last", type=int, default=2, help="recent turns never dropped"
    )
    p.add_argument("--reply-tokens", type=int, default=1000, help="max reply length")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
