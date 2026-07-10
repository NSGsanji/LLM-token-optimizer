"""Example: credential rotation, failover and provider routing.

Run with:  python examples/04_credentials_and_routing.py

No real network calls are made — a fake ``call_provider`` stands in for an SDK.
"""

from __future__ import annotations

from smarttokenoptimizer.credentials import CredentialPool
from smarttokenoptimizer.routing import CheapestPolicy, Provider, Router


def call_provider(provider: str, api_key: str) -> str:
    # Pretend to call the provider. Keys are never printed in full.
    return f"response from {provider}"


def main() -> None:
    # Two providers serving gpt-4o at different prices, each with two keys.
    openai = Provider(
        "openai",
        pool=CredentialPool(),
        models=["gpt-4o"],
        price_hint=12.5,
    )
    openrouter = Provider(
        "openrouter",
        pool=CredentialPool(),
        models=["gpt-4o"],
        price_hint=6.0,
    )
    openai.pool.add_key("sk-openai-key-1", provider="openai")
    openai.pool.add_key("sk-openai-key-2", provider="openai")
    openrouter.pool.add_key("sk-openrouter-key-1", provider="openrouter")

    # Cheapest policy prefers openrouter (6.0 < 12.5).
    router = Router([openai, openrouter], policy=CheapestPolicy())

    with router.dispatch(model="gpt-4o") as route:
        print(f"Routed to cheapest provider: {route.provider.name}")
        print(f"Using credential: {route.credential.masked_key}")
        print(call_provider(route.provider.name, route.key))

    # Simulate openrouter hitting a rate limit; the router fails over to openai.
    openrouter.pool.record_rate_limited(openrouter.pool.ids()[0], retry_after=60)
    with router.dispatch(model="gpt-4o") as route:
        print(f"After rate-limit, routed to: {route.provider.name}")

    print("\nProvider health:")
    for provider in (openai, openrouter):
        for h in provider.pool.health():
            print(
                f"  {provider.name:11} {h.id}  available={h.available}  "
                f"uses={h.uses}  rate_limited={h.rate_limited}"
            )


if __name__ == "__main__":
    main()
