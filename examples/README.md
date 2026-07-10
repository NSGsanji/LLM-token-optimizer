# Examples

Runnable, self-contained examples for SmartTokenOptimizer. None of them make
real network calls — provider SDKs are stubbed — so they run offline with only
the zero-dependency core installed.

```bash
pip install -e .          # from the repo root
python examples/01_token_counting.py
```

| Script | Shows |
| ------ | ----- |
| [`01_token_counting.py`](01_token_counting.py) | Counting tokens for strings and chat conversations. |
| [`02_optimize_conversation.py`](02_optimize_conversation.py) | Fitting a long conversation into a token budget with a composed strategy pipeline. |
| [`03_cost_and_cache.py`](03_cost_and_cache.py) | Estimating cost and caching responses, with usage analytics. |
| [`04_credentials_and_routing.py`](04_credentials_and_routing.py) | Credential rotation, rate-limit failover and cheapest-provider routing. |
| [`05_full_pipeline.py`](05_full_pipeline.py) | The end-to-end `OptimizingPipeline` combining every module. |

Each script has a `main()` and runs to completion with exit code `0`; they are
also exercised by the test suite (`tests/test_examples.py`) so they stay
working.
