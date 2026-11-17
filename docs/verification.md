# Verification

The complete suite passes with downloaded local CPU models enabled. It includes adapter/mock pagination and HTTP bounds, polling rollback/checkpoint/dedup, deterministic schema relations, full chunk reconstruction, model mismatch rejection, authorship bypass, CLI model-load bypass, socket-blocked lexical/neural queries, a fixture command implementing the background JSON protocol (not a real LLM), recency/contradictions, evidence/deletion/replay, name collisions, and cross-user/tenant isolation.

Final public-safe tree run: **55 tests passed** in 3.903 seconds, with no skipped neural test. Syntax compilation and dependency consistency checks also passed.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
POC_TEST_MODELS=1 .venv/bin/python -m unittest discover -s tests -v
python -m compileall -q poc tests
uv pip check --python .venv/bin/python
```

The synthetic fixture has **19 nodes, 15 edges, 10 text chunks and one active preference**. These measurements are a small functional experiment, not a scaling benchmark.

## Routing and retrieval

All three backends score **27/30 (90%)** on the same predeclared synthetic base-routing challenge set. The three misses are deliberate disclosed coverage gaps: “Tell me who opened DEMO-123”, “Who is the author of DEMO-123?”, and “Show publications on user memory”. They abstain rather than guessing. No router training or tuning against these results was performed. The other eleven end-to-end retrieval assertions pass on each backend, including actual personalized retrieval, no-memory fallback for another user, missing-entity refusal, ontology bypass and kind filtering. They are assertions on synthetic data, not relevance/recall estimates for customers.

The zero-dependency mode is lexical. Real neural measurements use all-MiniLM-L6-v2, with optional MS MARCO MiniLM-L6 cross-encoder; exact commits and dependency versions are pinned in the repository. Query-path socket blocking verifies offline operation even when the model objects are constructed. No real background LLM, authenticated live API, production source or hosted HydraDB instance was exercised.

## Reproduce latency

```sh
python -m poc.benchmark --output docs/evaluation-lexical.json
.venv/bin/python -m poc.benchmark --model models/encoder --output docs/evaluation-neural.json
.venv/bin/python -m poc.benchmark --model models/encoder --reranker models/reranker --output docs/evaluation-reranked.json
```

The JSON files contain first-of-route latency, twenty warm samples summarized by median/p95, initialization, and three fresh-process measurements. One CPU inference thread is configured, no concurrency or network is included, models are already downloaded, and OS page caches aren't flushed. Fresh-process wall time includes interpreter/import/model initialization; it must not be confused with a warm request. Exact authorship skips neural model loading even when model flags are supplied. On this small fixture, the cross-encoder adds substantial latency; it is optional, and no evidence here establishes a production relevance gain.

| Personalized path | First request (ms) | Warm median / p95 (ms) | Fresh-process median wall (ms) |
|---|---:|---:|---:|
| Lexical baseline | 0.119 | 0.069 / 0.079 | 52.105 |
| Dense MiniLM | 9.660 | 8.996 / 9.995 | 2517.887 |
| Dense + cross-encoder | 87.304 | 93.247 / 98.046 | 2657.679 |

Ontology-only authorship has a warm median of roughly 0.031 ms in all modes. Fresh-process timings vary with uncontrolled filesystem caches and host load. See the raw three samples rather than inferring startup differences from a tiny sample.

`docs/demo-results.json` contains actual source-grounded neural/reranked outputs and route traces for five demos. The wrong-user case may still retrieve shared tenant content, but it cannot use Alice's private alias; retrieval candidates are not generated factual answers.
