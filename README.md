# Ontology-aware retrieval POC

A single local reference for deterministic typed source structure, linked text retrieval, asynchronous user-alias learning, and a closed read-only router. All supplied data is synthetic. No production sources or hosted HydraDB database are accessed by the demo.

## Run without credentials or dependencies

From the repository root, with Python 3.12+:

```sh
python -m unittest discover -s tests -v
python -m poc.cli --db demo.db fixture
python -m poc.cli --db demo.db query 'Who authored Linear ticket DEMO123?'
python -m poc.cli --db demo.db query 'Search provenance in DEMO-123'
python -m poc.cli --db demo.db query 'What is turbolay?'
python -m poc.cli --db demo.db --user bob query 'What is turbolay?'
python -m poc.cli --db demo.db query 'Search hybrid retrieval' --no-personalization
python -m poc.cli --db demo.db evaluate
```

The default `lexical-cosine-v1` is a **lexical baseline, not vector/semantic search**. Query output contains grounded chunks/node relations, source URLs, offsets, strategy trace, logical stores and measured query latency. Exact authorship accesses only the ontology. Unsupported questions abstain; this is retrieval, not a general question-answering agent.

## Real local embeddings and optional reranking

```sh
uv venv
uv pip install --python .venv/bin/python torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-models.lock
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python -m poc.download_models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m poc.cli --db neural.db --model models/encoder fixture
.venv/bin/python -m poc.cli --db neural.db --model models/encoder --reranker models/reranker query 'What is turbolay?'
POC_TEST_MODELS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m poc.cli --db neural.db --model models/encoder --reranker models/reranker evaluate
```

Only the download command accesses Hugging Face. Exact model commits are in `poc/download_models.py`; model manifests are saved locally and embeddings record the same identity. A model mismatch fails closed; rebuild a **new** database when switching backend. Models stay outside git. CPU dependencies/model files need several hundred MB. Latency outputs assume one CPU inference thread; no GPU numbers are borrowed from model cards.

## Ingestion and separate background preference jobs

```sh
python -m poc.cli --db demo.db --tenant demo ingest linear fixtures/linear-export.json
python -m poc.cli --db demo.db --tenant demo --user alice learn fixtures/turns.json
python -m poc.cli --db demo.db --tenant demo --user alice forget --key turbolay
```

The illustrative `linear-export.json` is your own source-shaped export, not a supplied fixture; the four actual payload examples are inside `fixtures/sources.json`. `normalize` supports Linear issues/comments/replies; GitHub PR descriptions/discussion/review replies; Fireflies meetings/sentences/meeting-local speakers; arXiv publications/authors. JSON contracts live in `poc/adapters.py` and tests. Complete supplied bodies and raw source metadata are retained, with stable node/chunk links. arXiv supplies abstracts, not full PDFs.

`learn` defaults to the clearly labeled explicit-alias deterministic extractor. To use your own LLM client, set an environment variable `POC_EXTRACTOR_COMMAND` to a trusted executable command and pass `learn ... --llm-command-env POC_EXTRACTOR_COMMAND`. The client receives one JSON object on stdin (`instruction`, `untrusted_transcript`) and must write only a JSON list of `kind,key,value,quote,confidence` to stdout. It must handle its own environment-held credentials. No shell is used; no extractor is invoked by queries. Only exact supported first-person alias evidence is admitted even with an LLM. Run this separate job after new transcript events arrive; don't place it on the query path.

## Opt-in read-only live adapters

No live source is queried automatically. With consented test data and least-privilege environment credentials:

```sh
python -m poc.sources --db live.db --tenant your-tenant linear ENG-123
python -m poc.sources --db live.db --tenant your-tenant github owner/repo 123
python -m poc.sources --db live.db --tenant your-tenant fireflies explicit-transcript-id
python -m poc.sources --db research.db --tenant demo research --topic 'all:"preference memory"' --topic 'cat:cs.IR'
```

The respective variables are `LINEAR_API_KEY`, `GITHUB_TOKEN`, `FIREFLIES_API_KEY`; arXiv needs no credential. All endpoints are fixed HTTPS reads (GraphQL POSTs contain fixed `query`, never mutations), with timeouts, response limits, redirect refusal and pagination bounds. Re-run the research command manually for polling; there is no scheduler. The durable inbox lets failed ingestion replay safely. Overflow requires narrowing the topic, not silently advancing past missing data. APIs were mock-tested; authenticated live fetches were **not** exercised. Never run broad customer ingestion as a smoke test.

## Evidence and limitations

- [Design and primary-source comparison](docs/design.md) records memory-harness tradeoffs, licensing, injection boundaries, identity rules and future HydraDB mapping.
- `fixtures/heldout.json` is a predeclared synthetic routing corpus, including unseen paraphrases, bypasses, ambiguity, scope, isolation and unsupported queries. Its misses remain visible; it is not a production accuracy estimate.
- `docs/evaluation-*.json` and [verification](docs/verification.md) separate measured lexical/neural/reranked latency and process-cold assumptions. Reproduce with `python -m poc.benchmark` (lexical) or the neural commands documented there.

SQLite brute-force chunk scans are bounded to 10,000 chunks; graph scope to 1,000 nodes/depth three; retrieval/reranking to twenty candidates. There is no ANN index, authentication, per-document ACL, broad implicit preference inference, generated answer, full-paper fetch, or production deployment. Local tenant/user flags assume a trusted operator. Don't serve this as a public or shared multi-user API.
