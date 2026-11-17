import argparse
import json
import os
import shlex
import time
from pathlib import Path

from .adapters import normalize
from .evaluation import evaluate
from .preferences import CommandLLMExtractor, Preferences
from .retrieval import Engine, Router
from .store import CrossEncoderReranker, SentenceEncoder, Store


def main():
    parser = argparse.ArgumentParser(description="Synthetic ontology retrieval POC; explicit local operations only")
    parser.add_argument("--db", default="poc.db")
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--user", default="alice")
    parser.add_argument("--model", help="local pinned SentenceTransformer directory; never downloads during query")
    parser.add_argument("--reranker", help="local CrossEncoder directory")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("fixture")
    demo.add_argument("--sources", default="fixtures/sources.json")
    demo.add_argument("--turns", default="fixtures/turns.json")
    query = commands.add_parser("query")
    query.add_argument("text")
    query.add_argument("--no-personalization", action="store_true")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("source", choices=["linear", "github", "fireflies", "arxiv"])
    ingest.add_argument("file")
    learn = commands.add_parser("learn")
    learn.add_argument("file")
    learn.add_argument("--llm-command-env", help="name of env variable holding trusted background extractor command")
    delete = commands.add_parser("forget")
    delete.add_argument("--key")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--corpus", default="fixtures/heldout.json")
    args = parser.parse_args()
    start = time.perf_counter()
    bypass = args.command in {"learn", "forget"} or (args.command == "query" and Router().decide(args.text).route in {"ontology", "unsupported", "ambiguous"})
    encoder = SentenceEncoder(args.model) if args.model and not bypass else None
    store = Store(args.db, encoder)
    prefs = Preferences(store)
    reranker = CrossEncoderReranker(args.reranker) if args.reranker and not bypass else None
    engine = Engine(store, prefs, reranker)
    initialized_ms = (time.perf_counter()-start)*1000
    if args.command == "fixture":
        sources = json.loads(Path(args.sources).read_text())
        for source, payload in sources.items():
            store.ingest(args.tenant, *normalize(source, payload))
        for turn in json.loads(Path(args.turns).read_text()):
            prefs.add_turn(args.tenant, args.user, turn)
        output = {"learned": prefs.learn(args.tenant, args.user), "mode": "explicit deterministic fixture extraction, not LLM"}
    elif args.command == "ingest":
        nodes, edges = normalize(args.source, json.loads(Path(args.file).read_text()))
        store.ingest(args.tenant, nodes, edges)
        output = {"nodes": len(nodes), "edges": len(edges)}
    elif args.command == "learn":
        for turn in json.loads(Path(args.file).read_text()):
            prefs.add_turn(args.tenant, args.user, turn)
        extractor = CommandLLMExtractor(shlex.split(os.environ[args.llm_command_env])) if args.llm_command_env else None
        output = {"learned": prefs.learn(args.tenant, args.user, extractor)}
    elif args.command == "forget":
        prefs.delete(args.tenant, args.user, args.key)
        output = {"forgotten": args.key or "all user preferences and transcript evidence"}
    elif args.command == "evaluate":
        output = evaluate(engine, Path(args.corpus), args.tenant, args.user)
    else:
        output = engine.query(args.tenant, args.user, args.text, personalized=not args.no_personalization)
    output["initialization_ms"] = round(initialized_ms, 4)
    print(json.dumps(output, indent=2))
    store.close()


if __name__ == "__main__":
    main()
