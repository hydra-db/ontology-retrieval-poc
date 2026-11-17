"""Explicit read-only fetch/poll jobs, deliberately separate from query imports."""

import argparse
import json

from .adapters import fetch_fireflies, fetch_github, fetch_linear, normalize, poll_research
from .store import SentenceEncoder, Store


def main():
    parser = argparse.ArgumentParser(description="Opt-in bounded read-only source job; no write-back")
    parser.add_argument("--db", default="poc.db")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--model", help="same local model used to create this database")
    commands = parser.add_subparsers(dest="source", required=True)
    linear = commands.add_parser("linear")
    linear.add_argument("issue_id")
    github = commands.add_parser("github")
    github.add_argument("repo")
    github.add_argument("number", type=int)
    fireflies = commands.add_parser("fireflies")
    fireflies.add_argument("transcript_id")
    research = commands.add_parser("research")
    research.add_argument("--topic", action="append", required=True, help="arXiv query syntax; repeated at most 20 times")
    args = parser.parse_args()
    store = Store(args.db, SentenceEncoder(args.model) if args.model else None)
    if args.source == "linear":
        payload = fetch_linear(args.issue_id)
    elif args.source == "github":
        payload = fetch_github(args.repo, args.number)
    elif args.source == "fireflies":
        payload = fetch_fireflies(args.transcript_id)
    else:
        new = poll_research(args.db, args.topic)
        papers = [json.loads(row[0]) for row in store.db.execute("SELECT payload FROM research_poll_inbox ORDER BY paper_id")]
        nodes, edges = normalize("arxiv", {"papers": papers})
        store.ingest(args.tenant, nodes, edges)
        print(json.dumps({"enqueued": len(new), "replayed_public_inbox": len(papers), "nodes": len(nodes)}))
        store.close()
        return
    nodes, edges = normalize(args.source, payload)
    store.ingest(args.tenant, nodes, edges)
    print(json.dumps({"source": args.source, "nodes": len(nodes), "edges": len(edges)}))
    store.close()


if __name__ == "__main__":
    main()
