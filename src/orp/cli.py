"""Command line.

  orp serve [--host 127.0.0.1] [--port 8765]
  orp user add <name>                 create a user and print their API key once
  orp user list
  orp dream [--user NAME | --all] [--quiet-minutes 30]
  orp snippets --user NAME [--status active|superseded|conflict|all]
  orp render --user NAME --session ID   print a session as the analyzer sees it
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .store import Store


def _store() -> Store:
    return Store(os.environ.get("ORP_DB", "data/orp.db"))


def _user_id(store: Store, name: str) -> str:
    user_id = store.user_id_by_name(name)
    if not user_id:
        sys.exit(f"no user named {name!r}")
    return user_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    user = sub.add_parser("user")
    user_sub = user.add_subparsers(dest="user_cmd", required=True)
    user_sub.add_parser("add").add_argument("name")
    user_sub.add_parser("list")

    dream = sub.add_parser("dream")
    who = dream.add_mutually_exclusive_group(required=True)
    who.add_argument("--user")
    who.add_argument("--all", action="store_true")
    dream.add_argument("--quiet-minutes", type=int, default=30)
    dream.add_argument("--concurrency", type=int, default=8)

    snippets = sub.add_parser("snippets")
    snippets.add_argument("--user", required=True)
    snippets.add_argument("--status", default="active")

    render = sub.add_parser("render")
    render.add_argument("--user", required=True)
    render.add_argument("--session", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("orp.api:app_factory", factory=True, host=args.host, port=args.port)
        return

    store = _store()

    if args.cmd == "user":
        if args.user_cmd == "add":
            user_id, key = store.create_user(args.name)
            print(json.dumps({"user_id": user_id, "name": args.name, "api_key": key}, indent=2))
            print("Store this key now; it is not shown again.", file=sys.stderr)
        else:
            print(json.dumps(store.list_users(), indent=2))
        return

    if args.cmd == "dream":
        from .harness.lead import dream as run_dream
        from .llm import OpenAICompatLLM

        # Sampling the Qwen3.6 model card recommends for precise work with thinking on.
        sampling = {"temperature": 0.6, "max_tokens": 8192, "extra_body": {"top_p": 0.95, "top_k": 20}}
        analyzer = OpenAICompatLLM(os.environ["ORP_ANALYZER_MODEL"], **sampling)
        organizer = OpenAICompatLLM(
            os.environ.get("ORP_ORGANIZER_MODEL") or os.environ["ORP_ANALYZER_MODEL"], **sampling
        )
        users = store.list_users() if args.all else [{"id": _user_id(store, args.user), "name": args.user}]
        failed = False
        for u in users:
            report = run_dream(
                store,
                u["id"],
                analyzer,
                organizer,
                quiet_minutes=args.quiet_minutes,
                concurrency=args.concurrency,
                on_progress=lambda r: print(
                    f"  {r['input']}: {'ok' if r['ok'] else 'FAILED ' + r.get('reason', '')}", file=sys.stderr
                ),
            )
            report["user"] = u["name"]
            report["llm_usage"] = {"analyzer": analyzer.usage, "organizer": organizer.usage}
            print(json.dumps(report, indent=2, ensure_ascii=False))
            failed = failed or bool(report["failures"]) or "organizer_error" in report
        sys.exit(1 if failed else 0)

    if args.cmd == "snippets":
        status = None if args.status == "all" else args.status
        print(json.dumps(store.list_snippets(_user_id(store, args.user), status), indent=2, ensure_ascii=False))
        return

    if args.cmd == "render":
        from .transcripts import render_events

        user_id = _user_id(store, args.user)
        session = store.get_session(user_id, args.session)
        if not session:
            sys.exit(f"no session {args.session!r} for that user")
        print(render_events(store.get_events(user_id, args.session), session["client"]))


if __name__ == "__main__":
    main()
