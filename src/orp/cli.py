"""Command line. Run `orp <command> --help` for the options of each command.

  orp serve                     run the API
  orp user add <name>           create a user and print their API key once
  orp user list
  orp doctor                    check that every part needed for a test is in place
  orp backfill --user NAME      upload past Claude Code sessions from this machine
  orp status --user NAME        sessions received, what is waiting to be dreamed over, snippet counts
  orp dream --user NAME         run the dreaming harness
  orp snippets --user NAME      show the snippets
  orp runs --user NAME          show what the last dream did, step by step
  orp render --user NAME --session ID    print a session as the analyzer sees it
  orp reset --user NAME --yes   forget what was learned for a user (keeps transcripts)

Settings are read from the environment, and from a `.env` file in the current
directory when there is one (see `.env.example`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .store import Store

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "hooks" / "claude_code_upload.py"
KEY_FILE = Path.home() / ".orp" / "key"


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines into the environment. Existing variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _store() -> Store:
    return Store(os.environ.get("ORP_DB", "data/orp.db"))


def _user_id(store: Store, name: str) -> str:
    user_id = store.user_id_by_name(name)
    if not user_id:
        sys.exit(f"no user named {name!r}. Existing users: {[u['name'] for u in store.list_users()]}")
    return user_id


def _api_url() -> str:
    return os.environ.get("ORP_API_URL", "http://localhost:8765").rstrip("/")


def _api_key() -> str | None:
    return os.environ.get("ORP_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.is_file() else None)


def _llms():
    from .llm import OpenAICompatLLM

    for var in ("ORP_LLM_BASE_URL", "ORP_ANALYZER_MODEL"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set. Put it in .env (see .env.example), or run `orp doctor`.")
    # Sampling the Qwen3.6 model card recommends for precise work with thinking on.
    sampling = {"temperature": 0.6, "max_tokens": 8192, "extra_body": {"top_p": 0.95, "top_k": 20}}
    analyzer = OpenAICompatLLM(os.environ["ORP_ANALYZER_MODEL"], **sampling)
    organizer = OpenAICompatLLM(os.environ.get("ORP_ORGANIZER_MODEL") or os.environ["ORP_ANALYZER_MODEL"], **sampling)
    return analyzer, organizer


# ---- formatting -------------------------------------------------------------


def _snippet_line(s: dict) -> str:
    triggers = ", ".join(f'"{t}"' for t in s["trigger_terms"]) or "(always on)"
    if s["kind"] == "alias":
        targets = ", ".join(t["text"] + (f" [{t['entity_type']}]" if t.get("entity_type") else "") for t in s["targets"])
        arrow = "=" if s.get("direction") == "both" else "->"
        body = f"{triggers} {arrow} {targets}"
    else:
        body = f"{triggers} -> {s['field']} {s['op']} {json.dumps(s['value'], ensure_ascii=False)}  [{s['strength']}]"
    sessions = len({e["session_id"] for e in s["evidence"]})
    tail = f"{s['basis']}, {len(s['evidence'])} quote{'s' if len(s['evidence']) != 1 else ''} from {sessions} session{'s' if sessions != 1 else ''}"
    extra = ""
    if s["status"] == "superseded" and s.get("superseded_by"):
        extra = f"  (replaced by {s['superseded_by']}: {s.get('note') or ''})"
    if s["status"] == "conflict":
        extra = f"  (conflicts with {s.get('conflict_with')}: {s.get('note') or ''})"
    return f"{s['id']}  {s['status']:<10} {s['kind']:<10} {body}\n{'':14}{s['statement']}  ({tail}){extra}"


def print_snippets(snippets: list[dict], evidence: bool) -> None:
    if not snippets:
        print("no snippets")
        return
    for s in snippets:
        print(_snippet_line(s))
        if evidence:
            for e in s["evidence"]:
                when = f" at {e['said_at']}" if e.get("said_at") else ""
                print(f"{'':14}> \"{e['quote']}\"  ({e['session_id']}{when})")
        print()


def print_report(report: dict) -> None:
    print(
        f"run {report['run_id']}: {report['inputs_processed']} of {report['inputs_expected']} sessions processed "
        f"({report['inputs_skipped']} skipped without a model call), {len(report['failures'])} failed, "
        f"{report['snippets_in_force']} snippets in force"
    )
    for a in report["analyzers"]:
        if a.get("skipped"):
            continue
        print(f"\n  {a['input']}  ({a.get('pages', '?')} pages, {a.get('turns', '?')} turns, {len(a['written'])} written)")
        print(f"    {a.get('summary', '')}")
        _print_trace(a.get("trace", []))
    for f in report["failures"]:
        print(f"\n  FAILED {f['input']}: {f['reason']}")
        _print_trace(f.get("trace", []))
    skipped: dict[str, int] = {}
    for a in report["analyzers"]:
        if a.get("skipped"):
            skipped[a["skipped"]] = skipped.get(a["skipped"], 0) + 1
    if skipped:
        print("\n  skipped: " + ", ".join(f"{n} {why}" for why, n in skipped.items()))
    if report.get("organizer_error"):
        print(f"\n  ORGANIZER FAILED: {report['organizer_error']}")
    elif report["organized"]:
        print("\n  organizer:")
        for change in report["changes"] or ["no changes needed"]:
            print(f"    - {change}")
        _print_trace(report.get("organizer_trace", []))
    else:
        print("\n  organizer did not run (no analyzer wrote anything)")
    if report.get("llm_usage"):
        print(f"\n  model usage: {json.dumps(report['llm_usage'])}")


def _print_trace(trace: list[dict]) -> None:
    for step in trace:
        if step["tool"] in ("read_transcript", "list_snippets", "finish") and step["ok"]:
            continue
        detail = step.get("statement") or " ".join(f"{k}={v}" for k, v in step.items() if k not in ("tool", "ok", "error"))
        mark = "ok " if step["ok"] else "ERR"
        print(f"      {mark} {step['tool']}: {detail}" + (f"\n          {step['error']}" if not step["ok"] else ""))


# ---- commands ---------------------------------------------------------------


def cmd_status(store: Store, name: str) -> None:
    user_id = _user_id(store, name)
    sessions = store.list_sessions(user_id)
    main = [s for s in sessions if not s["meta"].get("parent_session_id")]
    waiting = [s for s in main if s["next_offset"] > s["dreamed_offset"]]
    events = sum(s["next_offset"] for s in sessions)
    print(f"user {name}: {len(main)} sessions (+{len(sessions) - len(main)} subagent transcripts), {events} events stored")
    print(f"waiting to be dreamed over: {len(waiting)} sessions")
    for s in waiting[-15:]:
        new = s["next_offset"] - s["dreamed_offset"]
        print(f"  {s['id']}  {new} new events, last event {s['last_event_at']}  {s['meta'].get('cwd', '')}")
    if len(waiting) > 15:
        print(f"  ... and {len(waiting) - 15} older")
    counts: dict[str, int] = {}
    for s in store.list_snippets(user_id):
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print("snippets: " + (", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "none"))
    print(f"unmapped notes: {len(store.list_unmapped(user_id))}")
    runs = store.list_runs(user_id)
    if runs:
        last = runs[-1]
        state = "finished " + last["finished_at"] if last["finished_at"] else "did not finish"
        print(f"dream runs: {len(runs)}, last started {last['started_at']} ({state})")


def cmd_backfill(name: str, directory: str, days: int, dry_run: bool) -> None:
    store = _store()
    _user_id(store, name)  # fail early on an unknown user
    key = _api_key()
    if not key:
        sys.exit(f"no API key: set ORP_API_KEY or put the key in {KEY_FILE}")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    root = Path(directory).expanduser()
    # Only top-level session files: the hook uploads each session's subagent files itself.
    files = sorted(
        (p for p in root.glob("*/*.jsonl") if p.stat().st_mtime >= cutoff), key=lambda p: p.stat().st_mtime
    )
    if not files:
        sys.exit(f"no session files changed in the last {days} days under {root}")
    print(f"{len(files)} session files changed in the last {days} days under {root}")
    if dry_run:
        for p in files:
            print(f"  {p}  ({p.stat().st_size // 1024} KB)")
        return
    env = {**os.environ, "ORP_API_URL": _api_url(), "ORP_API_KEY": key}
    for i, p in enumerate(files, 1):
        out = subprocess.run([sys.executable, str(HOOK), "--file", str(p)], env=env, capture_output=True, text=True)
        print(f"  [{i}/{len(files)}] {p.name}: {(out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr).strip() else 'nothing new'}")
    print("done. `orp status --user " + name + "` shows what is waiting.")


def cmd_doctor(user: str | None) -> None:
    import httpx

    ok_all = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        print(f"{'ok  ' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail else ""))

    db = os.environ.get("ORP_DB", "data/orp.db")
    store = _store()
    users = store.list_users()
    check("database", True, f"{db}, users: {[u['name'] for u in users] or 'none (run `orp user add <name>`)'}")

    url = _api_url()
    try:
        up = httpx.get(f"{url}/healthz", timeout=3).status_code == 200
    except httpx.HTTPError:
        up = False
    check("API running", up, url if up else f"nothing answers at {url} (run `orp serve`)")

    key = _api_key()
    if up and key:
        try:
            me = httpx.get(f"{url}/v1/me", headers={"Authorization": f"Bearer {key}"}, timeout=3)
            who = next((u["name"] for u in users if u["id"] == me.json().get("user_id")), None) if me.status_code == 200 else None
            check("API key", me.status_code == 200, f"belongs to user {who!r}" if who else "the API does not accept it")
            if user and who and who != user:
                check("key matches --user", False, f"the key belongs to {who!r}, not {user!r}")
        except httpx.HTTPError as exc:
            check("API key", False, str(exc))
    else:
        check("API key", bool(key), "found" if key else f"set ORP_API_KEY or create {KEY_FILE}")

    settings = Path.home() / ".claude" / "settings.json"
    try:
        hooks = json.loads(settings.read_text()).get("hooks", {})
        installed = [e for e in ("Stop", "SubagentStop", "PreCompact", "SessionEnd") if "upload" in json.dumps(hooks.get(e, []))]
    except (OSError, ValueError):
        installed = []
    check("Claude Code hook", "Stop" in installed, f"registered for {installed}" if installed else "not registered (see hooks/README.md)")

    base, model = os.environ.get("ORP_LLM_BASE_URL"), os.environ.get("ORP_ANALYZER_MODEL")
    if not base or not model:
        check("model endpoint", False, "ORP_LLM_BASE_URL and ORP_ANALYZER_MODEL are not set (see .env.example)")
    else:
        headers = {"Authorization": f"Bearer {os.environ.get('ORP_LLM_API_KEY', '')}", "User-Agent": "orp-harness/0.1"}
        try:
            r = httpx.get(f"{base.rstrip('/')}/models", headers=headers, timeout=15)
            names = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
            check("model endpoint", r.status_code == 200, base if r.status_code == 200 else f"{base} answered {r.status_code}")
            if r.status_code == 200:
                check("model served", model in names, model if model in names else f"{model!r} not in {names}")
                if model in names:
                    from .llm import OpenAICompatLLM

                    tool = {"type": "function", "function": {"name": "finish", "description": "Call this.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}}
                    msg = OpenAICompatLLM(model, max_tokens=2048).chat(
                        [{"role": "user", "content": "Call the finish tool with summary 'ready'."}], [tool]
                    )
                    called = bool(msg.get("tool_calls"))
                    check("model can call tools", called, "" if called else "the model answered in text; check the server's tool-call parser flags")
        except Exception as exc:  # any failure here is the answer
            check("model endpoint", False, f"{type(exc).__name__}: {exc}")
    print("\nall checks passed" if ok_all else "\nsome checks failed")
    sys.exit(0 if ok_all else 1)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="orp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    user = sub.add_parser("user", help="create or list users")
    user_sub = user.add_subparsers(dest="user_cmd", required=True)
    user_sub.add_parser("add").add_argument("name")
    user_sub.add_parser("list")

    doctor = sub.add_parser("doctor", help="check that every part needed for a test is in place")
    doctor.add_argument("--user")

    backfill = sub.add_parser("backfill", help="upload past Claude Code sessions from this machine")
    backfill.add_argument("--user", required=True, help="must be the user the API key belongs to")
    backfill.add_argument("--dir", default="~/.claude/projects", help="folder holding one folder per project")
    backfill.add_argument("--days", type=int, default=7, help="only sessions changed in the last N days")
    backfill.add_argument("--dry-run", action="store_true", help="list the files without uploading")

    status = sub.add_parser("status", help="what has been received and what is waiting")
    status.add_argument("--user", required=True)

    dream = sub.add_parser("dream", help="run the dreaming harness")
    who = dream.add_mutually_exclusive_group(required=True)
    who.add_argument("--user")
    who.add_argument("--all", action="store_true")
    dream.add_argument("--quiet-minutes", type=int, default=30, help="skip sessions with events newer than this (0 = take everything)")
    dream.add_argument("--concurrency", type=int, default=8)
    dream.add_argument("--session", action="append", help="only this session id (repeatable)")
    dream.add_argument("--limit", type=int, help="at most this many sessions, oldest first")
    dream.add_argument("--json", action="store_true", help="print the full report as JSON")

    snippets = sub.add_parser("snippets", help="show the snippets")
    snippets.add_argument("--user", required=True)
    snippets.add_argument("--status", default="active", help="active, superseded, conflict or all")
    snippets.add_argument("--evidence", action="store_true", help="show the quotes behind each snippet")
    snippets.add_argument("--json", action="store_true")

    runs = sub.add_parser("runs", help="show what the last dream did")
    runs.add_argument("--user", required=True)
    runs.add_argument("--json", action="store_true")

    render = sub.add_parser("render", help="print a session as the analyzer sees it")
    render.add_argument("--user", required=True)
    render.add_argument("--session", required=True)

    reset = sub.add_parser("reset", help="forget what was learned for a user; transcripts are kept")
    reset.add_argument("--user", required=True)
    reset.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("orp.api:app_factory", factory=True, host=args.host, port=args.port)
        return
    if args.cmd == "doctor":
        cmd_doctor(args.user)
        return
    if args.cmd == "backfill":
        cmd_backfill(args.user, args.dir, args.days, args.dry_run)
        return

    store = _store()

    if args.cmd == "user":
        if args.user_cmd == "add":
            user_id, key = store.create_user(args.name)
            print(json.dumps({"user_id": user_id, "name": args.name, "api_key": key}, indent=2))
            print("Store this key now; it is not shown again.", file=sys.stderr)
        else:
            print(json.dumps(store.list_users(), indent=2))

    elif args.cmd == "status":
        cmd_status(store, args.user)

    elif args.cmd == "dream":
        from .harness.lead import dream as run_dream

        analyzer, organizer = _llms()
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
                only_sessions=args.session,
                limit=args.limit,
                on_progress=lambda r: print(
                    f"  {r['input']}: {('skipped, ' + r['skipped']) if r.get('skipped') else 'ok' if r['ok'] else 'FAILED ' + r.get('reason', '')}",
                    file=sys.stderr,
                ),
            )
            report["user"] = u["name"]
            report["llm_usage"] = {"analyzer": dict(analyzer.usage), "organizer": dict(organizer.usage)}
            store.finish_run(report["run_id"], report)
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print_report(report)
            failed = failed or bool(report["failures"]) or "organizer_error" in report
        sys.exit(1 if failed else 0)

    elif args.cmd == "snippets":
        status_filter = None if args.status == "all" else args.status
        found = store.list_snippets(_user_id(store, args.user), status_filter)
        if args.json:
            print(json.dumps(found, indent=2, ensure_ascii=False))
        else:
            print_snippets(found, args.evidence)
            unmapped = store.list_unmapped(_user_id(store, args.user))
            if unmapped:
                print("unmapped notes (not used by search):")
                for note in unmapped:
                    print(f"  - {note['statement']}")

    elif args.cmd == "runs":
        all_runs = [r for r in store.list_runs(_user_id(store, args.user)) if r["report"]]
        if not all_runs:
            sys.exit("no finished dream runs for this user")
        if args.json:
            print(json.dumps(all_runs[-1]["report"], indent=2, ensure_ascii=False))
        else:
            print(f"started {all_runs[-1]['started_at']}, finished {all_runs[-1]['finished_at']}")
            print_report(all_runs[-1]["report"])

    elif args.cmd == "render":
        from .transcripts import render_events

        user_id = _user_id(store, args.user)
        session = store.get_session(user_id, args.session)
        if not session:
            sys.exit(f"no session {args.session!r} for that user")
        print(render_events(store.get_events(user_id, args.session), session["client"]))

    elif args.cmd == "reset":
        user_id = _user_id(store, args.user)
        if not args.yes:
            sys.exit(
                f"this deletes every snippet, unmapped note and run report of user {args.user!r} and marks all "
                "their sessions as not dreamed over. Transcripts are kept. Add --yes to do it."
            )
        print(json.dumps(store.reset_learning(user_id)))


if __name__ == "__main__":
    main()
