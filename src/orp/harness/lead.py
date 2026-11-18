"""The lead: plain code that runs one dream for one user.

It never reads a transcript or writes a snippet itself. It picks the sessions
that are ready, runs one analyzer per session in groups (analyzers inside a
group run in parallel; each group finishes before the next starts, so later
analyzers see earlier writes), then runs the organizer once.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..llm import LLM
from ..schema import field_guide
from ..store import Store
from ..transcripts import paginate, render_events
from .agent import run_agent
from .tools import AnalyzerTools, OrganizerTools, Resolver

PROMPTS = Path(__file__).parent / "prompts"
GROUP_SIZE = 20
PAGE_CHARS = 24000
# A transcript with less text than this cannot hold a preference worth a model call.
MIN_CHARS = 200
# Tool inputs and results longer than this are shortened for the analyzer. What
# the person wants shows in their own words, which are never shortened.
TOOL_CHARS = 1500


def analyzer_system() -> str:
    return (PROMPTS / "analyzer_system.md").read_text().replace("{field_guide}", field_guide())


def organizer_system() -> str:
    return (PROMPTS / "organizer_system.md").read_text()


def analyze_session(
    store: Store,
    llm: LLM,
    user_id: str,
    session: dict,
    resolver: Resolver | None = None,
    max_turns: int = 40,
) -> dict:
    """Run one analyzer over the part of one session not yet dreamed over."""
    session_id = session["id"]
    upto = session["next_offset"]
    # A subagent's transcript is stored, but its "user" is the parent agent, not
    # the person, so nothing in it is the person's preference.
    if session["meta"].get("parent_session_id"):
        return {"input": session_id, "ok": True, "upto": upto, "skipped": "subagent transcript", "written": []}
    events = store.get_events(user_id, session_id, 0, upto)
    full_text = render_events(events, session["client"], TOOL_CHARS)
    new_text = render_events(events, session["client"], TOOL_CHARS, start=session["dreamed_offset"])
    if len(new_text.strip()) < MIN_CHARS:
        return {"input": session_id, "ok": True, "upto": upto, "skipped": "too short", "written": []}

    pages = paginate(new_text, PAGE_CHARS)
    # Quotes are checked against the whole session, so a quote that sits in an
    # earlier part of the same session is still accepted.
    tools = AnalyzerTools(store, user_id, session_id, pages, full_text, resolver)
    task = (
        f"Read the transcript ({len(pages)} page{'s' if len(pages) != 1 else ''}) and record what it shows about "
        "how this person refers to things and which records they want when they search. Other analyzers are "
        "working on this person's other transcripts at the same time, so check what is already stored before "
        "writing. When done, call finish with a short summary of what you added."
    )
    result = run_agent(llm, analyzer_system(), task, tools.tools(), max_turns=max_turns)
    if not result.finished:
        return {
            "input": session_id,
            "ok": False,
            "reason": (
                f"analyzer kept resending a failing {result.result['stuck_on']} call: {result.result['error'][:200]}"
                if result.result.get("stuck_on")
                else f"analyzer did not finish within {max_turns} turns"
            ),
            "written": tools.written,
        }
    return {
        "input": session_id,
        "ok": True,
        "upto": upto,
        "summary": result.result.get("summary", ""),
        "written": tools.written,
        "turns": result.turns,
    }


def organize(store: Store, llm: LLM, user_id: str, max_turns: int = 40) -> dict:
    tools = OrganizerTools(store, user_id)
    task = (
        "All transcript analyzers have finished. Tidy this person's snippets: merge duplicates, and supersede "
        "or mark conflicts where two cannot both be true. When done, call finish with the list of changes."
    )
    result = run_agent(llm, organizer_system(), task, tools.tools(), max_turns=max_turns)
    if not result.finished:
        raise RuntimeError(f"organizer did not finish within {max_turns} turns")
    return {"changes": result.result.get("changes", []), "applied": tools.changes}


def dream(
    store: Store,
    user_id: str,
    analyzer_llm: LLM,
    organizer_llm: LLM,
    quiet_minutes: int = 30,
    group_size: int = GROUP_SIZE,
    concurrency: int = 8,
    resolver: Resolver | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Dream over every session of `user_id` that has gone quiet and has new events."""
    quiet_before = (datetime.now(timezone.utc) - timedelta(minutes=quiet_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sessions = store.sessions_to_dream(user_id, quiet_before)
    run_id = store.start_run(user_id)
    results: list[dict] = []

    def one(session: dict) -> dict:
        try:
            return analyze_session(store, analyzer_llm, user_id, session, resolver)
        except Exception as exc:  # one failed analyzer must not stop the run
            return {"input": session["id"], "ok": False, "reason": f"{type(exc).__name__}: {exc}", "written": []}

    for start in range(0, len(sessions), group_size):
        group = sessions[start : start + group_size]
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(group)))) as pool:
            for res in pool.map(one, group):
                # A session moves forward only when its analyzer finished, so a
                # failed one is picked up again by the next run.
                if res["ok"]:
                    store.mark_dreamed(user_id, res["input"], res["upto"])
                results.append(res)
                if on_progress:
                    on_progress(res)

    processed = [r for r in results if r["ok"]]
    analyzed = [r for r in processed if not r.get("skipped")]
    report: dict = {
        "run_id": run_id,
        "inputs_expected": len(sessions),
        "inputs_processed": len(processed),
        "inputs_skipped": len(processed) - len(analyzed),
        "failures": [{"input": r["input"], "reason": r["reason"]} for r in results if not r["ok"]],
        "organized": False,
        "changes": [],
        "analyzers": [
            {k: r[k] for k in ("input", "summary", "written", "turns", "skipped") if k in r} for r in processed
        ],
    }
    # The organizer runs whenever any analyzer wrote something, including one
    # that then failed: its partial writes still need tidying.
    if any(r["written"] for r in results):
        try:
            organized = organize(store, organizer_llm, user_id)
            report["organized"] = True
            report["changes"] = organized["changes"]
            report["changes_applied"] = organized["applied"]
        except Exception as exc:
            report["organizer_error"] = f"{type(exc).__name__}: {exc}"
    report["snippets_in_force"] = len(store.list_snippets(user_id, "active"))
    store.finish_run(run_id, report)
    return report
