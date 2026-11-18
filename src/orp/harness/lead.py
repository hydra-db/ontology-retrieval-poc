"""The lead: plain code that runs one dream for one user.

It never reads a transcript or writes a snippet itself. It picks the sessions
that are ready, runs one analyzer per session in groups (analyzers inside a
group run in parallel; each group finishes before the next starts, so later
analyzers see earlier writes), then runs the organizer once.
"""

from __future__ import annotations

import os
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
# What the analyzer is shown of each kind of block. The person's own messages
# are always shown in full; everything else is context, kept short so that a
# day-long session stays readable in a few runs. The stored transcript is not
# affected. Set ORP_ANALYZER_FULL=1 to show the analyzer everything.
ANALYZER_LIMITS = {
    "AGENT THINKING": 0,
    "AGENT": 500,
    "SKILL LOADED": 120,
    "SYSTEM": 200,
    "EVENT": 150,
}


def _view() -> dict:
    """How transcripts are rendered for the analyzer."""
    if os.environ.get("ORP_ANALYZER_FULL") == "1":
        return {}
    # Each run of tool calls and results becomes one line naming the tools used.
    return {"limits": ANALYZER_LIMITS, "collapse_tools": True}


# One analyzer run reads at most this many pages, because every page it reads
# stays in its context. A longer session is read in several runs, one after
# another, each seeing what the earlier ones stored.
PAGES_PER_RUN = 6


def analyzer_system() -> str:
    return (PROMPTS / "analyzer_system.md").read_text().replace("{field_guide}", field_guide())


def organizer_system() -> str:
    return (PROMPTS / "organizer_system.md").read_text()


def _trace(log: list[dict]) -> list[dict]:
    """What an agent did, small enough to keep in the run report."""
    out = []
    keys = ("page", "kind", "statement", "snippet_id", "old_id", "existing_id", "keep_id", "drop_id", "new_id", "a_id", "b_id")
    for entry in log:
        item: dict = {"tool": entry["tool"], "ok": entry["ok"]}
        args = entry.get("args") or {}
        item.update({k: args[k] for k in keys if k in args})
        if not entry["ok"]:
            item["error"] = str(entry.get("error", ""))[:300]
        out.append(item)
    return out


def analyze_session(
    store: Store,
    llm: LLM,
    user_id: str,
    session: dict,
    resolver: Resolver | None = None,
    max_turns: int = 40,
) -> dict:
    """Run the analyzer over the part of one session not yet dreamed over."""
    session_id = session["id"]
    upto = session["next_offset"]
    # A subagent's transcript is stored, but its "user" is the parent agent, not
    # the person, so nothing in it is the person's preference.
    if session["meta"].get("parent_session_id"):
        return {"input": session_id, "ok": True, "upto": upto, "skipped": "subagent transcript", "written": []}
    events = store.get_events(user_id, session_id, 0, upto)
    full_text = render_events(events, session["client"], TOOL_CHARS, **_view())
    new_text = render_events(events, session["client"], TOOL_CHARS, start=session["dreamed_offset"], **_view())
    if len(new_text.strip()) < MIN_CHARS:
        return {"input": session_id, "ok": True, "upto": upto, "skipped": "too short", "written": []}

    pages = paginate(new_text, PAGE_CHARS)
    windows = [pages[i : i + PAGES_PER_RUN] for i in range(0, len(pages), PAGES_PER_RUN)]
    written: list[str] = []
    summaries: list[str] = []
    trace: list[dict] = []
    turns = 0
    for n, window in enumerate(windows, 1):
        # Quotes are checked against the whole session, so a quote that sits in
        # another part of the same session is still accepted.
        tools = AnalyzerTools(store, user_id, session_id, window, full_text, resolver)
        part = f" This is part {n} of {len(windows)} of a long transcript." if len(windows) > 1 else ""
        task = (
            f"Read the transcript ({len(window)} page{'s' if len(window) != 1 else ''}) and record what it shows "
            "about how this person refers to things and which records they want when they search." + part + " Other "
            "analyzers are working on this person's other transcripts at the same time, so check what is already "
            "stored before writing. When done, call finish with a short summary of what you added."
        )
        result = run_agent(llm, analyzer_system(), task, tools.tools(), max_turns=max_turns)
        written += tools.written
        trace += _trace(result.tool_calls)
        turns += result.turns
        if not result.finished:
            stuck = result.result.get("stuck_on")
            reason = (
                f"analyzer kept resending a failing {stuck} call: {result.result['error'][:200]}"
                if stuck
                else f"analyzer did not finish within {max_turns} turns"
            )
            if len(windows) > 1:
                reason += f" (part {n} of {len(windows)})"
            return {"input": session_id, "ok": False, "reason": reason, "written": written, "trace": trace}
        summaries.append(result.result.get("summary", ""))
    return {
        "input": session_id,
        "ok": True,
        "upto": upto,
        "summary": " ".join(s for s in summaries if s),
        "written": written,
        "turns": turns,
        "pages": len(pages),
        "trace": trace,
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
    return {"changes": result.result.get("changes", []), "applied": tools.changes, "trace": _trace(result.tool_calls)}


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
    only_sessions: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Dream over the sessions of `user_id` that have gone quiet and have new events.

    `only_sessions` restricts the run to those session ids; `limit` caps how
    many sessions one run takes (the oldest first). Both are for controlled tests.
    """
    quiet_before = (datetime.now(timezone.utc) - timedelta(minutes=quiet_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sessions = store.sessions_to_dream(user_id, quiet_before)
    if only_sessions is not None:
        wanted = set(only_sessions)
        sessions = [s for s in sessions if s["id"] in wanted]
    if limit is not None:
        # Subagent transcripts are skipped without a model call, so they do not count.
        kept, real = [], 0
        for s in sessions:
            is_sub = bool(s["meta"].get("parent_session_id"))
            if is_sub or real < limit:
                kept.append(s)
                real += 0 if is_sub else 1
        sessions = kept
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
        "failures": [
            {"input": r["input"], "reason": r["reason"], **({"trace": r["trace"]} if r.get("trace") else {})}
            for r in results
            if not r["ok"]
        ],
        "organized": False,
        "changes": [],
        "analyzers": [
            {k: r[k] for k in ("input", "summary", "written", "turns", "pages", "skipped", "trace") if k in r}
            for r in processed
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
            report["organizer_trace"] = organized["trace"]
        except Exception as exc:
            report["organizer_error"] = f"{type(exc).__name__}: {exc}"
    report["snippets_in_force"] = len(store.list_snippets(user_id, "active"))
    store.finish_run(run_id, report)
    return report
