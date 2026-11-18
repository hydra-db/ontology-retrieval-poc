"""Harness tests with a scripted model: they check the code around the model
(tool checks, batching, supersede rules, bookkeeping), not model quality."""

import json

import pytest

from orp.harness.agent import Tool, run_agent
from orp.harness.lead import analyze_session, dream
from orp.harness.tools import AnalyzerTools, OrganizerTools
from orp.harness.agent import ToolError
from orp.store import Store

FILLER = "Please look at the failing ingestion test and tell me what is going wrong with the retry logic. " * 3


def user_event(text, ts="2026-11-17T10:00:00Z"):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def agent_event(text, ts="2026-11-17T10:00:05Z"):
    return {"type": "assistant", "timestamp": ts, "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def call(name, **args):
    return {"id": f"c_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """Replays a list of steps. Each step is a list of tool calls, or a function
    of the messages so far that returns one."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        step = self.steps.pop(0)
        calls = step(messages) if callable(step) else step
        return {"content": None, "tool_calls": calls}


ALIAS = dict(kind="alias", statement="User says 'turbolay' for hydradb.", trigger_terms=["turbolay"], basis="stated",
             targets=[{"text": "hydradb"}])
SHIPPED = dict(kind="constraint", statement="By 'shipped' the user means state is On main.", trigger_terms=["shipped"],
               basis="stated", field="state", op="eq", value="On main", strength="require")


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


@pytest.fixture()
def uid(store):
    return store.create_user("asha")[0]


def seed(store, uid, session_id, texts):
    events = [user_event(FILLER)] + [user_event(t) for t in texts] + [agent_event("Understood.")]
    store.append_events(uid, session_id, "claude-code", 0, events)
    return store.get_session(uid, session_id)


def test_run_agent_feeds_errors_back_and_stops_on_finish():
    def boom(**_):
        raise ToolError("bad input")

    llm = Scripted([
        [call("nope")],
        [call("boom")],
        [],  # no tool call: gets nudged
        [call("finish", summary="done")],
    ])
    result = run_agent(llm, "sys", "task", [Tool("boom", "", {"type": "object", "properties": {}}, boom)])
    assert result.finished and result.result == {"summary": "done"} and result.turns == 4
    last = llm.seen[-1]
    assert any(m["role"] == "tool" and "no tool named 'nope'" in m["content"] for m in last)
    assert any(m["role"] == "tool" and m["content"] == "ERROR: bad input" for m in last)
    assert last[-1]["role"] == "user" and "call finish" in last[-1]["content"]


def test_run_agent_stops_when_stuck_on_one_failing_call():
    def boom(**_):
        raise ToolError("bad input")

    llm = Scripted([[call("boom", x=1)] for _ in range(10)])
    result = run_agent(llm, "sys", "task", [Tool("boom", "", {"type": "object", "properties": {}}, boom)])
    assert result.finished is False and result.turns == 4 and result.result["stuck_on"] == "boom"
    assert "exactly this call before" in llm.seen[2][-1]["content"]


def test_run_agent_gives_up_after_max_turns():
    llm = Scripted([[] for _ in range(3)])
    assert run_agent(llm, "sys", "task", [], max_turns=3).finished is False


def test_analyzer_adds_snippet_with_checked_quote(store, uid):
    session = seed(store, uid, "s1", ["btw turbolay and hydradb are the same thing"])
    llm = Scripted([
        [call("read_transcript", page=1)],
        [call("list_snippets")],
        # a quote that is not in the transcript is rejected...
        [call("add_snippet", **ALIAS, quotes=[{"quote": "turbolay is our old name for hydradb"}])],
        # ...the corrected one is accepted, whitespace and case aside
        [call("add_snippet", **ALIAS, quotes=[{"quote": "Turbolay and  hydradb are the same thing", "said_at": "2026-11-17T10:00:00Z"}])],
        [call("finish", summary="Added one alias.")],
    ])
    res = analyze_session(store, llm, uid, session)
    assert res["ok"] and res["summary"] == "Added one alias." and len(res["written"]) == 1
    rejected = [m for m in llm.seen[-1] if m["role"] == "tool" and m["content"].startswith("ERROR")]
    assert len(rejected) == 1 and "not in the transcript" in rejected[0]["content"]
    (snip,) = store.list_snippets(uid, "active")
    assert snip["kind"] == "alias" and snip["targets"] == [{"text": "hydradb", "entity_type": None, "entity_id": None}]
    assert snip["evidence"][0]["session_id"] == "s1"
    assert "USER:" in llm.seen[1][-1]["content"] and "page 1 of 1" in llm.seen[1][-1]["content"]


def test_analyzer_schema_errors_reach_the_model(store, uid):
    seed(store, uid, "s1", ["by shipped I mean it's On main"])
    tools = AnalyzerTools(store, uid, "s1", ["p"], "by shipped I mean it's On main")
    q = [{"quote": "by shipped I mean it's On main"}]
    with pytest.raises(ToolError, match="field must be one of"):
        tools.add_snippet(**{**SHIPPED, "field": "status"}, quotes=q)
    with pytest.raises(ToolError, match="this is a constraint"):
        tools.add_snippet(**ALIAS, field="state", op="eq", value="On main", strength="require", quotes=q)
    assert tools.add_snippet(**SHIPPED, quotes=q)["added"].startswith("s_")


def test_resolver_fills_entity_id(store, uid):
    tools = AnalyzerTools(store, uid, "s1", ["p"], "I always call Chaithanya Kumar CK",
                          resolver=lambda etype, text: "person:42" if text == "Chaithanya Kumar" else None)
    out = tools.add_snippet(kind="alias", statement="User says 'CK' for Chaithanya Kumar.", trigger_terms=["CK"],
                            basis="stated", direction="one_way",
                            targets=[{"text": "Chaithanya Kumar", "entity_type": "person"}],
                            quotes=[{"quote": "I always call Chaithanya Kumar CK"}])
    assert store.get_snippet(uid, out["added"])["targets"][0]["entity_id"] == "person:42"


def test_supersede_and_conflict_rules(store, uid):
    text = "by shipped I mean it's On main. actually shipped now means Released."
    tools = AnalyzerTools(store, uid, "s1", ["p"], text)
    old = tools.add_snippet(**SHIPPED, quotes=[{"quote": "by shipped I mean it's On main"}])["added"]
    new_args = {**SHIPPED, "value": "Released", "statement": "By 'shipped' the user means state is Released."}
    q = [{"quote": "actually shipped now means Released"}]

    # an inferred snippet may not replace a stated one
    with pytest.raises(ToolError, match="cannot replace"):
        tools.supersede_snippet(old_id=old, reason="changed", **{**new_args, "basis": "inferred"}, quotes=q)
    assert store.get_snippet(uid, old)["status"] == "active"

    out = tools.supersede_snippet(old_id=old, reason="user changed the meaning", **new_args, quotes=q)
    kept = store.get_snippet(uid, old)
    assert kept["status"] == "superseded" and kept["superseded_by"] == out["added"]
    assert [s["id"] for s in store.list_snippets(uid, "active")] == [out["added"]]

    third = tools.flag_conflict(existing_id=out["added"], note="two meanings of shipped",
                                **{**SHIPPED, "basis": "inferred"}, quotes=[{"quote": "by shipped I mean it's On main"}])
    assert store.get_snippet(uid, out["added"])["status"] == "conflict"
    assert store.get_snippet(uid, third["added"])["conflict_with"] == out["added"]


def test_add_evidence_raises_basis_and_adds_terms(store, uid):
    tools = AnalyzerTools(store, uid, "s2", ["p"], "turbolay, i.e. hydradb")
    sid = store.add_snippet(uid, {**ALIAS, "basis": "inferred", "direction": "both",
                                  "evidence": [{"session_id": "s1", "quote": "turbolay"}]})["id"]
    tools.add_evidence(snippet_id=sid, quotes=[{"quote": "turbolay, i.e. hydradb"}], trigger_terms=["turbo lay", "Turbolay"], basis="stated")
    snip = store.get_snippet(uid, sid)
    assert snip["basis"] == "stated" and snip["trigger_terms"] == ["turbolay", "turbo lay"]
    assert [e["session_id"] for e in snip["evidence"]] == ["s1", "s2"]
    with pytest.raises(ToolError, match="no snippet"):
        tools.add_evidence(snippet_id="s_missing", quotes=[{"quote": "turbolay"}])


def test_organizer_tools(store, uid):
    def add(draft, session="s1", **over):
        return store.add_snippet(uid, {"direction": "both", **draft, **over,
                                       "evidence": [{"session_id": session, "quote": "q " + session}]})["id"]

    a1, a2 = add(ALIAS), add(ALIAS, "s2", trigger_terms=["turbo lay"])
    c1, c2 = add(SHIPPED), add(SHIPPED, "s2", value="Released", basis="inferred")
    org = OrganizerTools(store, uid)

    with pytest.raises(ToolError, match="cannot be merged"):
        org.merge_snippets(keep_id=a1, drop_id=c1)
    with pytest.raises(ToolError, match="not duplicates"):
        org.merge_snippets(keep_id=c1, drop_id=c2)
    with pytest.raises(ToolError, match="cannot replace"):
        org.supersede(old_id=c1, new_id=c2, reason="newer")

    org.merge_snippets(keep_id=a1, drop_id=a2)
    merged = store.get_snippet(uid, a1)
    assert merged["trigger_terms"] == ["turbolay", "turbo lay"] and len(merged["evidence"]) == 2
    assert store.get_snippet(uid, a2)["status"] == "superseded"
    with pytest.raises(ToolError, match="already superseded"):
        org.merge_snippets(keep_id=a1, drop_id=a2)

    org.mark_conflict(a_id=c1, b_id=c2, note="two meanings of shipped")
    assert {s["id"] for s in store.list_snippets(uid, "conflict")} == {c1, c2}
    assert not any(t.name in ("add_snippet", "add_unmapped") for t in org.tools())


def test_dream_end_to_end(store, uid):
    seed(store, uid, "s1", ["btw turbolay and hydradb are the same thing"])
    seed(store, uid, "s2", ["remember: turbolay and hydradb are the same thing"])
    seed(store, uid, "s3", ["nothing to learn here"])
    store.append_events(uid, "tiny", "claude-code", 0, [user_event("hi")])
    store.append_events(uid, "s1:agent1", "claude-code", 0, [user_event(FILLER + " turbolay and hydradb are the same thing")],
                        meta={"parent_session_id": "s1"})

    class Model:
        """Analyzer: add the alias if the transcript has it. s3's analyzer never finishes."""

        def chat(self, messages, tools):
            system = messages[0]["content"]
            results = [m["content"] for m in messages if m["role"] == "tool"]
            if system.startswith("You tidy"):
                if not results:
                    return {"tool_calls": [call("list_snippets")]}
                ids = [s["id"] for s in json.loads(results[0])["snippets"]]
                if len(results) == 1 and len(ids) == 2:
                    return {"tool_calls": [call("merge_snippets", keep_id=ids[0], drop_id=ids[1])]}
                return {"tool_calls": [call("finish", changes=["merged a duplicate alias"])]}
            if not results:
                return {"tool_calls": [call("read_transcript", page=1)]}
            if "nothing to learn" in results[0]:
                return {"tool_calls": [call("list_snippets")]}  # loops until max_turns
            if len(results) == 1:
                return {"tool_calls": [call("add_snippet", **ALIAS, quotes=[{"quote": "turbolay and hydradb are the same thing"}])]}
            return {"tool_calls": [call("finish", summary="added alias")]}

    model = Model()
    report = dream(store, uid, model, model, quiet_minutes=-1, group_size=2, concurrency=2)

    assert report["inputs_expected"] == 5
    assert report["inputs_processed"] == 4 and report["inputs_skipped"] == 2
    assert {a["input"]: a.get("skipped") for a in report["analyzers"]}["s1:agent1"] == "subagent transcript"
    assert [f["input"] for f in report["failures"]] == ["s3"]
    assert report["organized"] and report["changes"] == ["merged a duplicate alias"]
    assert report["snippets_in_force"] == 1
    (snip,) = store.list_snippets(uid, "active")
    assert {e["session_id"] for e in snip["evidence"]} == {"s1", "s2"}

    # finished sessions are not picked up again; the failed one is
    again = [s["id"] for s in store.sessions_to_dream(uid, "2999-01-01T00:00:00Z")]
    assert again == ["s3"]
    assert store.list_runs(uid)[0]["report"]["run_id"] == report["run_id"]


def test_dream_with_nothing_ready(store, uid):
    report = dream(store, uid, None, None)
    assert report["inputs_expected"] == 0 and report["organized"] is False and report["failures"] == []


def test_long_session_is_read_in_parts(store, uid, monkeypatch):
    from orp.harness import lead

    monkeypatch.setattr(lead, "PAGE_CHARS", 400)
    monkeypatch.setattr(lead, "PAGES_PER_RUN", 2)
    events = [user_event(f"message number {i}. " + FILLER) for i in range(12)]
    events.append(user_event("btw turbolay and hydradb are the same thing"))
    store.append_events(uid, "long", "claude-code", 0, events)
    seen_tasks, pages_read = [], []

    class Model:
        def chat(self, messages, tools):
            results = [m["content"] for m in messages if m["role"] == "tool"]
            total = int(messages[1]["content"].split("(")[1].split(" page")[0])
            if len(results) == 0:
                seen_tasks.append(messages[1]["content"])
            if len(results) < total:
                return {"tool_calls": [call("read_transcript", page=len(results) + 1)]}
            text = " ".join(results)
            pages_read.append(total)
            if "turbolay" in text and len(results) == total:
                return {"tool_calls": [call("add_snippet", **ALIAS, quotes=[{"quote": "turbolay and hydradb are the same thing"}])]}
            return {"tool_calls": [call("finish", summary=f"part done")]}

    res = analyze_session(store, Model(), uid, store.get_session(uid, "long"))
    assert res["ok"] and res["pages"] > 2 and len(seen_tasks) == -(-res["pages"] // 2)
    assert "part 1 of" in seen_tasks[0] and all(n <= 2 for n in pages_read)
    assert len(res["written"]) == 1
    assert [t["tool"] for t in res["trace"] if t["tool"] == "add_snippet"] == ["add_snippet"]
    assert res["trace"][-1]["tool"] == "finish"


def test_dream_only_sessions_and_limit(store, uid):
    for name in ("a", "b", "c"):
        seed(store, uid, name, ["nothing"])
    store.append_events(uid, "a:sub", "claude-code", 0, [user_event(FILLER)], meta={"parent_session_id": "a"})

    class Model:
        def chat(self, messages, tools):
            return {"tool_calls": [call("finish", summary="nothing")]}

    r = dream(store, uid, Model(), Model(), quiet_minutes=-1, only_sessions=["b"])
    assert [a["input"] for a in r["analyzers"]] == ["b"]
    r = dream(store, uid, Model(), Model(), quiet_minutes=-1, limit=1)
    real = [a["input"] for a in r["analyzers"] if not a.get("skipped")]
    assert len(real) == 1 and r["inputs_expected"] == 2  # one real session plus the subagent transcript
    assert len(store.sessions_to_dream(uid, "2999-01-01T00:00:00Z")) == 1


def test_failure_keeps_trace_with_error(store, uid):
    session = seed(store, uid, "s1", ["by shipped I mean it's On main"])
    bad = dict(SHIPPED, field="status")
    llm = Scripted([[call("add_snippet", **bad, quotes=[{"quote": "by shipped I mean it's On main"}])] for _ in range(6)])
    res = analyze_session(store, llm, uid, session)
    assert not res["ok"] and "kept resending a failing add_snippet" in res["reason"]
    assert res["trace"][0]["ok"] is False and "field must be one of" in res["trace"][0]["error"]
