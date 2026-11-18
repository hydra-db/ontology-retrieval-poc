import json

import pytest

from orp import cli
from orp.store import Store

ALIAS = dict(kind="alias", statement="User says 'turbolay' for hydradb.", trigger_terms=["turbolay"], basis="stated",
             targets=[{"text": "hydradb", "entity_type": None, "entity_id": None}], direction="both",
             evidence=[{"session_id": "s1", "quote": "turbolay and hydradb are the same thing", "said_at": None}])
SHIPPED = dict(kind="constraint", statement="By 'shipped' the user means state is On main.", trigger_terms=["shipped"],
               basis="stated", field="state", op="eq", value="On main", strength="require",
               evidence=[{"session_id": "s2", "quote": "by shipped I mean On main", "said_at": "2026-11-05T10:03:00Z"}])


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORP_DB", str(tmp_path / "t.db"))
    store = Store(tmp_path / "t.db")
    uid, _ = store.create_user("asha")
    return store, uid


def test_load_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("ORP_X", raising=False)
    monkeypatch.setenv("ORP_KEEP", "from-env")
    (tmp_path / ".env").write_text("# comment\nORP_X='hello world'\nexport ORP_Y=1\nORP_KEEP=from-file\nnot a line\n")
    cli.load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["ORP_X"] == "hello world" and os.environ["ORP_Y"] == "1" and os.environ["ORP_KEEP"] == "from-env"
    monkeypatch.delenv("ORP_X"); monkeypatch.delenv("ORP_Y")


def test_snippets_output(db, capsys):
    store, uid = db
    a = store.add_snippet(uid, ALIAS)["id"]
    b = store.add_snippet(uid, SHIPPED)["id"]
    store.supersede(uid, b, a, "changed")
    cli.main(["snippets", "--user", "asha", "--status", "all", "--evidence"])
    out = capsys.readouterr().out
    assert '"turbolay" = hydradb' in out and '"shipped" -> state eq "On main"  [require]' in out
    assert "superseded" in out and f"replaced by {a}" in out
    assert '> "by shipped I mean On main"  (s2 at 2026-11-05T10:03:00Z)' in out
    cli.main(["snippets", "--user", "asha", "--json"])
    assert [s["id"] for s in json.loads(capsys.readouterr().out)] == [a]


def test_status_and_reset(db, capsys):
    store, uid = db
    store.append_events(uid, "s1", "claude-code", 0, [{"n": 1}, {"n": 2}], meta={"cwd": "/work"})
    store.append_events(uid, "s1:sub", "claude-code", 0, [{"n": 1}], meta={"parent_session_id": "s1"})
    store.add_snippet(uid, ALIAS)
    store.mark_dreamed(uid, "s1", 2)
    cli.main(["status", "--user", "asha"])
    out = capsys.readouterr().out
    assert "1 sessions (+1 subagent transcripts), 3 events stored" in out
    assert "waiting to be dreamed over: 0 sessions" in out and "snippets: 1 active" in out

    with pytest.raises(SystemExit) as stop:
        cli.main(["reset", "--user", "asha"])
    assert "--yes" in str(stop.value) and len(store.list_snippets(uid)) == 1
    cli.main(["reset", "--user", "asha", "--yes"])
    assert store.list_snippets(uid) == [] and store.get_session(uid, "s1")["dreamed_offset"] == 0
    assert store.get_events(uid, "s1") == [{"n": 1}, {"n": 2}]  # transcripts are kept


def test_unknown_user_and_missing_model_settings(db, monkeypatch):
    with pytest.raises(SystemExit, match="no user named"):
        cli.main(["status", "--user", "nobody"])
    monkeypatch.delenv("ORP_LLM_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="ORP_LLM_BASE_URL is not set"):
        cli.main(["dream", "--user", "asha"])


def test_print_report_shows_writes_errors_and_skips(capsys):
    cli.print_report({
        "run_id": "r_1", "inputs_expected": 3, "inputs_processed": 2, "inputs_skipped": 1, "snippets_in_force": 1,
        "failures": [{"input": "s3", "reason": "analyzer did not finish within 40 turns",
                      "trace": [{"tool": "add_snippet", "ok": False, "statement": "X", "error": "ERROR: bad"}]}],
        "analyzers": [
            {"input": "s1", "summary": "Added one alias.", "written": ["s_1"], "turns": 4, "pages": 1,
             "trace": [{"tool": "read_transcript", "ok": True, "page": 1}, {"tool": "add_snippet", "ok": True, "statement": "User says X."}]},
            {"input": "s1:sub", "skipped": "subagent transcript", "written": []},
        ],
        "organized": True, "changes": [], "organizer_trace": [],
    })
    out = capsys.readouterr().out
    assert "2 of 3 sessions processed" in out and "ok  add_snippet: User says X." in out
    assert "read_transcript" not in out and "FAILED s3" in out and "ERROR: bad" in out
    assert "skipped: 1 subagent transcript" in out and "no changes needed" in out
