import pytest
from fastapi.testclient import TestClient

from orp.api import create_app
from orp.store import OffsetGap, Store


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


@pytest.fixture()
def client(store):
    _, key = store.create_user("asha")
    c = TestClient(create_app(store))
    c.headers["Authorization"] = f"Bearer {key}"
    return c


def test_append_is_idempotent_and_ordered(store):
    uid, _ = store.create_user("asha")
    assert store.append_events(uid, "s1", "claude-code", 0, [{"n": 0}, {"n": 1}]) == 2
    # resending an overlapping batch stores only the new events
    assert store.append_events(uid, "s1", "claude-code", 1, [{"n": 1}, {"n": 2}]) == 3
    assert store.append_events(uid, "s1", "claude-code", 0, [{"n": 0}]) == 3
    assert [e["n"] for e in store.get_events(uid, "s1")] == [0, 1, 2]
    with pytest.raises(OffsetGap) as gap:
        store.append_events(uid, "s1", "claude-code", 9, [{"n": 9}])
    assert gap.value.next_offset == 3


def test_sessions_are_per_user(store):
    a, _ = store.create_user("asha")
    b, _ = store.create_user("bo")
    store.append_events(a, "same-id", "claude-code", 0, [{"who": "a"}])
    store.append_events(b, "same-id", "claude-code", 0, [{"who": "b"}, {"who": "b"}])
    assert store.next_offset(a, "same-id") == 1
    assert store.next_offset(b, "same-id") == 2
    assert store.get_events(a, "same-id") == [{"who": "a"}]


def test_sessions_to_dream_and_mark(store):
    uid, _ = store.create_user("asha")
    store.append_events(uid, "s1", "claude-code", 0, [{"n": 0}])
    assert store.sessions_to_dream(uid, "2000-01-01T00:00:00Z") == []  # not quiet yet
    ready = store.sessions_to_dream(uid, "2999-01-01T00:00:00Z")
    assert [s["id"] for s in ready] == ["s1"]
    store.mark_dreamed(uid, "s1", 1)
    assert store.sessions_to_dream(uid, "2999-01-01T00:00:00Z") == []
    store.append_events(uid, "s1", "claude-code", 1, [{"n": 1}])
    assert store.sessions_to_dream(uid, "2999-01-01T00:00:00Z")[0]["dreamed_offset"] == 1


def test_api_auth(store):
    c = TestClient(create_app(store))
    assert c.get("/v1/me").status_code == 401
    assert c.get("/v1/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/healthz").json() == {"ok": True}


def test_api_upload_contract(client):
    assert client.get("/v1/sessions/s1").json() == {"session_id": "s1", "next_offset": 0}
    r = client.post("/v1/sessions/s1/events", json={"client": "claude-code", "offset": 0, "events": [{"a": 1}, {"a": 2}], "meta": {"cwd": "/x"}})
    assert r.json() == {"next_offset": 2}
    r = client.post("/v1/sessions/s1/events", json={"offset": 5, "events": [{"a": 6}]})
    assert r.status_code == 409 and r.json() == {"next_offset": 2}
    r = client.post("/v1/sessions/s1/events", json={"offset": 1, "events": [{"a": 2}, {"a": 3}]})
    assert r.json() == {"next_offset": 3}
    sessions = client.get("/v1/sessions").json()["sessions"]
    assert sessions[0]["meta"] == {"cwd": "/x"} and sessions[0]["next_offset"] == 3


def test_api_snippets_only_own(store, client):
    other, _ = store.create_user("bo")
    store.add_snippet(other, dict(kind="alias", statement="Bo says 'x' for y.", trigger_terms=["x"], basis="stated",
                                  targets=[{"text": "y"}], direction="both", evidence=[{"session_id": "s", "quote": "x is y"}]))
    assert client.get("/v1/snippets").json() == {"snippets": []}
