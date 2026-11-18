"""Runs hooks/claude_code_upload.py as a subprocess against a stdlib HTTP stub
that implements the ORP session-upload contract. All fixtures are synthetic."""

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "claude_code_upload.py"
API_KEY = "test-key"


class Stub:
    """In-memory server: GET /v1/sessions/{id}, POST /v1/sessions/{id}/events."""

    def __init__(self):
        self.sessions = {}      # session_id -> list of stored events
        self.posts = []         # (session_id, body) for every accepted-or-rejected POST
        self.gets = []          # session ids asked via GET
        self.headers = []       # headers of every request
        self.forced_offset = {} # session_id -> server-side next_offset override (to provoke 409)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _session(self):
                parts = self.path.split("/")
                assert parts[1:3] == ["v1", "sessions"], self.path
                return unquote(parts[3]), parts[4:]

            def _auth(self):
                stub.headers.append(dict(self.headers))
                if self.headers.get("Authorization") != "Bearer " + API_KEY:
                    self._send(401, {"detail": "bad key"})
                    return False
                return True

            def do_GET(self):
                if not self._auth():
                    return
                sid, rest = self._session()
                stub.gets.append(sid)
                self._send(200, {"session_id": sid, "next_offset": stub.next_offset(sid)})

            def do_POST(self):
                if not self._auth():
                    return
                sid, rest = self._session()
                assert rest == ["events"], self.path
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.posts.append((sid, body))
                nxt = stub.next_offset(sid)
                if body["offset"] > nxt:  # gap: client is ahead of the server
                    self._send(409, {"next_offset": nxt})
                    return
                stub.forced_offset.pop(sid, None)
                store = stub.sessions.setdefault(sid, [])
                for i, ev in enumerate(body["events"]):
                    if body["offset"] + i >= len(store):  # skip events below next_offset
                        store.append(ev)
                self._send(200, {"next_offset": len(store)})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def next_offset(self, sid):
        return self.forced_offset.get(sid, len(self.sessions.get(sid, [])))

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.fixture
def env(tmp_path, stub):
    return make_env(tmp_path, stub.url)


def make_env(tmp_path, url, key=API_KEY):
    e = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "ORP_STATE_DIR": str(tmp_path / "state"),
        "ORP_HOOK_LOG": str(tmp_path / "hook.log"),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    if url is not None:
        e["ORP_API_URL"] = url
    if key is not None:
        e["ORP_API_KEY"] = key
    return e


def line(i, **extra):
    return json.dumps({"type": "user", "uuid": "u%d" % i, "cwd": "/work/proj",
                       "message": {"role": "user", "content": "message %d" % i}, **extra}) + "\n"


def make_transcript(tmp_path, n, session_id="sess-1"):
    path = tmp_path / "projects" / "-work-proj" / (session_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line(i) for i in range(n)))
    return path


def run_hook(env, payload=None, args=(), stdin=None):
    if stdin is None:
        stdin = json.dumps(payload) if payload is not None else ""
    return subprocess.run([sys.executable, str(HOOK), *args], input=stdin, env=env,
                          capture_output=True, text=True, timeout=60)


def hook_payload(path, event="Stop", session_id="sess-1", **extra):
    return {"session_id": session_id, "transcript_path": str(path), "cwd": "/work/proj",
            "hook_event_name": event, **extra}


def uuids(events):
    return [e.get("uuid") for e in events]


# ------------------------------------------------------------------- tests

def test_first_upload(tmp_path, stub, env):
    path = make_transcript(tmp_path, 3)
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    assert uuids(stub.sessions["sess-1"]) == ["u0", "u1", "u2"]
    assert stub.gets == ["sess-1"]  # no local state yet -> asked the server
    sid, body = stub.posts[0]
    assert body["client"] == "claude-code" and body["offset"] == 0
    assert body["events"][0] == json.loads(line(0))  # raw object, unmodified
    assert body["meta"] == {"cwd": "/work/proj", "transcript_path": str(path), "hook_event": "Stop"}
    for h in stub.headers:
        assert h["Authorization"] == "Bearer " + API_KEY
        assert h["User-Agent"] == "orp-hook/0.1"
    state = json.loads((tmp_path / "state" / "sess-1.json").read_text())
    assert state["next_offset"] == 3 and state["byte_pos"] == path.stat().st_size


def test_incremental_upload_sends_only_new_lines(tmp_path, stub, env):
    path = make_transcript(tmp_path, 3)
    run_hook(env, hook_payload(path))
    with path.open("a") as f:
        f.write(line(3) + line(4))
    r = run_hook(env, hook_payload(path, event="SessionEnd", reason="other"))
    assert (r.returncode, r.stdout) == (0, "")
    assert len(stub.posts) == 2
    sid, body = stub.posts[1]
    assert body["offset"] == 3 and uuids(body["events"]) == ["u3", "u4"]
    assert body["meta"]["hook_event"] == "SessionEnd"
    assert uuids(stub.sessions["sess-1"]) == ["u0", "u1", "u2", "u3", "u4"]
    assert stub.gets == ["sess-1"]  # second run used the local state file, no GET

    # nothing new -> no request at all
    run_hook(env, hook_payload(path))
    assert len(stub.posts) == 2 and stub.gets == ["sess-1"]


def test_falls_back_to_get_when_state_is_missing(tmp_path, stub, env):
    path = make_transcript(tmp_path, 4)
    stub.sessions["sess-1"] = [json.loads(line(0)), json.loads(line(1))]  # server already has 2
    r = run_hook(env, hook_payload(path))
    assert r.returncode == 0
    assert stub.gets == ["sess-1"]
    assert stub.posts[0][1]["offset"] == 2 and uuids(stub.posts[0][1]["events"]) == ["u2", "u3"]


def test_trailing_partial_line_is_ignored_until_complete(tmp_path, stub, env):
    path = make_transcript(tmp_path, 2)
    full = line(2)
    with path.open("a") as f:
        f.write(full[:25])  # still being written: no newline yet
    run_hook(env, hook_payload(path))
    assert uuids(stub.sessions["sess-1"]) == ["u0", "u1"]
    with path.open("a") as f:
        f.write(full[25:])
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    assert stub.sessions["sess-1"][2] == json.loads(full)
    assert stub.posts[-1][1]["offset"] == 2


def test_invalid_json_line_sent_as_unparsed(tmp_path, stub, env):
    path = make_transcript(tmp_path, 1)
    with path.open("a") as f:
        f.write("{this is not json\n")
        f.write("[1, 2]\n")
        f.write(line(3))
    run_hook(env, hook_payload(path))
    stored = stub.sessions["sess-1"]
    assert len(stored) == 4  # offsets stay aligned with line numbers
    assert stored[1] == {"type": "unparsed", "raw": "{this is not json"}
    assert stored[2] == {"type": "unparsed", "raw": "[1, 2]"}
    assert stored[3]["uuid"] == "u3"


def test_409_resync_and_retry_once(tmp_path, stub, env):
    path = make_transcript(tmp_path, 3)
    run_hook(env, hook_payload(path))
    # Server lost the last two events (e.g. restored from backup); local state says 3.
    del stub.sessions["sess-1"][1:]
    with path.open("a") as f:
        f.write(line(3))
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    offsets = [b["offset"] for _, b in stub.posts]
    assert offsets == [0, 3, 1]  # first run, rejected with 409, resent from server's offset
    assert uuids(stub.sessions["sess-1"]) == ["u0", "u1", "u2", "u3"]
    state = json.loads((tmp_path / "state" / "sess-1.json").read_text())
    assert state["next_offset"] == 4


def test_batches_of_at_most_200_events(tmp_path, stub, env):
    path = make_transcript(tmp_path, 450)
    run_hook(env, hook_payload(path))
    assert [(b["offset"], len(b["events"])) for _, b in stub.posts] == [(0, 200), (200, 200), (400, 50)]
    assert len(stub.sessions["sess-1"]) == 450


def test_batches_split_at_about_one_megabyte(tmp_path, stub, env):
    path = make_transcript(tmp_path, 0)
    path.write_text("".join(line(i, pad="x" * 300_000) for i in range(7)))
    run_hook(env, hook_payload(path))
    assert [len(b["events"]) for _, b in stub.posts] == [3, 3, 1]
    assert uuids(stub.sessions["sess-1"]) == ["u%d" % i for i in range(7)]


def test_subagent_transcripts_uploaded_as_own_sessions(tmp_path, stub, env):
    path = make_transcript(tmp_path, 2)
    sub_dir = path.parent / "sess-1" / "subagents"
    wf_dir = sub_dir / "workflows" / "wf_1"
    wf_dir.mkdir(parents=True)
    (sub_dir / "agent-abc123.jsonl").write_text(line(10, isSidechain=True, agentId="abc123") + line(11))
    (sub_dir / "agent-abc123.meta.json").write_text(json.dumps({"agentType": "Explore", "description": "look around"}))
    (wf_dir / "agent-def456.jsonl").write_text(line(20))
    (wf_dir / "journal.jsonl").write_text(json.dumps({"type": "launched"}) + "\n")

    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    assert set(stub.sessions) == {"sess-1", "sess-1:abc123", "sess-1:def456", "sess-1:wf_1.journal"}
    assert uuids(stub.sessions["sess-1:abc123"]) == ["u10", "u11"]
    meta = next(b["meta"] for sid, b in stub.posts if sid == "sess-1:abc123")
    assert meta["parent_session_id"] == "sess-1"
    assert meta["agent_id"] == "abc123"
    assert meta["agent"] == {"agentType": "Explore", "description": "look around"}
    assert meta["transcript_path"] == str(sub_dir / "agent-abc123.jsonl")
    main_meta = next(b["meta"] for sid, b in stub.posts if sid == "sess-1")
    assert "parent_session_id" not in main_meta


def test_subagent_stop_uploads_agent_file_and_main_only(tmp_path, stub, env):
    path = make_transcript(tmp_path, 2)
    sub_dir = path.parent / "sess-1" / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-abc123.jsonl").write_text(line(10))
    (sub_dir / "agent-other.jsonl").write_text(line(30))
    payload = hook_payload(path, event="SubagentStop", agent_id="abc123", agent_type="Explore",
                           agent_transcript_path=str(sub_dir / "agent-abc123.jsonl"))
    r = run_hook(env, payload)
    assert (r.returncode, r.stdout) == (0, "")
    assert set(stub.sessions) == {"sess-1", "sess-1:abc123"}
    assert stub.posts[0][0] == "sess-1:abc123"
    assert stub.posts[0][1]["meta"]["hook_event"] == "SubagentStop"


def test_manual_backfill(tmp_path, stub, env):
    path = make_transcript(tmp_path, 3, session_id="sess-9")
    r = run_hook(env, args=["--file", str(path)])
    assert (r.returncode, r.stdout) == (0, "")
    assert uuids(stub.sessions["sess-9"]) == ["u0", "u1", "u2"]
    meta = stub.posts[0][1]["meta"]
    assert meta["hook_event"] == "backfill" and meta["cwd"] == "/work/proj"

    r = run_hook(env, args=["--file", str(path), "--session-id", "custom-id"])
    assert (r.returncode, r.stdout) == (0, "")
    assert uuids(stub.sessions["custom-id"]) == ["u0", "u1", "u2"]


@pytest.mark.parametrize("missing", ["ORP_API_URL", "ORP_API_KEY"])
def test_missing_env_exits_zero_silently(tmp_path, stub, env, missing):
    path = make_transcript(tmp_path, 2)
    del env[missing]
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")
    assert stub.posts == [] and stub.gets == []
    assert not (tmp_path / "hook.log").exists()


def test_server_down_exits_zero_and_prints_nothing(tmp_path):
    with socket.socket() as s:  # grab a port nobody is listening on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = make_env(tmp_path, "http://127.0.0.1:%d" % port)
    path = make_transcript(tmp_path, 2)
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    assert "sess-1" in (tmp_path / "hook.log").read_text()  # the failure was logged
    assert not (tmp_path / "state" / "sess-1.json").exists()


def test_bad_key_exits_zero_and_logs(tmp_path, stub):
    env = make_env(tmp_path, stub.url, key="wrong")
    path = make_transcript(tmp_path, 2)
    r = run_hook(env, hook_payload(path))
    assert (r.returncode, r.stdout) == (0, "")
    assert stub.sessions == {}
    assert "401" in (tmp_path / "hook.log").read_text()


@pytest.mark.parametrize("stdin", ["", "not json at all", "[1, 2, 3]", '{"session_id": "x"}',
                                   '{"session_id": "x", "transcript_path": "/nonexistent/x.jsonl"}'])
def test_garbage_hook_input_exits_zero(tmp_path, stub, env, stdin):
    r = run_hook(env, stdin=stdin)
    assert (r.returncode, r.stdout) == (0, "")
    assert stub.posts == []


def test_runs_under_system_python3(tmp_path, stub, env):
    system_python = Path("/usr/bin/python3")
    if not system_python.exists():
        pytest.skip("no /usr/bin/python3")
    path = make_transcript(tmp_path, 2)
    r = subprocess.run([str(system_python), str(HOOK)], input=json.dumps(hook_payload(path)), env=env,
                       capture_output=True, text=True, timeout=60)
    assert (r.returncode, r.stdout) == (0, "")
    assert uuids(stub.sessions["sess-1"]) == ["u0", "u1"]
