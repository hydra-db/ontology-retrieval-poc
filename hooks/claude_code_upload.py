#!/usr/bin/env python3
"""Claude Code hook: upload new session-transcript lines to the ORP API.

Stdlib only; runs with the system python3 (3.8+). Two ways to invoke it:

  as a hook   Claude Code pipes the hook JSON on stdin (session_id,
              transcript_path, hook_event_name, cwd, and for SubagentStop also
              agent_id / agent_transcript_path).
  backfill    python3 hooks/claude_code_upload.py --file <path.jsonl> [--session-id X]

Config (env): ORP_API_URL, ORP_API_KEY. If either is missing the script exits 0
without doing anything. Optional: ORP_STATE_DIR (default ~/.orp/state),
ORP_HOOK_LOG (default ~/.orp/hook.log), ORP_TIME_BUDGET (seconds, default 20
for hooks, 600 for backfill).

Offsets: a session's offset is the number of complete JSONL lines already
uploaded, so event N is always line N of the file. Every complete line becomes
exactly one event; a line that is not a JSON object is sent as
{"type": "unparsed", "raw": "<line>"} to keep that alignment. A trailing line
without a newline is still being written and is left for the next run.

Where subagent activity lives (checked against Claude Code 2.1.x files):
  * The main transcript is <project-dir>/<session-id>.jsonl. It holds only the
    parent's side of a subagent run: the `Agent` tool_use and its tool_result
    (the subagent's final report).
  * Everything the subagent itself did (its prompt, thinking, tool calls, tool
    results) is in a separate file next to the main transcript:
        <project-dir>/<session-id>/subagents/agent-<agent_id>.jsonl
    with a sibling agent-<agent_id>.meta.json (agentType, description,
    toolUseId of the parent's Agent call, parentAgentId for nested agents).
    Lines there carry isSidechain=true, agentId=<agent_id> and the PARENT's
    sessionId.
  * Agents started by a workflow sit one level deeper,
        <session-id>/subagents/workflows/<wf_id>/agent-<agent_id>.jsonl
    alongside a journal.jsonl (launched / started / result lines).
  * SubagentStop passes the subagent file as `agent_transcript_path`;
    `transcript_path` is always the main session file.
So this script uploads the main file as session `<session_id>` and every .jsonl
under <session-id>/subagents/ as its own session `<session_id>:<agent_id>`
(`<session_id>:<wf_id>.journal` for workflow journals), with
meta.parent_session_id set.

This must never disturb the user's session: nothing is written to stdout (for
PreCompact, stdout would be injected as compaction instructions), every error
is caught and logged, and the exit code is always 0.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import fcntl
except ImportError:  # not on Windows; uploads then run without the lock
    fcntl = None

CLIENT = "claude-code"
USER_AGENT = "orp-hook/0.1"
MAX_BATCH_EVENTS = 200
MAX_BATCH_BYTES = 1_000_000
REQUEST_TIMEOUT = 5.0
HOOK_TIME_BUDGET = 20.0
BACKFILL_TIME_BUDGET = 600.0
LOG_MAX_BYTES = 256 * 1024
LOG_KEEP_BYTES = 64 * 1024
AGENT_META_KEYS = ("agentType", "description", "toolUseId", "parentAgentId", "spawnDepth", "isFork", "workflowPhase")


def _orp_home():
    return os.path.join(os.path.expanduser("~"), ".orp")


def _log_path():
    return os.environ.get("ORP_HOOK_LOG") or os.path.join(_orp_home(), "hook.log")


def _state_dir():
    return os.environ.get("ORP_STATE_DIR") or os.path.join(_orp_home(), "state")


def log(msg):
    """Append one line to the hook log; trim the log when it grows past LOG_MAX_BYTES."""
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            if os.path.getsize(path) > LOG_MAX_BYTES:
                with open(path, "rb") as f:
                    f.seek(-LOG_KEEP_BYTES, os.SEEK_END)
                    tail = f.read()
                with open(path, "wb") as f:
                    f.write(tail[tail.find(b"\n") + 1:])
        except OSError:
            pass
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(path, "a", encoding="utf-8") as f:
            f.write("%s pid=%d %s\n" % (stamp, os.getpid(), str(msg).replace("\n", " ")[:2000]))
    except Exception:
        pass


# ---------------------------------------------------------------- local state

def _state_path(session_id):
    return os.path.join(_state_dir(), re.sub(r"[^A-Za-z0-9._-]", "_", session_id) + ".json")


def load_state(session_id, path):
    """Return (next_offset, byte_pos) saved for this session+file, or None."""
    try:
        with open(_state_path(session_id), encoding="utf-8") as f:
            st = json.load(f)
        if st.get("session_id") == session_id and st.get("path") == path:
            return int(st["next_offset"]), int(st["byte_pos"])
    except Exception:
        pass
    return None


def save_state(session_id, path, next_offset, byte_pos):
    try:
        target = _state_path(session_id)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = "%s.%d.tmp" % (target, os.getpid())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id, "path": path, "next_offset": next_offset, "byte_pos": byte_pos}, f)
        os.replace(tmp, target)
    except Exception as e:
        log("state save failed for %s: %r" % (session_id, e))


class _Lock(object):
    """Non-blocking per-session lock so overlapping hook runs (e.g. several
    SubagentStop events at once) do not upload the same lines twice."""

    def __init__(self, session_id):
        self.path = _state_path(session_id) + ".lock"
        self.fh = None

    def acquire(self):
        if fcntl is None:
            return True
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.fh = open(self.path, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except Exception:
            self.release()
            return False

    def release(self):
        try:
            if self.fh is not None:
                self.fh.close()
        except Exception:
            pass
        self.fh = None


# ------------------------------------------------------------------- API client

class ServerError(Exception):
    """The server answered in a way that will not improve for the next file."""


class Api(object):
    def __init__(self, base_url, api_key, deadline):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.deadline = deadline

    def time_left(self):
        return self.deadline - time.monotonic()

    def _request(self, method, session_id, suffix="", body=None):
        """Return (status, parsed-json-or-None). Raises on network errors / timeouts."""
        left = self.time_left()
        if left <= 0.05:
            raise TimeoutError("time budget exhausted")
        url = "%s/v1/sessions/%s%s" % (self.base, urllib.parse.quote(session_id, safe=""), suffix)
        headers = {"Authorization": "Bearer " + self.key, "User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=min(REQUEST_TIMEOUT, left)) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = None
        return status, parsed

    def get_offset(self, session_id):
        status, body = self._request("GET", session_id)
        if status == 200 and isinstance(body, dict) and isinstance(body.get("next_offset"), int):
            return body["next_offset"]
        raise ServerError("GET session -> HTTP %s" % status)

    def post_events(self, session_id, offset, events, meta):
        body = {"client": CLIENT, "offset": offset, "events": events, "meta": meta}
        return self._request("POST", session_id, "/events", body)


# ------------------------------------------------------------- transcript file

def _seek_to_line(f, n):
    """Position f at the start of complete line n (0-based). False if the file has fewer."""
    f.seek(0)
    for _ in range(n):
        if not f.readline().endswith(b"\n"):
            return False
    return True


def _parse_line(raw):
    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"type": "unparsed", "raw": text}


def _read_batch(f):
    """Read up to one batch of complete lines from the current position.
    Leaves f positioned right after the last line returned."""
    events, size = [], 0
    while len(events) < MAX_BATCH_EVENTS:
        pos = f.tell()
        raw = f.readline()
        if not raw.endswith(b"\n"):  # EOF or a line still being written
            f.seek(pos)
            break
        if events and size + len(raw) > MAX_BATCH_BYTES:
            f.seek(pos)
            break
        events.append(_parse_line(raw))
        size += len(raw)
    return events


def upload_file(api, session_id, path, meta):
    """Upload the not-yet-uploaded complete lines of one transcript file.
    Returns the number of events the server accepted in this run."""
    lock = _Lock(session_id)
    if not lock.acquire():
        return 0
    try:
        return _upload_locked(api, session_id, path, meta)
    finally:
        lock.release()


def _upload_locked(api, session_id, path, meta):
    size = os.path.getsize(path)
    state = load_state(session_id, path)
    if state is not None and state[1] == size:
        return 0  # nothing new since the last run
    sent = 0
    with open(path, "rb") as f:
        if state is not None and state[1] <= size:
            offset = state[0]
            f.seek(state[1])
        else:  # no state (or the file shrank): ask the server where it is
            offset = api.get_offset(session_id)
            if not _seek_to_line(f, offset):
                log("%s: server next_offset %d is past the end of %s" % (session_id, offset, path))
                return 0
        if meta.get("cwd") is None:
            meta = dict(meta, cwd=_peek_cwd(f))
        resynced = False
        while api.time_left() > 0:
            events = _read_batch(f)
            if not events:
                save_state(session_id, path, offset, f.tell())
                break
            status, body = api.post_events(session_id, offset, events, meta)
            server_next = body.get("next_offset") if isinstance(body, dict) else None
            expected = offset + len(events)
            if status == 200 and server_next in (None, expected):
                sent += len(events)
                offset = expected
                save_state(session_id, path, offset, f.tell())
                resynced = False
                continue
            if status in (200, 409) and isinstance(server_next, int) and not resynced:
                # Server is at a different offset than we assumed: jump there and retry once.
                resynced = True
                offset = server_next
                if not _seek_to_line(f, offset):
                    log("%s: resync to %d is past the end of %s" % (session_id, offset, path))
                    break
                continue
            msg = "POST offset=%d n=%d -> HTTP %s %r" % (offset, len(events), status, body)
            if status in (401, 403) or status >= 500:
                raise ServerError(msg)  # server-wide problem: stop the whole run
            log("%s: %s" % (session_id, msg))  # problem with this batch only
            break
    return sent


def _peek_cwd(f):
    """Best-effort cwd for backfill: the first `cwd` field in the next few lines."""
    pos = f.tell()
    try:
        for _ in range(50):
            raw = f.readline()
            if not raw.endswith(b"\n"):
                break
            ev = _parse_line(raw)
            if isinstance(ev.get("cwd"), str):
                return ev["cwd"]
    except Exception:
        pass
    finally:
        f.seek(pos)
    return None


# ------------------------------------------------------------ subagent discovery

def _sub_name(path):
    """Session-id suffix for a file under <session>/subagents/: the agent id for
    agent-<id>.jsonl, otherwise '<parent dir>.<stem>' (e.g. 'wf_ab12.journal')."""
    stem = os.path.basename(path)
    if stem.endswith(".jsonl"):
        stem = stem[:-len(".jsonl")]
    if stem.startswith("agent-") and len(stem) > len("agent-"):
        return stem[len("agent-"):], True
    return "%s.%s" % (os.path.basename(os.path.dirname(path)), stem), False


def _agent_meta(path):
    try:
        with open(path[:-len(".jsonl")] + ".meta.json", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: raw[k] for k in AGENT_META_KEYS if k in raw}
    except Exception:
        return {}


def _subagent_target(parent_id, path, base_meta):
    name, is_agent = _sub_name(path)
    meta = dict(base_meta, transcript_path=path, parent_session_id=parent_id)
    if is_agent:
        meta["agent_id"] = name
        info = _agent_meta(path)
        if info:
            meta["agent"] = info
    return ("%s:%s" % (parent_id, name), path, meta)


def _subagent_files(main_path):
    root = os.path.join(main_path[:-len(".jsonl")] if main_path.endswith(".jsonl") else main_path, "subagents")
    if not os.path.isdir(root):
        return []
    files = glob.glob(os.path.join(glob.escape(root), "**", "*.jsonl"), recursive=True)

    def mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    return sorted(files, key=mtime, reverse=True)  # most recently active first


def build_targets(session_id, main_path, base_meta, first_sub=None, sweep=True):
    """[(session_id, path, meta)] for the main file and its subagent files."""
    targets, seen = [], set()

    def add(t):
        real = os.path.realpath(t[1])
        if real not in seen and os.path.isfile(t[1]):
            seen.add(real)
            targets.append(t)

    if first_sub:
        add(_subagent_target(session_id, first_sub, base_meta))
    if main_path:
        add((session_id, main_path, dict(base_meta, transcript_path=main_path)))
        if sweep:
            for p in _subagent_files(main_path):
                add(_subagent_target(session_id, p, base_meta))
    return targets


# ------------------------------------------------------------------------- main

def _budget(default):
    try:
        return float(os.environ["ORP_TIME_BUDGET"])
    except Exception:
        return default


def _expand(p):
    return os.path.abspath(os.path.expanduser(p)) if isinstance(p, str) and p else None


def run(argv):
    api_url = (os.environ.get("ORP_API_URL") or "").strip()
    api_key = (os.environ.get("ORP_API_KEY") or "").strip()
    if not api_url or not api_key:
        return

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--file")
    parser.add_argument("--session-id")
    args = parser.parse_args(argv)

    if args.file:  # manual backfill
        path = _expand(args.file)
        base_meta = {"cwd": None, "hook_event": "backfill"}
        marker = os.sep + "subagents" + os.sep
        if not args.session_id and marker in path:
            # Pointed straight at a subagent file: keep the <parent>:<agent> id scheme.
            parent = os.path.basename(path.split(marker)[0])
            targets = [_subagent_target(parent, path, base_meta)]
        else:
            stem = os.path.basename(path)
            sid = args.session_id or (stem[:-len(".jsonl")] if stem.endswith(".jsonl") else stem)
            targets = build_targets(sid, path, base_meta)
        budget = _budget(BACKFILL_TIME_BUDGET)
    else:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            return
        sid = payload.get("session_id")
        main_path = _expand(payload.get("transcript_path"))
        if not isinstance(sid, str) or not sid or not main_path:
            log("hook input without session_id/transcript_path: keys=%s" % sorted(payload))
            return
        event = payload.get("hook_event_name")
        base_meta = {"cwd": payload.get("cwd"), "hook_event": event}
        # SubagentStop can fire many times in quick succession, so it uploads only the
        # finished subagent's file plus the main file; the other events sweep every
        # subagent file (unchanged files cost one stat + one small state read).
        targets = build_targets(sid, main_path, base_meta,
                                first_sub=_expand(payload.get("agent_transcript_path")),
                                sweep=(event != "SubagentStop"))
        budget = _budget(HOOK_TIME_BUDGET)

    api = Api(api_url, api_key, time.monotonic() + budget)
    total = 0
    for session_id, path, meta in targets:
        if api.time_left() <= 0:
            log("time budget used up; remaining files wait for the next run")
            break
        try:
            total += upload_file(api, session_id, path, meta)
        except Exception as e:
            log("%s: %s: %r" % (session_id, type(e).__name__, e))
            if not isinstance(e, (FileNotFoundError, PermissionError)):
                break  # server unreachable or rejecting us: do not repeat that for every file
    if args.file:
        sys.stderr.write("uploaded %d events from %d file(s)\n" % (total, len(targets)))


def main():
    real_stdout = sys.stdout
    try:
        # Nothing may reach stdout: Claude Code interprets hook stdout.
        sys.stdout = open(os.devnull, "w")
    except Exception:
        pass
    try:
        run(sys.argv[1:])
    except BaseException as e:  # includes SystemExit from argparse and KeyboardInterrupt
        if not (isinstance(e, SystemExit) and e.code in (0, None)):
            log("fatal: %s: %r" % (type(e).__name__, e))
    finally:
        sys.stdout = real_stdout
    sys.exit(0)


if __name__ == "__main__":
    main()
