"""SQLite storage for users, raw transcript events, snippets and dream runs.

Raw events are stored exactly as the client sent them. Snippets are never
deleted: a replaced snippet is marked superseded and kept.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(id),
  client TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',
  next_offset INTEGER NOT NULL DEFAULT 0,
  dreamed_offset INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  last_event_at TEXT NOT NULL,
  PRIMARY KEY (user_id, id)
);
CREATE TABLE IF NOT EXISTS events (
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  offset INTEGER NOT NULL,
  body TEXT NOT NULL,
  received_at TEXT NOT NULL,
  PRIMARY KEY (user_id, session_id, offset)
);
CREATE TABLE IF NOT EXISTS snippets (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  kind TEXT NOT NULL,
  statement TEXT NOT NULL,
  trigger_terms TEXT NOT NULL,
  body TEXT NOT NULL,
  evidence TEXT NOT NULL,
  basis TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  superseded_by TEXT,
  conflict_with TEXT,
  note TEXT,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS snippets_user ON snippets(user_id, status);
CREATE TABLE IF NOT EXISTS unmapped (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  statement TEXT NOT NULL,
  evidence TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  report TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class OffsetGap(Exception):
    """The client sent events starting beyond what the server has."""

    def __init__(self, next_offset: int):
        super().__init__(f"expected offset <= {next_offset}")
        self.next_offset = next_offset


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    # ---- users and keys -------------------------------------------------

    def create_user(self, name: str) -> tuple[str, str]:
        """Create a user and return (user_id, api_key). The key is shown once."""
        user_id = "u_" + uuid.uuid4().hex[:12]
        key = "orp_" + secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.execute("INSERT INTO users VALUES (?,?,?)", (user_id, name, now_iso()))
                self._db.execute("INSERT INTO api_keys VALUES (?,?,?)", (_hash_key(key), user_id, now_iso()))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return user_id, key

    def user_for_key(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT user_id FROM api_keys WHERE key_hash=?", (_hash_key(key),)).fetchone()
        return row["user_id"] if row else None

    def list_users(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM users ORDER BY created_at")]

    def user_id_by_name(self, name: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None

    # ---- transcripts ----------------------------------------------------

    def next_offset(self, user_id: str, session_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT next_offset FROM sessions WHERE user_id=? AND id=?", (user_id, session_id)
            ).fetchone()
        return row["next_offset"] if row else 0

    def append_events(
        self, user_id: str, session_id: str, client: str, offset: int, events: list[Any], meta: dict | None = None
    ) -> int:
        """Append events starting at `offset`. Events the server already has are skipped."""
        ts = now_iso()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT next_offset, meta FROM sessions WHERE user_id=? AND id=?", (user_id, session_id)
                ).fetchone()
                have = row["next_offset"] if row else 0
                if offset > have:
                    raise OffsetGap(have)
                fresh = events[have - offset :]
                for i, event in enumerate(fresh):
                    self._db.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?)",
                        (user_id, session_id, have + i, json.dumps(event, ensure_ascii=False), ts),
                    )
                new_next = have + len(fresh)
                if row is None:
                    self._db.execute(
                        "INSERT INTO sessions (id,user_id,client,meta,next_offset,created_at,last_event_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (session_id, user_id, client, json.dumps(meta or {}), new_next, ts, ts),
                    )
                elif fresh or meta:
                    merged = {**json.loads(row["meta"]), **(meta or {})}
                    self._db.execute(
                        "UPDATE sessions SET next_offset=?, meta=?, last_event_at=CASE WHEN ? THEN ? ELSE last_event_at END "
                        "WHERE user_id=? AND id=?",
                        (new_next, json.dumps(merged), 1 if fresh else 0, ts, user_id, session_id),
                    )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return new_next

    def get_events(self, user_id: str, session_id: str, start: int = 0, end: int | None = None) -> list[Any]:
        q = "SELECT body FROM events WHERE user_id=? AND session_id=? AND offset>=?"
        args: list[Any] = [user_id, session_id, start]
        if end is not None:
            q += " AND offset<?"
            args.append(end)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY offset", args).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def get_session(self, user_id: str, session_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM sessions WHERE user_id=? AND id=?", (user_id, session_id)).fetchone()
        return _session(row) if row else None

    def list_sessions(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions WHERE user_id=? ORDER BY last_event_at", (user_id,)
            ).fetchall()
        return [_session(r) for r in rows]

    def sessions_to_dream(self, user_id: str, quiet_before: str) -> list[dict]:
        """Sessions with events not yet dreamed over, and no new event since `quiet_before`."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions WHERE user_id=? AND next_offset>dreamed_offset AND last_event_at<=? "
                "ORDER BY last_event_at",
                (user_id, quiet_before),
            ).fetchall()
        return [_session(r) for r in rows]

    def mark_dreamed(self, user_id: str, session_id: str, offset: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET dreamed_offset=MAX(dreamed_offset, ?) WHERE user_id=? AND id=?",
                (offset, user_id, session_id),
            )

    # ---- snippets -------------------------------------------------------

    def add_snippet(self, user_id: str, draft: dict) -> dict:
        snippet_id = "s_" + uuid.uuid4().hex[:10]
        ts = now_iso()
        body = {k: draft.get(k) for k in ("targets", "direction", "field", "op", "value", "strength")}
        with self._lock:
            self._db.execute(
                "INSERT INTO snippets (id,user_id,kind,statement,trigger_terms,body,evidence,basis,first_seen,last_seen) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    snippet_id,
                    user_id,
                    draft["kind"],
                    draft["statement"],
                    json.dumps(draft["trigger_terms"]),
                    json.dumps(body),
                    json.dumps(draft["evidence"]),
                    draft["basis"],
                    ts,
                    ts,
                ),
            )
        return self.get_snippet(user_id, snippet_id)  # type: ignore[return-value]

    def get_snippet(self, user_id: str, snippet_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM snippets WHERE user_id=? AND id=?", (user_id, snippet_id)).fetchone()
        return _snippet(row) if row else None

    def list_snippets(self, user_id: str, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM snippets WHERE user_id=?", [user_id]
        if status:
            q += " AND status=?"
            args.append(status)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY first_seen, id", args).fetchall()
        return [_snippet(r) for r in rows]

    def add_evidence(self, user_id: str, snippet_id: str, evidence: list[dict], basis: str | None = None) -> dict:
        with self._lock:
            current = self.get_snippet(user_id, snippet_id)
            if current is None:
                raise KeyError(snippet_id)
            seen = {(e["session_id"], e["quote"]) for e in current["evidence"]}
            merged = current["evidence"] + [e for e in evidence if (e["session_id"], e["quote"]) not in seen]
            # "stated" outranks "inferred": once the user has said it, it stays stated.
            new_basis = "stated" if "stated" in (current["basis"], basis) else current["basis"]
            self._db.execute(
                "UPDATE snippets SET evidence=?, basis=?, last_seen=? WHERE user_id=? AND id=?",
                (json.dumps(merged), new_basis, now_iso(), user_id, snippet_id),
            )
            return self.get_snippet(user_id, snippet_id)  # type: ignore[return-value]

    def add_trigger_terms(self, user_id: str, snippet_id: str, terms: list[str]) -> None:
        with self._lock:
            current = self.get_snippet(user_id, snippet_id)
            if current is None:
                raise KeyError(snippet_id)
            have = {t.lower() for t in current["trigger_terms"]}
            merged = current["trigger_terms"] + [t for t in terms if t.lower() not in have]
            self._db.execute(
                "UPDATE snippets SET trigger_terms=? WHERE user_id=? AND id=?",
                (json.dumps(merged), user_id, snippet_id),
            )

    def supersede(self, user_id: str, old_id: str, new_id: str, note: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE snippets SET status='superseded', superseded_by=?, note=? WHERE user_id=? AND id=?",
                (new_id, note, user_id, old_id),
            )

    def mark_conflict(self, user_id: str, a_id: str, b_id: str, note: str) -> None:
        with self._lock:
            for one, other in ((a_id, b_id), (b_id, a_id)):
                self._db.execute(
                    "UPDATE snippets SET status='conflict', conflict_with=?, note=? WHERE user_id=? AND id=?",
                    (other, note, user_id, one),
                )

    def add_unmapped(self, user_id: str, statement: str, evidence: list[dict]) -> str:
        unmapped_id = "n_" + uuid.uuid4().hex[:10]
        with self._lock:
            self._db.execute(
                "INSERT INTO unmapped VALUES (?,?,?,?,?)",
                (unmapped_id, user_id, statement, json.dumps(evidence), now_iso()),
            )
        return unmapped_id

    def list_unmapped(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM unmapped WHERE user_id=? ORDER BY created_at", (user_id,)).fetchall()
        return [{**dict(r), "evidence": json.loads(r["evidence"])} for r in rows]

    # ---- runs -----------------------------------------------------------

    def start_run(self, user_id: str) -> str:
        run_id = "r_" + uuid.uuid4().hex[:10]
        with self._lock:
            self._db.execute("INSERT INTO runs (id,user_id,started_at) VALUES (?,?,?)", (run_id, user_id, now_iso()))
        return run_id

    def finish_run(self, run_id: str, report: dict) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE runs SET finished_at=?, report=? WHERE id=?", (now_iso(), json.dumps(report), run_id)
            )

    def list_runs(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM runs WHERE user_id=? ORDER BY started_at", (user_id,)).fetchall()
        return [{**dict(r), "report": json.loads(r["report"]) if r["report"] else None} for r in rows]


def _session(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["meta"] = json.loads(d["meta"])
    return d


def _snippet(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["trigger_terms"] = json.loads(d["trigger_terms"])
    d["evidence"] = json.loads(d["evidence"])
    body = json.loads(d.pop("body"))
    keys = ("targets", "direction") if d["kind"] == "alias" else ("field", "op", "value", "strength")
    for k in keys:
        d[k] = body.get(k)
    return d
