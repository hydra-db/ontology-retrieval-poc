"""HTTP API.

Every request carries `Authorization: Bearer <key>`; the key identifies the
user, so no endpoint takes a user id.

  GET  /v1/me
  GET  /v1/sessions/{session_id}          -> {"session_id", "next_offset"}
  POST /v1/sessions/{session_id}/events   -> {"next_offset"}   (409 on an offset gap)
  GET  /v1/sessions
  GET  /v1/snippets?status=active
  GET  /v1/unmapped
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .store import OffsetGap, Store

MAX_EVENTS_PER_POST = 1000


class EventBatch(BaseModel):
    client: str = "claude-code"
    offset: int = Field(ge=0)
    events: list[Any] = Field(max_length=MAX_EVENTS_PER_POST)
    meta: dict[str, Any] | None = None


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store(os.environ.get("ORP_DB", "data/orp.db"))
    app = FastAPI(title="ontology-retrieval-poc")
    app.state.store = store

    def user(authorization: str | None = Header(default=None)) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer key")
        user_id = store.user_for_key(authorization[7:].strip())
        if not user_id:
            raise HTTPException(401, "unknown key")
        return user_id

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/v1/me")
    def me(user_id: str = Depends(user)) -> dict:
        return {"user_id": user_id}

    @app.get("/v1/sessions/{session_id}")
    def session_offset(session_id: str, user_id: str = Depends(user)) -> dict:
        return {"session_id": session_id, "next_offset": store.next_offset(user_id, session_id)}

    @app.post("/v1/sessions/{session_id}/events")
    def append_events(session_id: str, batch: EventBatch, user_id: str = Depends(user)):
        try:
            next_offset = store.append_events(
                user_id, session_id, batch.client, batch.offset, batch.events, batch.meta
            )
        except OffsetGap as gap:
            return JSONResponse(status_code=409, content={"next_offset": gap.next_offset})
        return {"next_offset": next_offset}

    @app.get("/v1/sessions")
    def sessions(user_id: str = Depends(user)) -> dict:
        return {"sessions": store.list_sessions(user_id)}

    @app.get("/v1/snippets")
    def snippets(status: str | None = "active", user_id: str = Depends(user)) -> dict:
        return {"snippets": store.list_snippets(user_id, None if status in (None, "", "all") else status)}

    @app.get("/v1/unmapped")
    def unmapped(user_id: str = Depends(user)) -> dict:
        return {"unmapped": store.list_unmapped(user_id)}

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn orp.api:app_factory --factory`."""
    return create_app()
