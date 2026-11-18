"""The tools the analyzer and organizer call.

All checks live here, in code: the quote must be in the transcript, the draft
must fit the schema, and an inferred snippet may not replace a stated one.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from ..schema import ENTITY_TYPES, FIELDS, SnippetDraft
from ..store import Store
from ..transcripts import normalize_ws
from .agent import FINISH_ANALYZER, FINISH_ORGANIZER, Tool, ToolError

# Resolves an alias target's name to an ontology node id. None until the
# ontology store exists; an unresolved target is still used as a plain term.
Resolver = Callable[[str, str], str | None]

_QUOTES = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "description": "Exact words copied from the transcript, one sentence or less."},
            "said_at": {"type": "string", "description": "The timestamp shown at the start of that transcript block, if any."},
        },
        "required": ["quote"],
    },
    "description": "Where the transcript shows this. Each quote must appear in the transcript word for word.",
}

_DRAFT_PROPS: dict[str, Any] = {
    "kind": {"type": "string", "enum": ["alias", "constraint"]},
    "statement": {"type": "string", "description": "One plain present-tense sentence, e.g. \"User says 'turbolay' for HydraDB.\""},
    "trigger_terms": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The user's own words that should make this snippet fire when they appear in a query. Empty only for an always-on 'prefer' constraint.",
    },
    "basis": {"type": "string", "enum": ["stated", "inferred"], "description": "stated: the user said it. inferred: you concluded it from what they did."},
    "quotes": _QUOTES,
    "targets": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The name the user's word stands for."},
                "entity_type": {"type": "string", "enum": ENTITY_TYPES, "description": "Set only if the target is one of these kinds of thing."},
            },
            "required": ["text"],
        },
        "description": "Alias only.",
    },
    "direction": {"type": "string", "enum": ["both", "one_way"], "description": "Alias only. one_way: the trigger term means the target, but not the reverse."},
    "field": {"type": "string", "enum": sorted(FIELDS), "description": "Constraint only."},
    "op": {"type": "string", "enum": ["eq", "not_eq", "in", "within_days", "before", "after"], "description": "Constraint only."},
    "value": {"description": "Constraint only. A string, a list of strings for 'in', or a number of days for 'within_days'."},
    "strength": {"type": "string", "enum": ["require", "prefer"], "description": "Constraint only. require drops records that do not match; prefer only ranks matching records higher."},
}
_DRAFT_REQUIRED = ["kind", "statement", "trigger_terms", "basis", "quotes"]


def _brief(s: dict) -> dict:
    """A snippet as the agents see it."""
    out = {k: s[k] for k in ("id", "kind", "statement", "trigger_terms", "basis", "status")}
    if s["kind"] == "alias":
        out["targets"] = [{k: v for k, v in t.items() if v} for t in s["targets"]]
        out["direction"] = s["direction"]
    else:
        out.update({k: s[k] for k in ("field", "op", "value", "strength")})
    out["evidence"] = [
        {"session_id": e["session_id"], "quote": e["quote"], **({"said_at": e["said_at"]} if e.get("said_at") else {})}
        for e in s["evidence"]
    ]
    for k in ("conflict_with", "note"):
        if s.get(k):
            out[k] = s[k]
    return out


class AnalyzerTools:
    """Tools for one analyzer working on one session's transcript."""

    def __init__(
        self,
        store: Store,
        user_id: str,
        session_id: str,
        pages: list[str],
        full_text: str,
        resolver: Resolver | None = None,
    ):
        self.store = store
        self.user_id = user_id
        self.session_id = session_id
        self.pages = pages
        self._haystack = normalize_ws(full_text).lower()
        self.resolver = resolver
        self.written: list[str] = []

    # -- helpers ----------------------------------------------------------

    def _evidence(self, quotes: Any) -> list[dict]:
        if not isinstance(quotes, list) or not quotes:
            raise ToolError("quotes must be a non-empty list of {quote, said_at}")
        evidence = []
        for q in quotes:
            quote = q.get("quote") if isinstance(q, dict) else q
            if not isinstance(quote, str) or len(normalize_ws(quote)) < 3:
                raise ToolError("each quote must be text copied from the transcript")
            if normalize_ws(quote).lower() not in self._haystack:
                raise ToolError(
                    f"this quote is not in the transcript word for word: {quote!r}. "
                    "Copy the exact words from the transcript, or drop the snippet if the transcript does not say it."
                )
            said_at = q.get("said_at") if isinstance(q, dict) else None
            evidence.append({"session_id": self.session_id, "quote": normalize_ws(quote), "said_at": said_at or None})
        return evidence

    def _draft(self, args: dict) -> dict:
        args = dict(args)
        evidence = self._evidence(args.pop("quotes", None))
        try:
            draft = SnippetDraft(**args, evidence=evidence).model_dump()
        except ValidationError as exc:
            problems = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
            raise ToolError(f"the snippet does not fit the schema: {problems}") from exc
        except TypeError as exc:
            raise ToolError(f"unknown argument: {exc}") from exc
        if draft["kind"] == "alias" and self.resolver:
            for target in draft["targets"]:
                if target.get("entity_type"):
                    target["entity_id"] = self.resolver(target["entity_type"], target["text"])
        return draft

    def _existing(self, snippet_id: str) -> dict:
        snippet = self.store.get_snippet(self.user_id, snippet_id)
        if snippet is None:
            raise ToolError(f"no snippet with id {snippet_id!r}. Call list_snippets to see the ids.")
        return snippet

    # -- tools ------------------------------------------------------------

    def read_transcript(self, page: int = 1) -> str:
        if not isinstance(page, int) or page < 1 or page > len(self.pages):
            raise ToolError(f"page must be between 1 and {len(self.pages)}")
        more = f" Call read_transcript(page={page + 1}) for the next page." if page < len(self.pages) else " This is the last page."
        return f"[page {page} of {len(self.pages)}.{more}]\n\n{self.pages[page - 1]}"

    def list_snippets(self) -> dict:
        snippets = [s for s in self.store.list_snippets(self.user_id) if s["status"] != "superseded"]
        return {"count": len(snippets), "snippets": [_brief(s) for s in snippets]}

    def add_snippet(self, **args: Any) -> dict:
        snippet = self.store.add_snippet(self.user_id, self._draft(args))
        self.written.append(snippet["id"])
        return {"added": snippet["id"]}

    def add_evidence(self, snippet_id: str, quotes: Any, trigger_terms: list[str] | None = None, basis: str | None = None) -> dict:
        self._existing(snippet_id)
        if basis not in (None, "stated", "inferred"):
            raise ToolError("basis must be 'stated' or 'inferred'")
        self.store.add_evidence(self.user_id, snippet_id, self._evidence(quotes), basis)
        if trigger_terms:
            self.store.add_trigger_terms(self.user_id, snippet_id, [" ".join(str(t).split()) for t in trigger_terms if str(t).strip()])
        self.written.append(snippet_id)
        return {"updated": snippet_id}

    def supersede_snippet(self, old_id: str, reason: str, **args: Any) -> dict:
        old = self._existing(old_id)
        draft = self._draft(args)
        if old["basis"] == "stated" and draft["basis"] == "inferred":
            raise ToolError(
                "an inferred snippet cannot replace one the user stated. "
                "Use flag_conflict to keep both and mark the conflict."
            )
        new = self.store.add_snippet(self.user_id, draft)
        self.store.supersede(self.user_id, old_id, new["id"], reason)
        self.written.append(new["id"])
        return {"added": new["id"], "superseded": old_id}

    def flag_conflict(self, existing_id: str, note: str, **args: Any) -> dict:
        self._existing(existing_id)
        new = self.store.add_snippet(self.user_id, self._draft(args))
        self.store.mark_conflict(self.user_id, existing_id, new["id"], note)
        self.written.append(new["id"])
        return {"added": new["id"], "conflict_with": existing_id}

    def add_unmapped(self, statement: str, quotes: Any) -> dict:
        if not isinstance(statement, str) or len(statement.strip()) < 8:
            raise ToolError("statement must be one plain sentence")
        return {"added": self.store.add_unmapped(self.user_id, statement.strip(), self._evidence(quotes))}

    def tools(self) -> list[Tool]:
        draft = {"type": "object", "properties": _DRAFT_PROPS, "required": _DRAFT_REQUIRED}
        return [
            Tool(
                "read_transcript",
                "Read one page of the transcript. Pages start at 1.",
                {"type": "object", "properties": {"page": {"type": "integer", "minimum": 1}}, "required": ["page"]},
                self.read_transcript,
            ),
            Tool(
                "list_snippets",
                "List every snippet already stored for this user that is still in force, with its evidence.",
                {"type": "object", "properties": {}},
                self.list_snippets,
            ),
            Tool("add_snippet", "Store a new snippet. Use only when no stored snippet already says this.", draft, self.add_snippet),
            Tool(
                "add_evidence",
                "The transcript supports a snippet that is already stored: attach the new quotes to it. Optionally add new trigger terms, or raise its basis to 'stated'.",
                {
                    "type": "object",
                    "properties": {
                        "snippet_id": {"type": "string"},
                        "quotes": _QUOTES,
                        "trigger_terms": {"type": "array", "items": {"type": "string"}},
                        "basis": {"type": "string", "enum": ["stated", "inferred"]},
                    },
                    "required": ["snippet_id", "quotes"],
                },
                self.add_evidence,
            ),
            Tool(
                "supersede_snippet",
                "The transcript plainly corrects a stored snippet: store the corrected snippet and mark the old one superseded. The old one is kept.",
                {
                    "type": "object",
                    "properties": {
                        "old_id": {"type": "string"},
                        "reason": {"type": "string", "description": "What changed, in one sentence."},
                        **_DRAFT_PROPS,
                    },
                    "required": ["old_id", "reason", *_DRAFT_REQUIRED],
                },
                self.supersede_snippet,
            ),
            Tool(
                "flag_conflict",
                "The transcript disagrees with a stored snippet and it is not clear which is right: store the new one and mark both as in conflict.",
                {
                    "type": "object",
                    "properties": {
                        "existing_id": {"type": "string"},
                        "note": {"type": "string", "description": "What the two disagree about, in one sentence."},
                        **_DRAFT_PROPS,
                    },
                    "required": ["existing_id", "note", *_DRAFT_REQUIRED],
                },
                self.flag_conflict,
            ),
            Tool(
                "add_unmapped",
                "Record a durable preference that is neither an alias nor a constraint on one of the listed fields. It is kept for people to review and is not used by search.",
                {
                    "type": "object",
                    "properties": {"statement": {"type": "string"}, "quotes": _QUOTES},
                    "required": ["statement", "quotes"],
                },
                self.add_unmapped,
            ),
            FINISH_ANALYZER,
        ]


class OrganizerTools:
    """Tools for the organizer. It reorganizes; it cannot create snippets."""

    def __init__(self, store: Store, user_id: str):
        self.store = store
        self.user_id = user_id
        self.changes: list[dict] = []

    def _get(self, snippet_id: str) -> dict:
        snippet = self.store.get_snippet(self.user_id, snippet_id)
        if snippet is None:
            raise ToolError(f"no snippet with id {snippet_id!r}")
        if snippet["status"] == "superseded":
            raise ToolError(f"snippet {snippet_id} is already superseded")
        return snippet

    def list_snippets(self) -> dict:
        snippets = [s for s in self.store.list_snippets(self.user_id) if s["status"] != "superseded"]
        return {"count": len(snippets), "snippets": [_brief(s) for s in snippets]}

    def merge_snippets(self, keep_id: str, drop_id: str) -> dict:
        if keep_id == drop_id:
            raise ToolError("keep_id and drop_id are the same snippet")
        keep, drop = self._get(keep_id), self._get(drop_id)
        if keep["kind"] != drop["kind"]:
            raise ToolError("an alias and a constraint cannot be merged")
        if keep["kind"] == "constraint":
            same = all(keep[k] == drop[k] for k in ("field", "op", "value", "strength"))
            if not same:
                raise ToolError(
                    "these two constraints do not have the same field, op, value and strength, so they are not "
                    "duplicates. Use supersede if one corrects the other, or mark_conflict."
                )
        self.store.add_evidence(self.user_id, keep_id, drop["evidence"], drop["basis"])
        self.store.add_trigger_terms(self.user_id, keep_id, drop["trigger_terms"])
        self.store.supersede(self.user_id, drop_id, keep_id, "duplicate, merged")
        self.changes.append({"merged": drop_id, "into": keep_id})
        return {"merged": drop_id, "into": keep_id}

    def supersede(self, old_id: str, new_id: str, reason: str) -> dict:
        if old_id == new_id:
            raise ToolError("old_id and new_id are the same snippet")
        old, new = self._get(old_id), self._get(new_id)
        if old["basis"] == "stated" and new["basis"] == "inferred":
            raise ToolError("an inferred snippet cannot replace one the user stated. Use mark_conflict.")
        self.store.supersede(self.user_id, old_id, new_id, reason)
        self.changes.append({"superseded": old_id, "by": new_id})
        return {"superseded": old_id, "by": new_id}

    def mark_conflict(self, a_id: str, b_id: str, note: str) -> dict:
        if a_id == b_id:
            raise ToolError("a_id and b_id are the same snippet")
        self._get(a_id), self._get(b_id)
        self.store.mark_conflict(self.user_id, a_id, b_id, note)
        self.changes.append({"conflict": [a_id, b_id]})
        return {"conflict": [a_id, b_id]}

    def tools(self) -> list[Tool]:
        return [
            Tool("list_snippets", "List every snippet still in force for this user, with its evidence.", {"type": "object", "properties": {}}, self.list_snippets),
            Tool(
                "merge_snippets",
                "Two snippets say the same thing: move the evidence and trigger terms of drop_id onto keep_id and retire drop_id. Nothing is deleted.",
                {"type": "object", "properties": {"keep_id": {"type": "string"}, "drop_id": {"type": "string"}}, "required": ["keep_id", "drop_id"]},
                self.merge_snippets,
            ),
            Tool(
                "supersede",
                "new_id is plainly the later correction of old_id: mark old_id superseded. It is kept.",
                {
                    "type": "object",
                    "properties": {"old_id": {"type": "string"}, "new_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["old_id", "new_id", "reason"],
                },
                self.supersede,
            ),
            Tool(
                "mark_conflict",
                "Two snippets cannot both be true and it is not clear which is right: mark both as in conflict.",
                {
                    "type": "object",
                    "properties": {"a_id": {"type": "string"}, "b_id": {"type": "string"}, "note": {"type": "string"}},
                    "required": ["a_id", "b_id", "note"],
                },
                self.mark_conflict,
            ),
            FINISH_ORGANIZER,
        ]
