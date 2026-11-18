"""Typed snippet schema.

A snippet is one thing learned about how a user refers to or wants records.
There are two kinds, split by what the snippet can do to search results:

- alias: gives a thing another name. The right-hand side is a term or an entity.
  It can only add ways to match; it never removes a record.
- constraint: says which records count or come first. The right-hand side is a
  condition on a field. strength=require drops non-matching records,
  strength=prefer only boosts matching ones.

Everything the model writes goes through `validate_snippet`; the model never
writes to the store directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Shared field list. Every constraint is written against one of these fields.
# Both stores expose the same fields: ontology nodes as properties, vector-store
# chunks as metadata copied from their node. Provisional until the ontology
# schema is fixed.
FIELDS: dict[str, dict[str, Any]] = {
    "source": {
        "description": "Which system the record came from.",
        "values": ["linear", "github", "fireflies", "paper"],
        "ops": ["eq", "not_eq", "in"],
    },
    "record_type": {
        "description": "What kind of record it is.",
        "values": ["ticket", "comment", "pull_request", "review", "meeting", "paper"],
        "ops": ["eq", "not_eq", "in"],
    },
    "state": {
        "description": "Workflow state of a ticket or pull request, as the source names it.",
        "values": None,
        "ops": ["eq", "not_eq", "in"],
        "record_types": ["ticket", "pull_request"],
    },
    "team": {
        "description": "Team that owns the record.",
        "values": None,
        "ops": ["eq", "not_eq", "in"],
        "record_types": ["ticket", "comment"],
    },
    "project": {
        "description": "Project or repository the record belongs to.",
        "values": None,
        "ops": ["eq", "not_eq", "in"],
        "record_types": ["ticket", "comment", "pull_request", "review"],
    },
    "person": {
        "description": "Person who wrote, owns or spoke in the record.",
        "values": None,
        "ops": ["eq", "not_eq", "in"],
    },
    "organization": {
        "description": "Customer or outside organization the record is about.",
        "values": None,
        "ops": ["eq", "not_eq", "in"],
    },
    "date": {
        "description": "When the record was created or last changed.",
        "values": None,
        "ops": ["within_days", "before", "after"],
    },
}

ENTITY_TYPES = ["person", "team", "project", "organization", "ticket", "pull_request", "meeting", "paper"]

Kind = Literal["alias", "constraint"]
Status = Literal["active", "superseded", "conflict"]
Basis = Literal["stated", "inferred"]


class Evidence(BaseModel):
    session_id: str
    quote: str = Field(min_length=3)
    said_at: str | None = None


class AliasTarget(BaseModel):
    """What the user's word stands for.

    `text` is the name as the rest of the world writes it. If the thing is an
    entity in the ontology store, `entity_type` says which type. `entity_id` is
    filled in by code once the name resolves to a node; the model never sets it,
    and an unresolved target is still usable as a plain term.
    """

    text: str = Field(min_length=1)
    entity_type: str | None = None
    entity_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "AliasTarget":
        self.text = " ".join(self.text.split())
        if not self.text:
            raise ValueError("a target needs text: the name the user's word stands for")
        if self.entity_type is not None and self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}, or left out for a plain term")
        if self.entity_id and not self.entity_type:
            raise ValueError("entity_id needs entity_type")
        return self


class SnippetDraft(BaseModel):
    """What the analyzer submits. Ids, dates, status and counters are set by code."""

    kind: Kind
    statement: str = Field(min_length=8, max_length=300)
    trigger_terms: list[str] = Field(default_factory=list)
    basis: Basis = "stated"
    evidence: list[Evidence] = Field(min_length=1)

    # alias only
    targets: list[AliasTarget] = Field(default_factory=list)
    direction: Literal["both", "one_way"] | None = None

    # constraint only
    field: str | None = None
    op: str | None = None
    value: Any = None
    strength: Literal["require", "prefer"] | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> "SnippetDraft":
        self.trigger_terms = _clean_terms(self.trigger_terms)
        if self.kind == "alias":
            if not self.trigger_terms:
                raise ValueError("an alias needs at least one trigger term: the user's own word for the thing")
            if not self.targets:
                raise ValueError("an alias needs at least one target: the term or entity it stands for")
            if any(v is not None for v in (self.field, self.op, self.value, self.strength)):
                raise ValueError(
                    "an alias has no field, op, value or strength. If the right-hand side is a "
                    "condition on a field, this is a constraint"
                )
            self.direction = self.direction or "both"
            lowered = {t.lower() for t in self.trigger_terms}
            for target in self.targets:
                if target.text.lower() in lowered and len(self.targets) == 1:
                    raise ValueError("the alias target is the same as its trigger term")
        else:
            if self.targets or self.direction:
                raise ValueError(
                    "a constraint has no targets or direction. If the right-hand side is just "
                    "another name for the same thing, this is an alias"
                )
            if self.field not in FIELDS:
                raise ValueError(f"field must be one of {sorted(FIELDS)}")
            spec = FIELDS[self.field]
            if self.op not in spec["ops"]:
                raise ValueError(f"op for field '{self.field}' must be one of {spec['ops']}")
            if self.strength not in ("require", "prefer"):
                raise ValueError("a constraint needs strength: 'require' or 'prefer'")
            self.value = _check_value(self.field, self.op, self.value)
            if not self.trigger_terms and self.strength == "require":
                raise ValueError(
                    "a constraint with no trigger terms is always on, and an always-on constraint "
                    "may only be 'prefer'. Add the trigger terms, or change strength to 'prefer'"
                )
        return self


def _clean_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        t = " ".join(str(term).split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _check_value(field: str, op: str, value: Any) -> Any:
    spec = FIELDS[field]
    if value is None or value == "" or value == []:
        raise ValueError("a constraint needs a value")
    if op == "within_days":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("value for within_days must be a positive number of days")
        return int(value)
    if op in ("before", "after"):
        if not isinstance(value, str):
            raise ValueError(f"value for {op} must be an ISO date string, e.g. 2026-10-01")
        return value
    if op == "in":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("value for 'in' must be a list of strings")
        values = value
    else:
        if not isinstance(value, str):
            raise ValueError(f"value for '{op}' must be a string")
        values = [value]
    allowed = spec["values"]
    if allowed is not None:
        bad = [v for v in values if v not in allowed]
        if bad:
            raise ValueError(f"value(s) {bad} not allowed for field '{field}'; allowed: {allowed}")
    return value


def field_guide() -> str:
    """The field list as text, for the agent prompts."""
    lines = []
    for name, spec in FIELDS.items():
        values = f" Allowed values: {', '.join(spec['values'])}." if spec["values"] else ""
        only = f" Only exists on: {', '.join(spec['record_types'])}." if spec.get("record_types") else ""
        lines.append(f"- {name}: {spec['description']}{values}{only} Ops: {', '.join(spec['ops'])}.")
    return "\n".join(lines)
