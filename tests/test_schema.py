import pytest
from pydantic import ValidationError

from orp.schema import SnippetDraft, field_guide

EV = [{"session_id": "s1", "quote": "turbolay and hydradb are the same thing"}]


def alias(**over):
    base = dict(kind="alias", statement="User says 'turbolay' for hydradb.", trigger_terms=["turbolay"],
                targets=[{"text": "hydradb"}], evidence=EV)
    return SnippetDraft(**{**base, **over})


def constraint(**over):
    base = dict(kind="constraint", statement="By 'shipped' the user means state is On main.",
                trigger_terms=["shipped"], field="state", op="eq", value="On main", strength="require", evidence=EV)
    return SnippetDraft(**{**base, **over})


def test_alias_ok_and_defaults():
    d = alias(trigger_terms=["turbolay", " Turbolay ", "turbo  lay"])
    assert d.direction == "both"
    assert d.trigger_terms == ["turbolay", "turbo lay"]


def test_alias_entity_target():
    d = alias(trigger_terms=["CK"], targets=[{"text": "Chaithanya  Kumar", "entity_type": "person"}], direction="one_way")
    assert d.targets[0].text == "Chaithanya Kumar"
    with pytest.raises(ValidationError):
        alias(targets=[{"text": "x", "entity_type": "spaceship"}])


@pytest.mark.parametrize("over", [
    {"trigger_terms": []},
    {"targets": []},
    {"field": "state", "op": "eq", "value": "On main", "strength": "require"},
    {"targets": [{"text": "Turbolay"}]},
    {"evidence": []},
])
def test_alias_rejected(over):
    with pytest.raises(ValidationError):
        alias(**over)


def test_constraint_ok():
    assert constraint().value == "On main"
    assert constraint(field="date", op="within_days", value=14.0, trigger_terms=["recent"]).value == 14
    assert constraint(field="source", op="eq", value="linear", strength="prefer", trigger_terms=["roadmap"]).strength == "prefer"
    # always on is allowed only as prefer
    assert constraint(field="project", op="eq", value="ingestion", strength="prefer", trigger_terms=[]).trigger_terms == []


@pytest.mark.parametrize("over", [
    {"field": "colour"},
    {"op": "within_days"},
    {"field": "source", "value": "slack"},
    {"field": "date", "op": "within_days", "value": -3},
    {"field": "date", "op": "within_days", "value": True},
    {"op": "in", "value": "On main"},
    {"strength": None},
    {"trigger_terms": []},
    {"targets": [{"text": "x"}]},
    {"value": ""},
])
def test_constraint_rejected(over):
    with pytest.raises(ValidationError):
        constraint(**over)


def test_field_guide_lists_every_field():
    guide = field_guide()
    for name in ("source", "record_type", "state", "team", "project", "person", "organization", "date"):
        assert f"- {name}:" in guide
