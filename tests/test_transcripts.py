"""Tests for orp.transcripts. All fixtures are synthetic."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orp.transcripts import normalize_ws, paginate, render_events  # noqa: E402

TS = "2026-11-17T10:00:00.123Z"


def user(content, **extra):
    return {"type": "user", "timestamp": TS, "uuid": "u1", "sessionId": "s1",
            "message": {"role": "user", "content": content}, **extra}


def assistant(*blocks, **extra):
    return {"type": "assistant", "timestamp": TS, "uuid": "a1", "sessionId": "s1",
            "message": {"role": "assistant", "model": "test-model", "content": list(blocks)}, **extra}


def tool_use(tool_id, name, tool_input):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def tool_result(tool_id, content, is_error=False):
    return user([{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": is_error}])


def blocks_of(text):
    return text.split("\n\n")


# ------------------------------------------------------------------ labels

def test_user_and_agent_labels_with_timestamp():
    out = render_events([
        user("please sort the list"),
        assistant({"type": "text", "text": "Sorted it."}),
    ])
    assert blocks_of(out) == [
        "[2026-11-17T10:00:00Z] USER: please sort the list",
        "[2026-11-17T10:00:00Z] AGENT: Sorted it.",
    ]


def test_no_timestamp_means_no_prefix():
    ev = user("hello")
    del ev["timestamp"]
    assert render_events([ev]) == "USER: hello"


def test_thinking_label_and_empty_thinking_skipped():
    out = render_events([
        assistant({"type": "thinking", "thinking": "the user wants ascending order", "signature": "sig"}),
        assistant({"type": "thinking", "thinking": "", "signature": "sig-only"}),
    ])
    assert out == "[2026-11-17T10:00:00Z] AGENT THINKING: the user wants ascending order"


def test_tool_call_and_result_mapped_by_id():
    out = render_events([
        assistant(tool_use("toolu_1", "Bash", {"command": "ls -la", "description": "list"})),
        assistant(tool_use("toolu_2", "Read", {"file_path": "/tmp/a.txt"})),
        tool_result("toolu_2", "file body"),
        tool_result("toolu_1", [{"type": "text", "text": "a.txt"}, {"type": "image", "source": {}}]),
        tool_result("toolu_missing", "orphan", is_error=True),
    ])
    b = blocks_of(out)
    assert b[0] == '[2026-11-17T10:00:00Z] TOOL CALL Bash: {"command":"ls -la","description":"list"}'
    assert b[1] == '[2026-11-17T10:00:00Z] TOOL CALL Read: {"file_path":"/tmp/a.txt"}'
    assert b[2] == "[2026-11-17T10:00:00Z] TOOL RESULT (Read): file body"
    assert b[3] == "[2026-11-17T10:00:00Z] TOOL RESULT (Bash): a.txt\n[image]"
    assert b[4] == "[2026-11-17T10:00:00Z] TOOL RESULT (unknown): [error] orphan"


def test_skill_loaded_uses_skill_tool_name():
    body = "Base directory for this skill: /home/u/.claude/skills/other-dir\n\n# Tidy\nKeep answers short."
    out = render_events([
        assistant(tool_use("toolu_s", "Skill", {"skill": "tidy-answers"})),
        tool_result("toolu_s", "Launching skill: tidy-answers"),
        user([{"type": "text", "text": body}], isMeta=True, sourceToolUseID="toolu_s"),
    ])
    b = out.split("\n\n[")
    assert b[1].startswith("2026-11-17T10:00:00Z] TOOL RESULT (Skill): Launching skill")
    assert b[2].startswith("2026-11-17T10:00:00Z] SKILL LOADED tidy-answers: Base directory for this skill:")
    assert "Keep answers short." in b[2]


def test_skill_loaded_falls_back_to_directory_name():
    body = "Base directory for this skill: /home/u/.claude/skills/plain-speak\n\nExplain simply."
    out = render_events([user([{"type": "text", "text": body}], isMeta=True)])
    assert out.startswith("[2026-11-17T10:00:00Z] SKILL LOADED plain-speak: ")


def test_system_label_for_injected_user_lines():
    out = render_events([
        user("<task-notification>\n<status>completed</status>\n</task-notification>",
             origin={"kind": "task-notification"}),
        user("Scheduled check: look at the queue", isMeta=True),
        user("<local-command-stdout>Set model</local-command-stdout>"),
        user([{"type": "text", "text": "[Request interrupted by user]"}]),
        {"type": "system", "subtype": "compact_boundary", "content": "Conversation compacted", "timestamp": TS},
        assistant({"type": "text", "text": "API Error: overloaded"}, isApiErrorMessage=True),
    ])
    b = blocks_of(out)
    assert len(b) == 6
    assert all(x.startswith("[2026-11-17T10:00:00Z] SYSTEM: ") for x in b)
    assert "USER:" not in out and "AGENT:" not in out


def test_slash_command_and_image_message_stay_user():
    out = render_events([
        user("<command-name>/compact</command-name>\n<command-args></command-args>", origin={"kind": "human"}),
        user([{"type": "text", "text": "what is wrong in this screenshot"}, {"type": "image", "source": {}}]),
    ])
    b = blocks_of(out)
    assert b[0].startswith("[2026-11-17T10:00:00Z] USER: <command-name>/compact")
    assert b[1] == "[2026-11-17T10:00:00Z] USER: what is wrong in this screenshot\n[image]"


def test_message_typed_mid_turn_is_user():
    def queued(mode, prompt):
        return {"type": "attachment", "timestamp": TS,
                "attachment": {"type": "queued_command", "commandMode": mode, "prompt": prompt}}

    out = render_events([
        queued("prompt", "also add a test for it"),
        queued("prompt", [{"type": "text", "text": "and use tabs"}]),
        queued("task-notification", "<task-notification>done</task-notification>"),
    ])
    assert blocks_of(out) == [
        "[2026-11-17T10:00:00Z] USER: also add a test for it",
        "[2026-11-17T10:00:00Z] USER: and use tabs",
        "[2026-11-17T10:00:00Z] SYSTEM: <task-notification>done</task-notification>",
    ]


def test_event_label_for_kept_extras():
    out = render_events([
        user("Summary of the earlier conversation: sorted a list.", isCompactSummary=True),
        {"type": "attachment", "timestamp": TS,
         "attachment": {"type": "file", "filename": "/tmp/notes.txt", "content": "remember the milk"}},
        {"type": "attachment", "timestamp": TS, "attachment": {"type": "brand_new_kind", "value": 7}},
        {"type": "system", "subtype": "api_error", "timestamp": TS, "error": {"status": 529}},
    ])
    b = blocks_of(out)
    assert b[0] == "[2026-11-17T10:00:00Z] EVENT compact_summary: Summary of the earlier conversation: sorted a list."
    assert b[1] == "[2026-11-17T10:00:00Z] EVENT attachment.file: /tmp/notes.txt\nremember the milk"
    assert b[2] == '[2026-11-17T10:00:00Z] EVENT attachment.brand_new_kind: {"value":7}'
    assert b[3] == '[2026-11-17T10:00:00Z] EVENT system.api_error: {"subtype":"api_error","error":{"status":529}}'


def test_bookkeeping_lines_are_dropped():
    out = render_events([
        {"type": "file-history-snapshot", "messageId": "m", "snapshot": {}},
        {"type": "queue-operation", "operation": "enqueue", "content": "queued text", "timestamp": TS},
        {"type": "api-request-blob", "hash": "h", "message": {"role": "user", "content": "copy"}},
        {"type": "ai-title", "aiTitle": "Sorting"},
        {"type": "permission-mode", "permissionMode": "default"},
        {"type": "system", "subtype": "turn_duration", "durationMs": 5, "timestamp": TS},
        {"type": "attachment", "timestamp": TS, "attachment": {"type": "total_tokens_reminder", "text": "x"}},
        {"type": "attachment", "timestamp": TS, "attachment": {"type": "skill_listing", "content": "x"}},
    ])
    assert out == ""


# -------------------------------------------------------------- truncation

def test_long_tool_result_is_shortened():
    text = "A" * 3000 + "B" * 3000
    out = render_events([
        assistant(tool_use("t1", "Bash", {"command": "cat big"})),
        tool_result("t1", text),
    ], max_tool_chars=1000)
    body = blocks_of(out)[1].split("TOOL RESULT (Bash): ", 1)[1]
    assert body == "A" * 500 + "\n…[5000 chars omitted]…\n" + "B" * 500


def test_long_tool_input_is_shortened():
    out = render_events([assistant(tool_use("t1", "Write", {"content": "x" * 10000}))], max_tool_chars=200)
    body = out.split("TOOL CALL Write: ", 1)[1]
    head, marker, tail = body.split("\n")
    assert head.startswith('{"content":"xxx') and len(head) == 100 and len(tail) == 100
    full_len = len(json.dumps({"content": "x" * 10000}, separators=(",", ":")))
    assert marker == "…[%d chars omitted]…" % (full_len - 200)


def test_user_agent_and_thinking_text_never_shortened():
    long_user, long_agent, long_think = "u" * 9000, "a" * 9000, "t" * 9000
    out = render_events([
        user(long_user),
        assistant({"type": "thinking", "thinking": long_think}),
        assistant({"type": "text", "text": long_agent}),
    ], max_tool_chars=100)
    assert long_user in out and long_agent in out and long_think in out
    assert "omitted" not in out


def test_short_tool_result_untouched_at_exact_limit():
    out = render_events([tool_result("t", "z" * 50)], max_tool_chars=50)
    assert out.endswith("TOOL RESULT (unknown): " + "z" * 50)


# ----------------------------------------------------------- unknown shapes

def test_unknown_shapes_do_not_raise():
    events = [
        {"type": "user"},                                   # no message
        {"type": "user", "message": {"content": 42}},       # wrong content type
        {"type": "assistant", "message": None},
        {"type": "assistant", "message": {"content": ["bare string", {"type": "mystery", "x": 1}]}},
        {"type": "attachment"},
        {"type": "unparsed", "raw": "{not json"},
        {"no_type": True},
        {"type": "launched"},
        "just a string",
        None,
        17,
    ]
    out = render_events(events)
    b = blocks_of(out)
    assert 'EVENT user: {"message":{"content":42}}' in b
    assert "EVENT unparsed: {not json" in b
    assert 'EVENT unknown: {"no_type":true}' in b
    assert "EVENT launched: {}" in b
    assert 'EVENT unknown: "just a string"' in b
    assert "EVENT unknown: null" in b
    assert 'EVENT assistant.mystery: {"type":"mystery","x":1}' in b
    assert len(b) == 12


def test_unknown_client_falls_back_to_event_json():
    out = render_events([
        {"type": "msg", "timestamp": "2026-11-17T10:00:00Z", "role": "user", "text": "hi"},
        {"kind": "other"},
    ], client="some-other-agent")
    assert blocks_of(out) == [
        '[2026-11-17T10:00:00Z] EVENT msg: {"role":"user","text":"hi"}',
        'EVENT unknown: {"kind":"other"}',
    ]


def test_empty_input():
    assert render_events([]) == ""
    assert paginate("") == []


# --------------------------------------------------------------- pagination

def test_paginate_splits_only_on_block_boundaries():
    events = []
    for i in range(30):
        events.append(user("question %d\n\nwith a blank line inside" % i))
        events.append(assistant({"type": "text", "text": "answer %d " % i + "word " * 40}))
    text = render_events(events)
    all_blocks = blocks_of_rendered(events)
    pages = paginate(text, page_chars=1200)
    assert len(pages) > 3
    assert all(len(p) <= 1200 for p in pages)
    assert "\n\n".join(pages) == text
    # every page is a whole number of blocks
    rebuilt = []
    for p in pages:
        n = 0
        while p:
            blk = all_blocks[len(rebuilt)]
            assert p.startswith(blk), "page does not start at a block boundary"
            p = p[len(blk):].removeprefix("\n\n")
            rebuilt.append(blk)
            n += 1
        assert n >= 1
    assert rebuilt == all_blocks


def blocks_of_rendered(events):
    return [render_events([e]) for e in events]


def test_paginate_single_oversized_block_is_cut_alone():
    small = render_events([user("short one")])
    big = render_events([user("line of text\n" * 400)])
    text = "\n\n".join([small, big, small])
    pages = paginate(text, page_chars=1000)
    assert pages[0] == small and pages[-1] == small
    assert all(len(p) <= 1000 for p in pages)
    assert "".join(pages[1:-1]) == big
    assert len(pages) > 3


def test_paginate_fits_in_one_page():
    text = render_events([user("a"), assistant({"type": "text", "text": "b"})])
    assert paginate(text) == [text]


# ------------------------------------------------------------- normalize_ws

def test_normalize_ws():
    assert normalize_ws("  a\n\n b\t\tc \r\n d e ") == "a b c d e"
    assert normalize_ws("") == ""
    quote = "keep   answers\nshort"
    rendered = render_events([user("Please keep answers\n   short from now on")])
    assert normalize_ws(quote) in normalize_ws(rendered)
