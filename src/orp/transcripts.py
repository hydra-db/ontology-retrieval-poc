"""Render raw agent-session events as labelled plain text for the LLM harness.

Stdlib only. The raw events are whatever the client wrote (for Claude Code: the
parsed lines of its session JSONL); nothing here mutates them.

Output is a sequence of blocks separated by one blank line. Each block starts
with an optional `[timestamp] ` and one of these labels:

    USER:                      something the human typed (incl. messages typed mid-turn)
    AGENT:                     assistant text
    AGENT THINKING:            assistant thinking (only when the text was recorded)
    TOOL CALL <tool>:          tool input as compact JSON
    TOOL RESULT (<tool>):      tool output, mapped back to the tool via tool_use id
    SKILL LOADED <name>:       skill instructions injected into the conversation
    SYSTEM:                    harness-injected text (task notifications, scheduled
                               prompts, local command output, interruptions, ...)
    EVENT <type>:              anything else worth keeping, as text or compact JSON

Shortening: tool inputs, tool results, skill bodies and EVENT payloads longer
than `max_tool_chars` keep their first and last halves with a line
`…[N chars omitted]…` between. USER, AGENT, AGENT THINKING and SYSTEM text is
never shortened.
"""

from __future__ import annotations

import json
import re
from typing import Any

OMIT_TEMPLATE = "…[{n} chars omitted]…"

# Claude Code line types that are dropped entirely: pure bookkeeping with no
# conversational content of their own.
#   api-request / api-request-blob / api-request-shape
#       wire-level copies of the requests sent to the model; they repeat the
#       user / assistant / tool content that the normal lines already carry
#   queue-operation
#       enqueue / dequeue records of the input queue; the queued text shows up
#       again as a `user` line or a `queued_command` attachment
#   file-history-snapshot / file-history-delta   undo checkpoints for edited files
#   ai-title / custom-title / summary / last-prompt   session title and resume pointers
#   mode / permission-mode / atis-latch / isolation-latch / session-owner /
#   cost-state / tools-used / pr-link / frame-link /
#   artifact-comment-monitor / artifact-autoreact-ledger / progress
#       UI state, accounting and link records
_DROP_TYPES = frozenset({
    "api-request", "api-request-blob", "api-request-shape", "queue-operation",
    "file-history-snapshot", "file-history-delta",
    "ai-title", "custom-title", "summary", "last-prompt",
    "mode", "permission-mode", "atis-latch", "isolation-latch", "session-owner",
    "cost-state", "tools-used", "pr-link", "frame-link",
    "artifact-comment-monitor", "artifact-autoreact-ledger", "progress",
})

# `system` subtypes dropped: per-turn timing and stop-hook bookkeeping.
_DROP_SYSTEM_SUBTYPES = frozenset({"turn_duration", "stop_hook_summary"})

# `attachment` types dropped: context plumbing the harness re-sends to the model
# (tool / agent / skill listings, environment and date, token counters,
# reminders, prompt snapshots). None of it is something the user or agent said.
_DROP_ATTACHMENT_TYPES = frozenset({
    "deferred_tools_delta", "deferred_tools_record", "agent_listing_delta",
    "mcp_instructions_delta", "skill_listing", "dynamic_skill", "task_reminder",
    "total_tokens_reminder", "silent_turn_reminder", "environment", "model",
    "auto_mode", "instructions", "session_context", "date", "date_change",
    "credential_org", "remote_session_change", "prompt_snapshot", "thinking_drop",
    "session_cron_carry", "command_permissions", "compact_file_reference",
    "read_truncation_notice",
})

# Envelope fields left out of EVENT fallbacks (ids and per-line metadata).
_ENVELOPE_KEYS = frozenset({
    "type", "timestamp", "uuid", "parentUuid", "logicalParentUuid", "sessionId",
    "session_id", "cwd", "version", "gitBranch", "userType", "entrypoint", "slug",
    "isSidechain", "agentId", "promptId", "requestId",
})

# A plain-string user message starting with one of these tags was produced by
# the harness, not typed by the human.
_SYSTEM_TAGS = (
    "<local-command-stdout>", "<local-command-stderr>", "<local-command-caveat>",
    "<bash-stdout>", "<bash-stderr>", "<task-notification>", "<system-reminder>",
)
_SKILL_PREFIX = "Base directory for this skill:"

_LABEL_RE = (
    r"(?:\[[^\]\n]{1,40}\] )?"
    r"(?:USER|AGENT|AGENT THINKING|SYSTEM|TOOL CALL [^\n]*?|TOOL RESULT \([^\n]*?\)"
    r"|SKILL LOADED [^\n]*?|EVENT [^\n]*?):"
)
_BOUNDARY_RE = re.compile(r"\n\n(?=" + _LABEL_RE + ")")
_WS_RE = re.compile(r"\s+")
_FRACTION_RE = re.compile(r"\.\d+(?=Z$|[+-]\d\d:?\d\d$|$)")


def normalize_ws(s: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return _WS_RE.sub(" ", s).strip()


def paginate(text: str, page_chars: int = 24000) -> list[str]:
    """Split rendered text into pages of at most `page_chars`, cutting only
    between blocks. A single block longer than a page is cut on its own."""
    if not text:
        return []
    page_chars = max(1, page_chars)
    pages: list[str] = []
    current = ""
    for block in _BOUNDARY_RE.split(text):
        if len(block) > page_chars:
            if current:
                pages.append(current)
                current = ""
            pages.extend(_split_long(block, page_chars))
        elif not current:
            current = block
        elif len(current) + 2 + len(block) <= page_chars:
            current += "\n\n" + block
        else:
            pages.append(current)
            current = block
    if current:
        pages.append(current)
    return pages


def _split_long(block: str, page_chars: int) -> list[str]:
    out = []
    while len(block) > page_chars:
        cut = block.rfind("\n", page_chars // 2, page_chars)
        cut = cut + 1 if cut > 0 else page_chars
        out.append(block[:cut])
        block = block[cut:]
    if block:
        out.append(block)
    return out


def render_events(
    events: list[dict],
    client: str = "claude-code",
    max_tool_chars: int = 4000,
    start: int = 0,
    limits: dict[str, int] | None = None,
    collapse_tools: bool = False,
) -> str:
    """Turn raw events into labelled plain text, one block per item, in order.

    `start` renders only events[start:], while tool names are still looked up
    across the whole list, so a tool result whose call came earlier keeps its name.

    `limits` shortens whole blocks by label, for a reader that only needs the
    gist of some kinds of block: {"AGENT": 800} keeps the first and last 400
    characters of each agent message, and a limit of 0 drops that kind of block.
    A label matches the longest key it starts with ("TOOL RESULT (Bash)" matches
    "TOOL RESULT"). USER blocks are never shortened or dropped.

    `collapse_tools` replaces each unbroken run of tool calls and results with
    one line naming the tools and how often each was called, e.g.
    "TOOLS USED: Bash x12, Read x5".
    """
    render = _render_claude_code if client == "claude-code" else None
    ctx = _Context(events if render else [], max_tool_chars)
    keys = sorted(limits or {}, key=len, reverse=True)
    blocks: list[str] = []
    run: dict[str, int] = {}  # tool name -> calls in the current run of tool blocks
    run_prefix = ""

    def flush_run() -> None:
        if run:
            blocks.append(run_prefix + "TOOLS USED: " + ", ".join(f"{name} x{n}" for name, n in run.items()))
            run.clear()

    for event in (events or [])[start:]:
        try:
            parts = render(event, ctx) if render else [_fallback(event, ctx)]
        except Exception:
            parts = [_fallback(event, ctx)]  # unexpected shape: never raise
        prefix = _timestamp_prefix(event)
        for label, body in parts:
            if collapse_tools and label.startswith(("TOOL CALL", "TOOL RESULT")):
                if label.startswith("TOOL CALL"):
                    if not run:
                        run_prefix = prefix
                    name = label[len("TOOL CALL") :].strip() or "unknown"
                    run[name] = run.get(name, 0) + 1
                continue
            flush_run()
            key = None if label == "USER" else next((k for k in keys if label.startswith(k)), None)
            if key is not None:
                limit = limits[key]  # type: ignore[index]
                if limit <= 0:
                    continue
                if len(body) > limit:
                    half = max(1, limit // 2)
                    body = body[:half] + "\n" + OMIT_TEMPLATE.format(n=len(body) - 2 * half) + "\n" + body[-half:]
            blocks.append(prefix + label + ":" + (" " + body if body else ""))
    flush_run()
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ helpers

class _Context:
    def __init__(self, events: Any, max_tool_chars: int):
        self.limit = max_tool_chars
        self.tool_names: dict[str, str] = {}   # tool_use id -> tool name
        self.skill_names: dict[str, str] = {}  # tool_use id of a Skill call -> skill name
        for event in events if isinstance(events, list) else []:
            try:
                for block in event["message"]["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                        self.tool_names[block["id"]] = str(block.get("name") or "unknown")
                        skill = (block.get("input") or {}).get("skill") if block.get("name") == "Skill" else None
                        if isinstance(skill, str) and skill:
                            self.skill_names[block["id"]] = skill
            except Exception:
                continue

    def shorten(self, s: str) -> str:
        if self.limit is None or self.limit <= 0 or len(s) <= self.limit:
            return s
        half = max(1, self.limit // 2)
        return s[:half] + "\n" + OMIT_TEMPLATE.format(n=len(s) - 2 * half) + "\n" + s[-half:]


def _compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def _timestamp_prefix(event: Any) -> str:
    ts = event.get("timestamp") if isinstance(event, dict) else None
    if not isinstance(ts, str) or not ts.strip() or len(ts) > 40:
        return ""
    return "[" + _FRACTION_RE.sub("", ts.strip()) + "] "


def _fallback(event: Any, ctx: _Context, kind: str | None = None) -> tuple[str, str]:
    if not isinstance(event, dict):
        return ("EVENT unknown", ctx.shorten(_compact(event)))
    if kind is None:
        kind = event.get("type") if isinstance(event.get("type"), str) and event.get("type") else "unknown"
    payload = {k: v for k, v in event.items() if k not in _ENVELOPE_KEYS}
    if set(payload) == {"raw"} and isinstance(payload["raw"], str):  # {"type":"unparsed","raw":...}
        return ("EVENT " + kind, ctx.shorten(payload["raw"]))
    return ("EVENT " + kind, ctx.shorten(_compact(payload)))


def _text_of(content: Any) -> str:
    """Flatten message / tool_result content (a string or a list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif not isinstance(block, dict):
                parts.append(_compact(block))
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block.get("type") in ("image", "document"):
                parts.append("[%s]" % block["type"])
            elif block.get("type") == "tool_reference":
                parts.append("[tool_reference: %s]" % block.get("tool_name", "?"))
            else:
                parts.append(_compact(block))
        return "\n".join(parts)
    return _compact(content)


# -------------------------------------------------------------- Claude Code

def _render_claude_code(event: Any, ctx: _Context) -> list[tuple[str, str]]:
    if not isinstance(event, dict):
        return [_fallback(event, ctx)]
    kind = event.get("type")
    if kind in _DROP_TYPES:
        return []
    if kind == "user":
        return _render_user(event, ctx)
    if kind == "assistant":
        return _render_assistant(event, ctx)
    if kind == "system":
        return _render_system(event, ctx)
    if kind == "attachment":
        return _render_attachment(event, ctx)
    return [_fallback(event, ctx)]


def _render_user(event: dict, ctx: _Context) -> list[tuple[str, str]]:
    content = event["message"]["content"]
    meta = bool(event.get("isMeta"))
    origin = event.get("origin")
    origin_kind = origin.get("kind") if isinstance(origin, dict) else None
    # Not typed by the human: meta lines, task notifications, messages from peer sessions.
    injected = meta or origin_kind not in (None, "human")

    if event.get("isCompactSummary"):
        # Text the harness wrote to replace older turns after compaction.
        return [("EVENT compact_summary", ctx.shorten(_text_of(content)))]

    if isinstance(content, str):
        if not content.strip():
            return []
        if injected or content.lstrip().startswith(_SYSTEM_TAGS):
            return [("SYSTEM", content)]
        return [("USER", content)]  # includes slash commands (<command-name>…) and !bash input

    out: list[tuple[str, str]] = []
    typed: list[str] = []  # text and image blocks of one human message, kept together

    def flush() -> None:
        if typed:
            out.append(("USER", "\n".join(typed)))
            typed.clear()

    for block in content:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "tool_result":
            flush()
            name = ctx.tool_names.get(block.get("tool_use_id"), "unknown")
            body = ctx.shorten(_text_of(block.get("content")))
            out.append(("TOOL RESULT (%s)" % name, ("[error] " if block.get("is_error") else "") + body))
        elif btype == "text":
            text = block.get("text") or ""
            source_tool = event.get("sourceToolUseID")
            if not text.strip():
                continue
            if meta and (source_tool in ctx.skill_names or text.startswith(_SKILL_PREFIX)):
                flush()
                out.append(("SKILL LOADED " + _skill_name(text, source_tool, ctx), ctx.shorten(text)))
            elif meta and text.startswith("[Image:"):
                continue  # file-path note for a pasted image; the image itself is in the user line
            elif injected or text.startswith("[Request interrupted"):
                flush()
                out.append(("SYSTEM", text))
            else:
                typed.append(text)
        elif btype in ("image", "document"):
            if not injected:
                typed.append("[%s]" % btype)
        else:
            flush()
            out.append(("EVENT user.%s" % (btype or "unknown"), ctx.shorten(_compact(block))))
    flush()
    return out


def _skill_name(text: str, source_tool: Any, ctx: _Context) -> str:
    if source_tool in ctx.skill_names:
        return ctx.skill_names[source_tool]
    if text.startswith(_SKILL_PREFIX):
        path = text[len(_SKILL_PREFIX):].split("\n", 1)[0].strip().rstrip("/\\")
        name = re.split(r"[/\\]", path)[-1]
        if name:
            return name
    return "unknown"


def _render_assistant(event: dict, ctx: _Context) -> list[tuple[str, str]]:
    content = event["message"]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    api_error = bool(event.get("isApiErrorMessage"))  # synthetic "API Error: …" line, not model output
    out: list[tuple[str, str]] = []
    for block in content:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "text":
            if (block.get("text") or "").strip():
                out.append(("SYSTEM" if api_error else "AGENT", block["text"]))
        elif btype == "thinking":
            # Most thinking blocks are stored with an empty `thinking` string and only
            # a signature; those carry nothing readable and are skipped.
            if (block.get("thinking") or "").strip():
                out.append(("AGENT THINKING", block["thinking"]))
        elif btype == "redacted_thinking":
            continue
        elif btype == "tool_use":
            out.append(("TOOL CALL %s" % (block.get("name") or "unknown"), ctx.shorten(_compact(block.get("input")))))
        else:
            out.append(("EVENT assistant.%s" % (btype or "unknown"), ctx.shorten(_compact(block))))
    return out


def _render_system(event: dict, ctx: _Context) -> list[tuple[str, str]]:
    subtype = event.get("subtype")
    if subtype in _DROP_SYSTEM_SUBTYPES:
        return []
    content = event.get("content")
    if isinstance(content, str) and content.strip():
        # compact_boundary, away_summary, informational, local_command, scheduled_task_fire, ...
        return [("SYSTEM", content)]
    return [_fallback(event, ctx, "system.%s" % subtype if isinstance(subtype, str) and subtype else "system")]


def _render_attachment(event: dict, ctx: _Context) -> list[tuple[str, str]]:
    att = event["attachment"]
    atype = att.get("type")
    if atype in _DROP_ATTACHMENT_TYPES:
        return []
    if atype == "queued_command":
        # A message that arrived while the agent was busy. commandMode "prompt" is the
        # human typing mid-turn (often the only place that message is recorded);
        # other modes are background-task notifications.
        text = _text_of(att.get("prompt"))
        if not text.strip():
            return []
        human = att.get("commandMode") == "prompt" and not text.lstrip().startswith(_SYSTEM_TAGS)
        return [("USER" if human else "SYSTEM", text)]
    if atype == "hook_additional_context":
        text = _text_of(att.get("content"))
        return [("SYSTEM", text)] if text.strip() else []
    if atype == "invoked_skills":
        # Skills re-attached after compaction; their bodies were already shown when first loaded.
        names = [s.get("name") for s in att.get("skills") or [] if isinstance(s, dict) and s.get("name")]
        return [("EVENT attachment.invoked_skills", ", ".join(map(str, names)))] if names else []
    if atype in ("file", "nested_memory", "edited_text_file"):
        name = att.get("displayPath") or att.get("filename") or att.get("path") or ""
        body = att.get("content") if atype != "edited_text_file" else att.get("snippet")
        body = body if isinstance(body, str) else _compact(body)
        return [("EVENT attachment.%s" % atype, ctx.shorten((str(name) + "\n" + body).strip()))]
    payload = {k: v for k, v in att.items() if k != "type"}
    return [("EVENT attachment.%s" % (atype or "unknown"), ctx.shorten(_compact(payload)))]
