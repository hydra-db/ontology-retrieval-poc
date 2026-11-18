"""A small tool-calling loop: the model calls tools until it calls `finish`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm import LLM


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class AgentResult:
    finished: bool
    result: dict
    turns: int
    tool_calls: list[dict] = field(default_factory=list)


MAX_SAME_FAILURE = 4


class ToolError(Exception):
    """Raised by a tool to hand the model a message it can act on."""


def run_agent(llm: LLM, system: str, task: str, tools: list[Tool], max_turns: int = 40) -> AgentResult:
    """Run until the model calls `finish` or `max_turns` is reached.

    A tool that raises returns its error text to the model, so the model can
    correct the call. Only `finish` ends the run.
    """
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    log: list[dict] = []
    failed: dict[str, int] = {}  # identical failing call -> times sent

    for turn in range(1, max_turns + 1):
        msg = llm.chat(messages, specs)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls or None})
        if not calls:
            messages[-1].pop("tool_calls")
            messages.append(
                {
                    "role": "user",
                    "content": "Reply only by calling a tool. If the work is done, call finish.",
                }
            )
            continue
        for call in calls:
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                out, ok = f"ERROR: arguments were not valid JSON ({exc}). Send the call again.", False
                args = {}
            else:
                if name == "finish":
                    log.append({"tool": name, "args": args, "ok": True})
                    return AgentResult(True, args, turn, log)
                tool = by_name.get(name)
                if tool is None:
                    out, ok = f"ERROR: no tool named '{name}'. Tools: {sorted(by_name)}", False
                else:
                    try:
                        result = tool.fn(**args)
                        out = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                        ok = True
                    except ToolError as exc:
                        out, ok = f"ERROR: {exc}", False
                    except TypeError as exc:
                        out, ok = f"ERROR: wrong arguments for {name}: {exc}", False
            log.append({"tool": name, "args": args, "ok": ok})
            if not ok:
                key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
                failed[key] = failed.get(key, 0) + 1
                if failed[key] == 2:
                    out += (
                        " You sent exactly this call before and it failed the same way. "
                        "Change the arguments the error names, or leave this item out."
                    )
                elif failed[key] >= MAX_SAME_FAILURE:
                    # The model is stuck resending one call; stop instead of burning turns.
                    return AgentResult(False, {"stuck_on": name, "error": out}, turn, log)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": out})

    return AgentResult(False, {}, max_turns, log)


FINISH_ANALYZER = Tool(
    "finish",
    "Call once, when you are done. Ends your work.",
    {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "One paragraph: what the transcript was about and what you changed."}},
        "required": ["summary"],
    },
    lambda **_: None,
)

FINISH_ORGANIZER = Tool(
    "finish",
    "Call once, when you are done. Ends your work.",
    {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One line per change made. Start a line with 'not done:' for a change you could not make.",
            }
        },
        "required": ["changes"],
    },
    lambda **_: None,
)
