"""Client for an OpenAI-compatible chat endpoint (vLLM serving a Qwen model on RunPod).

Config from env:
  ORP_LLM_BASE_URL   e.g. https://<pod-id>-8000.proxy.runpod.net/v1
  ORP_LLM_API_KEY
  ORP_ANALYZER_MODEL, ORP_ORGANIZER_MODEL   model names as the server lists them
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return the assistant message: {"content": str|None, "tool_calls": [...]|None}."""
        ...


class LLMError(Exception):
    pass


class OpenAICompatLLM:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        retries: int = 3,
        extra_body: dict | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ["ORP_LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("ORP_LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.extra_body = extra_body or {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        # The RunPod proxy rejects some default client User-Agents, so set one.
        self._http = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "orp-harness/0.1"},
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=body)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise LLMError(f"{resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    # A 4xx will not get better on retry.
                    raise RuntimeError(f"LLM request rejected {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                usage = data.get("usage") or {}
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.usage["calls"] += 1
                return data["choices"][0]["message"]
            except (httpx.HTTPError, LLMError) as exc:
                last = exc
                time.sleep(2**attempt)
        raise LLMError(f"LLM call failed after {self.retries} attempts: {last}")
