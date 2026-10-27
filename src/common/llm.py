"""Thin Anthropic client for structured-JSON calls, with cost accounting and a disk cache.

Design notes:
  * Structured output uses `output_config.format` (JSON schema). Forced `tool_choice` is
    rejected (400) on the Claude 5.5 models, and it would be the wrong tool anyway.
  * No sampling parameters are sent (non-default values 400 on these models); randomness
    is whatever the model does at its default.
  * Successful replies are cached on disk keyed by the full request, so a crashed or
    repeated benchmark run does not pay twice and re-runs are reproducible. The cached
    reply stores the original usage and latency, so cost/latency reports stay faithful.
  * Refusals, truncation and unparsable JSON are returned as `data=None` with a reason; the
    solver counts them as failures rather than this layer hiding them behind a fallback model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .solver import Usage

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "llm"

# $ per million tokens (input, output). Cache reads bill at 0.1x input, cache writes at 1.25x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float:
    inp, out = PRICES[model]
    return (input_tokens * inp + cache_read * inp * 0.1 + cache_write * inp * 1.25
            + output_tokens * out) / 1e6


@dataclass
class LLMReply:
    data: dict | None  # parsed JSON, or None on any failure
    failure: str | None  # None | "refusal" | "max_tokens" | "invalid_json" | "api_error"
    usage: Usage  # for THIS call only
    latency_s: float
    from_cache: bool = False


class JSONClient(Protocol):
    def complete_json(self, *, model: str, system: str, user: str, schema: dict,
                      effort: str = "medium", max_tokens: int = 16000) -> LLMReply: ...


class AnthropicJSONClient:
    def __init__(self, cache_dir: Path | None = CACHE_DIR, max_retries: int = 4) -> None:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env (see .env.example)")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=max_retries)
        self.cache_dir = cache_dir

    def _key(self, **req) -> str:
        return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()

    def complete_json(self, *, model: str, system: str, user: str, schema: dict,
                      effort: str = "medium", max_tokens: int = 16000) -> LLMReply:
        req = dict(model=model, system=system, user=user, schema=schema,
                   effort=effort, max_tokens=max_tokens)
        path = self.cache_dir / f"{self._key(**req)}.json" if self.cache_dir else None
        if path and path.exists():
            hit = json.loads(path.read_text())
            return LLMReply(hit["data"], None, Usage(**hit["usage"]), hit["latency_s"], True)

        start = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": schema}},
            )
        except (self._anthropic.AuthenticationError, self._anthropic.PermissionDeniedError,
                self._anthropic.NotFoundError, self._anthropic.BadRequestError):
            raise  # configuration problems (key, billing, model id, request shape), not model failures
        except self._anthropic.APIError:  # rate limit / 5xx / network, after the SDK's retries
            return LLMReply(None, "api_error", Usage(), time.perf_counter() - start)
        latency = time.perf_counter() - start

        u = resp.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read_tokens=cache_read, api_calls=1,
            cost_usd=cost_usd(model, u.input_tokens, u.output_tokens, cache_read, cache_write),
        )
        if resp.stop_reason == "refusal":
            return LLMReply(None, "refusal", usage, latency)
        if resp.stop_reason == "max_tokens":
            return LLMReply(None, "max_tokens", usage, latency)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return LLMReply(None, "invalid_json", usage, latency)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"data": data, "usage": asdict(usage), "latency_s": latency}))
        return LLMReply(data, None, usage, latency)
