"""Local-model JSON client (Apple-silicon MLX): same interface as the Claude client, $0 per call.

Differences from the hosted client, and how they are handled:
  * No schema-constrained decoding, so the schema is put in the prompt and the reply is parsed
    leniently (`extract_json`: strips <think> blocks, finds the first valid JSON object). One
    retry with a corrective message; remaining failures come back as data=None / "invalid_json"
    and are counted by the solver like any other model failure.
  * Greedy decoding (temperature 0), so results are deterministic and reproducible.
  * Cost is recorded as $0; latency is real wall-clock time on this machine, which is what
    makes the cost-vs-latency comparison against hosted models meaningful.
  * `effort` is ignored. `thinking` (Qwen3's reasoning mode) is an explicit constructor flag:
    off by default because it multiplies tokens (and so latency) per puzzle.
"""
from __future__ import annotations

import json
import re
import time

from .llm import CACHE_DIR, LLMReply, ReplyCache
from .solver import Usage

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


def extract_json(text: str) -> dict | None:
    """First JSON object found in `text`, tolerating <think> blocks, prose and code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class LocalJSONClient:
    def __init__(self, thinking: bool = False, max_new_tokens: int = 2048,
                 cache_dir=CACHE_DIR) -> None:
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens * (4 if thinking else 1)
        self.cache = ReplyCache(cache_dir)
        self._loaded: dict[str, tuple] = {}

    def _model(self, name: str):
        if name not in self._loaded:
            from mlx_lm import load

            self._loaded[name] = load(name)
        return self._loaded[name]

    def _generate(self, name: str, messages: list[dict]) -> tuple[str, int, int]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._model(name)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=self.max_new_tokens,
                        sampler=make_sampler(temp=0.0))
        return text, len(tokenizer.encode(prompt)), len(tokenizer.encode(text))

    def complete_json(self, *, model: str, system: str, user: str, schema: dict,
                      effort: str = "medium", max_tokens: int = 16000) -> LLMReply:
        req = dict(backend=f"local-mlx-thinking={self.thinking}", model=model, system=system,
                   user=user, schema=schema, max_new_tokens=self.max_new_tokens)
        if (hit := self.cache.get(req)) is not None:
            return hit

        messages = [
            {"role": "system", "content": system + "\n\nReply with ONLY a JSON object that matches "
             "this JSON schema, with no other text:\n" + json.dumps(schema)},
            {"role": "user", "content": user},
        ]
        start = time.perf_counter()
        text, n_in, n_out = self._generate(model, messages)
        data = extract_json(text)
        if data is None:  # one corrective retry
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "That was not valid JSON for the schema. "
                          "Reply again with ONLY the JSON object."}]
            text2, i2, o2 = self._generate(model, messages)
            n_in, n_out, data = n_in + i2, n_out + o2, extract_json(text2)
        latency = time.perf_counter() - start
        usage = Usage(input_tokens=n_in, output_tokens=n_out, cost_usd=0.0, api_calls=1)
        if data is None:
            return LLMReply(None, "invalid_json", usage, latency)
        reply = LLMReply(data, None, usage, latency)
        self.cache.put(req, reply)
        return reply
