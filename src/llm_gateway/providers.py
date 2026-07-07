"""Provider adapters for llm_gatewayV2.

Each provider implements:
  async chat(messages, *, max_tokens, temperature, model, tools, tool_choice,
             reasoning, response_format, system_blocks) -> dict

The returned dict is normalised:
  {
    "text": str,
    "tool_calls": [ {"id","name","arguments"} ],
    "input_tokens": int, "output_tokens": int,
    "cache_creation_input_tokens": int, "cache_read_input_tokens": int,
    "stop_reason": "tool_use"|"end_turn"|"max_tokens",
    "model": str,
    "tool_call_dialect": "native"|"prompted_fallback"|"none",
    "reasoning_applied": bool,
  }

`messages` may include role="tool" entries with `tool_call_id` and `content`;
each adapter translates them to its native shape.
"""
from __future__ import annotations
import os, json, uuid, hashlib, re
from typing import AsyncIterator, Optional, Any
import httpx


class ProviderError(Exception):
    def __init__(self, msg, status=None, retryable=True):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _flatten_system(system_blocks) -> tuple[str, list[dict], bool]:
    """Returns (joined_text, raw_blocks, has_cache_marker)."""
    if system_blocks is None:
        return "", [], False
    if isinstance(system_blocks, str):
        return system_blocks, [{"text": system_blocks, "cache": False}], False
    blocks = []
    has_cache = False
    parts = []
    for b in system_blocks:
        if isinstance(b, dict):
            t = b.get("text", "")
            c = bool(b.get("cache", False))
        else:
            t = getattr(b, "text", "")
            c = bool(getattr(b, "cache", False))
        blocks.append({"text": t, "cache": c})
        parts.append(t)
        if c:
            has_cache = True
    return "\n".join(parts), blocks, has_cache


def _empty_result(model: str) -> dict:
    return {
        "text": "", "tool_calls": [],
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "stop_reason": "end_turn", "model": model,
        "tool_call_dialect": "none", "reasoning_applied": False,
    }


def _extract_rate_limit_headers(headers) -> dict:
    """Extract rate-limit info from provider response headers.

    Common header patterns:
    - x-ratelimit-remaining-requests / x-ratelimit-limit-requests  (Groq, OpenRouter, Nvidia)
    - x-ratelimit-remaining-tokens / x-ratelimit-limit-tokens
    - x-ratelimit-reset-requests / x-ratelimit-reset-tokens
    """
    out = {}
    def _int(key):
        v = headers.get(key)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
        return None

    out["rpm_remaining"] = _int("x-ratelimit-remaining-requests")
    out["rpm_limit"] = _int("x-ratelimit-limit-requests")
    out["tpm_remaining"] = _int("x-ratelimit-remaining-tokens")
    out["tpm_limit"] = _int("x-ratelimit-limit-tokens")
    # Some providers use daily limits in different headers
    out["rpd_remaining"] = _int("x-ratelimit-remaining-daily-requests")
    out["rpd_limit"] = _int("x-ratelimit-limit-daily-requests")
    # Filter out None values
    return {k: v for k, v in out.items() if v is not None}


# ────────────────────────────────────────────────────────────────────────────
# Base
# ────────────────────────────────────────────────────────────────────────────

class BaseProvider:
    name: str = ""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False) -> dict:
        raise NotImplementedError

    async def stream(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                     tools=None, tool_choice=None, reasoning=None, response_format=None,
                     system_blocks=None, cache_system=False) -> AsyncIterator[str]:
        # Default fallback: do non-streaming and yield once.
        result = await self.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                 model=model, tools=tools, tool_choice=tool_choice,
                                 reasoning=reasoning, response_format=response_format,
                                 system_blocks=system_blocks, cache_system=cache_system)
        if result["text"]:
            yield result["text"]


# ────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible providers
# ────────────────────────────────────────────────────────────────────────────

REASONING_MODEL_HINTS = ("gpt-oss", "qwen3-think", "deepseek-r1", "deepseek-r2",
                        "qwen3", "o1", "o3", "o4", "gpt-5")


def _model_supports_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in REASONING_MODEL_HINTS)


class OpenAICompatProvider(BaseProvider):
    capabilities = {
        "tools": True, "caching": True, "reasoning": False,
        "structured": True, "parallel_tools": True,
    }

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _translate_tools(self, tools):
        out = []
        for t in tools or []:
            d = t if isinstance(t, dict) else t.model_dump()
            out.append({
                "type": "function",
                "function": {
                    "name": d["name"],
                    "description": d.get("description", ""),
                    "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
                },
            })
        return out

    def _translate_messages(self, messages, system_text):
        """Translate canonical messages (incl role=tool) to OpenAI shape."""
        out = []
        if system_text:
            out.append({"role": "system", "content": system_text})
        for m in messages:
            r = m.get("role")
            if r == "system":
                # already prepended via system_text — but allow inline if no system_blocks
                if not system_text:
                    out.append({"role": "system", "content": m.get("content", "")})
                continue
            if r == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id") or m.get("id") or "",
                    "content": m.get("content", "") if isinstance(m.get("content"), str) else json.dumps(m.get("content")),
                })
                continue
            if r == "assistant" and m.get("tool_calls"):
                # Carry assistant tool_calls back through.
                tcs = []
                for tc in m["tool_calls"]:
                    tcs.append({
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    })
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
                continue
            out.append({"role": r, "content": m.get("content", "")})
        return out

    def _apply_response_format(self, body, response_format):
        if not response_format:
            return
        rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
        if rf.get("type") == "json_schema" and rf.get("schema"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": rf.get("name", "out"),
                    "schema": rf["schema"],
                    "strict": bool(rf.get("strict", True)),
                },
            }
        elif rf.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}

    def _apply_reasoning(self, body, reasoning, model):
        if not reasoning or reasoning == "off":
            return False
        if not _model_supports_reasoning(model):
            return False
        body["reasoning_effort"] = reasoning
        return True

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, _, _ = _flatten_system(system_blocks)
        body = {
            "model": m,
            "messages": self._translate_messages(messages, system_text),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = self._translate_tools(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice if isinstance(tool_choice, (str, dict)) else "auto"
        self._apply_response_format(body, response_format)
        reasoning_applied = self._apply_reasoning(body, reasoning, m)

        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
            if r.status_code != 200:
                # Some providers reject reasoning_effort or strict json_schema — retry without them.
                txt = r.text
                if reasoning_applied and "reasoning_effort" in txt:
                    body.pop("reasoning_effort", None)
                    reasoning_applied = False
                    r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
                if r.status_code != 200 and "json_schema" in (body.get("response_format") or {}).get("type", ""):
                    body["response_format"] = {"type": "json_object"}
                    # Some providers (e.g. Groq) require "json" in messages
                    # when using json_object response_format.
                    msgs = body.get("messages", [])
                    has_json_word = any("json" in (m.get("content", "") or "").lower() for m in msgs)
                    if not has_json_word and msgs:
                        if msgs[0]["role"] == "system":
                            msgs[0]["content"] = (msgs[0].get("content") or "") + "\nRespond in JSON format."
                        else:
                            msgs.insert(0, {"role": "system", "content": "Respond in JSON format."})
                    r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
                if r.status_code != 200:
                    raise ProviderError(
                        f"{self.name} HTTP {r.status_code}: {r.text}",
                        status=r.status_code,
                        retryable=(r.status_code not in (400, 401)),
                    )
            d = r.json()
            choice = (d.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            tool_calls_out = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {"_raw": args_str}
                tool_calls_out.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
            usage = d.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cache_read = details.get("cached_tokens", 0) or 0
            stop = choice.get("finish_reason") or "stop"
            stop_norm = "tool_use" if tool_calls_out else (
                "max_tokens" if stop == "length" else "end_turn"
            )
            # Parse structured output JSON if response_format was json_schema
            parsed = None
            if response_format and text:
                rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
                if rf.get("type") == "json_schema" or rf.get("schema"):
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        pass
            # V4: capture rate-limit headers for live tracking
            rl_headers = _extract_rate_limit_headers(r.headers)
            return {
                "text": text or "",
                "tool_calls": tool_calls_out,
                "input_tokens": usage.get("prompt_tokens", 0) or 0,
                "output_tokens": usage.get("completion_tokens", 0) or 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cache_read,
                "stop_reason": stop_norm,
                "model": m,
                "tool_call_dialect": "native",
                "reasoning_applied": reasoning_applied,
                "parsed": parsed,
                "rate_limit_headers": rl_headers,
            }

    async def stream(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                     tools=None, tool_choice=None, reasoning=None, response_format=None,
                     system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, _, _ = _flatten_system(system_blocks)
        body = {
            "model": m,
            "messages": self._translate_messages(messages, system_text),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = self._translate_tools(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice if isinstance(tool_choice, (str, dict)) else "auto"
        self._apply_response_format(body, response_format)
        self._apply_reasoning(body, reasoning, m)
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", f"{self.base_url}/chat/completions",
                                headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "ignore")
                    raise ProviderError(f"{self.name} HTTP {r.status_code}: {text}", status=r.status_code)
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        d = json.loads(payload)
                        delta = d["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
                        if delta.get("tool_calls"):
                            yield "[[TOOL_CALL_DELTA]] " + json.dumps(delta["tool_calls"])
                    except Exception:
                        continue


_GROQ_STRICT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
_GROQ_BESTEFFORT_MODELS = _GROQ_STRICT_MODELS | {
    "openai/gpt-oss-safeguard-20b", "meta-llama/llama-4-scout-17b-16e-instruct",
}


# JSON-Schema keywords that Groq/Cerebras strict mode rejects. These must be
# stripped before sending the schema; leaving them in causes the provider to
# reject the request or produce malformed constrained output.
_STRICT_DROP_KEYS = {"default", "maxLength", "maxItems", "title", "format", "examples", "$schema", "$defs", "definitions"}


def _groq_strict_schema(schema: dict) -> dict:
    """Prepare a JSON schema for Groq/Cerebras strict mode.

    Three transformations:
    1. Inline all ``$ref`` / ``$defs`` references (strict mode requires a
       single self-contained schema with no refs). Reuses the Gemini inliner.
    2. Strip keywords strict mode rejects (``default``, ``maxLength``,
       ``maxItems``, ``title``, ``format``, …).
    3. Make every object fully strict: all properties required,
       ``additionalProperties: false``, recursively.
    """
    s = _gemini_inline_refs(schema)
    s = _strip_strict_keys(s)
    return _make_strict(s)


def _strip_strict_keys(node):
    """Recursively remove JSON-Schema keywords that strict mode rejects."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _STRICT_DROP_KEYS:
                continue
            out[k] = _strip_strict_keys(v)
        return out
    if isinstance(node, list):
        return [_strip_strict_keys(x) for x in node]
    return node


def _make_strict(schema: dict) -> dict:
    """Recursively enforce strict constraints: all fields required,
    additionalProperties false, through nested objects and arrays."""
    s = dict(schema)
    if s.get("type") == "object" and "properties" in s:
        s["required"] = list(s["properties"].keys())
        s["additionalProperties"] = False
        s["properties"] = {
            k: _make_strict(v) for k, v in s["properties"].items()
        }
    if s.get("type") == "array" and "items" in s:
        s["items"] = _make_strict(s["items"])
    # anyOf/allOf/oneOf: recurse into each branch
    for combiner in ("anyOf", "allOf", "oneOf"):
        if combiner in s and isinstance(s[combiner], list):
            s[combiner] = [_make_strict(b) if isinstance(b, dict) else b for b in s[combiner]]
    return s


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True, "structured": True}
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://api.groq.com/openai/v1")

    def _apply_response_format(self, body, response_format):
        """Groq structured output: strict constrained decoding for supported models,
        best-effort json_schema for others, plain json_object as final fallback."""
        if not response_format:
            return
        rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
        m = body.get("model", self.model)
        if rf.get("type") == "json_schema" and rf.get("schema"):
            if m in _GROQ_STRICT_MODELS:
                # Strict constrained decoding — all fields required, no additionalProperties
                schema = _groq_strict_schema(rf["schema"])
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": rf.get("name", "out"),
                        "strict": True,
                        "schema": schema,
                    },
                }
            elif m in _GROQ_BESTEFFORT_MODELS:
                # Best-effort mode
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": rf.get("name", "out"),
                        "strict": False,
                        "schema": rf["schema"],
                    },
                }
            else:
                # Fallback: json_object mode + schema hint in system message
                body["response_format"] = {"type": "json_object"}
                msgs = body.get("messages", [])
                hint = f"\nRespond in JSON format matching this schema: {json.dumps(rf['schema'])}"
                if msgs and msgs[0]["role"] == "system":
                    msgs[0]["content"] = (msgs[0].get("content") or "") + hint
                else:
                    msgs.insert(0, {"role": "system", "content": hint.strip()})
        elif rf.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}
            msgs = body.get("messages", [])
            has_json_word = any("json" in (m.get("content", "") or "").lower() for m in msgs)
            if not has_json_word and msgs:
                if msgs[0]["role"] == "system":
                    msgs[0]["content"] = (msgs[0].get("content") or "") + "\nRespond in JSON format."
                else:
                    msgs.insert(0, {"role": "system", "content": "Respond in JSON format."})


class CerebrasProvider(OpenAICompatProvider):
    name = "cerebras"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True, "structured": True}
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://api.cerebras.ai/v1")

    def _apply_response_format(self, body, response_format):
        """Cerebras structured output: strict json_schema with additionalProperties
        false enforced recursively. Cerebras uses the same json_schema envelope as
        OpenAI but requires strict schema compliance."""
        if not response_format:
            return
        rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
        if rf.get("type") == "json_schema" and rf.get("schema"):
            schema = _groq_strict_schema(rf["schema"])  # reuse recursive strict helper
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": rf.get("name", "out"),
                    "strict": True,
                    "schema": schema,
                },
            }
        elif rf.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}
            msgs = body.get("messages", [])
            has_json_word = any("json" in (m.get("content", "") or "").lower() for m in msgs)
            if not has_json_word and msgs:
                if msgs[0]["role"] == "system":
                    msgs[0]["content"] = (msgs[0].get("content") or "") + "\nRespond in JSON format."
                else:
                    msgs.insert(0, {"role": "system", "content": "Respond in JSON format."})


class NvidiaProvider(OpenAICompatProvider):
    name = "nvidia"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True, "structured": True}
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://integrate.api.nvidia.com/v1")

    def _apply_response_format(self, body, response_format):
        """NVIDIA structured output via nvext.guided_json.
        Falls back to standard json_schema first, then nvext if rejected."""
        if not response_format:
            return
        rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
        if rf.get("type") == "json_schema" and rf.get("schema"):
            # Use nvext.guided_json — recommended by NVIDIA for reliability
            body["nvext"] = {"guided_json": rf["schema"]}
            # Also hint the model with response_format json_object
            body["response_format"] = {"type": "json_object"}
        elif rf.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True}
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://openrouter.ai/api/v1")

    def _headers(self):
        h = super()._headers()
        h["HTTP-Referer"] = "http://localhost"
        h["X-Title"] = "LLM Gateway V2"
        return h


class GitHubProvider(OpenAICompatProvider):
    name = "github"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True}
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://models.github.ai/inference")


class SGLangProvider(OpenAICompatProvider):
    name = "sglang"
    capabilities = {
        "tools": True, "caching": False, "reasoning": True,
        "structured": True, "parallel_tools": True,
    }
    def __init__(self, api_key, model, base_url):
        super().__init__(api_key, model, base_url)


# ────────────────────────────────────────────────────────────────────────────
# Gemini
# ────────────────────────────────────────────────────────────────────────────

class GeminiProvider(BaseProvider):
    name = "gemini"
    capabilities = {
        "tools": True, "caching": True, "reasoning": True,
        "structured": True, "parallel_tools": True,
    }

    def __init__(self, api_key, model, cache_store):
        super().__init__(api_key, model, "https://generativelanguage.googleapis.com/v1beta")
        self.cache_store = cache_store  # cache.GeminiCache

    def _translate_tools(self, tools):
        if not tools:
            return None
        decls = []
        for t in tools:
            d = t if isinstance(t, dict) else t.model_dump()
            decls.append({
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
            })
        return [{"function_declarations": decls}]

    def _translate_messages(self, messages):
        contents = []
        for m in messages:
            r = m.get("role")
            if r == "system":
                continue
            if r == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": m.get("tool_name") or m.get("name") or "tool",
                            "response": _coerce_obj(m.get("content")),
                        }
                    }],
                })
                continue
            if r == "assistant":
                parts = []
                if m.get("content"):
                    parts.append({"text": m["content"]})
                for tc in (m.get("tool_calls") or []):
                    part = {
                        "functionCall": {
                            "name": tc["name"],
                            "args": tc.get("arguments") or {},
                        }
                    }
                    meta = tc.get("provider_meta") or {}
                    if meta.get("thoughtSignature"):
                        part["thoughtSignature"] = meta["thoughtSignature"]
                    parts.append(part)
                if not parts:
                    parts = [{"text": ""}]
                contents.append({"role": "model", "parts": parts})
                continue
            content = m.get("content", "")
            contents.append({"role": "user", "parts": [{"text": content if isinstance(content, str) else json.dumps(content)}]})
        return contents

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, blocks, has_cache_marker = _flatten_system(system_blocks)
        cacheable_text = None
        if cache_system or has_cache_marker:
            # Concatenate the cacheable portion (or the entire system_text if cache_system bool).
            if has_cache_marker:
                cacheable_text = "\n".join(b["text"] for b in blocks if b["cache"])
            else:
                cacheable_text = system_text
        cache_name = None
        cache_create_tokens = 0
        cache_read_tokens = 0
        if cacheable_text and len(cacheable_text) > 1000:
            cache_name, cache_create_tokens = await self.cache_store.get_or_create(
                self.api_key, m, cacheable_text, self.base_url
            )
            if cache_name and cache_create_tokens == 0:
                # Reused an existing cache entry — count its tokens as cache_read.
                cache_read_tokens = len(cacheable_text) // 4

        body: dict[str, Any] = {
            "contents": self._translate_messages(messages),
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if cache_name:
            body["cachedContent"] = cache_name
            # Strip cached part from system instruction to avoid double-billing.
            remaining_sys = "\n".join(b["text"] for b in blocks if not b["cache"]) if has_cache_marker else ""
            if remaining_sys:
                body["systemInstruction"] = {"parts": [{"text": remaining_sys}]}
        elif system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        if tools:
            body["tools"] = self._translate_tools(tools)
            mode = "AUTO"
            if tool_choice == "none":
                mode = "NONE"
            elif isinstance(tool_choice, dict):
                mode = "ANY"
            body["toolConfig"] = {"function_calling_config": {"mode": mode}}

        if response_format:
            rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
            if rf.get("schema"):
                body["generationConfig"]["responseMimeType"] = "application/json"
                body["generationConfig"]["responseSchema"] = _gemini_clean_schema(rf["schema"])
            elif rf.get("type") == "json_object":
                body["generationConfig"]["responseMimeType"] = "application/json"

        reasoning_applied = False
        if reasoning and reasoning != "off":
            knob = _gemini_thinking_knob(m)
            if knob == "level":
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": reasoning}
                reasoning_applied = True
            elif knob == "budget":
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": _GEMINI_BUDGETS[reasoning]}
                reasoning_applied = True

        url = f"{self.base_url}/models/{m}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(url, json=body)
            if r.status_code != 200:
                # Retry stripping thinkingConfig / cachedContent on 400.
                if r.status_code == 400:
                    if reasoning_applied:
                        body["generationConfig"].pop("thinkingConfig", None)
                        reasoning_applied = False
                    if "cachedContent" in body and "cache" in r.text.lower():
                        body.pop("cachedContent", None)
                        if system_text:
                            body["systemInstruction"] = {"parts": [{"text": system_text}]}
                        cache_name = None
                        cache_read_tokens = 0
                    r = await c.post(url, json=body)
                if r.status_code != 200 and "response_schema" in r.text.lower():
                    # Schema still rejected — fall back to plain JSON mode.
                    body["generationConfig"].pop("responseSchema", None)
                    r = await c.post(url, json=body)
                if r.status_code != 200:
                    raise ProviderError(
                        f"gemini HTTP {r.status_code}: {r.text}",
                        status=r.status_code,
                        retryable=(r.status_code not in (400, 401)),
                    )
            d = r.json()
            cands = d.get("candidates") or []
            if not cands:
                raise ProviderError(f"gemini no candidates: {json.dumps(d)}", status=200, retryable=True)
            parts = cands[0].get("content", {}).get("parts", []) or []
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            tool_calls_out = []
            for p in parts:
                fc = p.get("functionCall") or p.get("function_call")
                if fc:
                    tc = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args") or fc.get("arguments") or {},
                    }
                    sig = p.get("thoughtSignature") or p.get("thought_signature")
                    if sig:
                        tc["provider_meta"] = {"thoughtSignature": sig}
                    tool_calls_out.append(tc)
            usage = d.get("usageMetadata") or {}
            in_tok = usage.get("promptTokenCount", 0) or 0
            out_tok = usage.get("candidatesTokenCount", 0) or 0
            cached_tok = usage.get("cachedContentTokenCount", 0) or 0
            if cached_tok and not cache_read_tokens:
                cache_read_tokens = cached_tok
            stop = (cands[0].get("finishReason") or "STOP").upper()
            stop_norm = "tool_use" if tool_calls_out else (
                "max_tokens" if stop == "MAX_TOKENS" else "end_turn"
            )
            # V4: Gemini rate-limit headers
            rl_headers = _extract_rate_limit_headers(r.headers)
            return {
                "text": text,
                "tool_calls": tool_calls_out,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cache_creation_input_tokens": cache_create_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "stop_reason": stop_norm,
                "model": m,
                "tool_call_dialect": "native",
                "reasoning_applied": reasoning_applied,
                "rate_limit_headers": rl_headers,
            }


def _gemini_supports_thinking(model: str) -> bool:
    return _gemini_thinking_knob(model) is not None


def _gemini_thinking_knob(model: str) -> Optional[str]:
    """Returns 'level' for thinkingLevel-capable models (2.5-pro, 3.x non-lite),
    'budget' for thinkingBudget-only models (2.5-flash), or None for non-thinking."""
    m = (model or "").lower()
    if "gemini" not in m:
        return None
    if "flash-lite" in m:
        return None
    if "2.5-pro" in m or "3-pro" in m or "3.1-pro" in m:
        return "level"
    if "3-flash" in m or "3.1-flash" in m:
        return "level"  # 3.x flash (non-lite) supports thinkingLevel
    if "2.5-flash" in m:
        return "budget"
    return None


_GEMINI_BUDGETS = {"low": 2048, "medium": 8192, "high": 24576}


def _gemini_inline_refs(schema: dict) -> dict:
    """Resolve `$ref` references to `$defs` / `definitions` inline.

    Pydantic emits refs for nested models. Gemini's responseSchema endpoint
    rejects `$ref`, so we inline before cleaning. This must run BEFORE
    `_gemini_clean_schema` (which strips `$defs`).
    """
    if not isinstance(schema, dict):
        return schema
    defs = dict(schema.get("$defs") or schema.get("definitions") or {})

    def walk(node, seen: frozenset[str] = frozenset()) -> dict | list:
        if isinstance(node, dict):
            if "$ref" in node:
                target = node["$ref"]
                name = None
                if target.startswith("#/$defs/"):
                    name = target.removeprefix("#/$defs/")
                elif target.startswith("#/definitions/"):
                    name = target.removeprefix("#/definitions/")
                if name and name in defs and name not in seen:
                    resolved = walk(defs[name], seen | {name})
                    extras = {k: walk(v, seen) for k, v in node.items() if k != "$ref"}
                    if isinstance(resolved, dict):
                        return {**resolved, **extras}
                    return resolved
                # Unresolvable ref — drop it; Gemini would 400 anyway.
                return {k: walk(v, seen) for k, v in node.items() if k != "$ref"}
            return {k: walk(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x, seen) for x in node]
        return node

    return walk(schema)  # type: ignore[return-value]


def _gemini_clean_schema(schema: dict) -> dict:
    """Strip JSON-Schema keys Gemini rejects, after inlining `$ref` / `$defs`."""
    schema = _gemini_inline_refs(schema)
    if not isinstance(schema, dict):
        return schema
    drop = {"additionalProperties", "$schema", "title", "definitions", "$defs", "examples", "default"}

    def strip(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in drop:
                    continue
                if k == "type" and isinstance(v, list):
                    # Convert ["string", "null"] → type: "string", nullable: true
                    types = [t for t in v if t != "null"]
                    out["type"] = types[0] if types else "string"
                    if "null" in v:
                        out["nullable"] = True
                else:
                    out[k] = strip(v)
            return out
        if isinstance(node, list):
            return [strip(x) for x in node]
        return node

    return strip(schema)


def _coerce_obj(v):
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {"text": v}
    return {"value": v}


# ────────────────────────────────────────────────────────────────────────────
# Ollama
# ────────────────────────────────────────────────────────────────────────────

OLLAMA_TOOL_MODELS = ("llama3.1", "llama3.2", "llama3.3", "qwen2.5", "qwen3",
                     "mistral-nemo", "mistral-small", "command-r", "firefunction")


def _ollama_native_tools(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in OLLAMA_TOOL_MODELS)


class OllamaProvider(BaseProvider):
    name = "ollama"
    capabilities = {
        "tools": True, "caching": False, "reasoning": False,
        "structured": True, "parallel_tools": False,
    }

    def __init__(self, model, base_url="http://localhost:11434"):
        super().__init__("", model, base_url)

    def _translate_messages(self, messages, system_text, prompted_fallback=False):
        out = []
        if system_text:
            out.append({"role": "system", "content": system_text})
        for m in messages:
            r = m.get("role")
            if r == "system":
                if not system_text:
                    out.append({"role": "system", "content": m.get("content", "")})
                continue
            if r == "tool":
                if prompted_fallback:
                    # Fold tool result into next user message.
                    out.append({
                        "role": "user",
                        "content": f"tool_result {{\"id\":\"{m.get('tool_call_id','')}\",\"output\":{json.dumps(m.get('content',''))}}}",
                    })
                else:
                    out.append({"role": "tool", "content": m.get("content", "") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))})
                continue
            if r == "assistant" and m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    tcs.append({"function": {"name": tc["name"], "arguments": tc.get("arguments") or {}}})
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
                continue
            out.append({"role": r, "content": m.get("content", "")})
        return out

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, _, _ = _flatten_system(system_blocks)
        native = _ollama_native_tools(m) and tools
        prompted_fallback = bool(tools) and not native

        if prompted_fallback:
            system_text = (system_text + "\n\n" if system_text else "") + _prompted_tool_system(tools)

        body = {
            "model": m,
            "messages": self._translate_messages(messages, system_text, prompted_fallback=prompted_fallback),
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        if native:
            body["tools"] = [{
                "type": "function",
                "function": {
                    "name": (t if isinstance(t, dict) else t.model_dump())["name"],
                    "description": (t if isinstance(t, dict) else t.model_dump()).get("description", ""),
                    "parameters": (t if isinstance(t, dict) else t.model_dump()).get("input_schema") or {"type": "object", "properties": {}},
                },
            } for t in tools]

        if response_format:
            rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
            if rf.get("schema"):
                body["format"] = rf["schema"]
            elif rf.get("type") == "json_object":
                body["format"] = "json"

        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{self.base_url}/api/chat", json=body)
            if r.status_code != 200:
                raise ProviderError(f"ollama HTTP {r.status_code}: {r.text}", status=r.status_code)
            d = r.json()
            msg = d.get("message", {}) or {}
            text = msg.get("content", "") or ""
            tool_calls_out = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {"_raw": args}
                tool_calls_out.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": fn.get("name", ""),
                    "arguments": args,
                })

            dialect = "native" if native else ("prompted_fallback" if prompted_fallback else "none")
            if prompted_fallback and not tool_calls_out:
                parsed = _parse_prompted_tool_call(text)
                if parsed:
                    tool_calls_out = [parsed]
                    text = ""

            return {
                "text": text,
                "tool_calls": tool_calls_out,
                "input_tokens": d.get("prompt_eval_count", 0) or 0,
                "output_tokens": d.get("eval_count", 0) or 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "stop_reason": "tool_use" if tool_calls_out else "end_turn",
                "model": m,
                "tool_call_dialect": dialect,
                "reasoning_applied": False,
                "rate_limit_headers": {},  # Ollama is local, no rate limits
            }


def _prompted_tool_system(tools):
    descs = []
    for t in tools:
        d = t if isinstance(t, dict) else t.model_dump()
        descs.append(f"- {d['name']}: {d.get('description','')} schema={json.dumps(d.get('input_schema') or {})}")
    return (
        "When you need to call a tool, respond with ONLY a JSON line of the form "
        '{"tool_call":{"name":"<tool>","arguments":{...}}}. '
        "Do not add prose. Tools available:\n" + "\n".join(descs)
    )


def _parse_prompted_tool_call(text: str):
    if not text:
        return None
    m = re.search(r'\{[\s\S]*"tool_call"[\s\S]*\}', text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        tc = d.get("tool_call") or {}
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": tc.get("name", ""),
            "arguments": tc.get("arguments") or {},
        }
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Per-model capability resolution
# ────────────────────────────────────────────────────────────────────────────

# Allow per-model overrides where defaults differ.
def model_capabilities(provider_name: str, model: str, default_caps: dict) -> dict:
    caps = dict(default_caps)
    m = (model or "").lower()
    if provider_name == "gemini":
        caps["reasoning"] = _gemini_supports_thinking(model)
    if provider_name == "ollama":
        caps["tools"] = True  # we always have prompted fallback
        caps["reasoning"] = False
    if provider_name in ("groq", "cerebras", "nvidia", "openrouter", "github"):
        caps["reasoning"] = _model_supports_reasoning(model)
    # Groq: structured output depends on model
    if provider_name == "groq":
        caps["structured"] = model in _GROQ_BESTEFFORT_MODELS or model in _GROQ_STRICT_MODELS
    return caps


def build_providers(cache_store):
    """Worker pool — the LLMs that do real work for the agent.

    V3 changes vs V2:
    - cerebras worker default: zai-glm-4.7 (was qwen-3-235b-a22b-instruct-2507, deprecating May 27 2026)
    - groq worker default: openai/gpt-oss-120b (was llama-3.3-70b-versatile, now moved to router pool)
    """
    out = {}
    if k := os.getenv("GEMINI_API_KEY"):
        out["gemini"] = GeminiProvider(k, os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), cache_store)
    if k := os.getenv("NVIDIA_API_KEY"):
        out["nvidia"] = NvidiaProvider(k, os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2"))
    if k := os.getenv("GROQ_API_KEY"):
        out["groq"] = GroqProvider(k, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    if k := os.getenv("CEREBRAS_API_KEY"):
        out["cerebras"] = CerebrasProvider(k, os.getenv("CEREBRAS_MODEL", "zai-glm-4.7"))
    if k := os.getenv("OPEN_ROUTER_API_KEY"):
        out["openrouter"] = OpenRouterProvider(k, os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"))
    if k := os.getenv("GITHUB_ACCESS_TOKEN"):
        out["github"] = GitHubProvider(k, os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"))
    if om := os.getenv("OLLAMA_MODEL"):
        out["ollama"] = OllamaProvider(om, os.getenv("OLLAMA_URL", "http://localhost:11434"))
    if sm := os.getenv("SGLANG_MODEL"):
        out["sglang"] = SGLangProvider(
            os.getenv("SGLANG_API_KEY", "sk-dummy"),
            sm,
            os.getenv("SGLANG_URL", "http://localhost:30000/v1"),
        )
    return out


# V3 router pool — small/fast LLMs used only for routing decisions.
# Separate from the worker pool: separate quotas, separate dashboard section,
# separate per-call markers. Routers receive a bounded envelope (token_count +
# 800-char sample) and emit a single word (TINY/LARGE/HUGE).
ROUTER_DEFAULTS = {
    # NOTE: On the test Cerebras account, gpt-oss-120b / zai-glm-4.7 / qwen-3-32b
    # all 404 (no entitlement despite docs). Only llama3.1-8b and the deprecating
    # qwen-3-235b respond. Using llama3.1-8b — small, fast, the natural router
    # shape. *** DEPRECATES MAY 27, 2026 *** — must update ROUTER_CEREBRAS_MODEL
    # before then, OR upgrade the Cerebras account to unlock gpt-oss-120b.
    "cerebras": "gpt-oss-120b",
    "groq": "llama-3.3-70b-versatile",
    "github": "microsoft/Phi-4-mini-instruct",
    "sglang": "Qwen/Qwen3.5-0.8B",
}


def build_router_providers():
    """Router pool — same provider classes as workers, but separate instances
    with router-specific (smaller/faster) model defaults. Uses the same API keys
    as workers; per-provider rate budgets are independent because the providers
    we picked (Cerebras, Groq, NVIDIA, GitHub) all meter per-model, not per-key.
    """
    out = {}
    if k := os.getenv("CEREBRAS_API_KEY"):
        out["cerebras"] = CerebrasProvider(k, os.getenv("ROUTER_CEREBRAS_MODEL", ROUTER_DEFAULTS["cerebras"]))
    if k := os.getenv("GROQ_API_KEY"):
        out["groq"] = GroqProvider(k, os.getenv("ROUTER_GROQ_MODEL", ROUTER_DEFAULTS["groq"]))
    if k := os.getenv("GITHUB_ACCESS_TOKEN"):
        out["github"] = GitHubProvider(k, os.getenv("ROUTER_GITHUB_MODEL", ROUTER_DEFAULTS["github"]))
    if sm := os.getenv("SGLANG_MODEL"):
        out["sglang"] = SGLangProvider(
            os.getenv("SGLANG_API_KEY", "sk-dummy"),
            os.getenv("ROUTER_SGLANG_MODEL", ROUTER_DEFAULTS["sglang"]),
            os.getenv("SGLANG_URL", "http://localhost:30000/v1"),
        )
    return out
