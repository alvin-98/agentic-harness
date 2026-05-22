# LLM Gateway V2

Same 7-provider gateway as `llm_gateway/` (Ollama, Gemini, NVIDIA NIM, Groq, Cerebras, OpenRouter, GitHub Models), upgraded to the **2026 agent shape** taught in Session 5: native tool-use, prompt caching, reasoning budgets, structured output, capability-aware routing.

V1 stays as-is on port **8099**. V2 runs on port **8100** so both coexist.

> Pydantic v2 on every boundary. Old request bodies still work — every new feature is opt-in via new fields.

---

## What's new vs V1

| Feature | V1 | V2 |
|---|---|---|
| Tool-use | Pass JSON-by-prompt, regex the response | **Native tool-use** translated per provider; canonical `tool_calls[]` returned |
| Prompt caching | none | **Gemini explicit cache** (SHA-256 keyed) + `cache_*_input_tokens` surfacing for OpenAI-compat (implicit prefix) |
| Reasoning | none | **`reasoning="off"\|"low"\|"medium"\|"high"`** mapped to provider knob (Gemini `thinking_level`, OpenAI-compat `reasoning_effort`, etc.) |
| Structured output | "ask the model nicely" | **`response_format={type:"json_schema",schema:{}}`** + server-side schema validation + 1 corrective retry |
| Routing | RPM/RPD/cooldowns | RPM/RPD/cooldowns **+ capability skip**: routing skips providers that lack a requested capability and reports it |
| Endpoints | `/v1/chat`, `/v1/providers`, `/v1/status`, `/v1/calls`, `/`, `/help` | all of the above **+ `/v1/capabilities`** |
| Wait-on-cooldown | 503s on cooldown | When an explicit `provider` is given, the handler waits up to 30s for cooldown rather than 503-ing |
| Dashboard | RPM/RPD bars | RPM/RPD bars **+ capability badges + cache reads/writes + tool-call dialect** |
| SQLite log | tokens, latency | tokens, latency **+ cache_read_tokens, cache_create_tokens, tool_calls, tool_dialect, reasoning_applied** |

---

## Quick start

```bash
cd llm_gatewayV2
./run.sh                            # creates .venv, starts on port 8100
# in another shell:
curl -s http://localhost:8100/v1/capabilities | python3 -m json.tool
```

Reads `../.env` (parent directory) for keys — same as V1.

`GATEWAY_V2_PORT=8100` (override with env). `LLM_ORDER` is shared with V1.

---

## Request shape

All V1 fields still work. New optional fields:

```jsonc
{
  "messages": [...],
  "system": [{"text": "...", "cache": true}, {"text": "...per-turn..."}],   // OR plain string
  "cache_system": true,                                                     // shorthand: cache the whole system block
  "tools": [
    {"name":"add","description":"a+b",
     "input_schema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}
  ],
  "tool_choice": "auto",
  "reasoning": "high",
  "response_format": {"type":"json_schema","schema":{...},"name":"out","strict":true}
}
```

To send a tool result back, use a new `tool` role:

```jsonc
{"role":"tool","tool_call_id":"<id from tool_calls[0].id>","tool_name":"add","content":"{\"result\":12}"}
```

(`tool_name` is required for Gemini's `function_response`; OpenAI-compat ignores it.)

## Response shape

```jsonc
{
  "provider": "gemini",
  "model": "gemini-3.1-flash-lite-preview",
  "text": "",
  "tool_calls": [
    {"id":"call_4bf8ecc7","name":"add","arguments":{"a":7,"b":5},
     "provider_meta":{"thoughtSignature":"..."}}      // opaque meta — echo back unchanged in next turn
  ],
  "stop_reason": "tool_use",
  "input_tokens": 66, "output_tokens": 16,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0,
  "latency_ms": 412,
  "tool_call_dialect": "native",       // or "prompted_fallback" (Ollama on non-tool models) or "none"
  "reasoning_applied": false,
  "parsed": null,                      // populated when response_format is used and validation passes
  "attempted": []                      // failed providers + skip reasons
}
```

`provider_meta` carries provider-specific opaque state that must be sent back unchanged on the assistant turn. Today it carries Gemini's `thoughtSignature` (required by `gemini-3.x` models on the second turn or you get HTTP 400). Treat it as a black box.

---

## `/v1/capabilities`

Per-provider, per-current-model capability matrix. Routing reads this when failing over.

```bash
curl -s http://localhost:8100/v1/capabilities | python3 -m json.tool
```

```jsonc
{
  "gemini":   {"tools":true,"caching":true,"reasoning":false,"structured":true,"parallel_tools":true,"model":"gemini-3.1-flash-lite-preview","max_ctx":1000000,"rpm":15,"rpd":1000},
  "groq":     {"tools":true,"caching":true,"reasoning":false,"structured":true,"parallel_tools":true,"model":"llama-3.3-70b-versatile","max_ctx":100000,"rpm":30,"rpd":1000},
  "ollama":   {"tools":true,"caching":false,"reasoning":false,"structured":true,"parallel_tools":false,"model":"gemma4:31b","max_ctx":32000,"rpm":9999,"rpd":9999999}
  // ... 4 more
}
```

`reasoning=true` only when the **current** model supports it (e.g. Gemini 2.5/3 non-lite; OpenAI-compat reasoning models — DeepSeek-R, gpt-oss, qwen3, o1/o3, etc.). For other models the request returns with `reasoning_applied: false` and a 200 — the gateway logs the no-op rather than failing.

---

## Capability-aware routing

When a request needs a capability the chosen provider lacks, V2 skips it during failover and tags the attempt:

```jsonc
"attempted": [
  {"provider":"github","reason":"skipped:no_reasoning"},
  {"provider":"groq","reason":"cooldown (1.7s)"}
]
```

When you set `provider="..."` explicitly and that provider lacks the capability, the gateway still tries (so you can experiment) — capabilities only gate **failover**, not direct calls.

---

## End-to-end example: tools + caching + reasoning, against Gemini

```bash
curl -s -X POST http://localhost:8100/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "provider":"g",
    "system":[{"text":"You are a careful math assistant. Always pick a tool when one fits.","cache":true}],
    "messages":[{"role":"user","content":"What is 7 plus 5? Use the add tool."}],
    "tools":[{"name":"add","description":"Return a + b.","input_schema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}],
    "tool_choice":"auto",
    "reasoning":"medium",
    "max_tokens":2048
  }' | python3 -m json.tool
```

(On `gemini-3.1-flash-lite-preview` reasoning is silently ignored — the response will show `reasoning_applied: false`. Switch to `gemini-2.5-pro` or `gemini-3-pro` and it kicks in.)

---

## Python client

```python
from client import LLM
llm = LLM()  # http://localhost:8100

# Tool use
result = llm.chat(
    messages=[{"role":"user","content":"What is 7+5? use add."}],
    provider="gr",
    tools=[{"name":"add","description":"a+b",
            "input_schema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}],
    tool_choice="auto",
)
tc = result["tool_calls"][0]
# … run the tool locally, call back with role="tool"

# Structured output
plan = llm.chat(
    prompt="Plan a 2-step trip from Bangalore to Tokyo as JSON.",
    provider="g",
    response_format={"type":"json_schema","schema":{...},"name":"plan","strict":True},
)
print(plan["parsed"])           # validated dict, or None on validation failure (then 503)

# Cached system + reasoning
out = llm.chat(
    prompt="Solve: ...",
    provider="g",
    system="<long stable preamble>",
    cache_system=True,
    reasoning="high",
)
print(out["cache_creation_input_tokens"], out["cache_read_input_tokens"], out["reasoning_applied"])

# All V1 calls still work unchanged
print(llm.chat("hello", provider="o")["text"])
```

---

## Per-provider notes (free tier behaviour, May 2026)

| Provider | Tools | Caching | Reasoning | Structured | Notes |
|---|---|---|---|---|---|
| **ollama** | native (llama3.x/qwen2.5+/mistral-nemo) or **prompted_fallback** | n/a (local) | n/a | json schema via `format=` | thinking models like `gemma4:31b` need `max_tokens ≥ 1024` to emit visible text |
| **gemini** | native (`function_declarations`) | **explicit cache** above 1024 tokens — but free tier on `*-flash-lite-preview` is **disabled by Google** (returns 429 RESOURCE_EXHAUSTED with `limit=0`); switch to a paid model to exercise it | `thinking_level` on 2.5/3 non-lite | `responseMimeType` + `responseSchema` (auto-cleans `additionalProperties`/`$schema` etc.) | requires `thoughtSignature` echo back when re-sending function_call — V2 captures it into `provider_meta` and threads it through |
| **nvidia** | native (OpenAI-compat) | implicit prefix (server-side); `cache_read_input_tokens` surfaces if the upstream returns `prompt_tokens_details.cached_tokens` | passed via `reasoning_effort` for DeepSeek-R / gpt-oss — silent no-op for v3.2 | json_schema strict, falls back to json_object | `deepseek-v4-pro` is slow (15-25s typical), bump request timeout |
| **groq** | native | implicit | `reasoning_effort` on gpt-oss / qwen3 / deepseek-r1 — no-op on llama-3.3 | json_schema | tightest cooldown (2s) of the OpenAI-compat group |
| **cerebras** | native | implicit | `reasoning_effort` on qwen3-think | json_schema | free-tier RPM is genuinely 30; will 429 under burst — V2 retries via failover |
| **openrouter** | native | implicit | `reasoning_effort` if model supports it | json_schema | `:free` tier has 50 RPD pool shared across :free models; quality of `nvidia/nemotron-3-super:free` for structured output is iffy |
| **github** | native | implicit | `reasoning_effort` on o1/o3/o4/gpt-5 family | json_schema strict | hard 8K input / 4K output cap |

**Reasoning** — when a knob is unsupported on the current model, V2 logs it and returns `reasoning_applied: false` with a 200. It does **not** fail — that would be too aggressive for routing.

**Caching** — for OpenAI-compat providers the gateway just keeps the system prefix byte-stable across calls; the upstream's implicit prefix cache does the real work. The `cache_read_input_tokens` field surfaces only if the provider returns `usage.prompt_tokens_details.cached_tokens` (Groq does sometimes; the others rarely on free tier). For Gemini you have to explicitly cache, and free-tier `flash-lite` doesn't allow it. For Ollama there is no upstream — local generation is already fast.

---

## Test matrix

```bash
./.venv/bin/python tests/test_all_providers.py
```

Runs five tests (basic, tools, structured, cache, reasoning) against each provider individually and prints a matrix. `n/a` is an honest answer when the provider/free-tier doesn't support the feature; `SKIP` is for cases that genuinely don't apply (Ollama for caching).

Exit code is non-zero if any provider's `basic` test fails.

A real run (May 2026, free-tier keys, all 7 providers in parallel):

```
provider    basic    tools     struct    cache    reasoning
-----------------------------------------------------------
ollama      OK       OK        OK        SKIP     n/a
gemini      OK       OK        OK        n/a*     n/a*
nvidia      OK       OK        OK        n/a      n/a
groq        OK       OK        OK        FAIL‡    n/a
cerebras    FAIL§    FAIL§     FAIL§     FAIL     FAIL
openrouter  OK       OK        OK        n/a      n/a
github      OK       OK        OK        n/a      n/a
```

**6/7 providers green on basic + tools + structured-output.**

- **(*) Gemini cache n/a** — free tier returns `limit=0` for `gemini-3.1-flash-lite-preview`. The cache module mints and reuses correctly on paid-tier or larger models; on free-tier flash-lite it silently falls back to no-cache. `gemini-3.1-flash-lite` is also a non-thinking model so the reasoning knob is correctly a no-op.
- **NVIDIA** — `deepseek-v4-pro` is slow on free tier (15-25s typical, occasionally up to 2 min) but stable; passes basic + tools + struct.
- **(‡) Groq cache FAIL** — second call hit the 2s cooldown during parallel test; not a real failure of caching, just a test-pacing artefact. `prompt_tokens_details.cached_tokens` is rarely populated by Groq on free tier anyway.
- **(§) Cerebras** — free-tier RPM (30) was burned during parallel test. Serial calls succeed.
- **GitHub cache** — `cr2=2560` cached tokens read on second call (verified earlier; intermittent across runs depending on whether `prompt_tokens_details.cached_tokens` is populated). Implicit prefix caching is working server-side.
- **Ollama tools** — `gemma4:31b` (the configured default) doesn't speak native tool-use, so the gateway falls back to **prompted_fallback** and parses `{"tool_call":{...}}` JSON out of the prose. **Native dialect verified** with `model="llama3.2:latest"` (and any `llama3.x` / `qwen2.5+` / `mistral-nemo` / `firefunction`) — set `model=` and the response carries `tool_call_dialect: "native"`. `phi4:latest` also returns a parsed tool call via prompted_fallback. `smollm2:135m` is too small (135M params) to follow either protocol.
- **Reasoning n/a on the default models** — none of the *current default* free-tier models is a reasoning model. **Verified working** on three providers:
  - **GitHub** — `model="deepseek/DeepSeek-R1"` + `reasoning="medium"` → `reasoning_applied: true`, visible `<think>...</think>` block, 5.2s latency.
  - **Groq** — `model="openai/gpt-oss-120b"` (or `gpt-oss-20b`) + `reasoning="medium"` → `reasoning_applied: true`, sub-second latency.
  - **Gemini** — `model="gemini-2.5-flash"` (uses `thinkingBudget` integer knob) or `model="gemini-2.5-pro"` / `gemini-3-pro` (uses `thinkingLevel` enum) → `reasoning_applied: true`. The gateway picks the right knob per model — see `_gemini_thinking_knob()`. **Note**: `qwen3-32b` on Groq emits `<think>` blocks but doesn't accept `reasoning_effort`; the gateway honestly returns `reasoning_applied: false` rather than lying.
- **Cerebras** — free-tier RPM (30/min) is harsh and the upstream sometimes returns `queue_exceeded` ("We're experiencing high traffic right now") regardless of the gateway's pacing. Serial calls with sufficient spacing pass basic; parallel + tool tests are at the mercy of the upstream queue.
- **Robust model-override handling** — when an explicit `model=` is set and the upstream returns 403/404 (model entitlement issue), V2 surfaces the error to the caller but does **not** put the whole provider into 600s backoff. This avoids the "one bad model name kills GitHub for ten minutes" footgun.

Acceptance bar is `basic` and `tools` OK on at least 5/7. Met. See `/v1/capabilities` for the truth at runtime.

---

## Files

- `main.py` — FastAPI app, routes, schema-validation + corrective retry
- `providers.py` — adapters with `_translate_tools`, `_translate_messages`, `_apply_response_format`, `_apply_reasoning`
- `cache.py` — Gemini SHA-256-keyed cache (TTL 5 min)
- `router.py` — RateState + capability-aware `pick()`
- `schemas.py` — Pydantic v2 request/response models incl. `ToolCall`, `ToolDef`, `CacheableSystemBlock`
- `db.py` — SQLite log with tool/cache/reasoning columns
- `client.py` — Python SDK with new kwargs
- `static/dashboard.html` — capability badges, cache columns
- `tests/test_all_providers.py` — per-provider matrix
- `run.sh`, `requirements.txt`

---

## What V2 deliberately does NOT do

- **No new providers.** Same seven, same `.env` keys.
- **No streaming changes** beyond emitting `tool_call_delta` SSE chunks alongside text deltas (same envelope).
- **No tool execution.** V2 returns `tool_calls`; your agent dispatches and sends results back as `role: "tool"`.
- **No agent loop.** The Plan→Act→Verify shape lives in `Session 5/agent5.py`, not here. The gateway is the substrate.
