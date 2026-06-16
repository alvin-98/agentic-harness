import os, time, json, re
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent.parent / ".env")
print(ROOT.parent.parent)
import db
import providers as P
from router import Router, RouterPool, DEFAULT_ROUTER_ORDER, LIMITS, SHORTCUTS, resolve
from cache import GeminiCache
from schemas import ChatRequest, ChatResponse, ToolCall, RouterDecision, EmbedRequest, EmbedResponse
import embedders as E
import config as C

DEFAULT_ORDER = ["sglang", "ollama", "gemini", "nvidia", "groq", "cerebras", "openrouter", "github"]
ORDER = [x.strip() for x in os.getenv("LLM_ORDER", ",".join(DEFAULT_ORDER)).split(",") if x.strip()]
ROUTER_ORDER = [x.strip() for x in os.getenv("ROUTER_ORDER", ",".join(DEFAULT_ROUTER_ORDER)).split(",") if x.strip()]
PORT = int(os.getenv("GATEWAY_V3_PORT", "8101"))

# Tier -> worker failover order. Loaded from models.yaml `tier_order`; these
# defaults are only used when the config file is missing or has no tier_order.
TIER_TO_ORDER = {
    "TINY":  ["github", "openrouter", "cerebras", "gemini", "ollama"],
    "LARGE": ["cerebras", "gemini", "openrouter", "ollama"],
    "HUGE":  ["openrouter", "groq"],
}

# Router envelope: cap the sample at ~800 chars (first 400 + last 400).
# Keeps router input under 400 tokens regardless of worker payload size, so
# routing decisions never burn through router quota on big prompts.
ROUTER_SAMPLE_HEAD = 400
ROUTER_SAMPLE_TAIL = 400
ROUTER_PROMPT = (
    "You are a routing classifier. Classify requests by size. Respond in JSON.\n"
    "Rules:\n"
    "- TINY: token_count below 1000 with simple factual content.\n"
    "- LARGE: token_count between 1000 and 8000, OR token_count below 1000 "
    "but content is dense (code, base64, multilingual, technical).\n"
    "- HUGE: token_count above 8000."
)

ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "tier": {"type": "string", "enum": ["TINY", "LARGE", "HUGE"]}
        },
        "required": ["tier"],
        "additionalProperties": False,
    },
    "name": "RouterDecision",
    "strict": True,
}


def _estimate_tokens(text: str) -> int:
    """words * 1.4 — deliberately rough. The router sample handles the cases
    where rough isn't good enough (code, CJK, base64)."""
    return int(len(text.split()) * 1.4)


def _build_sample(text: str) -> str:
    if len(text) <= ROUTER_SAMPLE_HEAD + ROUTER_SAMPLE_TAIL + 10:
        return text
    return text[:ROUTER_SAMPLE_HEAD] + "\n...\n" + text[-ROUTER_SAMPLE_TAIL:]


def _tier_from_count(tokens: int) -> str:
    """Deterministic fallback when the router LLM is unreachable or replies
    with garbage. Pure token-count rule, identical thresholds."""
    if tokens > 8000:
        return "HUGE"
    if tokens >= 1000:
        return "LARGE"
    return "TINY"


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags that reasoning models may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_tier(text: str) -> Optional[str]:
    # Strip thinking tags first
    text = _strip_think_tags(text or "")
    # Try JSON extraction first (reasoning models may wrap in JSON)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            tier = (obj.get("tier") or obj.get("classification") or "").upper()
            if tier in ("TINY", "LARGE", "HUGE"):
                return tier
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting JSON from mixed text
    m = re.search(r'\{[^}]*"tier"\s*:\s*"(TINY|LARGE|HUGE)"[^}]*\}', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: keyword search
    up = text.upper()
    for tier in ("HUGE", "LARGE", "TINY"):
        if tier in up:
            return tier
    return None


async def _classify_tier(req: ChatRequest, role: str, router_pool: RouterPool, prompt_text: str):
    """Run a router-LLM classification. Returns a RouterDecision (without
    chosen_worker_* fields, which are filled in by the caller after worker pick).

    Failover: try each router provider in order. Only fall back to the
    pure token-count rule when all routers in the pool have failed.
    """
    estimated = _estimate_tokens(prompt_text)

    # Short-circuit HUGE — no need to spend a router call.
    if estimated > 8000:
        return RouterDecision(
            role=role, tier="HUGE", estimated_tokens=estimated,
            router_provider="(skipped)", router_model="(skipped)",
            router_latency_ms=0, fallback_used=True,
        )

    sample = _build_sample(prompt_text)
    envelope = f"token_count: {estimated}\nsample:\n{sample}"
    call_role = f"router_{role}"

    last_provider = ""
    last_model = ""
    last_latency = 0
    router_attempts = []

    for name in router_pool.candidates():
        ok, why = router_pool.state[name].can_use(LIMITS[name], 400)
        if not ok:
            router_attempts.append({"provider": name, "model": router_pool.providers[name].model, "status": "skipped", "reason": why})
            continue
        provider = router_pool.providers[name]
        t0 = time.time()
        router_pool.state[name].record(0)
        last_provider = name
        last_model = provider.model
        try:
            result = await provider.chat(
                messages=[{"role": "user", "content": envelope}],
                system_blocks=ROUTER_PROMPT,
                max_tokens=32, temperature=0,
                model=None, tools=None, tool_choice=None,
                reasoning="off", response_format=ROUTER_RESPONSE_FORMAT,
                cache_system=False,
            )
            latency = int((time.time() - t0) * 1000)
            last_latency = latency
            tokens = (result.get("input_tokens") or 0) + (result.get("output_tokens") or 0)
            router_pool.state[name].tokens_today += tokens
            router_pool.state[name].tokens_minute.append((time.time(), tokens))
            # Extract tier from structured output, fallback to text parsing
            parsed = result.get("parsed")
            tier = (parsed.get("tier") or parsed.get("classification")) if parsed else None
            if tier:
                tier = tier.upper() if tier.upper() in ("TINY", "LARGE", "HUGE") else None
            if tier is None:
                raw_text = _strip_think_tags(result.get("text", ""))
                tier = _parse_tier(raw_text)
            # Sanity clamp: HUGE is only valid when the deterministic count
            # agrees. Small router LLMs occasionally hallucinate HUGE on small
            # inputs that look "dense" (URLs, JSON brackets, code fragments).
            # The 8000-token ceiling is hard — override the LLM here to keep
            # the request servable.
            if tier == "HUGE" and estimated <= 8000:
                tier = "LARGE"
            if tier is None:
                # Router returned text we couldn't classify — try the next router
                # rather than giving up immediately. Log this attempt as a soft
                # failure with the actual response captured.
                raw_text = result.get("text", "")
                router_attempts.append({
                    "provider": name, "model": result.get("model", provider.model),
                    "status": "unparseable", "latency_ms": latency,
                    "raw_reply": raw_text[:300],
                })
                db.log_call(provider=name, model=result.get("model", provider.model),
                            input_tokens=result.get("input_tokens", 0),
                            output_tokens=result.get("output_tokens", 0),
                            latency_ms=latency, status="error",
                            error=f"unparseable tier reply: {raw_text[:300]}",
                            prompt_chars=len(envelope),
                            call_role=call_role, router_decision="unparseable")
                continue
            router_attempts.append({
                "provider": name, "model": result.get("model", provider.model),
                "status": "ok", "tier": tier, "latency_ms": latency,
            })
            db.log_call(provider=name, model=result.get("model", provider.model),
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        latency_ms=latency, status="ok",
                        prompt_chars=len(envelope), response_chars=len(result.get("text", "")),
                        call_role=call_role, router_decision=tier)
            return RouterDecision(
                role=role, tier=tier, estimated_tokens=estimated,
                router_provider=name, router_model=result.get("model", provider.model),
                router_latency_ms=latency, fallback_used=False,
                router_attempts=router_attempts,
            )
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            last_latency = latency
            router_attempts.append({
                "provider": name, "model": provider.model,
                "status": "error", "latency_ms": latency,
                "error": str(e)[:300],
            })
            db.log_call(provider=name, model=provider.model,
                        status="error", error=str(e)[:500],
                        latency_ms=latency, call_role=call_role,
                        router_decision="error")
            # Move on to the next router. No backoff for routing — keep the
            # router pool aggressive since each call is cheap.
            continue

    # All routers in the pool failed — deterministic token-count fallback.
    return RouterDecision(
        role=role, tier=_tier_from_count(estimated), estimated_tokens=estimated,
        router_provider=last_provider or "(unavailable)",
        router_model=last_model or "(unavailable)",
        router_latency_ms=last_latency, fallback_used=True,
        router_attempts=router_attempts,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    # Load models.yaml config (V4)
    gateway_cfg = None
    if C.CONFIG_PATH.exists():
        try:
            gateway_cfg = C.load_config(C.CONFIG_PATH)
            # Override ORDER/ROUTER_ORDER from config if present
            if gateway_cfg.llm_order:
                global ORDER, ROUTER_ORDER, PORT
                ORDER = gateway_cfg.llm_order
            if gateway_cfg.router_order:
                ROUTER_ORDER = gateway_cfg.router_order
            if gateway_cfg.gateway_port:
                PORT = gateway_cfg.gateway_port
            if gateway_cfg.tier_order:
                global TIER_TO_ORDER
                TIER_TO_ORDER = gateway_cfg.tier_order
        except Exception as e:
            print(f"[WARNING] Failed to load models.yaml: {e}. Falling back to env vars.")
            gateway_cfg = None
    # Sync models from provider APIs at startup (discover new, prune stale)
    if gateway_cfg:
        try:
            sync_results = await C.sync_models_on_startup(gateway_cfg)
            if sync_results:
                for prov, info in sync_results.items():
                    print(f"  [sync] {prov}: discovered={info['discovered']}, "
                          f"added={info['added']}, removed={info['removed']}")
        except Exception as e:
            print(f"[WARNING] Model sync at startup failed: {e}")

    app.state.gateway_config = gateway_cfg
    app.state.cache = GeminiCache(ttl_seconds=300)
    app.state.providers = P.build_providers(app.state.cache)
    app.state.router = Router(app.state.providers, ORDER, config=gateway_cfg)
    app.state.router_providers = P.build_router_providers()
    app.state.router_pool = RouterPool(app.state.router_providers, ROUTER_ORDER)
    app.state.embedders, app.state.embed_order = E.build_embedders()
    yield


app = FastAPI(title="LLM Gateway V3", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def _normalize_messages(req: ChatRequest):
    if req.messages:
        return list(req.messages)
    msgs = []
    msgs.append({"role": "user", "content": req.prompt or ""})
    return msgs


def _system_blocks(req: ChatRequest):
    """Returns the system_blocks payload to hand to the provider adapter."""
    if req.system is None:
        return None
    if isinstance(req.system, str):
        if req.cache_system:
            return [{"text": req.system, "cache": True}]
        return req.system
    return [b.model_dump() if hasattr(b, "model_dump") else b for b in req.system]


def _est_tokens(messages, system_blocks, max_tokens):
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    if isinstance(system_blocks, str):
        chars += len(system_blocks)
    elif isinstance(system_blocks, list):
        for b in system_blocks:
            chars += len(b.get("text", "") if isinstance(b, dict) else "")
    return chars // 4 + max_tokens


def _backoff_for(err: Exception, has_model_override: bool = False):
    msg = str(err).lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "queue" in msg: return 15, "server queue full"
        if "quota" in msg or "rpm" in msg or "per minute" in msg: return 60, "RPM quota burned"
        if "rpd" in msg or "per day" in msg or "daily" in msg: return 3600, "RPD quota burned"
        return 30, "rate limited"
    if status and 500 <= status < 600: return 20, f"upstream {status}"
    if status == 408 or "timeout" in msg: return 10, "timeout"
    if status in (401, 403):
        # When the caller explicitly picked a model, 403/404 likely means
        # "this model not available to your account" rather than "key dead".
        # Don't blackball the whole provider for 10 minutes.
        if has_model_override:
            return 0, ""
        return 600, "auth error"
    if status == 404 and has_model_override:
        return 0, ""
    return 0, ""


def _attempts_str(attempts):
    return "; ".join(f"{a['provider']}:{a['reason']}" for a in attempts)


def _required_caps(req: ChatRequest):
    caps = []
    if req.tools: caps.append("tools")
    if req.reasoning and req.reasoning != "off": caps.append("reasoning")
    if req.response_format: caps.append("structured")
    return caps


def _validate_structured(text: str, schema: dict):
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"output is not JSON: {e}")
    Draft202012Validator(schema).validate(obj)
    return obj


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    router = app.state.router
    router_pool = app.state.router_pool
    messages = _normalize_messages(req)
    system_blocks = _system_blocks(req)
    prompt_text = "".join(str(m.get("content", "")) for m in messages)
    est = _est_tokens(messages, system_blocks, req.max_tokens)
    explicit_override = bool(req.provider)
    required_caps = _required_caps(req)

    # V3: auto_route runs a router-LLM classifier first and uses tier-specific
    # failover order. Explicit `provider` overrides routing (caller knows best).
    router_decision: Optional[RouterDecision] = None
    if req.auto_route and not req.provider:
        router_decision = await _classify_tier(req, req.auto_route, router_pool, prompt_text)
        # Replace failover order with the tier-specific one, intersected with
        # what's actually wired in this gateway.
        tier_order = TIER_TO_ORDER[router_decision.tier]
        candidates = [p for p in tier_order if p in router.order]
    else:
        candidates = router.candidates(req.provider) if req.provider else list(router.order)

    if req.provider and not candidates:
        raise HTTPException(400, f"unknown provider '{req.provider}'. Try one of: {list(router.providers)} or shortcuts {list(SHORTCUTS)}")

    all_attempts = []
    last_err = None

    # When explicit provider is requested and the only blocker is cooldown,
    # wait briefly rather than 503-ing — this is what users intuitively expect.
    if explicit_override and len(candidates) == 1:
        import asyncio as _asyncio
        deadline = time.time() + 30
        while time.time() < deadline:
            name, _ = router.pick(est, candidates, required_caps=required_caps)
            if name is not None:
                break
            cd = router.state[candidates[0]].snapshot(LIMITS[candidates[0]])["cooldown_remaining"]
            if cd <= 0 or cd > 30:
                break
            await _asyncio.sleep(min(cd + 0.05, 5))

    # V4: determine max iterations based on total model slots across all candidates
    max_iters = len(candidates) + 1
    if router.config:
        total_slots = sum(len(router.config.providers[c].models) for c in candidates if c in router.config.providers)
        max_iters = max(total_slots + 1, max_iters)

    for _ in range(max_iters):
        name, atts = router.pick(est, candidates, required_caps=required_caps)
        all_attempts.extend(atts)
        if name is None:
            break

        provider = router.providers[name]
        # V4: use router-picked model when no explicit req.model override
        picked_model = router._last_picked_model if router.config else None
        use_model = req.model or picked_model  # req.model wins if set explicitly
        t0 = time.time()
        router.state[name].record(0)

        try:
            if req.stream:
                async def gen():
                    try:
                        agg = []
                        async for chunk in provider.stream(messages,
                                                          max_tokens=req.max_tokens,
                                                          temperature=req.temperature,
                                                          model=use_model,
                                                          tools=req.tools,
                                                          tool_choice=req.tool_choice,
                                                          reasoning=req.reasoning,
                                                          response_format=req.response_format,
                                                          system_blocks=system_blocks,
                                                          cache_system=bool(req.cache_system)):
                            agg.append(chunk)
                            if chunk.startswith("[[TOOL_CALL_DELTA]]"):
                                yield f"data: {json.dumps({'provider': name, 'tool_call_delta': chunk[len('[[TOOL_CALL_DELTA]] '):]})}\n\n"
                            else:
                                yield f"data: {json.dumps({'provider': name, 'delta': chunk})}\n\n"
                        text = "".join(agg)
                        latency = int((time.time() - t0) * 1000)
                        db.log_call(provider=name, model=use_model or provider.model,
                                    latency_ms=latency, status="ok",
                                    prompt_chars=len(prompt_text), response_chars=len(text),
                                    override=req.provider, attempted=_attempts_str(all_attempts))
                        yield f"data: {json.dumps({'done': True, 'provider': name})}\n\n"
                    except Exception as e:
                        db.log_call(provider=name, model=use_model or provider.model,
                                    status="error", error=str(e)[:500],
                                    latency_ms=int((time.time() - t0) * 1000),
                                    prompt_chars=len(prompt_text),
                                    override=req.provider, attempted=_attempts_str(all_attempts))
                        yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")

            result = await provider.chat(messages,
                                         max_tokens=req.max_tokens,
                                         temperature=req.temperature,
                                         model=use_model,
                                         tools=req.tools,
                                         tool_choice=req.tool_choice,
                                         reasoning=req.reasoning,
                                         response_format=req.response_format,
                                         system_blocks=system_blocks,
                                         cache_system=bool(req.cache_system))
            latency = int((time.time() - t0) * 1000)

            # Optional: validate structured output and (single) retry on failure.
            parsed = None
            if req.response_format and req.response_format.schema_ and not result["tool_calls"]:
                try:
                    parsed = _validate_structured(result["text"], req.response_format.schema_)
                except (ValueError, ValidationError) as ve:
                    # one corrective retry
                    fix_msgs = list(messages) + [
                        {"role": "assistant", "content": result["text"]},
                        {"role": "user", "content": f"Your previous reply did not match the required JSON schema: {ve}. Reply ONLY with valid JSON conforming to the schema."},
                    ]
                    result = await provider.chat(fix_msgs,
                                                 max_tokens=req.max_tokens,
                                                 temperature=0,
                                                 model=use_model,
                                                 response_format=req.response_format,
                                                 system_blocks=system_blocks,
                                                 cache_system=bool(req.cache_system))
                    try:
                        parsed = _validate_structured(result["text"], req.response_format.schema_)
                    except (ValueError, ValidationError) as ve2:
                        raise HTTPException(503, f"structured output failed validation: {ve2}")

            tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
            router.state[name].tokens_today += tokens
            router.state[name].tokens_minute.append((time.time(), tokens))
            # V4: record per-model state and update live limits from headers
            actual_model = result["model"] or use_model or provider.model
            if router.config:
                router.record_call(name, actual_model, tokens)
                rl = result.get("rate_limit_headers") or {}
                if rl:
                    router.update_live_limits(
                        name, actual_model,
                        rpm_remaining=rl.get("rpm_remaining"),
                        rpd_remaining=rl.get("rpd_remaining"),
                        rpm_limit=rl.get("rpm_limit"),
                        rpd_limit=rl.get("rpd_limit"),
                    )
                C.maybe_flush_state(router.config)
            if router_decision is not None:
                router_decision.chosen_worker_provider = name
                router_decision.chosen_worker_model = actual_model
            db.log_call(provider=name, model=actual_model,
                        input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                        cache_create_tokens=result["cache_creation_input_tokens"],
                        cache_read_tokens=result["cache_read_input_tokens"],
                        latency_ms=latency, status="ok",
                        prompt_chars=len(prompt_text), response_chars=len(result["text"]),
                        override=req.provider, attempted=_attempts_str(all_attempts),
                        tool_calls=len(result["tool_calls"]),
                        reasoning_applied=result["reasoning_applied"],
                        tool_dialect=result["tool_call_dialect"],
                        call_role="worker",
                        router_decision=router_decision.tier if router_decision else None)
            return ChatResponse(
                provider=name,
                model=result["model"],
                text=result["text"],
                tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
                stop_reason=result["stop_reason"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cache_creation_input_tokens=result["cache_creation_input_tokens"],
                cache_read_input_tokens=result["cache_read_input_tokens"],
                latency_ms=latency,
                tool_call_dialect=result["tool_call_dialect"],
                reasoning_applied=result["reasoning_applied"],
                parsed=parsed,
                attempted=all_attempts,
                router_decision=router_decision,
            ).model_dump()

        except P.ProviderError as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            err_model = use_model or provider.model
            if secs > 0:
                router.mark_model_unavailable(name, err_model, secs, reason)
            db.log_call(provider=name, model=err_model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text),
                        override=req.provider, attempted=_attempts_str(all_attempts))
            tag = f"failed: {str(e)[:100]}"
            if secs > 0: tag += f" → backoff {secs:.0f}s ({reason})"
            all_attempts.append({"provider": name, "model": err_model, "reason": tag})
            if explicit_override or not getattr(e, "retryable", True):
                raise HTTPException(502, f"{name} failed: {e}")
            # V4: don't remove provider from candidates — other models may be available
            if not router.config:
                candidates = [c for c in candidates if c != name]
            continue
        except HTTPException:
            raise
        except Exception as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            err_model = use_model or provider.model
            if secs > 0:
                router.mark_model_unavailable(name, err_model, secs, reason)
            db.log_call(provider=name, model=err_model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text),
                        override=req.provider, attempted=_attempts_str(all_attempts))
            all_attempts.append({"provider": name, "model": err_model, "reason": f"exception: {str(e)[:120]}"})
            if explicit_override:
                raise HTTPException(502, f"{name} failed: {e}")
            if not router.config:
                candidates = [c for c in candidates if c != name]
            continue

    raise HTTPException(503, f"all providers unavailable. attempts: {all_attempts}. last_error: {last_err}")


@app.post("/v1/embed")
async def embed(req: EmbedRequest):
    """Single V7 endpoint. Failover ring runs nomic → configured fallback.
    `provider` pins the choice (returns 502 on failure with no fallback).
    Rejects inputs over MAX_INPUT_CHARS with 413 — caller must chunk."""
    embedders = app.state.embedders
    if not embedders:
        raise HTTPException(503, "no embedding providers configured")

    if len(req.text) > E.MAX_INPUT_CHARS:
        raise HTTPException(
            413,
            f"text is {len(req.text)} chars; embed input is capped at "
            f"{E.MAX_INPUT_CHARS} chars (~{E.MAX_INPUT_CHARS // 4} tokens). "
            f"Chunk the input and embed each chunk.",
        )

    t0 = time.time()
    try:
        name, result, attempts, latency = await E.embed_with_failover(
            embedders, req.text, req.task_type, explicit=req.provider
        )
    except E.EmbedderError as e:
        latency = int((time.time() - t0) * 1000)
        db.log_call(
            provider=req.provider or "(any)",
            model="(none)",
            status="error",
            error=str(e)[:500],
            latency_ms=latency,
            prompt_chars=len(req.text),
            override=req.provider,
            call_role="embed",
        )
        if req.provider:
            if e.status == 429:
                raise HTTPException(429, f"{req.provider} rate-limited: {e}")
            if e.status == 400:
                raise HTTPException(400, str(e))
            raise HTTPException(502, f"{req.provider} embed failed: {e}")
        raise HTTPException(503, str(e))

    db.log_call(
        provider=name,
        model=result["model"],
        status="ok",
        latency_ms=latency,
        prompt_chars=len(req.text),
        override=req.provider,
        attempted=_attempts_str(attempts),
        call_role="embed",
        embed_dim=result["dim"],
    )
    return EmbedResponse(
        provider=name,
        model=result["model"],
        embedding=result["embedding"],
        dim=result["dim"],
        latency_ms=latency,
        attempted=attempts,
    ).model_dump()


@app.get("/v1/embedders")
async def list_embedders():
    return {
        "order": app.state.embed_order,
        "models": {e.name: e.model for e in app.state.embedders},
        "fixed_dim": E.EMBED_DIM,
        "max_input_chars": E.MAX_INPUT_CHARS,
        "backoff_steps_s": E.BACKOFF_STEPS,
        "live": {e.name: e.state.snapshot() for e in app.state.embedders},
        "today": db.aggregate(call_role="embed"),
    }


@app.get("/v1/providers")
async def list_providers():
    r = app.state.router
    return {
        "order": r.order,
        "providers": list(r.providers.keys()),
        "shortcuts": SHORTCUTS,
        "limits": LIMITS,
        "models": {n: p.model for n, p in r.providers.items()},
    }


@app.get("/v1/capabilities")
async def capabilities():
    r = app.state.router
    out = {}
    for name, p in r.providers.items():
        caps = dict(getattr(p, "capabilities", {}))
        # per-model overrides
        caps = P.model_capabilities(name, p.model, caps)
        caps["model"] = p.model
        caps.update({
            "max_ctx": LIMITS[name]["max_ctx"],
            "rpm": LIMITS[name]["rpm"],
            "rpd": LIMITS[name]["rpd"],
        })
        out[name] = caps
    return out


@app.get("/v1/status")
async def status():
    r = app.state.router
    return {"order": r.order, "live": r.all_status(),
            "today": db.aggregate(call_role="worker"), "limits": LIMITS}


@app.get("/v1/routers")
async def routers():
    """V3: router pool — separate from the worker pool. Shows which router LLMs
    are wired, the failover order, and live rate-state."""
    rp = app.state.router_pool
    return {
        "order": rp.order,
        "providers": list(rp.providers.keys()),
        "models": {n: p.model for n, p in rp.providers.items()},
        "live": rp.all_status(),
        "today": db.aggregate(call_role="router"),
        "limits": {k: LIMITS[k] for k in rp.providers},
        "tier_to_order": TIER_TO_ORDER,
    }


@app.get("/v1/config")
async def get_config():
    """V4: return current gateway config state."""
    cfg = app.state.gateway_config
    if not cfg:
        return {"loaded": False}
    return {
        "loaded": True,
        "failover_strategy": cfg.failover_strategy,
        "llm_order": cfg.llm_order,
        "router_order": cfg.router_order,
        "providers": {
            pname: {
                "models": [
                    {"id": s.id, "rpm_limit": s.rpm_limit, "rpd_limit": s.rpd_limit,
                     "max_ctx": s.max_ctx, "structured": s.structured,
                     "type": s.type, "visible": s.visible,
                     "exhausted": s.exhausted, "rpm_remaining": s.rpm_remaining,
                     "rpd_remaining": s.rpd_remaining}
                    for s in pcfg.models if s.visible
                ],
                "active_model": pcfg.active_model.id if pcfg.active_model else None,
                "active_index": pcfg.active_index(),
            }
            for pname, pcfg in cfg.providers.items()
        },
    }


@app.post("/v1/config/strategy")
async def set_strategy(body: dict):
    """V4: switch failover strategy at runtime."""
    cfg = app.state.gateway_config
    if not cfg:
        raise HTTPException(400, "models.yaml not loaded")
    strategy = body.get("strategy", "").lower()
    if strategy not in ("breadth", "depth"):
        raise HTTPException(400, "strategy must be 'breadth' or 'depth'")
    cfg.failover_strategy = strategy
    C.save_state(cfg)
    return {"ok": True, "failover_strategy": strategy}


@app.post("/v1/config/reload")
async def reload_config():
    """V4: hot-reload models.yaml from disk."""
    cfg = app.state.gateway_config
    if not cfg:
        raise HTTPException(400, "models.yaml not loaded")
    new_cfg = C.reload_if_changed(cfg)
    app.state.gateway_config = new_cfg
    app.state.router.config = new_cfg
    # Update order if changed
    if new_cfg.llm_order:
        app.state.router.order = [p for p in new_cfg.llm_order if p in app.state.router.providers]
    return {"ok": True, "changed": new_cfg is not cfg}


@app.post("/v1/config/update")
async def update_models():
    """V4: discover models from all configured providers and merge into config.

    Discovers models from Groq, NVIDIA, and OpenRouter (free models only).
    New models get visible=False by default. Existing models are not overwritten.
    """
    cfg = app.state.gateway_config
    if not cfg:
        raise HTTPException(400, "models.yaml not loaded")

    results = {}
    any_added = False

    # OpenRouter: free models only
    or_key = os.getenv("OPEN_ROUTER_API_KEY", "")
    if or_key and "openrouter" in cfg.providers:
        discovered = await C.discover_openrouter_free_models(or_key)
        added = C.merge_discovered_models(cfg, discovered, provider_name="openrouter")
        results["openrouter"] = {"discovered": len(discovered), "added": added}
        if added:
            any_added = True

    # Groq: all models
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and "groq" in cfg.providers:
        discovered = await C.discover_groq_models(groq_key)
        added = C.merge_discovered_models(cfg, discovered, provider_name="groq")
        results["groq"] = {"discovered": len(discovered), "added": added}
        if added:
            any_added = True

    # NVIDIA: all models
    nv_key = os.getenv("NVIDIA_API_KEY", "")
    if nv_key and "nvidia" in cfg.providers:
        discovered = await C.discover_nvidia_models(nv_key)
        added = C.merge_discovered_models(cfg, discovered, provider_name="nvidia")
        results["nvidia"] = {"discovered": len(discovered), "added": added}
        if added:
            any_added = True

    # Gemini: all models supporting generateContent
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and "gemini" in cfg.providers:
        discovered = await C.discover_gemini_models(gemini_key)
        added = C.merge_discovered_models(cfg, discovered, provider_name="gemini")
        results["gemini"] = {"discovered": len(discovered), "added": added}
        if added:
            any_added = True

    if any_added:
        C.save_state(cfg)
    return {"ok": True, "results": results}


@app.post("/v1/config/discover")
async def discover_models_compat():
    """V4 compat: redirect old discover endpoint to update."""
    return await update_models()


@app.get("/v1/calls")
async def calls(limit: int = 100, provider: Optional[str] = None, status: Optional[str] = None):
    return db.recent(limit=limit, provider=provider, status=status)


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(ROOT / "static" / "dashboard.html"))


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    return FileResponse(str(ROOT / "static" / "help.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
