"""Capability-aware router with multi-model failover.

V4: Each provider can have multiple models (a "model ring"). Two strategies:
- breadth-first: try model[0] across all providers, then model[1] across all, etc.
- depth-first: exhaust all models in provider[0], then all in provider[1], etc.

Rate state is tracked per (provider, model) pair. Provider headers take precedence
over local counters when available."""
from __future__ import annotations
import time, asyncio
from collections import deque, defaultdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import GatewayConfig, ModelSlot, ProviderConfig


LIMITS = {
    "ollama":     {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0,   "max_ctx": 32000},
    "sglang":     {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0,   "max_ctx": 32000},
    "cerebras":   {"rpm": 5,   "rpd": 2400,    "tpm": 30000,    "cooldown": 2,   "max_ctx": 8000,    "tokens_per_day": 1_000_000},
    "groq":       {"rpm": 30,   "rpd": 1000,    "tpm": 6000,     "cooldown": 2,   "max_ctx": 100000},
    "nvidia":     {"rpm": 40,   "rpd": 9999,    "tpm": 100000,   "cooldown": 2,   "max_ctx": 100000},
    "gemini":     {"rpm": 15,   "rpd": 1000,    "tpm": 250000,   "cooldown": 4,   "max_ctx": 1000000},
    "openrouter": {"rpm": 20,   "rpd": 50,      "tpm": 99999999, "cooldown": 3,   "max_ctx": 100000},
    "github":     {"rpm": 10,   "rpd": 50,      "tpm": 99999999, "cooldown": 6,   "max_ctx": 8000},
}

SHORTCUTS = {
    "g": "gemini", "gem": "gemini", "gemini": "gemini",
    "n": "nvidia", "nv": "nvidia", "nvidia": "nvidia",
    "o": "ollama", "oll": "ollama", "ollama": "ollama",
    "gr": "groq", "groq": "groq",
    "c": "cerebras", "cer": "cerebras", "cerebras": "cerebras",
    "or": "openrouter", "opr": "openrouter", "openrouter": "openrouter",
    "gh": "github", "ghb": "github", "github": "github",
    "sg": "sglang", "sgl": "sglang", "sglang": "sglang",
}


def resolve(name):
    if not name:
        return None
    return SHORTCUTS.get(name.lower())


class RateState:
    def __init__(self):
        self.calls_minute = deque()
        self.tokens_minute = deque()
        self.calls_today = 0
        self.tokens_today = 0
        self.day_start = self._day_start()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    @staticmethod
    def _day_start():
        now = time.time()
        return now - (now % 86400)

    def gc(self):
        now = time.time()
        if now - self.day_start >= 86400:
            self.calls_today = 0
            self.tokens_today = 0
            self.day_start = self._day_start()
        cutoff = now - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()
        while self.tokens_minute and self.tokens_minute[0][0] < cutoff:
            self.tokens_minute.popleft()

    def can_use(self, limits, est_tokens=0):
        self.gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        wait = limits["cooldown"] - (now - self.last_call)
        if wait > 0:
            return False, f"cooldown ({wait:.1f}s)"
        if len(self.calls_minute) >= limits["rpm"]:
            return False, "RPM limit"
        if self.calls_today >= limits["rpd"]:
            return False, "RPD limit"
        tpm = sum(t for _, t in self.tokens_minute)
        if tpm + est_tokens > limits["tpm"]:
            return False, "TPM limit"
        if "tokens_per_day" in limits and self.tokens_today + est_tokens > limits["tokens_per_day"]:
            return False, "daily token cap"
        return True, None

    def record(self, tokens):
        now = time.time()
        self.calls_minute.append(now)
        self.tokens_minute.append((now, tokens))
        self.calls_today += 1
        self.tokens_today += tokens
        self.last_call = now

    def mark_unavailable(self, seconds: float, reason: str):
        self.unavailable_until = time.time() + seconds
        self.unavailable_reason = reason

    def snapshot(self, limits):
        self.gc()
        now = time.time()
        tpm = sum(t for _, t in self.tokens_minute)
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": limits["rpm"],
            "rpd_used": self.calls_today,
            "rpd_limit": limits["rpd"],
            "tpm_used": tpm,
            "tpm_limit": limits["tpm"],
            "tokens_today": self.tokens_today,
            "tokens_per_day": limits.get("tokens_per_day"),
            "cooldown": limits["cooldown"],
            "cooldown_remaining": max(0, limits["cooldown"] - (now - self.last_call)) if self.last_call else 0,
            "last_call": self.last_call,
            "backoff_remaining": max(0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
        }


class Router:
    """V4 multi-model router. Each provider can have multiple models.

    The `model_state` dict is keyed by (provider_name, model_id) and tracks
    per-model rate state. The legacy `state` dict (keyed by provider name) is
    kept for backward compat with RouterPool and dashboard.
    """
    def __init__(self, providers: dict, order: list[str], config: Optional[GatewayConfig] = None):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self.state = defaultdict(RateState)  # legacy per-provider state
        self.model_state: dict[tuple[str, str], RateState] = {}  # (provider, model) → RateState
        self.config = config
        self.lock = asyncio.Lock()
        self._last_picked_model: Optional[str] = None  # set by pick() when config is active

    def _get_model_state(self, provider: str, model: str) -> RateState:
        key = (provider, model)
        if key not in self.model_state:
            self.model_state[key] = RateState()
        return self.model_state[key]

    def _strategy(self) -> str:
        return self.config.failover_strategy if self.config else "legacy"

    def _record(self, name, model, slot_idx, outcome, reason, rate):
        """Build one enriched attempt record. Keeps `provider`/`reason` keys so
        the legacy `_attempts_str` helper keeps working, while adding the fields
        the observability UI needs to reconstruct the full failover flow."""
        return {
            "provider": name,
            "model": model,
            "slot_index": slot_idx,
            "strategy": self._strategy(),
            "outcome": outcome,  # skipped | selected (mutated later to success/error)
            "reason": reason,
            "rate": rate,
            "ts": time.time(),
        }

    def candidates(self, override=None):
        if override:
            r = resolve(override)
            return [r] if r and r in self.providers else []
        return list(self.order)

    def pick(self, est_tokens, candidates, required_caps: list[str] | None = None):
        """Legacy single-model pick. Used when config is not loaded or for
        backward compat. Falls through to pick_model when config is available."""
        if self.config:
            result = self.pick_model(est_tokens, candidates, required_caps=required_caps)
            self._last_picked_model = result[1]
            if result[0] is not None:
                return result[0], result[2]
            return None, result[2]
        self._last_picked_model = None

        attempts = []
        for name in candidates:
            limits = LIMITS[name]
            prov = self.providers[name]
            caps = getattr(prov, "capabilities", {})
            if required_caps:
                missing = [c for c in required_caps if not caps.get(c)]
                if missing:
                    attempts.append({"provider": name, "reason": f"skipped:no_{missing[0]}"})
                    continue
            if est_tokens > limits["max_ctx"]:
                attempts.append({"provider": name, "reason": f"prompt {est_tokens} > max_ctx {limits['max_ctx']}"})
                continue
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def pick_model(self, est_tokens, candidates, required_caps: list[str] | None = None,
                   ) -> tuple[Optional[str], Optional[str], list[dict]]:
        """Pick provider AND model. Returns (provider, model_id, attempts).

        Uses the failover strategy from config:
        - breadth: try slot[i] across all providers, then slot[i+1], etc.
        - depth: try all slots in provider[0], then all in provider[1], etc.
        """
        if not self.config:
            name, attempts = self.pick(est_tokens, candidates, required_caps)
            model = self.providers[name].model if name else None
            return name, model, attempts

        strategy = self.config.failover_strategy
        attempts = []

        if strategy == "depth":
            return self._pick_depth(est_tokens, candidates, required_caps, attempts)
        else:
            return self._pick_breadth(est_tokens, candidates, required_caps, attempts)

    def _pick_breadth(self, est_tokens, candidates, required_caps, attempts
                      ) -> tuple[Optional[str], Optional[str], list[dict]]:
        """Breadth-first: try model[i] across all providers, then model[i+1]."""
        if not self.config:
            return None, None, attempts

        # Find max number of model slots across all candidate providers
        max_slots = 0
        for name in candidates:
            pcfg = self.config.providers.get(name)
            if pcfg:
                max_slots = max(max_slots, len(pcfg.models))
        if max_slots == 0:
            max_slots = 1  # fallback for providers without config

        for slot_idx in range(max_slots):
            for name in candidates:
                result = self._try_provider_slot(name, slot_idx, est_tokens, required_caps, attempts)
                if result is not None:
                    return result
        return None, None, attempts

    def _pick_depth(self, est_tokens, candidates, required_caps, attempts
                    ) -> tuple[Optional[str], Optional[str], list[dict]]:
        """Depth-first: exhaust all models in provider[0], then provider[1]."""
        if not self.config:
            return None, None, attempts

        for name in candidates:
            pcfg = self.config.providers.get(name)
            if not pcfg:
                # Provider not in config — try with default model
                result = self._try_provider_slot(name, 0, est_tokens, required_caps, attempts)
                if result is not None:
                    return result
                continue
            for slot_idx in range(len(pcfg.models)):
                result = self._try_provider_slot(name, slot_idx, est_tokens, required_caps, attempts)
                if result is not None:
                    return result
        return None, None, attempts

    def _try_provider_slot(self, name: str, slot_idx: int, est_tokens: int,
                           required_caps: list[str] | None, attempts: list[dict]
                           ) -> Optional[tuple[str, str, list[dict]]]:
        """Try a specific (provider, slot_index). Returns (name, model, attempts) or None."""
        prov = self.providers.get(name)
        if not prov:
            return None

        pcfg = self.config.providers.get(name) if self.config else None
        if not pcfg or slot_idx >= len(pcfg.models):
            # No config or slot out of range — only try slot 0 with legacy limits
            if slot_idx > 0:
                return None
            limits = LIMITS.get(name, {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0, "max_ctx": 32000})
            model_id = prov.model
            state = self.state[name]
            caps = getattr(prov, "capabilities", {})
            if required_caps:
                missing = [c for c in required_caps if not caps.get(c)]
                if missing:
                    attempts.append(self._record(name, model_id, slot_idx, "skipped", f"skipped:no_{missing[0]}", state.snapshot(limits)))
                    return None
            if est_tokens > limits["max_ctx"]:
                attempts.append(self._record(name, model_id, slot_idx, "skipped", f"prompt {est_tokens} > max_ctx {limits['max_ctx']}", state.snapshot(limits)))
                return None
            ok, why = state.can_use(limits, est_tokens)
            if ok:
                attempts.append(self._record(name, model_id, slot_idx, "selected", "picked", state.snapshot(limits)))
                return name, model_id, attempts
            attempts.append(self._record(name, model_id, slot_idx, "skipped", why, state.snapshot(limits)))
            return None

        slot = pcfg.models[slot_idx]
        limits = slot.limits_dict()
        ms = self._get_model_state(name, slot.id)

        # Skip non-visible models
        if not slot.visible:
            attempts.append(self._record(name, slot.id, slot_idx, "skipped", "not visible", ms.snapshot(limits)))
            return None

        # Skip exhausted models
        if slot.exhausted:
            attempts.append(self._record(name, slot.id, slot_idx, "skipped", "exhausted", ms.snapshot(limits)))
            return None

        # Check capabilities
        caps = getattr(prov, "capabilities", {})
        if required_caps:
            for cap in required_caps:
                if cap == "structured" and slot.structured is not None:
                    if not slot.structured:
                        attempts.append(self._record(name, slot.id, slot_idx, "skipped", "skipped:no_structured (model)", ms.snapshot(limits)))
                        return None
                elif not caps.get(cap):
                    attempts.append(self._record(name, slot.id, slot_idx, "skipped", f"skipped:no_{cap}", ms.snapshot(limits)))
                    return None

        # Check context length
        if est_tokens > limits["max_ctx"]:
            attempts.append(self._record(name, slot.id, slot_idx, "skipped", f"prompt {est_tokens} > max_ctx {limits['max_ctx']}", ms.snapshot(limits)))
            return None

        # Check provider-reported remaining (headers take precedence)
        if slot.rpm_remaining is not None and slot.rpm_remaining <= 0:
            attempts.append(self._record(name, slot.id, slot_idx, "skipped", "RPM exhausted (header)", ms.snapshot(limits)))
            return None
        if slot.rpd_remaining is not None and slot.rpd_remaining <= 0:
            attempts.append(self._record(name, slot.id, slot_idx, "skipped", "RPD exhausted (header)", ms.snapshot(limits)))
            return None

        # Check local rate state
        ok, why = ms.can_use(limits, est_tokens)
        if ok:
            attempts.append(self._record(name, slot.id, slot_idx, "selected", "picked", ms.snapshot(limits)))
            return name, slot.id, attempts
        attempts.append(self._record(name, slot.id, slot_idx, "skipped", why, ms.snapshot(limits)))
        return None

    def record_call(self, provider: str, model: str, tokens: int = 0):
        """Record a call for both legacy and per-model state."""
        self.state[provider].record(tokens)
        self._get_model_state(provider, model).record(tokens)

    def mark_model_unavailable(self, provider: str, model: str, seconds: float, reason: str):
        """Mark a specific model as unavailable (backoff)."""
        self._get_model_state(provider, model).mark_unavailable(seconds, reason)
        # Also mark on legacy state if it's the only/active model
        if self.config:
            pcfg = self.config.providers.get(provider)
            if pcfg and pcfg.active_model and pcfg.active_model.id == model:
                self.state[provider].mark_unavailable(seconds, reason)
        else:
            self.state[provider].mark_unavailable(seconds, reason)

    def mark_model_exhausted(self, provider: str, model: str):
        """Mark a model slot as exhausted in config (persisted to YAML)."""
        if not self.config:
            return
        pcfg = self.config.providers.get(provider)
        if not pcfg:
            return
        for slot in pcfg.models:
            if slot.id == model:
                slot.exhausted = True
                break

    def update_live_limits(self, provider: str, model: str,
                           rpm_remaining: Optional[int] = None,
                           rpd_remaining: Optional[int] = None,
                           rpm_limit: Optional[int] = None,
                           rpd_limit: Optional[int] = None):
        """Update a model slot's live limits from provider response headers."""
        if not self.config:
            return
        pcfg = self.config.providers.get(provider)
        if not pcfg:
            return
        for slot in pcfg.models:
            if slot.id == model:
                if rpm_remaining is not None:
                    slot.rpm_remaining = rpm_remaining
                if rpd_remaining is not None:
                    slot.rpd_remaining = rpd_remaining
                if rpm_limit is not None:
                    slot.rpm_limit = rpm_limit
                if rpd_limit is not None:
                    slot.rpd_limit = rpd_limit
                slot.last_header_update = time.time()
                # Auto-mark exhausted if headers say 0 remaining
                if (rpm_remaining is not None and rpm_remaining <= 0) or \
                   (rpd_remaining is not None and rpd_remaining <= 0):
                    slot.exhausted = True
                break

    def get_active_model(self, provider: str) -> Optional[str]:
        """Get the currently active (first non-exhausted) model for a provider."""
        if self.config:
            pcfg = self.config.providers.get(provider)
            if pcfg:
                am = pcfg.active_model
                return am.id if am else None
        return self.providers[provider].model if provider in self.providers else None

    def all_status(self):
        """Return status for all providers, including model ring info."""
        out = {}
        for name in self.providers:
            limits = LIMITS.get(name, {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0, "max_ctx": 32000})
            out[name] = self.state[name].snapshot(limits)
            out[name]["model"] = self.providers[name].model
            out[name]["capabilities"] = getattr(self.providers[name], "capabilities", {})
            # Add model ring info if config is available
            if self.config:
                pcfg = self.config.providers.get(name)
                if pcfg:
                    ring = []
                    for slot in pcfg.models:
                        if not slot.visible:
                            continue
                        ms = self._get_model_state(name, slot.id)
                        slot_snap = ms.snapshot(slot.limits_dict())
                        slot_snap["model"] = slot.id
                        slot_snap["exhausted"] = slot.exhausted
                        slot_snap["rpm_remaining_header"] = slot.rpm_remaining
                        slot_snap["rpd_remaining_header"] = slot.rpd_remaining
                        ring.append(slot_snap)
                    out[name]["model_ring"] = ring
                    out[name]["active_model"] = pcfg.active_model.id if pcfg.active_model else None
                    out[name]["active_index"] = pcfg.active_index()
        return out


# -----------------------------------------------------------------------------
# V3 Router pool — separate failover ring for routing-decision LLM calls.
# Same rate-state machinery, separate state dict so router quotas never compete
# with worker quotas (provider keys are shared but providers meter per-model).
# -----------------------------------------------------------------------------

DEFAULT_ROUTER_ORDER = ["cerebras", "groq", "github", "sglang"]  # nvidia excluded: no structured output support


class RouterPool:
    """Failover ring for router-LLM calls. Mirrors `Router` but for the
    Perception/Memory/Decision routing classifiers. Each call is logged with
    a call_role marker (router_perception | router_memory | router_decision)
    so the dashboard can show router activity separately from worker activity.
    """
    def __init__(self, providers: dict, order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self.state = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def candidates(self):
        return list(self.order)

    def pick(self, est_tokens=400):
        """Pick first available router provider. Caps require nothing — router
        LLMs only need to emit one word, no tools/reasoning/structured needed."""
        attempts = []
        for name in self.candidates():
            limits = LIMITS[name]
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def all_status(self):
        out = {}
        for name in self.providers:
            out[name] = self.state[name].snapshot(LIMITS[name])
            out[name]["model"] = self.providers[name].model
        return out
