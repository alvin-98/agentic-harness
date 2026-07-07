"""Configuration loader for LLM Gateway V4.

Reads models.yaml as single source of truth for:
- Provider model lists and failover order
- Per-model rate limits
- Live rate-limit state (written back from provider headers)
- Failover strategy (breadth/depth)
"""
from __future__ import annotations
import os, time, threading, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

log = logging.getLogger(__name__)


CONFIG_PATH = Path(__file__).parent.parent.parent / "models.yaml"


@dataclass
class ModelSlot:
    """A single model within a provider's failover ring."""
    id: str
    rpm_limit: int = 9999
    rpd_limit: int = 9999999
    tpm_limit: int = 99999999
    cooldown: float = 0
    max_ctx: int = 32000
    tokens_per_day: Optional[int] = None
    structured: Optional[bool] = None  # None = inherit from provider
    type: str = "placeholder"  # text, image, sound, embedding, etc.
    visible: bool = False  # only visible models are usable through the gateway
    # Live state (written back to YAML from provider headers)
    rpm_remaining: Optional[int] = None
    rpd_remaining: Optional[int] = None
    exhausted: bool = False
    # Transient (not persisted)
    last_header_update: float = 0.0

    def limits_dict(self) -> dict:
        """Return a dict compatible with the old LIMITS format."""
        d = {
            "rpm": self.rpm_limit,
            "rpd": self.rpd_limit,
            "tpm": self.tpm_limit,
            "cooldown": self.cooldown,
            "max_ctx": self.max_ctx,
        }
        if self.tokens_per_day is not None:
            d["tokens_per_day"] = self.tokens_per_day
        return d


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    name: str
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    discover_free: bool = False
    models: list[ModelSlot] = field(default_factory=list)

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @property
    def resolved_base_url(self) -> str:
        if self.base_url_env:
            return os.getenv(self.base_url_env, self.base_url)
        return self.base_url

    @property
    def active_model(self) -> Optional[ModelSlot]:
        """First visible, non-exhausted model in the ring."""
        for slot in self.models:
            if slot.visible and not slot.exhausted:
                return slot
        return self.models[0] if self.models else None

    def active_index(self) -> int:
        for i, slot in enumerate(self.models):
            if slot.visible and not slot.exhausted:
                return i
        return 0


@dataclass
class RouterEntry:
    """A router pool entry."""
    provider: str
    model: str


@dataclass
class GatewayConfig:
    """Top-level gateway configuration."""
    failover_strategy: str = "breadth"  # "breadth" or "depth"
    gateway_port: int = 8101
    llm_order: list[str] = field(default_factory=list)
    router_order: list[str] = field(default_factory=list)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routers: dict[str, RouterEntry] = field(default_factory=dict)
    tier_order: dict[str, list[str]] = field(default_factory=dict)
    _path: Optional[Path] = field(default=None, repr=False)
    _mtime: float = field(default=0.0, repr=False)


def _parse_model_slot(d: dict) -> ModelSlot:
    return ModelSlot(
        id=d["id"],
        rpm_limit=d.get("rpm_limit", 9999),
        rpd_limit=d.get("rpd_limit", 9999999),
        tpm_limit=d.get("tpm_limit", 99999999),
        cooldown=d.get("cooldown", 0),
        max_ctx=d.get("max_ctx", 32000),
        tokens_per_day=d.get("tokens_per_day"),
        structured=d.get("structured"),
        type=d.get("type", "placeholder"),
        visible=d.get("visible", False),
        rpm_remaining=d.get("rpm_remaining"),
        rpd_remaining=d.get("rpd_remaining"),
        exhausted=d.get("exhausted", False),
    )


def _parse_provider(name: str, d: dict) -> ProviderConfig:
    models = [_parse_model_slot(m) for m in d.get("models", [])]
    return ProviderConfig(
        name=name,
        api_key_env=d.get("api_key_env", ""),
        base_url=d.get("base_url", ""),
        base_url_env=d.get("base_url_env", ""),
        discover_free=d.get("discover_free", False),
        models=models,
    )


def load_config(path: Path = CONFIG_PATH) -> GatewayConfig:
    """Load models.yaml and return a typed GatewayConfig."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    providers = {}
    for name, pdata in raw.get("providers", {}).items():
        providers[name] = _parse_provider(name, pdata)

    routers = {}
    for name, rdata in raw.get("routers", {}).items():
        routers[name] = RouterEntry(provider=name, model=rdata.get("model", ""))

    cfg = GatewayConfig(
        failover_strategy=raw.get("failover_strategy", "breadth"),
        gateway_port=raw.get("gateway_port", 8101),
        llm_order=raw.get("llm_order", []),
        router_order=raw.get("router_order", []),
        providers=providers,
        routers=routers,
        tier_order=raw.get("tier_order", {}),
        _path=path,
        _mtime=path.stat().st_mtime if path.exists() else 0,
    )
    return cfg


def _slot_to_dict(slot: ModelSlot) -> dict:
    """Serialize a ModelSlot to a dict suitable for YAML persistence."""
    d: dict = {"id": slot.id}
    if slot.rpm_limit != 9999:
        d["rpm_limit"] = slot.rpm_limit
    if slot.rpd_limit != 9999999:
        d["rpd_limit"] = slot.rpd_limit
    if slot.tpm_limit != 99999999:
        d["tpm_limit"] = slot.tpm_limit
    if slot.cooldown:
        d["cooldown"] = slot.cooldown
    if slot.max_ctx != 32000:
        d["max_ctx"] = slot.max_ctx
    if slot.tokens_per_day is not None:
        d["tokens_per_day"] = slot.tokens_per_day
    if slot.structured is not None:
        d["structured"] = slot.structured
    d["type"] = slot.type
    d["visible"] = slot.visible
    if slot.rpm_remaining is not None:
        d["rpm_remaining"] = slot.rpm_remaining
    if slot.rpd_remaining is not None:
        d["rpd_remaining"] = slot.rpd_remaining
    if slot.exhausted:
        d["exhausted"] = slot.exhausted
    return d


def save_state(cfg: GatewayConfig, path: Optional[Path] = None):
    """Write back the full model list and live rate-limit state to models.yaml.

    Updates existing model entries and appends newly discovered ones.
    """
    path = path or cfg._path or CONFIG_PATH
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if "providers" not in raw:
        raw["providers"] = {}

    for pname, pcfg in cfg.providers.items():
        if pname not in raw["providers"]:
            raw["providers"][pname] = {}
        raw_models = raw["providers"][pname].get("models", [])

        # Build index of existing raw models by id for update
        raw_by_id = {m["id"]: m for m in raw_models if "id" in m}

        new_raw_models = []
        for slot in pcfg.models:
            if slot.id in raw_by_id:
                # Update only live runtime state on existing entries.
                # User-authored fields (type, visible, limits, max_ctx, ...)
                # are owned by the YAML file — never overwrite them from the
                # in-memory config, otherwise the background flusher would
                # silently revert manual edits to models.yaml.
                entry = raw_by_id[slot.id]
                entry["rpm_remaining"] = slot.rpm_remaining
                entry["rpd_remaining"] = slot.rpd_remaining
                entry["exhausted"] = slot.exhausted
                new_raw_models.append(entry)
            else:
                # Append newly discovered model
                new_raw_models.append(_slot_to_dict(slot))

        raw["providers"][pname]["models"] = new_raw_models

    # Update strategy if changed at runtime
    raw["failover_strategy"] = cfg.failover_strategy

    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    cfg._mtime = path.stat().st_mtime


def has_changed(cfg: GatewayConfig) -> bool:
    """Check if models.yaml has been modified since last load."""
    path = cfg._path or CONFIG_PATH
    if not path.exists():
        return False
    return path.stat().st_mtime > cfg._mtime


def reload_if_changed(cfg: GatewayConfig, force: bool = False) -> GatewayConfig:
    """Hot-reload: re-read models.yaml if it changed on disk.

    Preserves transient runtime state (like last_header_update) and any
    unflushed live rate-limit state (rpm_remaining, rpd_remaining,
    exhausted) for matching slots.

    Pass force=True to bypass the mtime check — used by the explicit
    /v1/config/reload endpoint so a user click always re-reads the file,
    even if the background flusher recently bumped _mtime.
    """
    if not force and not has_changed(cfg):
        return cfg
    new_cfg = load_config(cfg._path or CONFIG_PATH)
    # Preserve transient + unflushed live state from old config
    for pname, old_pcfg in cfg.providers.items():
        if pname not in new_cfg.providers:
            continue
        new_pcfg = new_cfg.providers[pname]
        # Index new slots by id for O(1) lookup (order may differ)
        new_by_id = {s.id: s for s in new_pcfg.models}
        for old_slot in old_pcfg.models:
            new_slot = new_by_id.get(old_slot.id)
            if new_slot is None:
                continue
            new_slot.last_header_update = old_slot.last_header_update
            # Only carry forward live state the new config didn't specify,
            # so an explicit edit in the YAML still wins.
            if new_slot.rpm_remaining is None:
                new_slot.rpm_remaining = old_slot.rpm_remaining
            if new_slot.rpd_remaining is None:
                new_slot.rpd_remaining = old_slot.rpd_remaining
            if not new_slot.exhausted:
                new_slot.exhausted = old_slot.exhausted
    return new_cfg


# ─── Background state flusher ────────────────────────────────────────────────

_flush_lock = threading.Lock()
_last_flush = 0.0
FLUSH_INTERVAL = 30.0  # seconds


def maybe_flush_state(cfg: GatewayConfig):
    """Flush state to disk if enough time has passed since last flush."""
    global _last_flush
    now = time.time()
    if now - _last_flush < FLUSH_INTERVAL:
        return
    with _flush_lock:
        if now - _last_flush < FLUSH_INTERVAL:
            return
        _last_flush = now
        try:
            save_state(cfg)
        except Exception:
            pass  # non-critical — state will be flushed on next cycle


# ─── Multi-provider model discovery ──────────────────────────────────────────

async def discover_openrouter_free_models(api_key: str) -> list[dict]:
    """Fetch free models from OpenRouter API.

    Returns list of dicts with keys: id, name, context_length, supports_tools,
    supports_structured, type.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("https://openrouter.ai/api/v1/models", headers=headers)
        if r.status_code != 200:
            log.warning("OpenRouter model discovery failed: HTTP %d", r.status_code)
            return []
        data = r.json().get("data", [])

    free = []
    for m in data:
        pricing = m.get("pricing") or {}
        prompt_cost = float(pricing.get("prompt", "1") or "1")
        completion_cost = float(pricing.get("completion", "1") or "1")
        if prompt_cost == 0 and completion_cost == 0:
            supported = set(m.get("supported_parameters") or [])
            # Try to extract type from architecture modality
            modality = (m.get("architecture") or {}).get("modality", "")
            model_type = "text" if "text" in modality.lower() else "placeholder"
            free.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": (m.get("top_provider") or {}).get("context_length")
                                  or m.get("context_length", 32000),
                "supports_tools": "tools" in supported,
                "supports_structured": "response_format" in supported
                                       or "structured_output" in supported,
                "type": model_type,
            })
    return free


async def discover_groq_models(api_key: str) -> list[dict]:
    """Fetch all models from Groq API.

    Returns list of dicts with keys: id, context_length, type.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("https://api.groq.com/openai/v1/models", headers=headers)
        if r.status_code != 200:
            log.warning("Groq model discovery failed: HTTP %d", r.status_code)
            return []
        data = r.json().get("data", [])

    models = []
    for m in data:
        model_id = m.get("id", "")
        if not model_id:
            continue
        ctx = m.get("context_window", 32000)
        model_type = "placeholder"
        if m.get("object") == "model":
            model_type = "text"
        models.append({
            "id": model_id,
            "context_length": ctx,
            "type": model_type,
        })
    return models


async def discover_nvidia_models(api_key: str) -> list[dict]:
    """Fetch all models from NVIDIA API.

    Returns list of dicts with keys: id, context_length, type.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("https://integrate.api.nvidia.com/v1/models", headers=headers)
        if r.status_code != 200:
            log.warning("NVIDIA model discovery failed: HTTP %d", r.status_code)
            return []
        data = r.json().get("data", [])

    models = []
    for m in data:
        model_id = m.get("id", "")
        if not model_id:
            continue
        ctx = m.get("max_model_len") or m.get("context_length", 32000)
        model_type = "placeholder"
        if m.get("object") == "model":
            model_type = "text"
        models.append({
            "id": model_id,
            "context_length": ctx,
            "type": model_type,
        })
    return models


async def discover_gemini_models(api_key: str) -> list[dict]:
    """Fetch available models from Gemini API (v1beta ListModels).

    Returns list of dicts with keys: id, context_length, type, supports_structured.
    Only includes models that support generateContent.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
    models = []
    async with httpx.AsyncClient(timeout=30) as c:
        while url:
            r = await c.get(url)
            if r.status_code != 200:
                log.warning("Gemini model discovery failed: HTTP %d", r.status_code)
                return []
            data = r.json()
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                model_id = (m.get("name") or "").replace("models/", "")
                if not model_id:
                    continue
                ctx = m.get("inputTokenLimit", 32000)
                models.append({
                    "id": model_id,
                    "context_length": ctx,
                    "supports_structured": True,
                    "type": "text",
                })
            next_token = data.get("nextPageToken")
            if next_token:
                base = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
                url = f"{base}&pageToken={next_token}"
            else:
                url = None
    return models


def merge_discovered_models(cfg: GatewayConfig, discovered: list[dict],
                            provider_name: str = "openrouter",
                            max_merge: int = 100) -> int:
    """Merge discovered models into a provider's config.

    Only adds models not already in the config. New models get visible=False.
    Returns count of models added.
    """
    pcfg = cfg.providers.get(provider_name)
    if not pcfg:
        return 0

    # Default limits by provider
    default_limits = {
        "openrouter": {"rpm_limit": 20, "rpd_limit": 50, "tpm_limit": 99999999, "cooldown": 3},
        "groq": {"rpm_limit": 30, "rpd_limit": 1000, "tpm_limit": 30000, "cooldown": 2},
        "nvidia": {"rpm_limit": 40, "rpd_limit": 9999, "tpm_limit": 100000, "cooldown": 2},
        "gemini": {"rpm_limit": 15, "rpd_limit": 1000, "tpm_limit": 250000, "cooldown": 4},
    }
    limits = default_limits.get(provider_name, {"rpm_limit": 20, "rpd_limit": 1000, "tpm_limit": 99999999, "cooldown": 2})

    existing_ids = {s.id for s in pcfg.models}
    added = 0
    for m in discovered:
        if m["id"] in existing_ids:
            continue
        if added >= max_merge:
            break
        slot = ModelSlot(
            id=m["id"],
            rpm_limit=limits["rpm_limit"],
            rpd_limit=limits["rpd_limit"],
            tpm_limit=limits["tpm_limit"],
            cooldown=limits["cooldown"],
            max_ctx=m.get("context_length", 32000),
            structured=m.get("supports_structured"),
            type=m.get("type", "placeholder"),
            visible=False,  # new discoveries default to hidden
        )
        pcfg.models.append(slot)
        existing_ids.add(m["id"])
        added += 1

    if added:
        log.info("Merged %d new models into %s config", added, provider_name)
    return added


def sync_provider_models(cfg: GatewayConfig, discovered: list[dict],
                         provider_name: str) -> tuple[int, int]:
    """Sync a provider's model list against the live discovery results.

    - Removes models no longer reported by the provider API.
    - Adds newly discovered models with visible=False.
    Returns (added_count, removed_count).
    """
    pcfg = cfg.providers.get(provider_name)
    if not pcfg:
        return 0, 0

    discovered_ids = {m["id"] for m in discovered}

    # Prune: remove models that are no longer available upstream.
    kept: list[ModelSlot] = []
    removed = 0
    for slot in pcfg.models:
        if slot.id in discovered_ids:
            kept.append(slot)
        else:
            log.info("pruning stale model %s/%s (was visible=%s)",
                     provider_name, slot.id, slot.visible)
            removed += 1

    pcfg.models = kept

    # Add new models
    added = merge_discovered_models(cfg, discovered, provider_name=provider_name,
                                    max_merge=500)

    if added or removed:
        log.info("sync_provider_models: %s — added %d, removed %d",
                 provider_name, added, removed)
    return added, removed


async def sync_models_on_startup(cfg: GatewayConfig) -> dict[str, dict]:
    """Discover models from all configured providers and sync models.yaml.

    Called once at gateway startup. For each provider with an API key:
    1. Fetches the current model catalogue from the provider.
    2. Removes stale models no longer upstream (unless user-pinned visible).
    3. Adds newly available models (visible=False by default).
    4. Persists the result to models.yaml.

    Returns a summary dict per provider: {"added": N, "removed": N, "discovered": N}.
    """
    results: dict[str, dict] = {}

    # OpenRouter
    or_key = os.getenv("OPEN_ROUTER_API_KEY", "")
    if or_key and "openrouter" in cfg.providers:
        try:
            discovered = await discover_openrouter_free_models(or_key)
            added, removed = sync_provider_models(cfg, discovered, "openrouter")
            results["openrouter"] = {"discovered": len(discovered), "added": added, "removed": removed}
        except Exception as e:
            log.warning("startup sync failed for openrouter: %s", e)

    # Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and "groq" in cfg.providers:
        try:
            discovered = await discover_groq_models(groq_key)
            added, removed = sync_provider_models(cfg, discovered, "groq")
            results["groq"] = {"discovered": len(discovered), "added": added, "removed": removed}
        except Exception as e:
            log.warning("startup sync failed for groq: %s", e)

    # NVIDIA
    nv_key = os.getenv("NVIDIA_API_KEY", "")
    if nv_key and "nvidia" in cfg.providers:
        try:
            discovered = await discover_nvidia_models(nv_key)
            added, removed = sync_provider_models(cfg, discovered, "nvidia")
            results["nvidia"] = {"discovered": len(discovered), "added": added, "removed": removed}
        except Exception as e:
            log.warning("startup sync failed for nvidia: %s", e)

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and "gemini" in cfg.providers:
        try:
            discovered = await discover_gemini_models(gemini_key)
            added, removed = sync_provider_models(cfg, discovered, "gemini")
            results["gemini"] = {"discovered": len(discovered), "added": added, "removed": removed}
        except Exception as e:
            log.warning("startup sync failed for gemini: %s", e)

    # Persist
    if any(r.get("added", 0) or r.get("removed", 0) for r in results.values()):
        save_state(cfg)
        log.info("startup model sync complete — persisted to models.yaml")
    else:
        log.info("startup model sync complete — no changes")

    return results
