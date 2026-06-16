"""Python client for LLM Gateway V3. Adds auto_route kwarg on top of V2."""
import os, json, httpx, time, structlog
from typing import Any, Optional

logger = structlog.get_logger(__name__)

DEFAULT_URL = os.getenv("LLM_GATEWAY_V3_URL", "http://localhost:8101")


class LLM:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, prompt: str = None, *,
             messages: Optional[list] = None,
             system: Any = None,
             provider: str = None, model: str = None,
             max_tokens: int = 2048, temperature: float = 0.7,
             tools: Optional[list] = None,
             tool_choice: Any = None,
             cache_system: Optional[bool] = None,
             reasoning: Optional[str] = None,
             response_format: Any = None,
             auto_route: Optional[str] = None) -> dict:
        body = {
            "prompt": prompt, "messages": messages, "system": system,
            "provider": provider, "model": model,
            "max_tokens": max_tokens, "temperature": temperature, "stream": False,
            "tools": tools, "tool_choice": tool_choice,
            "cache_system": cache_system, "reasoning": reasoning,
            "response_format": response_format,
            "auto_route": auto_route,
        }
        body = {k: v for k, v in body.items() if v is not None}
        url = f"{self.base_url}/v1/chat"
        last_error_body = None
        for attempt in range(3):
            r = httpx.post(url, json=body, timeout=self.timeout)
            if r.status_code in (502, 503, 429):
                try:
                    last_error_body = r.json()
                except Exception:
                    last_error_body = {"raw": r.text[:500]}
                logger.warning("llm_chat_retry",
                               attempt=attempt + 1,
                               status=r.status_code,
                               provider=provider,
                               model=model,
                               error=last_error_body)
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            r.raise_for_status()
            resp = r.json()
            logger.debug("llm_chat_complete",
                         provider=resp.get("provider"),
                         model=resp.get("model"),
                         latency_ms=resp.get("latency_ms"),
                         input_tokens=resp.get("input_tokens"),
                         output_tokens=resp.get("output_tokens"))
            return resp
        logger.error("llm_chat_failed",
                     provider=provider,
                     model=model,
                     status=r.status_code,
                     error=last_error_body)
        r.raise_for_status()

    def stream(self, prompt: str = None, *, messages=None, system=None,
               provider: str = None, model: str = None,
               max_tokens: int = 2048, temperature: float = 0.7,
               tools=None, tool_choice=None,
               cache_system=None, reasoning=None, response_format=None):
        body = {
            "prompt": prompt, "messages": messages, "system": system,
            "provider": provider, "model": model,
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "tools": tools, "tool_choice": tool_choice,
            "cache_system": cache_system, "reasoning": reasoning,
            "response_format": response_format,
        }
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream("POST", f"{self.base_url}/v1/chat", json=body, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if "delta" in d:
                    yield d["delta"]
                if d.get("done") or d.get("error"):
                    return

    def capabilities(self):
        return httpx.get(f"{self.base_url}/v1/capabilities", timeout=30).json()

    def embed(self, text: str,
              task_type: str = "retrieval_document",
              provider: Optional[str] = None) -> dict:
        """Returns {provider, model, embedding, dim, latency_ms, attempted}."""
        body = {"text": text, "task_type": task_type}
        if provider:
            body["provider"] = provider
        r = httpx.post(f"{self.base_url}/v1/embed", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def ask(prompt: str, provider: str = None, **kw) -> str:
    return LLM().chat(prompt, provider=provider, **kw)["text"]


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    print(ask("Say hello in one short line.", provider=p))
