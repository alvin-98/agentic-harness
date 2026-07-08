"""
MCP server for the agent.

Eleven tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file,
    index_document, search_knowledge

web_search:      Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results.
fetch_url:       crawl4ai for HTML pages; httpx + pypdf for .pdf URLs
                 (redirect-following download with PDF text extraction).
index_document:  Chunks a sandbox file or artifact and writes the chunks as
                 fact records into Memory, where they become FAISS-searchable.
search_knowledge: Vector search over indexed facts.

Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Allow importing agents package (Memory uses relative imports)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This process speaks MCP over stdio — stdout is the JSON-RPC transport and
# must carry ONLY protocol frames.  The LLM gateway client (imported transitively
# via agents.memory) logs with structlog, whose default PrintLogger writes to
# stdout.  Re-route structlog to stderr before any import that could trigger a
# log call, so debug lines like "llm_chat_complete" don't corrupt the stream.
import structlog
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from agents.memory import Memory
from agents.artifacts import ArtifactStore
from agents.schemas import Kind

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

load_dotenv(Path(__file__).parent.parent.parent / ".env")

mcp = FastMCP("eagv3-s6-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    hits: list[dict] = []
    with DDGS() as ddgs:
        for backend in ("auto", "html", "lite"):
            try:
                hits = list(ddgs.text(query, max_results=max_results, backend=backend))
            except Exception:
                hits = []
            if hits:
                break
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]


async def _download_file(url: str, timeout: int = 30) -> dict:
    """Download a binary file (e.g. PDF) via httpx and extract text.

    Uses httpx with follow_redirects=True to handle 301/302 responses that
    crawl4ai's headless Chromium pipeline cannot process for non-HTML content.
    For PDFs, text is extracted with pypdf."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    raw_bytes = r.content

    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)
        return {
            "status": r.status_code,
            "content_type": "application/pdf",
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
            "url": str(r.url),
        }
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
        return {
            "status": r.status_code,
            "content_type": content_type,
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
            "url": str(r.url),
        }


async def _crawl4ai_fetch(url: str) -> dict:
    from crawl4ai import (
        AsyncWebCrawler,
        CrawlerRunConfig,
        DefaultMarkdownGenerator,
        PruningContentFilter,
    )

    # Prune boilerplate (nav bars, sidebars, footers, ad blocks) at the source so
    # the artifact holds article content instead of page chrome. PruningContentFilter
    # scores DOM blocks by text/link density and drops low-signal ones; the cleaned
    # result surfaces as markdown.fit_markdown.
    run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
        ),
    )

    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url, config=run_config)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
    # serializes as {} because its real field is private. Pull the raw string
    # out and force a plain str so FastMCP serializes correctly.
    # Prefer fit_markdown (boilerplate-pruned) over raw_markdown (full page).
    md = r.markdown
    raw = (
        getattr(md, "fit_markdown", None)
        or getattr(md, "raw_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (Tavily primary, DDG fallback). Hard-capped at 5 results. Example: web_search("python asyncio tutorial", 3)."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")
    results = _ddg_search(query, max_results)
    if results:
        _bump("duckduckgo")
    else:
        _bump("duckduckgo", "errors")
    return results


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> dict:
    """Fetch the content of a URL as text. For HTML pages, returns boilerplate-pruned markdown via crawl4ai (headless Chromium) — nav bars, sidebars, and ads are stripped, only main content is returned. For PDFs, downloads via httpx (following redirects) and extracts text with pypdf. PDFs are detected by a .pdf suffix, a /pdf/ path segment (e.g. arxiv.org/pdf/2602.06791), or as a fallback when crawl4ai returns near-empty content for a non-HTML resource. Example: fetch_url("https://example.com"). Example: fetch_url("https://arxiv.org/pdf/2602.06791")."""
    lower = url.lower()
    if lower.endswith(".pdf") or "/pdf/" in lower:
        return await _download_file(url, timeout=timeout)
    return await _crawl4ai_fetch(url)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }

# ── Memory instance (singleton for this subprocess) ─────────────────────────
_memory = Memory()


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into word-level chunks with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Singleton artifact store for resolving art: handles in index_document.
_artifact_store = ArtifactStore()


def _read_for_index(path: str) -> tuple[str, str]:
    """Return (content, source_label) for an indexable file or artifact.

    Accepts both sandbox file paths and `art:` artifact handles. When given
    an artifact handle, the bytes are loaded from the artifact store and
    decoded as UTF-8."""
    if path.startswith("art:"):
        return _artifact_store.get_bytes(path).decode("utf-8", errors="replace"), path
    p = _safe(path)
    return p.read_text(encoding="utf-8"), f"sandbox:{path}"


@mcp.tool()
def index_document(path: str, chunk_size: int = 400, overlap: int = 80) -> dict:
    """Chunk a sandbox file or artifact and write the chunks into Memory as
    DOCUMENT records with LLM-generated semantic descriptors, where they
    become FAISS-searchable for later queries via search_knowledge.
    Use this when the content must be searchable across later turns or runs.
    For one-shot inspection of a file's contents, use read_file.
    Accepts sandbox file paths or `art:` artifact handles.
    Re-indexing a previously indexed source replaces its chunks (deduped)
    rather than appending duplicates."""
    text, source = _read_for_index(path)
    if not text.strip():
        return {"ok": True, "path": path, "source": source, "chunks": 0}

    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"ok": True, "path": path, "source": source, "chunks": 0}

    # Dedupe: clear any prior DOCUMENT chunks for this source before re-indexing.
    removed = _memory.delete_by_source(source)

    path_stem = path if path.startswith("art:") else Path(path).stem
    ids = []
    for i, chunk in enumerate(chunks):
        item = _memory.add_document_chunk(
            chunk=chunk,
            source=source,
            path_stem=path_stem,
            chunk_index=i,
            total_chunks=len(chunks),
        )
        ids.append(item.id)

    return {"ok": True, "path": path, "source": source, "chunks": len(chunks), "memory_ids": ids, "replaced": removed}


@mcp.tool()
def search_knowledge(query: str, k: int = 5) -> list[dict]:
    """Vector search over previously indexed document chunks. Use this rather
    than re-fetching or re-reading source files when Memory already
    contains indexed chunks for the topic. Queries the DOCUMENT kind
    specifically, so it does not surface the agent's general facts,
    preferences, or tool outcomes."""
    results = _memory.read(query, top_k=k, kinds=[Kind.DOCUMENT])
    return [
        {
            "id": item.id,
            "descriptor": item.descriptor,
            "value": item.value,
            "score": item.confidence,
            "source": item.source,
        }
        for item in results
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
