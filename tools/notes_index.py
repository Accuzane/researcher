"""Semantic retrieval over saved notes, with a keyword fallback.

Deliberately dependency-light: embeddings come from Ollama (`nomic-embed-text`
runs locally and costs nothing), vectors are cached in a JSON sidecar, and
similarity is a numpy dot product. No vector database to run, no extra service
to keep alive. For a personal research corpus of a few hundred notes this is
comfortably fast enough, and it degrades to keyword scoring if the embedding
model isn't pulled -- so the QA path never hard-fails on a missing dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any

from config import settings
from tools.file_tools import all_notes

logger = logging.getLogger("notes_index")

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

_embedder = None
_embedder_failed = False


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries where possible, hard-split where not."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i:i + size])
            continue
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


def _chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

def _get_embedder():
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    try:
        from langchain_ollama import OllamaEmbeddings

        _embedder = OllamaEmbeddings(model=settings.embed_model)
        return _embedder
    except Exception as exc:  # noqa: BLE001
        logger.info("embeddings unavailable (%s); using keyword search", exc)
        _embedder_failed = True
        return None


def set_embedder(embedder) -> None:
    """Inject an embedder. Used by tests; pass None to force keyword search."""
    global _embedder, _embedder_failed
    _embedder = embedder
    _embedder_failed = embedder is None


def _embed(texts: list[str]) -> list[list[float]] | None:
    embedder = _get_embedder()
    if embedder is None or not texts:
        return None
    try:
        return embedder.embed_documents(texts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding failed, falling back to keywords: %s", exc)
        return None


def _embed_query(text: str) -> list[float] | None:
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        return embedder.embed_query(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("query embedding failed: %s", exc)
        return None


# --------------------------------------------------------------------------
# index persistence
# --------------------------------------------------------------------------

def _load_cache() -> dict[str, list[float]]:
    path = settings.index_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    try:
        settings.index_path.parent.mkdir(parents=True, exist_ok=True)
        settings.index_path.write_text(json.dumps(cache), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not persist index: %s", exc)


def build_index() -> list[dict[str, Any]]:
    """Chunk every note and attach a cached or freshly computed vector."""
    cache = _load_cache()
    entries: list[dict[str, Any]] = []
    pending: list[tuple[int, str]] = []

    for note in all_notes():
        for chunk in chunk_text(note["body"]):
            cid = _chunk_id(chunk)
            entry = {
                "id": cid,
                "topic": note["topic"],
                "slug": note["slug"],
                "text": chunk,
                "vector": cache.get(cid),
            }
            if entry["vector"] is None:
                pending.append((len(entries), chunk))
            entries.append(entry)

    if pending:
        vectors = _embed([text for _, text in pending])
        if vectors:
            for (index, _), vector in zip(pending, vectors, strict=False):
                entries[index]["vector"] = vector
                cache[entries[index]["id"]] = vector
            _save_cache(cache)

    return entries


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


_WORD_RE = re.compile(r"[a-z0-9]+")


def _keyword_score(query: str, text: str) -> float:
    q = set(_WORD_RE.findall(query.lower()))
    t = set(_WORD_RE.findall(text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def search_notes_index(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Top-k passages across every saved note.

    Uses embeddings when available and keyword overlap otherwise, so callers
    get results either way rather than an error about a missing model.
    """
    entries = build_index()
    if not entries:
        return []

    query_vector = _embed_query(query)

    if query_vector and all(e["vector"] for e in entries):
        scored = [(_cosine(query_vector, e["vector"]), e) for e in entries]
    else:
        scored = [(_keyword_score(query, e["text"]), e) for e in entries]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {**entry, "score": round(score, 4)}
        for score, entry in scored[:k]
        if score > 0
    ]
