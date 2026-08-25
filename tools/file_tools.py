"""Note storage: Markdown files with structured frontmatter.

Three changes from a naive implementation, each of which was a real problem:

* Filenames are slugified through a character allowlist. The topic string is
  model-generated, which makes it untrusted input, and `topic.replace(" ", "_")`
  happily lets `../` walk out of the notes directory.
* Notes are `.md`, because the content is Markdown.
* Every note carries frontmatter -- query, sub-questions, sources, confidence,
  model. Without it the corpus is a pile of prose you can't filter, and a note
  that failed review is indistinguishable from one that passed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from config import settings

logger = logging.getLogger("notes")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

MAX_SLUG_LEN = 80


def notes_dir() -> Path:
    settings.notes_dir.mkdir(parents=True, exist_ok=True)
    return settings.notes_dir


def slugify(topic: str) -> str:
    """Filesystem-safe slug from arbitrary model output.

    Allowlist, not denylist: everything outside [a-z0-9] collapses to a hyphen.
    That makes traversal (`../`), absolute paths, null bytes, and Windows
    reserved characters structurally impossible rather than individually
    filtered.
    """
    slug = _SLUG_STRIP.sub("-", (topic or "").lower()).strip("-")
    slug = slug[:MAX_SLUG_LEN].strip("-")
    return slug or "untitled"


def note_path(topic: str) -> Path:
    return notes_dir() / f"{slugify(topic)}.md"


# --------------------------------------------------------------------------
# frontmatter (deliberately minimal -- we own both ends of this format)
# --------------------------------------------------------------------------

def _dump_frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {json.dumps(str(item), ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {json.dumps(str(value), ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def _load_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, Any] = {}
    current_key: str | None = None
    for line in match.group(1).splitlines():
        if line.startswith("  - ") and current_key:
            meta.setdefault(current_key, [])
            if isinstance(meta[current_key], list):
                meta[current_key].append(_scalar(line[4:].strip()))
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            meta[key] = _scalar(value) if value else []
    return meta, match.group(2)


def _scalar(value: str) -> str:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, str) else value
    except json.JSONDecodeError:
        return value


# --------------------------------------------------------------------------
# read / write
# --------------------------------------------------------------------------

def write_note(topic: str, body: str, meta: dict[str, Any] | None = None) -> Path:
    """Write a note, overwriting any existing note on the same topic."""
    path = note_path(topic)
    full_meta = {
        "topic": topic,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
    }
    path.write_text(
        f"{_dump_frontmatter(full_meta)}\n\n{body.strip()}\n", encoding="utf-8"
    )
    logger.info("saved note -> %s", path.name)
    return path


def all_notes() -> list[dict[str, Any]]:
    """Every note with its metadata and body, newest first."""
    out = []
    for path in sorted(notes_dir().glob("*.md")):
        if path.name.startswith("."):
            continue
        try:
            meta, body = _load_frontmatter(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("could not read %s: %s", path.name, exc)
            continue
        out.append(
            {
                "slug": path.stem,
                "path": str(path),
                "topic": meta.get("topic", path.stem.replace("-", " ")),
                "meta": meta,
                "body": body,
            }
        )
    out.sort(key=lambda n: str(n["meta"].get("created", "")), reverse=True)
    return out


def find_note(topic: str) -> dict[str, Any] | None:
    """Exact slug match, then a forgiving containment match.

    Exact-filename lookup alone is why a knowledge base like this normally
    feels broken: it requires the caller to reproduce, character for character,
    a topic string that a model invented at save time.
    """
    target = slugify(topic)
    notes = all_notes()

    for note in notes:
        if note["slug"] == target:
            return note

    target_words = {w for w in target.split("-") if len(w) > 2}
    if not target_words:
        return None

    best, best_score = None, 0.0
    for note in notes:
        words = {w for w in note["slug"].split("-") if len(w) > 2}
        if not words:
            continue
        score = len(target_words & words) / len(target_words | words)
        if score > best_score:
            best, best_score = note, score
    return best if best_score >= 0.34 else None


# --------------------------------------------------------------------------
# tools exposed to the QA agent
# --------------------------------------------------------------------------

@tool
def load_note(topic: str) -> str:
    """Load a saved research note by topic. Tolerates approximate topic names."""
    note = find_note(topic)
    if note is None:
        known = ", ".join(n["topic"] for n in all_notes()) or "none"
        return f"No note found for '{topic}'. Saved topics: {known}"
    return note["body"]


@tool
def list_notes() -> str:
    """List the topics of all saved research notes."""
    notes = all_notes()
    if not notes:
        return "No notes saved yet."
    return "\n".join(
        f"- {n['topic']} (saved {str(n['meta'].get('created', 'unknown'))[:10]})"
        for n in notes
    )


@tool
def search_notes(query: str) -> str:
    """Search saved research notes for passages relevant to a question."""
    from tools.notes_index import search_notes_index

    hits = search_notes_index(query, k=4)
    if not hits:
        return "No relevant passages found in the saved notes."
    return "\n\n---\n\n".join(
        f"From note '{h['topic']}':\n{h['text']}" for h in hits
    )
