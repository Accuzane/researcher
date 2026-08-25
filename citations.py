"""Citation handling -- entirely deterministic, no model in the loop.

This is the module that makes the difference between "the model wrote some
URLs" and "these claims are tied to documents we actually retrieved". Every
function here is pure and unit-testable, which is the point: verifying a
citation is a mechanical check, and asking an LLM to do it is how you get
confident, well-formatted, fabricated sources.

Citation lifecycle:
  1. A sub-researcher sees its own sources numbered 1..N and cites `[1]`.
  2. `remap_local_citations` rewrites those to global `[S1a2b3c4d]` ids.
     Out-of-range numbers are dropped here and reported.
  3. The reporter copies `[S…]` markers through into the final report.
  4. `validate` flags any marker not backed by the registry.
  5. `renumber` turns surviving markers into `[1]`, `[2]` and emits a
     bibliography containing only sources that were actually cited.
"""

from __future__ import annotations

import re

from state import Source

GLOBAL_CITE_RE = re.compile(r"\[(S[0-9a-f]{8})\]")
LOCAL_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def find_citations(text: str) -> list[str]:
    """Global citation ids in order of first appearance, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for match in GLOBAL_CITE_RE.finditer(text or ""):
        sid = match.group(1)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def remap_local_citations(
    text: str,
    local_sources: list[Source],
) -> tuple[str, list[str]]:
    """Rewrite `[1]`-style markers into global `[S…]` ids.

    `local_sources` must be in exactly the order the model was shown them.
    Returns the rewritten text and a list of problems found.

    Models are asked to emit small integers rather than hashes because that is
    the one part of this they can do reliably.
    """
    issues: list[str] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [n.strip() for n in match.group(1).split(",")]
        ids: list[str] = []
        for raw in numbers:
            try:
                index = int(raw)
            except ValueError:
                continue
            if 1 <= index <= len(local_sources):
                ids.append(f"[{local_sources[index - 1]['id']}]")
            else:
                issues.append(
                    f"cited source [{index}] but only {len(local_sources)} "
                    f"were provided"
                )
        return "".join(ids)

    return LOCAL_CITE_RE.sub(replace, text or ""), issues


def validate(text: str, sources: list[Source]) -> list[str]:
    """Report citation ids that no source in the registry backs."""
    known = {src["id"] for src in sources}
    return [
        f"citation [{sid}] has no source in the registry"
        for sid in find_citations(text)
        if sid not in known
    ]


def strip_unknown(text: str, sources: list[Source]) -> str:
    """Remove citation markers with no backing source."""
    known = {src["id"] for src in sources}
    return GLOBAL_CITE_RE.sub(
        lambda m: m.group(0) if m.group(1) in known else "", text or ""
    )


def renumber(text: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Turn `[S…]` markers into reader-facing `[1]`, `[2]`, in order of use.

    Returns the rewritten text plus the bibliography -- only sources actually
    cited, each appearing exactly once, numbered to match.
    """
    by_id = {src["id"]: src for src in sources}
    order: dict[str, int] = {}
    bibliography: list[Source] = []

    for sid in find_citations(text):
        if sid in by_id and sid not in order:
            order[sid] = len(order) + 1
            bibliography.append(by_id[sid])

    def replace(match: re.Match[str]) -> str:
        sid = match.group(1)
        return f"[{order[sid]}]" if sid in order else ""

    return GLOBAL_CITE_RE.sub(replace, text or ""), bibliography


def format_bibliography(bibliography: list[Source]) -> str:
    """Numbered source list matching the in-text markers."""
    if not bibliography:
        return "_No sources were cited._"
    lines = []
    for i, src in enumerate(bibliography, start=1):
        title = (src.get("title") or src["url"]).strip()
        lines.append(f"{i}. [{title}]({src['url']})")
    return "\n".join(lines)


def coverage(text: str, sources: list[Source]) -> dict[str, float | int]:
    """Cheap mechanical quality signal, useful in logs and in evaluation."""
    cited = [sid for sid in find_citations(text) if sid in {s["id"] for s in sources}]
    return {
        "sources_retrieved": len(sources),
        "sources_cited": len(cited),
        "cited_fraction": round(len(cited) / len(sources), 3) if sources else 0.0,
    }
