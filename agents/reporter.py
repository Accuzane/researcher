"""Report writing: one synthesis pass over everything gathered.

Single-pass and single-agent on purpose. Parallelising section writing sounds
appealing and produces disjoint output -- sections repeat each other, contradict
each other, and cite the same source under different numbers. Research
parallelises; synthesis does not.

Everything mechanical about the citations happens here in code: validation
against the registry, stripping markers nothing backs, renumbering to reader-
facing `[1]`, `[2]`, and building the bibliography from what was actually cited.
"""

from __future__ import annotations

import logging

import citations
from llm import chat, smart
from state import Source, SubQuestion

logger = logging.getLogger("reporter")

REPORT_SYSTEM = """You write research reports from gathered findings.

Structure, in Markdown:

## Summary
Three to five sentences answering the question directly. Lead with the answer,
not with background.

## Findings
One `###` subsection per sub-question, using a descriptive heading rather than
the question text. Synthesise; do not paste the findings back.

## Confidence and limitations
What the evidence supports well, what it supports weakly, and what could not be
established. Be concrete about which parts are thin.

Rules:
- Citation markers look exactly like [S1a2b3c4]. Copy them character for
  character from the findings you were given.
- Never invent a citation marker. Never cite a bare URL. If a claim has no
  marker in the findings, either drop the claim or state it as unverified.
- Every paragraph in Summary and Findings carries at least one marker.
- Where sources disagree, say so and cite both.
- No preamble, no sign-off. Start with `## Summary`."""


def _format_findings(sub_questions: list[SubQuestion]) -> str:
    blocks = []
    for sq in sub_questions:
        findings = (sq.get("findings") or "").strip()
        if not findings:
            continue
        status = sq.get("status", "pending")
        blocks.append(
            f"### {sq['id']}: {sq['question']}\n"
            f"(status: {status})\n\n{findings}"
        )
    return "\n\n".join(blocks) if blocks else "(no findings were gathered)"


def write_report(
    query: str,
    brief: str,
    sub_questions: list[SubQuestion],
    sources: list[Source],
) -> tuple[str, list[str]]:
    """Produce the final report and any citation problems found.

    The returned report has reader-facing numbered citations and a bibliography
    listing only sources that survived validation.
    """
    if not sources:
        return (
            f"# {query}\n\n"
            "No sources could be retrieved for this question, so there is "
            "nothing to report. Check network access and `TAVILY_API_KEY`.",
            ["no sources retrieved"],
        )

    body = chat(
        smart(),
        REPORT_SYSTEM,
        f"Question: {query}\n\nResearch brief: {brief}\n\n"
        f"Findings by sub-question:\n\n{_format_findings(sub_questions)}",
    )

    issues = citations.validate(body, sources)
    for issue in issues:
        logger.warning("%s", issue)

    body = citations.strip_unknown(body, sources)
    body, bibliography = citations.renumber(body, sources)

    stats = citations.coverage(body, sources)
    unanswered = [sq["id"] for sq in sub_questions if sq.get("status") != "answered"]

    footer = [
        "",
        "## Sources",
        "",
        citations.format_bibliography(bibliography),
        "",
        "---",
        "",
        f"_{stats['sources_retrieved']} sources retrieved, "
        f"{len(bibliography)} cited._",
    ]
    if unanswered:
        footer.append(
            f"_Sub-questions left unresolved: {', '.join(unanswered)}._"
        )
    if issues:
        footer.append(
            f"_{len(issues)} citation(s) were dropped as unverifiable._"
        )

    report = f"# {query}\n\n{body.strip()}\n" + "\n".join(footer) + "\n"
    return report, issues
