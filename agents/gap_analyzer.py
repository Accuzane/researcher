"""Gap analysis: what is still unsupported, and what query would fix it.

This replaces the usual pass/fail critic. A boolean verdict can only tell you
to try again, which is why a retry loop built on one repeats the same query and
gets the same answer. A gap analyser returns *which* sub-question is still
unsupported and *what to search for next*, which is the difference between
iterating and re-rolling.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from llm import smart, structured
from state import Gap, SubQuestion

logger = logging.getLogger("gap_analyzer")

GAP_SYSTEM = """You audit research coverage.

You are given a research brief and, for each sub-question, the findings
gathered so far. For each sub-question decide whether the findings genuinely
answer it on the evidence presented.

Mark a sub-question as a gap when:
- the findings are missing, vague, or hedged
- the findings answer a narrower or different question than the one asked
- a central claim carries no citation

Do not mark a gap merely because more detail could exist. The bar is whether
the question is answered, not whether it is exhausted.

For each gap, give one specific web search query that would close it. Make it
narrower or differently angled than what has clearly been tried -- repeating
the original phrasing is useless."""


class GapItem(BaseModel):
    sub_question_id: str = Field(description="The id, e.g. Q2")
    reason: str = Field(description="What is missing, in one sentence")
    follow_up_query: str = Field(description="A specific search query to close it")


class GapReport(BaseModel):
    gaps: list[GapItem] = Field(default_factory=list)


def _format(sub_questions: list[SubQuestion]) -> str:
    blocks = []
    for sq in sub_questions:
        findings = sq.get("findings") or "(nothing gathered)"
        blocks.append(f"{sq['id']}: {sq['question']}\nFindings: {findings}")
    return "\n\n".join(blocks)


def find_gaps(brief: str, sub_questions: list[SubQuestion]) -> list[Gap]:
    """Identify sub-questions still needing work.

    On a model failure we fall back to the mechanical signal the
    sub-researchers already computed: anything not marked `answered` is a gap.
    That keeps the loop honest even when the analyser itself misbehaves.
    """
    known_ids = {sq["id"] for sq in sub_questions}
    mechanical = [
        Gap(
            sub_question_id=sq["id"],
            reason="No supported findings were produced.",
            follow_up_query=sq["question"],
        )
        for sq in sub_questions
        if sq.get("status") != "answered"
    ]

    report = structured(
        smart(),
        GapReport,
        GAP_SYSTEM,
        f"Research brief: {brief}\n\n{_format(sub_questions)}",
        default=GapReport(gaps=[]),
    )

    gaps: list[Gap] = []
    seen: set[str] = set()
    for item in report.gaps:
        sid = item.sub_question_id.strip()
        if sid not in known_ids or sid in seen:
            continue
        seen.add(sid)
        gaps.append(
            Gap(
                sub_question_id=sid,
                reason=item.reason.strip(),
                follow_up_query=(item.follow_up_query or "").strip(),
            )
        )

    # Union with the mechanical signal: the analyser can add gaps it spotted,
    # but it cannot clear a sub-question that produced nothing citable.
    for gap in mechanical:
        if gap["sub_question_id"] not in seen:
            gaps.append(gap)

    for gap in gaps:
        logger.info(
            "gap %s: %s -> %r",
            gap["sub_question_id"],
            gap["reason"],
            gap["follow_up_query"],
        )
    return gaps
