"""One sub-question, researched end to end in isolated context.

Structured as a fixed pipeline -- generate queries, retrieve, synthesise --
rather than a free-running ReAct loop. Two reasons. An 8B model driving its own
multi-step search loop wanders and burns tokens; and putting the iteration in
the *graph* (the gap loop) instead of inside an agent makes it inspectable,
budgetable, and testable.

The compression step at the end is not decoration. Raw page text must never
reach the parent state -- that is exactly how a multi-agent research system
accumulates its way into context-window failure.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

import citations
from config import settings
from llm import chat, fast, smart, structured
from state import Source, SubQuestion
from tools.search import search_many

logger = logging.getLogger("sub_researcher")

# Written when a round retrieves nothing. Named so later rounds can recognise
# it as a placeholder rather than treating it as evidence worth keeping.
NO_SOURCES = "No sources could be retrieved for this question."

QUERY_SYSTEM = """You write web search queries.

Given a research question, produce {n} distinct search queries that would
surface documents answering it.

Rules:
- Use the keywords a relevant page would actually contain, not question phrasing.
- Vary the angle between queries; do not paraphrase one query {n} times.
- Include a year only when recency genuinely matters.
- No operators, no quotes, no site: filters."""

SYNTHESIS_SYSTEM = """You extract findings from source documents.

You are given one research question and numbered sources. Write a tight,
factual answer drawn only from those sources.

Rules:
- Cite every factual claim with the source number in square brackets: [1], [2].
- A sentence with no citation must be removed. There are no exceptions.
- Never state anything the sources do not support. Do not fill gaps from
  memory.
- If the sources do not answer the question, say exactly what is missing.
- Prefer specifics: figures, dates, names, direct findings.
- 150-250 words. No preamble, no restating the question, no closing summary."""


class SearchQueries(BaseModel):
    queries: list[str] = Field(description="Distinct web search queries")


def _make_queries(question: str, extra_context: str = "") -> list[str]:
    n = settings.queries_per_sub_question
    result = structured(
        fast(),
        SearchQueries,
        QUERY_SYSTEM.format(n=n),
        f"Research question: {question}\n{extra_context}".strip(),
        default=SearchQueries(queries=[question]),
    )
    queries = [q.strip() for q in result.queries if q and q.strip()][:n]
    return queries or [question]


def _format_sources(sources: list[Source], total_budget: int = 12000) -> str:
    """Number sources for the prompt and share the character budget between them."""
    if not sources:
        return "(no sources retrieved)"
    per_source = max(600, total_budget // len(sources))
    blocks = []
    for i, src in enumerate(sources, start=1):
        text = src["text"][:per_source]
        blocks.append(
            f"[{i}] {src['title']}\nURL: {src['url']}\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def research_sub_question(
    sub_question: SubQuestion,
    brief: str,
    round_index: int = 0,
    follow_up_query: str | None = None,
) -> tuple[SubQuestion, list[Source], int]:
    """Research one sub-question.

    Returns the updated sub-question, the sources it retrieved, and how many
    searches it spent -- the caller adds that to the run's budget.
    """
    question = sub_question["question"]
    context = f"Overall research brief: {brief}" if brief else ""

    queries = [follow_up_query] if follow_up_query else _make_queries(question, context)
    logger.info("%s: %s", sub_question["id"], queries)

    sources = search_many(queries)
    if not sources:
        # A follow-up that retrieves nothing must leave the earlier round's
        # work alone. `merge_sub_questions` takes the later non-empty value
        # field by field, so writing the placeholder here would overwrite cited
        # findings and flip `answered` back to `unanswered` -- losing evidence
        # that is still sitting in the source registry. An empty search is also
        # what a backend timeout looks like, since `search()` swallows failures.
        previous = (sub_question.get("findings") or "").strip()
        if previous and previous != NO_SOURCES:
            logger.info(
                "%s: follow-up found nothing; keeping round-%s findings",
                sub_question["id"],
                sub_question.get("round"),
            )
            return ({**sub_question, "round": round_index}, [], len(queries))

        return (
            {
                **sub_question,
                "status": "unanswered",
                "findings": NO_SOURCES,
                "round": round_index,
            },
            [],
            len(queries),
        )

    raw_findings = chat(
        smart(),
        SYNTHESIS_SYSTEM,
        f"Research question: {question}\n\nSources:\n\n{_format_sources(sources)}",
    )

    # The model cites [1], [2] against its own numbered list; rewrite those to
    # global ids so the citation survives being merged with every other
    # sub-agent's findings.
    findings, issues = citations.remap_local_citations(raw_findings, sources)
    for issue in issues:
        logger.warning("%s: %s", sub_question["id"], issue)

    # A follow-up round adds to what we already have rather than replacing it.
    # Overwriting would mean a narrower second search could *lose* a good
    # first-round answer -- the loop is supposed to accumulate evidence.
    previous = (sub_question.get("findings") or "").strip()
    if follow_up_query and previous and previous != NO_SOURCES:
        findings = f"{previous}\n\nAdditional findings:\n{findings.strip()}"

    cited = citations.find_citations(findings)
    answered = bool(cited) and len(findings.split()) >= 30

    logger.info(
        "%s: %d sources, %d cited, %s",
        sub_question["id"],
        len(sources),
        len(cited),
        "answered" if answered else "unanswered",
    )

    return (
        {
            **sub_question,
            "status": "answered" if answered else "unanswered",
            "findings": findings,
            "source_ids": cited,
            "round": round_index,
        },
        sources,
        len(queries),
    )
