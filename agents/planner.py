"""Planning: turn a question into a research brief and sub-questions.

This is the step that separates a researcher from a search wrapper. One query
producing one search produces one perspective; decomposing first means the
sub-questions can be researched independently, in parallel, in isolated
context, and the gap analyser afterwards has something concrete to check
coverage *against*.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from config import settings
from llm import smart, structured
from state import SubQuestion

logger = logging.getLogger("planner")

PLANNER_SYSTEM = """You are a research lead scoping a piece of work.

Given a question, produce:
- topic: a short noun phrase naming the subject, suitable as a filename
- brief: 2-3 sentences stating what a good answer must establish
- sub_questions: {min}-{max} specific, independently researchable questions

Rules for sub_questions:
- Each must be answerable on its own, without the answers to the others.
- Together they must cover the whole question, with no redundancy.
- Prefer concrete and checkable over broad and interpretive.
- If the question implies comparison, give each item its own sub-question.
- If recency matters, say so explicitly in the sub-question text.

Do not answer the question. Plan the work."""


class ResearchPlan(BaseModel):
    topic: str = Field(description="Short noun phrase naming the subject")
    brief: str = Field(description="What a good answer must establish")
    sub_questions: list[str] = Field(description="Independently researchable questions")


def plan(query: str) -> tuple[str, str, list[SubQuestion]]:
    """Return (topic, brief, sub_questions).

    Falls back to treating the original query as a single sub-question, so a
    planning failure degrades to the old one-shot behaviour rather than
    stopping the run.
    """
    fallback = ResearchPlan(topic=query[:60], brief=query, sub_questions=[query])

    result = structured(
        smart(),
        ResearchPlan,
        PLANNER_SYSTEM.format(
            min=settings.min_sub_questions, max=settings.max_sub_questions
        ),
        f"Question: {query}",
        default=fallback,
    )

    questions = [q.strip() for q in result.sub_questions if q and q.strip()]
    questions = questions[: settings.max_sub_questions] or [query]

    sub_questions: list[SubQuestion] = [
        SubQuestion(
            id=f"Q{i}",
            question=question,
            status="pending",
            findings="",
            source_ids=[],
            round=0,
        )
        for i, question in enumerate(questions, start=1)
    ]

    topic = (result.topic or query)[:80].strip()
    brief = (result.brief or query).strip()
    logger.info("topic=%r, %d sub-questions", topic, len(sub_questions))
    for sq in sub_questions:
        logger.info("  %s %s", sq["id"], sq["question"])

    return topic, brief, sub_questions
