"""Question answering over the saved corpus.

Kept as a genuine tool-calling agent, because unlike the research path this one
really does benefit from deciding what to look up. The tools it gets are the
retrieval-shaped ones -- semantic search first, whole-note loading second --
rather than the exact-filename lookup that makes a notes corpus unusable.
"""

from __future__ import annotations

from agents.base import run_agent_loop
from llm import fast
from tools.file_tools import list_notes, load_note, search_notes

QA_SYSTEM = """You answer questions from a personal research corpus.

Method:
1. Call search_notes with the user's question. This searches the contents of
   every saved note and is almost always the right first move.
2. Call load_note if you need a whole note for context.
3. Call list_notes only when the user asks what has been researched.

Rules:
- Answer only from what the tools return. Do not use your own knowledge of the
  subject, even when you are confident.
- If the notes do not cover it, say so plainly and name what is saved instead.
- Quote figures and specifics exactly as the notes state them.
- Keep the citation markers and source links that appear in the notes."""

QA_TOOLS = [search_notes, load_note, list_notes]


def answer_from_notes(question: str) -> str:
    return run_agent_loop(
        QA_SYSTEM,
        f"Question: {question}",
        fast().bind_tools(QA_TOOLS),
        QA_TOOLS,
        agent_name="qa",
    )
