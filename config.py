"""Central configuration.

Everything tunable lives here or in `.env`. The budgets in particular are the
difference between a research run that terminates and one that doesn't, so they
are explicit rather than buried in the graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- models ---------------------------------------------------------
    # FAST does high-volume mechanical work: per-source extraction, query
    # generation. SMART does the judgment calls: planning, gap analysis,
    # synthesis. They default to the same model so this runs unchanged on a
    # single-model Ollama install, but pointing SMART at something larger
    # (qwen3:32b, or an API model) is the single biggest quality lever here
    # after retrieval.
    fast_model: str
    smart_model: str
    embed_model: str
    temperature: float
    num_ctx: int

    # --- research budgets ------------------------------------------------
    min_sub_questions: int
    max_sub_questions: int
    max_rounds: int          # gap-driven iterations before we stop
    max_searches: int        # hard ceiling on total search calls per run
    max_sources: int         # hard ceiling on sources kept in the registry
    queries_per_sub_question: int
    results_per_query: int
    source_chars: int        # per-source text truncation before it hits a prompt

    # --- agent loop safety ------------------------------------------------
    react_max_iters: int
    tool_timeout_s: float

    # --- paths -------------------------------------------------------------
    notes_dir: Path
    index_path: Path
    checkpoint_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        fast = os.getenv("FAST_MODEL", "qwen3:8b")
        return cls(
            fast_model=fast,
            smart_model=os.getenv("SMART_MODEL", fast),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            temperature=_env_float("TEMPERATURE", 0.1),
            num_ctx=_env_int("NUM_CTX", 8192),
            min_sub_questions=_env_int("MIN_SUB_QUESTIONS", 3),
            max_sub_questions=_env_int("MAX_SUB_QUESTIONS", 6),
            max_rounds=_env_int("MAX_ROUNDS", 3),
            max_searches=_env_int("MAX_SEARCHES", 24),
            max_sources=_env_int("MAX_SOURCES", 40),
            queries_per_sub_question=_env_int("QUERIES_PER_SUB_QUESTION", 2),
            results_per_query=_env_int("RESULTS_PER_QUERY", 4),
            source_chars=_env_int("SOURCE_CHARS", 6000),
            react_max_iters=_env_int("REACT_MAX_ITERS", 10),
            tool_timeout_s=_env_float("TOOL_TIMEOUT_S", 45.0),
            notes_dir=Path(os.getenv("NOTES_DIR", str(ROOT / "notes"))),
            index_path=Path(os.getenv("INDEX_PATH", str(ROOT / "notes" / ".index.json"))),
            checkpoint_path=Path(os.getenv("CHECKPOINT_PATH", str(ROOT / ".checkpoints.sqlite"))),
        )


settings = Settings.from_env()
