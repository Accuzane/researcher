"""Test doubles.

The point of these is that the whole graph can be exercised -- fan-out,
reducers, the gap loop, budgets, citation validation -- with no Ollama running
and no Tavily key. A research agent whose control flow you can only test by
making live API calls is one you will never actually test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm as llm_module  # noqa: E402
from config import settings  # noqa: E402
from state import Source, source_id  # noqa: E402
from tools import notes_index, search  # noqa: E402


# --------------------------------------------------------------------------
# fake chat model
# --------------------------------------------------------------------------

class FakeLLM:
    """Duck-typed chat model driven by a handler function.

    The handler receives (system_prompt, user_prompt, schema) and returns
    either a string (for `chat`) or a dict (for `structured`). `schema` is None
    for plain text calls.
    """

    def __init__(self, handler, record: list | None = None):
        self.handler = handler
        self.record = record if record is not None else []
        self.bound_tools: list = []

    # -- plain text -------------------------------------------------------
    def invoke(self, messages):
        system, user = _split(messages)
        self.record.append(("text", system, user))
        result = self.handler(system, user, None)
        if isinstance(result, AIMessage):
            return result
        return AIMessage(content=str(result))

    # -- structured -------------------------------------------------------
    def with_structured_output(self, schema):
        return _StructuredView(self, schema)

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self


class _StructuredView:
    def __init__(self, parent: FakeLLM, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        system, user = _split(messages)
        self.parent.record.append(("structured", self.schema.__name__, user))
        result = self.parent.handler(system, user, self.schema)
        if result is None:
            raise RuntimeError("fake structured-output failure")
        if isinstance(result, self.schema):
            return result
        return self.schema.model_validate(result)


def _split(messages) -> tuple[str, str]:
    system = next(
        (m.content for m in messages if m.__class__.__name__ == "SystemMessage"), ""
    )
    user = next(
        (m.content for m in messages if m.__class__.__name__ == "HumanMessage"), ""
    )
    return str(system), str(user)


# --------------------------------------------------------------------------
# fake search
# --------------------------------------------------------------------------

def make_sources(*specs: tuple[str, str, str], query: str = "q") -> list[Source]:
    """Build Source records from (url, title, text) triples."""
    return [
        Source(
            id=source_id(url),
            url=url,
            title=title,
            text=text,
            query=query,
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
        for url, title, text in specs
    ]


class FakeSearch:
    """Records every query and returns canned results."""

    def __init__(self, results_by_query=None, default=None):
        self.results_by_query = results_by_query or {}
        self.default = default if default is not None else []
        self.queries: list[str] = []

    def __call__(self, query: str, max_results: int):
        self.queries.append(query)
        results = self.results_by_query.get(query, self.default)
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "raw_content": r["text"],
            }
            for r in results
        ][:max_results]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point every path at a temp dir and keep budgets small and predictable."""
    monkeypatch.setattr(settings, "notes_dir", tmp_path / "notes")
    monkeypatch.setattr(settings, "index_path", tmp_path / "notes" / ".index.json")
    monkeypatch.setattr(settings, "checkpoint_path", tmp_path / "cp.sqlite")
    monkeypatch.setattr(settings, "max_rounds", 2)
    monkeypatch.setattr(settings, "max_searches", 20)
    monkeypatch.setattr(settings, "max_sources", 40)
    monkeypatch.setattr(settings, "queries_per_sub_question", 1)
    monkeypatch.setattr(settings, "results_per_query", 3)
    monkeypatch.setattr(settings, "react_max_iters", 4)
    notes_index.set_embedder(None)  # force deterministic keyword search
    yield
    llm_module.reset_models()
    search.reset_backend()


@pytest.fixture
def fake_search():
    backend = FakeSearch()
    search.set_backend(backend)
    return backend


@pytest.fixture
def install_llm():
    """Install a FakeLLM as both tiers; returns the call record."""

    def _install(handler):
        record: list = []
        model = FakeLLM(handler, record)
        llm_module.set_models(model, model)
        return record

    return _install
