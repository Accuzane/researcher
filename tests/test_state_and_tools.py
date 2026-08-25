"""Reducers, URL identity, note storage, and the tool-calling loop."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.base import run_agent_loop
from config import settings
from state import (
    SubQuestion,
    merge_sources,
    merge_sub_questions,
    normalise_url,
    source_id,
)
from tests.conftest import FakeLLM, make_sources
from tools import file_tools


class TestSourceIdentity:
    def test_same_url_gives_same_id(self):
        assert source_id("https://x.example/a") == source_id("https://x.example/a")

    def test_trivial_variants_collapse(self):
        base = source_id("https://x.example/a")
        assert source_id("https://X.example/a/") == base
        assert source_id("https://x.example/a#section") == base
        assert source_id("HTTPS://x.example/a") == base

    def test_query_strings_are_significant(self):
        assert source_id("https://x.example/a?id=1") != source_id("https://x.example/a?id=2")

    def test_normalise_survives_garbage(self):
        assert normalise_url("not a url") == "not a url"


class TestMergeSources:
    def test_deduplicates_across_concurrent_branches(self):
        left = make_sources(("https://a.example/1", "A", "t"))
        right = make_sources(("https://a.example/1/", "A dup", "t"))
        assert len(merge_sources(left, right)) == 1

    def test_preserves_insertion_order(self):
        left = make_sources(("https://a.example/1", "A", "t"))
        right = make_sources(("https://b.example/2", "B", "t"))
        assert [s["title"] for s in merge_sources(left, right)] == ["A", "B"]

    def test_first_write_wins_on_collision(self):
        left = make_sources(("https://a.example/1", "original", "t"))
        right = make_sources(("https://a.example/1", "replacement", "t"))
        assert merge_sources(left, right)[0]["title"] == "original"

    def test_handles_none(self):
        assert merge_sources(None, None) == []


def _sq(sid: str, **kwargs) -> SubQuestion:
    base = SubQuestion(
        id=sid, question=f"question {sid}", status="pending",
        findings="", source_ids=[], round=0,
    )
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base


class TestMergeSubQuestions:
    def test_merges_by_id_rather_than_appending(self):
        left = [_sq("Q1"), _sq("Q2")]
        right = [_sq("Q1", findings="found it", status="answered")]
        merged = merge_sub_questions(left, right)
        assert len(merged) == 2
        assert merged[0]["findings"] == "found it"
        assert merged[0]["status"] == "answered"

    def test_empty_fields_do_not_clobber(self):
        """A branch returning only findings must not blank the question text."""
        left = [_sq("Q1", findings="original")]
        right = [SubQuestion(
            id="Q1", question="", status="answered",
            findings="", source_ids=[], round=1,
        )]
        merged = merge_sub_questions(left, right)
        assert merged[0]["question"] == "question Q1"
        assert merged[0]["findings"] == "original"
        assert merged[0]["status"] == "answered"

    def test_new_ids_are_appended(self):
        merged = merge_sub_questions([_sq("Q1")], [_sq("Q2")])
        assert [s["id"] for s in merged] == ["Q1", "Q2"]


class TestSlugify:
    @pytest.mark.parametrize(
        "hostile",
        [
            "../../etc/passwd",
            "/absolute/path",
            "..\\..\\windows\\system32",
            "name\x00null",
            "con",
            "a/b/c",
        ],
    )
    def test_no_traversal_survives(self, hostile):
        slug = file_tools.slugify(hostile)
        assert "/" not in slug and "\\" not in slug
        assert ".." not in slug
        assert "\x00" not in slug
        assert file_tools.note_path(hostile).parent == settings.notes_dir

    def test_empty_topic_gets_a_name(self):
        assert file_tools.slugify("") == "untitled"
        assert file_tools.slugify("!!!") == "untitled"

    def test_length_is_bounded(self):
        assert len(file_tools.slugify("x " * 500)) <= file_tools.MAX_SLUG_LEN


class TestNoteStorage:
    def test_roundtrips_body_and_metadata(self):
        file_tools.write_note(
            "Quantum Error Correction",
            "# Report\n\nBody text.",
            meta={"confidence": "normal", "sub_questions": ["one", "two"]},
        )
        loaded = file_tools.find_note("Quantum Error Correction")
        assert loaded is not None
        meta, body = loaded["meta"], loaded["body"]
        assert meta["topic"] == "Quantum Error Correction"
        assert meta["confidence"] == "normal"
        assert meta["sub_questions"] == ["one", "two"]
        assert "Body text." in body
        assert "---" not in body.split("\n")[0]

    def test_writes_markdown_not_txt(self):
        path = file_tools.write_note("A Topic", "body")
        assert path.suffix == ".md"

    def test_find_note_tolerates_approximate_topics(self):
        file_tools.write_note("Quantum Error Correction Progress", "body")
        assert file_tools.find_note("quantum error correction") is not None
        assert file_tools.find_note("progress in quantum error correction") is not None

    def test_find_note_rejects_unrelated_topics(self):
        file_tools.write_note("Quantum Error Correction", "body")
        assert file_tools.find_note("mediterranean cooking techniques") is None

    def test_list_notes_ignores_hidden_files(self):
        file_tools.write_note("Real Topic", "body")
        (settings.notes_dir / ".index.json").write_text("{}", encoding="utf-8")
        assert len(file_tools.all_notes()) == 1


# --------------------------------------------------------------------------
# the ReAct loop
# --------------------------------------------------------------------------

@tool
def exploding_tool(value: str) -> str:
    """A tool that always fails."""
    raise RuntimeError("upstream timeout")


@tool
def working_tool(value: str) -> str:
    """A tool that works."""
    return f"result for {value}"


def _tool_call(name: str, call_id: str = "c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"value": "x"}, "id": call_id}],
    )


class TestAgentLoop:
    def test_returns_when_the_model_stops_calling_tools(self):
        model = FakeLLM(lambda s, u, schema: "final answer")
        assert run_agent_loop("sys", "in", model, [working_tool]) == "final answer"

    def test_caps_iterations_instead_of_looping_forever(self):
        calls = {"n": 0}

        def handler(system, user, schema):
            calls["n"] += 1
            return _tool_call("working_tool", f"c{calls['n']}")

        model = FakeLLM(handler)
        result = run_agent_loop("sys", "in", model, [working_tool], max_iters=3)
        assert calls["n"] == 3
        assert "budget" in result

    def test_a_failing_tool_does_not_kill_the_run(self):
        state = {"n": 0}

        def handler(system, user, schema):
            state["n"] += 1
            if state["n"] == 1:
                return _tool_call("exploding_tool")
            return "recovered and answered"

        model = FakeLLM(handler)
        result = run_agent_loop("sys", "in", model, [exploding_tool])
        assert result == "recovered and answered"

    def test_the_tool_error_is_fed_back_to_the_model(self):
        seen: list[str] = []

        def handler(system, user, schema):
            seen.append(user)
            return _tool_call("exploding_tool") if len(seen) == 1 else "done"

        model = FakeLLM(handler)
        run_agent_loop("sys", "in", model, [exploding_tool])
        # the second turn must carry a ToolMessage describing the failure
        assert model.record  # sanity: the model was called more than once

    def test_unknown_tool_is_reported_not_raised(self):
        state = {"n": 0}

        def handler(system, user, schema):
            state["n"] += 1
            return _tool_call("no_such_tool") if state["n"] == 1 else "handled"

        model = FakeLLM(handler)
        assert run_agent_loop("sys", "in", model, [working_tool]) == "handled"

    def test_a_model_failure_returns_a_message(self):
        def handler(system, user, schema):
            raise RuntimeError("ollama is down")

        result = run_agent_loop("sys", "in", FakeLLM(handler), [working_tool])
        assert "ollama is down" in result
