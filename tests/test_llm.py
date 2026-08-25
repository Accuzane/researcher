"""JSON extraction and structured-output fallbacks.

These exist because local models in the 8B range routinely wrap JSON in prose,
in fences, or in reasoning blocks, and a research run must survive all three.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm import extract_json, strip_reasoning, structured
from tests.conftest import FakeLLM


class Example(BaseModel):
    name: str
    count: int = 0


class TestStripReasoning:
    def test_removes_think_blocks(self):
        assert strip_reasoning("<think>hmm</think>Answer") == "Answer"

    def test_removes_multiline_think_blocks(self):
        assert strip_reasoning("<think>\nline\nline\n</think>\nAnswer") == "Answer"

    def test_truncated_think_block_does_not_swallow_everything(self):
        # A generation cut off mid-reasoning has an opening tag and no closing
        # one; naive regex removal would return the empty string.
        assert strip_reasoning("Answer first<think>unterminated") == "Answer first"

    def test_handles_empty(self):
        assert strip_reasoning("") == ""


class TestExtractJson:
    def test_bare_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_wrapped_in_prose(self):
        assert extract_json('Sure! {"a": 1} Hope that helps.') == {"a": 1}

    def test_object_after_reasoning(self):
        assert extract_json('<think>plan</think>{"a": 1}') == {"a": 1}

    def test_nested_objects_are_not_truncated(self):
        assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_two_objects_does_not_greedily_span_both(self):
        # The naive r"\{.*\}" pattern matches from the first brace to the last
        # and produces unparseable garbage here.
        assert extract_json('{"a": 1} and then {"b": 2}') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"a": "literal } brace"}') == {"a": "literal } brace"}

    def test_arrays(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_raises_when_there_is_no_json(self):
        with pytest.raises(ValueError):
            extract_json("no json at all")


class TestStructured:
    def test_uses_native_structured_output_when_it_works(self):
        model = FakeLLM(lambda s, u, schema: {"name": "ok", "count": 1})
        result = structured(model, Example, "sys", "user", Example(name="fallback"))
        assert result.name == "ok"

    def test_falls_back_to_json_extraction(self):
        def handler(system, user, schema):
            if schema is not None:
                return None  # native structured output fails
            return 'Here you go: {"name": "recovered", "count": 2}'

        result = structured(FakeLLM(handler), Example, "sys", "user", Example(name="fb"))
        assert result.name == "recovered"
        assert result.count == 2

    def test_falls_back_to_default_when_everything_fails(self):
        def handler(system, user, schema):
            if schema is not None:
                return None
            return "I refuse to produce JSON."

        result = structured(FakeLLM(handler), Example, "sys", "user", Example(name="fb"))
        assert result.name == "fb"

    def test_never_raises_on_a_dead_model(self):
        def handler(system, user, schema):
            raise RuntimeError("connection refused")

        result = structured(FakeLLM(handler), Example, "sys", "user", Example(name="fb"))
        assert result.name == "fb"
