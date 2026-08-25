"""Model construction and structured-output handling.

Two things matter here.

Tiering: `fast()` and `smart()` are separate so the expensive judgment steps
(planning, gap analysis, synthesis) can run on a bigger model than the
high-volume extraction steps. They collapse to the same model by default.

Never parsing free text you could have constrained: `structured()` asks for a
schema, and falls back through progressively dumber strategies rather than
raising. Local models in the 8B range fail schema-following often enough that
a bare `.with_structured_output()` call is not safe on its own.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from config import settings

logger = logging.getLogger("llm")

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------
# model factory
# --------------------------------------------------------------------------

_cache: dict[str, BaseChatModel] = {}


def _build(model: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        # Ollama enables thinking by default for models that support it, and it
        # is not reliably suppressible per-model from the client side. We ask,
        # and strip_reasoning() cleans up whatever still arrives in content.
        reasoning=False,
    )


def fast() -> BaseChatModel:
    if "fast" not in _cache:
        _cache["fast"] = _build(settings.fast_model)
    return _cache["fast"]


def smart() -> BaseChatModel:
    if "smart" not in _cache:
        _cache["smart"] = _build(settings.smart_model)
    return _cache["smart"]


def set_models(fast_model: BaseChatModel, smart_model: BaseChatModel | None = None) -> None:
    """Inject models directly. Used by the tests to run the graph without Ollama."""
    _cache["fast"] = fast_model
    _cache["smart"] = smart_model or fast_model


def reset_models() -> None:
    _cache.clear()


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def strip_reasoning(text: str) -> str:
    """Remove <think> blocks that reasoning models leak into content."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    # An unterminated <think> (truncated generation) would otherwise swallow
    # the whole response, so drop everything from the opening tag onward only
    # if no closing tag exists anywhere.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


def _content_to_text(content: Any) -> str:
    """Chat models may return a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def chat(llm: BaseChatModel, system: str, user: str) -> str:
    """One-shot prompt returning clean text."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return strip_reasoning(_content_to_text(response.content))


def extract_json(text: str) -> Any:
    """Pull the first complete JSON value out of a model response.

    Handles ```json fences, prose wrapped around the object, and reasoning
    blocks. Scans for balanced delimiters rather than regex-matching from the
    first `{` to the last `}` -- that greedy form happily spans two unrelated
    objects and produces garbage.
    """
    cleaned = strip_reasoning(text)

    fenced = _FENCE_RE.search(cleaned)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)

    for candidate in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            if start == -1:
                continue
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start:i + 1])
                        except json.JSONDecodeError:
                            break
    raise ValueError("no parseable JSON in response")


def structured(
    llm: BaseChatModel,
    schema: type[T],
    system: str,
    user: str,
    default: T,
) -> T:
    """Get a validated `schema` instance, or `default`.

    Three attempts, cheapest-to-recover first:
      1. native structured output (tool/JSON-schema constrained decoding)
      2. plain generation, then JSON extraction and validation
      3. the caller's default

    This never raises. A research run should degrade, not die, because one
    model call came back malformed.
    """
    try:
        result = llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.debug("structured output failed for %s: %s", schema.__name__, exc)

    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    retry_system = (
        f"{system}\n\n"
        f"Respond with a single JSON object matching this schema, and nothing else:\n"
        f"{schema_hint}"
    )
    try:
        raw = chat(llm, retry_system, user)
        return schema.model_validate(extract_json(raw))
    except (ValueError, ValidationError, Exception) as exc:  # noqa: BLE001
        logger.warning(
            "falling back to default for %s: %s", schema.__name__, exc
        )
        return default
