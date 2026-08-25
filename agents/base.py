"""The shared tool-calling loop.

Same shape as a textbook ReAct loop, with the two things a textbook version
leaves out and production reliably punishes you for:

* a hard iteration cap, so a model that keeps calling the same tool ends the
  turn instead of the process;
* per-tool error containment, so one search timeout costs you a tool result
  rather than the entire graph run.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from config import settings
from llm import strip_reasoning, _content_to_text


def run_agent_loop(
    system_prompt: str,
    user_input: str,
    llm_with_tools: BaseChatModel,
    tools: Sequence[BaseTool],
    agent_name: str = "agent",
    max_iters: int | None = None,
) -> str:
    """Run the model until it answers, it stops calling tools, or we cut it off.

    Always returns a string. Exhausting the iteration budget yields the best
    text produced so far rather than raising, because a partial answer is more
    useful to the caller than a stack trace.
    """
    logger = logging.getLogger(agent_name)
    limit = max_iters or settings.react_max_iters
    by_name = {tool.name: tool for tool in tools}

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    last_text = ""

    for iteration in range(limit):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("model call failed: %s", exc)
            return last_text or f"The model call failed: {exc}"

        messages.append(response)
        text = strip_reasoning(_content_to_text(response.content))
        if text:
            last_text = text

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return last_text

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id") or name
            logger.info("iter %d: %s(%s)", iteration + 1, name, args)

            tool = by_name.get(name)
            if tool is None:
                messages.append(
                    ToolMessage(
                        content=(
                            f"Error: tool '{name}' does not exist. "
                            f"Available tools: {', '.join(by_name) or 'none'}."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue

            messages.append(
                ToolMessage(content=_invoke_tool(tool, args, logger), tool_call_id=call_id)
            )

    logger.warning("hit the %d-iteration cap; returning best effort", limit)
    return last_text or (
        "I wasn't able to finish this within the tool-call budget."
    )


def _invoke_tool(tool: BaseTool, args: dict[str, Any], logger: logging.Logger) -> str:
    """Run one tool. A failure becomes a message the model can react to."""
    try:
        result = tool.invoke(args)
    except Exception as exc:  # noqa: BLE001 - failures are fed back, not raised
        logger.warning("%s failed: %s", tool.name, exc)
        return (
            f"Error: '{tool.name}' failed with {type(exc).__name__}: {exc}. "
            f"Try different arguments, a different tool, or answer without it."
        )
    text = str(result)
    logger.info("%s -> %s", tool.name, text[:120].replace("\n", " "))
    return text
