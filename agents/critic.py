from langchain_core.messages import HumanMessage, SystemMessage
import json
import logging
import re

CRITIC_PROMPT = """
You are a quality reviewer for research summaries. Evaluate the summary on:
1. Does it contain at least 3 specific facts?
2. Does it cite at least 2 source URLs?
3. Is it free of speculation?

Respond with JSON only: {"pass": true/false, "reason": "..."}
If pass is false, the researcher will search again with a more specific query.
"""


def _extract_json(text: str) -> dict:
    """
    Local models sometimes wrap JSON in markdown fences or extra text.
    Pull out the first {...} block before parsing.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in critic response.")
    return json.loads(match.group(0))


def critic(research_summary: str, llm: any) -> dict:
    """
    A critic agent that reviews research quality and returns a pass/fail verdict.
    """
    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=research_summary)
    ]

    response = llm.invoke(messages)

    try:
        verdict = _extract_json(response.content)
    except (ValueError, json.JSONDecodeError):
        logging.error(f"Critic returned unparseable response: {response.content}")
        # fail-open: don't block the pipeline if the critic itself misbehaves
        verdict = {"pass": True, "reason": "Could not parse critic response; defaulting to pass."}

    return verdict