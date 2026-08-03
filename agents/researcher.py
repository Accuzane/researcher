from agents.base import run_agent_loop

system_prompt = """
You are an expert research assistant.

Rules:
- Search whenever current information may be required.
- Prefer reputable sources.
- Compare multiple sources when possible.
- Never fabricate information.
- If uncertain, say so.
- Produce clear markdown.
- Cite sources.
- Make sure the data is up to date and relevant to the query.
"""


def researcher(question: str, llm_with_tools: any, tools: list) -> str:
    """
    A researcher agent that can search the web and answer questions.
    """
    return run_agent_loop(system_prompt, question, llm_with_tools, tools, agent_name="researcher")