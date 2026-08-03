from agents.base import run_agent_loop

QA_PROMPT = """
You are a knowledge base assistant. You answer questions using notes that
have been previously saved. Always call load_note or list_notes first before
answering. If no relevant note exists, say so clearly — do not make up an answer.
"""


def qa_agent(topic: str, llm_with_tools: any, tools: list) -> str:
    """
    A question-answering agent that uses saved notes to answer questions.
    """
    user_input = f"Question: {topic}"
    return run_agent_loop(QA_PROMPT, user_input, llm_with_tools, tools, agent_name="qa_agent")