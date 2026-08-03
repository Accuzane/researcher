import logging
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.researcher import researcher
from agents.notes_writer import note_writer
from agents.qa_agent import qa_agent
from agents.critic import critic
from tools.file_tools import save_note, load_note, list_notes

load_dotenv()

logger = logging.getLogger("supervisor")


# --- llm + tools setup ---
llm = ChatOllama(model="qwen3:8b")

research_tools = [TavilySearch(max_results=3)]
research_llm = llm.bind_tools(research_tools)

note_tools = [save_note]
note_writer_llm = llm.bind_tools(note_tools)

qa_tools = [load_note, list_notes]
qa_llm = llm.bind_tools(qa_tools)


# --- shared state ---
class AgentState(TypedDict):
    query: str
    topic: str
    research: str
    answer: str
    saved_path: str
    next_agent: str
    needs_research: bool
    retry_count: int


# --- node functions ---
def researcher_node(state: AgentState) -> dict:
    result = researcher(state["query"], research_llm, research_tools)
    return {"research": result}


def note_writer_node(state: AgentState) -> dict:
    path = note_writer(state["topic"], state["research"], note_writer_llm, note_tools)
    return {"saved_path": path}


def qa_node(state: AgentState) -> dict:
    result = qa_agent(state["query"], qa_llm, qa_tools)
    return {"answer": result}


def critic_node(state: AgentState) -> dict:
    verdict = critic(state["research"], llm)
    if verdict.get("pass"):
        return {"needs_research": False}
    return {"needs_research": True, "retry_count": state.get("retry_count", 0) + 1}


SUPERVISOR_PROMPT = """
You are a routing supervisor. Given a user query, decide which agent should handle it.

Rules:
- If the query asks to research or find information about a new topic -> "researcher"
- If the query asks a question about something already researched -> "qa"
- If the query asks what topics have been saved -> "qa"
- If research was just completed and needs saving -> "note_writer"

Respond with ONLY one word: researcher, note_writer, or qa.
"""


def supervisor_node(state: AgentState) -> dict:
    saved_topics = list_notes.invoke({})

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"Already saved notes on: {saved_topics}\n\nUser query: {state['query']}")
    ]
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    logger.info(f"routed to: {decision}")
    return {"next_agent": decision}


def route_after_critic(state: AgentState) -> str:
    # cap retries so a stubborn critic can't loop forever
    if state.get("needs_research") and state.get("retry_count", 0) < 2:
        return "researcher"
    return "note_writer"


# --- graph wiring ---
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("note_writer", note_writer_node)
graph.add_node("qa", qa_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {"researcher": "researcher", "note_writer": "note_writer", "qa": "qa"}
)

graph.add_edge("researcher", "critic")

graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {"researcher": "researcher", "note_writer": "note_writer"}
)

graph.add_edge("note_writer", "qa")
graph.add_edge("qa", END)


# --- compile with checkpointing + human-in-the-loop pause ---
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["note_writer"])