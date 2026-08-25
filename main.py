"""CLI.

Three things it does differently from the obvious version:

* A fresh thread id per query. Reusing one leaves the graph parked on an
  earlier checkpoint and quietly carries state into the next question.
* The report is printed before anything is asked about saving. Displaying and
  persisting are separate decisions; answering "no" to a save prompt should
  never discard work you just waited minutes for.
* Progress streams as nodes complete, because a real research run takes long
  enough that silence reads as a hang.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from config import settings
from graph.research_graph import compile_graph
from graph.router import route
from state import new_state
from tools.file_tools import write_note

LOG_FORMAT = "%(asctime)s  %(name)-16s %(message)s"

NODE_LABELS = {
    "plan": "planning sub-questions",
    "research": "researching",
    "gaps": "checking coverage",
    "report": "writing report",
}


def run_research(app, query: str) -> dict:
    """Run the graph, streaming progress, and return the final state."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final: dict = {}

    for chunk in app.stream(new_state(query), config=config, stream_mode="updates"):
        for node, update in chunk.items():
            label = NODE_LABELS.get(node, node)
            detail = ""
            if node == "plan":
                detail = f" ({len(update.get('sub_questions', []))} sub-questions)"
            elif node == "research":
                sqs = update.get("sub_questions") or [{}]
                detail = f" {sqs[0].get('id', '')}: {sqs[0].get('status', '')}"
            elif node == "gaps":
                gaps = update.get("gaps", [])
                stop = update.get("stop_reason")
                detail = f" ({stop})" if stop else f" ({len(gaps)} gap(s), another round)"
            print(f"  · {label}{detail}", flush=True)

    final = app.get_state(config).values
    return final


def maybe_save(state: dict) -> None:
    """Offer to save. Declining costs nothing -- the report is already printed."""
    report = state.get("report", "")
    if not report.strip():
        return
    try:
        choice = input("\nSave this as a note? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if choice not in {"y", "yes"}:
        return

    sub_questions = state.get("sub_questions", [])
    unresolved = [sq["id"] for sq in sub_questions if sq.get("status") != "answered"]
    path = write_note(
        state.get("topic") or state.get("query", "untitled"),
        report,
        meta={
            "query": state.get("query", ""),
            "model": settings.smart_model,
            "rounds": state.get("round", 1),
            "sources_retrieved": len(state.get("sources", [])),
            "confidence": "low" if unresolved else "normal",
            "unresolved_sub_questions": unresolved,
            "sub_questions": [sq["question"] for sq in sub_questions],
        },
    )
    print(f"Saved to {path}")


def answer_once(app, query: str) -> dict:
    """Route one question and return the resulting state."""
    if route(query) == "notes":
        from agents.qa import answer_from_notes

        print("  · searching saved notes", flush=True)
        return {"query": query, "answer": answer_from_notes(query), "report": ""}
    return run_research(app, query)


def show(state: dict) -> None:
    text = state.get("report") or state.get("answer") or "(no output produced)"
    print("\n" + "=" * 72)
    print(text.strip())
    print("=" * 72)
    for issue in state.get("citation_issues", []):
        print(f"  ! {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local deep-research agent.")
    parser.add_argument("query", nargs="*", help="Ask once and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument("--save", action="store_true", help="Save without prompting.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    app = compile_graph()

    if args.query:
        state = answer_once(app, " ".join(args.query))
        show(state)
        if args.save:
            write_note(state.get("topic") or state["query"], state.get("report", ""))
        else:
            maybe_save(state)
        return 0

    print(f"Research agent  ·  {settings.smart_model}  ·  Ctrl-D to exit")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0

        try:
            state = answer_once(app, query)
        except KeyboardInterrupt:
            print("\ninterrupted")
            continue
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            logging.getLogger("main").exception("run failed")
            print(f"\nThat run failed: {exc}")
            continue

        show(state)
        maybe_save(state)


if __name__ == "__main__":
    raise SystemExit(main())
