import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from graph.supervisor import app


def maybe_save_note(config):
    state = app.get_state(config)
    if state.next == ("note_writer",):
        choice = input("Save this note? (y/n): ").strip().lower()
        if choice == "y":
            return app.invoke(None, config=config)
        return None
    return None


def main():
    thread_id = 1
    while True:
        query = input("\nAsk me anything (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "query": query,
            "topic": "",
            "research": "",
            "answer": "",
            "saved_path": "",
            "next_agent": "",
            "needs_research": False,
            "retry_count": 0,
        }

        result = app.invoke(state, config=config)
        resumed = maybe_save_note(config)
        final = resumed if resumed else result

        print(f"\n{'-'*40}")
        print(final.get("answer", "(no answer)"))
        print("-" * 40)


if __name__ == "__main__":
    main()