from graph.supervisor import app
def save_note(config):
    state = app.get_state(config)
    if state.next == ("note_writer",):
        choice = input("Do you want to save this note? (y/n): ")
        result = app.invoke(None, config=config) if choice.lower() == "y" else "Note discarded."
        return result
    return None
while True:
    query = input("Ask me anything (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    config = {
        "configurable": {
            "thread_id": 1,
        }
    }

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
    save_note(config)

    print(result.get("answer", result))