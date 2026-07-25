from graph.supervisor import app

def save_note(config):
    choice = input("Do you want to save this note? (y/n): ")
    result = app.invoke(None, config=config) if choice.lower() == "y" else "Note discarded."
    return result

config = {
    "configurable": {
        "thread_id": 1,
    }
}

state = {
    "query": input("Ask me anything: "),
    "topic": "",
    "research": "",
    "answer": "",
    "saved_path": "",
    "next_agent": "",
    "needs_research": False,
    "retry_count": 0,
}

result = app.invoke(state, config=config)
print(save_note(config))