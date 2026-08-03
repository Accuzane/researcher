from agents.base import run_agent_loop

NOTE_WRITER_PROMPT = """
You are a note-taking assistant.

Format every note in Markdown.

Use this structure:

# {Topic}

## Summary

...

## Key Points

- ...
- ...
- ...

## Sources

- ...

Then call the save_note tool with the formatted note.
"""


def note_writer(topic: str, content: str, llm_with_tools: any, tools: list) -> str:
    """
    A note-taking agent that saves research summaries to local files.
    """
    user_input = f"Topic: {topic}\nContent: {content}"
    return run_agent_loop(NOTE_WRITER_PROMPT, user_input, llm_with_tools, tools, agent_name="notes_writer")