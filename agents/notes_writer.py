import logging
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# System prompt for the note writer agent
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
    
    messages = [
        SystemMessage(content=NOTE_WRITER_PROMPT),
        HumanMessage(content=f"Topic: {topic}\nContent: {content}")
    ]
    
    while True:
        # Ask the LLM what to do
        response = llm_with_tools.invoke(messages)

        # Save the response into memory
        messages.append(response)

        # Did the model request a tool?
        if response.tool_calls:
            for tool_call in response.tool_calls:
                logging.info(f"Tool call requested: {tool_call['name']} with args: {tool_call['args']}")

                # Find the requested tool
                tool = next(
                   ( tool
                    for tool in tools
                    if tool.name == tool_call["name"]
                   ),
                   None
                )
                
                if tool is None:
                    raise ValueError(f"Tool {tool_call['name']} not found in available tools.")

                # Run the tool
                tool_result = tool.invoke(
                    tool_call["args"]
                )
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )
        else:
            return response.content    



