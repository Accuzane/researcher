from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import logging


# System prompt for the researcher agent
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
- make sure the data is up to date and relevant to the query.
"""



# researcher agent
def researcher(question: str, llm_with_tools: any, tools: list) -> str:
    """
    A researcher agent that can search the web and answer questions.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
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
                    (tool
                    for tool in tools
                    if tool.name == tool_call["name"]),
                    None
                )
                
                if tool is None:
                    raise ValueError(f"Tool {tool_call['name']} not found.")


                # Run the tool
                tool_result = tool.invoke(
                    tool_call["args"]
                )


                logging.info("Search completed.")


                # Give result back to LLM
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )


        else:
            # the final response
            return response.content

            
            


