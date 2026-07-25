from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import logging

QA_PROMPT = """
You are a knowledge base assistant. You answer questions using notes that
have been previously saved. Always call load_note or list_notes first before
answering. If no relevant note exists, say so clearly — do not make up an answer.
"""

def qa_agent(topic: str, llm_with_tools: any, tools: list) -> str:
    """
    A question-answering agent that uses saved notes to answer questions.
    """
    
    messages = [
        SystemMessage(content=QA_PROMPT),
        HumanMessage(content=f"Question: {topic}")
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
                    logging.error(f"Tool {tool_call['name']} not found.")
                    continue

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