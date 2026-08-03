import logging
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def run_agent_loop(system_prompt: str, user_input: str, llm_with_tools: any, tools: list, agent_name: str = "agent") -> str:
    """
    Shared ReAct loop used by every tool-calling agent.
    Sends messages to the model, runs any requested tools, and repeats
    until the model returns a final answer instead of a tool request.
    """
    logger = logging.getLogger(agent_name)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            logger.info(f"calling {tool_call['name']}({tool_call['args']})")

            tool = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool is None:
                logger.error(f"tool '{tool_call['name']}' not found")
                messages.append(ToolMessage(
                    content=f"Error: tool '{tool_call['name']}' does not exist.",
                    tool_call_id=tool_call["id"]
                ))
                continue

            tool_result = tool.invoke(tool_call["args"])
            logger.info(f"{tool_call['name']} returned: {str(tool_result)[:80]}")

            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))