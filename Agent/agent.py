"""
LangGraph ReAct Agent with Tool Calling.

This module implements a simple ReAct agent using LangGraph with three tools:
- get_weather: Returns weather information for a city
- calculate: Evaluates mathematical expressions
- search_knowledge_base: Searches company policies and products

Usage:
    from agent import run_agent

    result = run_agent("What's the weather in San Francisco?")
    print(result["output"])
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# =============================================================================
# Tools
# =============================================================================


@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""
    weather_data = {
        "new york": "New York: 45°F, cloudy with a chance of rain.",
        "san francisco": "San Francisco: 62°F, sunny and clear.",
        "london": "London: 50°F, overcast with light drizzle.",
        "tokyo": "Tokyo: 55°F, partly cloudy.",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}."
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 3 * 4'"""
    try:
        # Restricted eval for safety
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for company policies and product information."""
    knowledge_base = {
        "refund": (
            "Our refund policy allows returns within 30 days for a full refund. "
            "Items must be unused and in original packaging."
        ),
        "shipping": (
            "Standard shipping takes 5-7 business days. "
            "Express shipping (2-day) is available for $9.99."
        ),
        "warranty": (
            "All electronics come with a 1-year manufacturer warranty. "
            "Extended warranty available for $49.99."
        ),
        "hours": (
            "Customer support is available Monday-Friday 9AM-6PM EST. "
            "Weekend support via email only."
        ),
    }
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return "No relevant information found in the knowledge base."


# =============================================================================
# Agent Factory
# =============================================================================

TOOLS = [get_weather, calculate, search_knowledge_base]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools to answer "
    "user questions accurately and concisely."
)


def create_agent(model_name: str = "gpt-4o-mini") -> Any:
    """
    Create and return a LangGraph ReAct agent.

    Args:
        model_name: OpenAI model to use for the agent.

    Returns:
        Configured ReAct agent instance.
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    return create_react_agent(llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(query: str, model_name: str = "gpt-4o-mini") -> dict:
    """
    Run the agent on a query and return structured results.

    Args:
        query: The user's question.
        model_name: OpenAI model to use.

    Returns:
        Dictionary with keys:
            - input: Original query
            - output: Final text response
            - tools_called: List of tool calls with name, args, and output
    """
    agent = create_agent(model_name)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Extract tool calls from message history
    tools_called = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append({
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"],
                })
        # Match tool outputs to tool calls
        if msg.type == "tool":
            for t in tools_called:
                if t["id"] == msg.tool_call_id:
                    t["output"] = msg.content
                    break

    return {
        "input": query,
        "output": result["messages"][-1].content,
        "tools_called": tools_called,
    }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import json

    demo_queries = [
        "What's the weather in San Francisco?",
        "Calculate 15% tip on a $85 dinner bill.",
        "What is your refund policy?",
    ]

    for query in demo_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        result = run_agent(query)
        print(f"Output: {result['output']}")
        print(f"Tools: {json.dumps(result['tools_called'], indent=2)}")
