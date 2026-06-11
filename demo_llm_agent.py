"""
demo_llm_agent.py — Demo of UCGALPGAgent with tools.

Demonstrates a single LLM-powered agent with local/premium compatibility
executing a multi-step task involving a calculator tool and memory retrieval.

Author: Antigravity AI
"""

import sys
import os
sys.path.insert(0, ".")

import torch
from agents.llm_agent import UCGALPGAgent
from agents.tool_registry import Tool, ToolRegistry

# Force UTF-8 encoding on stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def calculator_tool(expression: str) -> str:
    """Evaluates a basic math expression safely."""
    # Clean expression of any non-math characters
    cleaned = "".join(c for c in expression if c in "0123456789+-*/(). ")
    try:
        # Evaluate math expression safely
        res = eval(cleaned, {"__builtins__": None}, {})
        return str(res)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def main():
    print("=" * 70)
    print("🚀 UCGA LLM-Powered Cognitive Agent Demo")
    print("=" * 70)

    # 1. Initialize Tool Registry and register a calculator tool
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluates mathematical expressions. Input: string like '2 + 2'.",
            func=calculator_tool,
        )
    )

    # 2. Instantiate the agent in mock mode (works offline without API keys)
    agent = UCGALPGAgent(
        name="JARvis-LLM",
        instructions="You are a precise reasoning assistant. Solve queries using tools when needed.",
        tool_registry=registry,
        mock=True,
    )

    print(f"\nAgent initialized: {agent.name}")
    print(registry.list_tools())

    # 3. Add some persistent memories
    agent.persistent_memory.append("Fact: the target value to multiply by is 10.")
    agent.persistent_memory.append("Context: user is testing math capabilities.")

    # 4. Run a query
    query = "Calculate 2 + 2 and retrieve the target value from memory to multiply the result by."
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    final_response = agent.run(query)

    print("-" * 50)
    print("Final Agent Response:")
    print(final_response)

    print("\n✓ LLM agent demo complete.")


if __name__ == "__main__":
    main()
