"""
demo_multi_agent.py — Demo of Multi-Agent cooperative workflows.

Demonstrates collaborative problem solving with a writer agent and an
editor/critic agent passing context via the shared Message Bus.

Author: Antigravity AI
"""

import sys
import os
sys.path.insert(0, ".")

from agents.multi_agent import MultiAgentOrchestrator

# Force UTF-8 encoding on stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def main():
    print("=" * 70)
    print("🚀 UCGA Multi-Agent Collaboration Demo")
    print("=" * 70)

    # 1. Initialize orchestrator in mock/test mode
    orchestrator = MultiAgentOrchestrator(mock=True)

    # 2. Register specialized agents
    orchestrator.register_agent(
        role="writer",
        instructions="You are a creative writer. Write a short paragraph about the requested topic.",
    )
    orchestrator.register_agent(
        role="editor",
        instructions="You are an editor. Critique the input draft and rewrite it to make it more professional.",
    )

    # 3. Define the collaborative workflow
    workflow = ["writer", "editor"]
    topic = "The beauty of recursive cognitive neural graphs."
    print(f"\nTask: Write and edit a paragraph about: '{topic}'")
    print("-" * 50)

    # 4. Run the workflow
    final_output = orchestrator.run_collaborative_workflow(task=topic, workflow=workflow)

    print("-" * 50)
    print("Final Processed Output (from Editor):")
    print(final_output)

    print("\n✓ Multi-agent collaboration demo complete.")


if __name__ == "__main__":
    main()
