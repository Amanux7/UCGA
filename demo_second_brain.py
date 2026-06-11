"""
demo_second_brain.py — Standalone demonstration of the UCGA Second Brain.

Pre-populates the memory stores and runs a query through the 4-step dual-process
cognitive loop (Context Ingestion, Reasoning, Critique, and Execution).
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from second_brain.config import SecondBrainConfig
from second_brain.agents.orchestrator import Orchestrator


def main():
    print("=" * 70)
    print("🧠 UCGA SECOND BRAIN — COGNITIVE ARCHITECTURE DEMO")
    print("=" * 70)

    # 1. Configuration (Mock mode for offline demo run)
    config = SecondBrainConfig(
        llm_provider="mock",
        mock_mode=True,
        critic_enabled=True,
        data_dir=Path("data/second_brain_demo_run"),
    )
    
    # Ensure directories are created
    config.ensure_dirs()

    # 2. Initialize the Central Orchestrator
    orchestrator = Orchestrator(config)
    
    # Clear any old data in demo folder to start fresh
    orchestrator.vector_store.clear()
    
    print("\n[Memory Ingestion] Pre-populating semantic and episodic memory stores...")
    
    # 3. Add entities and relations to Knowledge Graph (Semantic Memory)
    orchestrator.knowledge_graph.add_entity(
        "UCGA", 
        type="Cognitive Architecture", 
        description="Unified Cognitive Graph Architecture, a modular, differentiable cognitive framework."
    )
    orchestrator.knowledge_graph.add_entity(
        "SecondBrain", 
        type="System", 
        description="Dual-process implementation with episodic vector DB, semantic KG, and metacognitive critique."
    )
    orchestrator.knowledge_graph.add_relation("SecondBrain", "UCGA", "extends")
    
    # 4. Add facts to Vector Store (Episodic Memory)
    orchestrator.vector_store.add("Episode 1: The user initialized the UCGA cognitive loop.")
    orchestrator.vector_store.add("Episode 2: Second Brain was configured with PythonInterpreter, FileWriter, and WebSearch tools.")
    
    print(f"  Episodic Memory Size: {orchestrator.vector_store.count()} items")
    print(f"  Semantic Memory Size: {orchestrator.knowledge_graph.get_stats()['entities']} entities")

    # 5. Run a user query
    query = "Inspect the relationship between SecondBrain and UCGA, and run a python code snippet to compute 100 * 5."
    print(f"\n[Processing Query]: '{query}'")
    print("-" * 70)

    # To show the complete trace in mock mode, we mock the completions.
    # Turn 1: Orchestrator runs python_interpreter tool
    # Turn 2: Orchestrator provides draft solution
    # Critic approves draft (mocked/simulated) and Orchestrator outputs final result
    mock_responses = [
        (
            "<thinking>\n"
            "I need to query the relationship between SecondBrain and UCGA in the retrieved memory.\n"
            "Retrieved memory indicates that 'SecondBrain extends UCGA'.\n"
            "Now, I must run the python code to compute 100 * 5.\n"
            "</thinking>\n"
            "<tool_call>\n"
            '{"tool": "python_interpreter", "arguments": {"code": "print(100 * 5)"}}\n'
            "</tool_call>"
        ),
        (
            "<thinking>\n"
            "The python interpreter tool returned '500'.\n"
            "SecondBrain extends UCGA and the calculation result is 500.\n"
            "I will generate the draft solution.\n"
            "</thinking>\n"
            "<draft_solution>\n"
            "SecondBrain is a dual-process system that extends UCGA. The product of 100 and 5 is 500.\n"
            "</draft_solution>\n"
            "<final_output>\n"
            "Based on retrieved memory, SecondBrain extends the UCGA cognitive architecture. "
            "The mathematical computation of 100 * 5 executed successfully via the python_interpreter, returning 500."
            "</final_output>"
        )
    ]
    
    # Mock the LLM completion
    from unittest.mock import MagicMock
    orchestrator.llm.completion = MagicMock(side_effect=mock_responses)
    
    # Mock critic feedback to return APPROVED
    if orchestrator.critic:
        orchestrator.critic.review = MagicMock(return_value={
            "status": "APPROVED",
            "issues": [],
            "suggestions": ["Draft looks accurate."],
            "raw": "<critic_feedback><status>APPROVED</status></critic_feedback>"
        })

    # Run the loop!
    response = orchestrator.process(query)
    
    print(response)
    print("-" * 70)

    # 6. Show updated stats
    stats = orchestrator.get_memory_stats()
    print("\n📊 Updated Memory Stats:")
    print(f"  Episodic Entries: {stats['episodic_memory_count']}")
    print(f"  Knowledge Graph:  {stats['knowledge_graph']}")
    print("\n✓ Second Brain demo run completed successfully!")


if __name__ == "__main__":
    main()
