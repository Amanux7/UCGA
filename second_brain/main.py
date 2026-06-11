"""
UCGA Second Brain — Interactive CLI

Provides a REPL-style command-line interface for interacting with the
dual-process cognitive architecture. Supports slash-commands for memory
inspection, knowledge graph management, and runtime configuration.

Usage:
    python -m second_brain.main [--mock] [--provider {ollama,openai,mock}]
                                [--model MODEL] [--no-critic] [--data-dir DIR]

Author: Aman Singh
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path so that both top-level packages
# (``agents``, ``second_brain``) are importable when running directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from second_brain.config import SecondBrainConfig
from second_brain.agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║          🧠  UCGA SECOND BRAIN  🧠                      ║
║     Dual-Process Cognitive Architecture v1.0             ║
║                                                          ║
║  Commands:                                               ║
║    /memory   — Show memory statistics                    ║
║    /graph    — Show knowledge graph entities              ║
║    /learn    — Add a fact to the knowledge graph          ║
║    /clear    — Clear episodic memory                      ║
║    /help     — Show this help                             ║
║    /quit     — Exit                                       ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="UCGA Second Brain CLI")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no LLM needed)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "mock"],
        default="auto",
        help="LLM provider to use (default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="LLM model name (auto-detected if empty)",
    )
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable the Critic Agent",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/second_brain",
        help="Data directory for persisted state",
    )
    return parser.parse_args()


def handle_command(cmd: str, orchestrator: Orchestrator) -> None:
    """Handle a slash-command entered by the user.

    Args:
        cmd: The full command string (including leading ``/``).
        orchestrator: The active Orchestrator instance.
    """
    if cmd in ("/quit", "/exit"):
        print("Goodbye! 🧠")
        sys.exit(0)

    elif cmd == "/memory":
        stats = orchestrator.get_memory_stats()
        print(
            f"\n📊 Memory Stats:\n"
            f"  Episodic: {stats['episodic_memory_count']} entries\n"
            f"  Knowledge Graph: {stats['knowledge_graph']}\n"
        )

    elif cmd == "/graph":
        stats = orchestrator.knowledge_graph.get_stats()
        print(f"\n🕸️  Knowledge Graph: {stats}")
        # Show up to 20 nodes if the internal graph is accessible
        if hasattr(orchestrator.knowledge_graph, "_graph"):
            nodes = list(orchestrator.knowledge_graph._graph.nodes())[:20]
            for node in nodes:
                print(f"  • {node}")
        print()

    elif cmd.startswith("/learn"):
        # Syntax: /learn entity_name: fact description
        parts = cmd[6:].strip().split(":", 1)
        if len(parts) == 2:
            name, fact = parts[0].strip(), parts[1].strip()
            orchestrator.knowledge_graph.add_entity(name, description=fact)
            orchestrator.knowledge_graph.save()
            print(f"  ✅ Learned: {name} → {fact}")
        else:
            print("  Usage: /learn entity_name: description")

    elif cmd == "/clear":
        # Guard: only call clear if available
        if hasattr(orchestrator.vector_store, "clear"):
            orchestrator.vector_store.clear()
            print("  🗑️  Episodic memory cleared.")
        else:
            print("  ⚠️ clear() not supported by the current vector store.")

    elif cmd == "/help":
        print(BANNER)

    else:
        print(f"  Unknown command: {cmd}. Type /help for options.")


def main() -> None:
    """Entry point: parse args, build config, run interactive REPL."""
    args = parse_args()

    config = SecondBrainConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        mock_mode=args.mock or args.provider == "mock",
        critic_enabled=not args.no_critic,
        data_dir=Path(args.data_dir),
    )

    print(BANNER)
    print(f"  Provider: {config.llm_provider} | Model: {config.llm_model or 'auto'}")
    print(f"  Critic:   {'ON' if config.critic_enabled else 'OFF'}")
    print(f"  Mock:     {'ON' if config.mock_mode else 'OFF'}")
    print()

    orchestrator = Orchestrator(config)

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 🧠")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(user_input, orchestrator)
            continue

        print("\n⏳ Processing through cognitive loop...\n")
        try:
            response = orchestrator.process(user_input)
            print(f"\n🧠 Second Brain:\n{response}\n")
        except Exception as exc:
            logger.error("Processing error: %s", exc, exc_info=True)
            print(f"\n❌ Error: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
