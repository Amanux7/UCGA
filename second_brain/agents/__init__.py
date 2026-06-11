"""
UCGA Second Brain - Agents Layer

Provides the cognitive agents for the Second Brain architecture:
- CriticAgent: Metacognitive reviewer for draft solutions
- Orchestrator: Central dual-process cognitive loop coordinator

Author: Aman Singh
"""

from second_brain.agents.critic import CriticAgent
from second_brain.agents.orchestrator import Orchestrator

__all__ = ["CriticAgent", "Orchestrator"]
