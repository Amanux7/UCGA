"""
UCGA Second Brain - Memory Package.

Provides episodic memory (VectorStore) and semantic memory (KnowledgeGraph)
implementations for the cognitive architecture.

Author: Aman Singh
"""

from second_brain.memory.vector_store import VectorStore
from second_brain.memory.knowledge_graph import KnowledgeGraph

__all__ = ["VectorStore", "KnowledgeGraph"]
