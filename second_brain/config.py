"""
Configuration module for the UCGA Second Brain.

Provides a centralized dataclass-based configuration for all subsystems
including LLM, memory, cognitive loop, and tool settings.

Author: Aman Singh
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class SecondBrainConfig:
    """Central configuration for the Second Brain cognitive architecture.

    Attributes:
        llm_provider: LLM backend to use ("ollama", "openai", "mock", "auto").
        llm_model: Model name; auto-detected if left empty.
        mock_mode: If True, use mock LLM responses for testing.
        data_dir: Root directory for all persisted data.
        chroma_collection: Name of the ChromaDB collection for episodic memory.
        knowledge_graph_file: Filename for the serialized knowledge graph.
        embedding_model: Sentence-transformer model name for embeddings.
        max_cognitive_loops: Maximum iterations for the cognitive loop.
        critic_enabled: Whether the internal critic is active.
        tools_enabled: List of tool names available to the agent.
        files_output_dir: Subdirectory name for file outputs.
    """

    # LLM
    llm_provider: str = "auto"  # "ollama", "openai", "mock", "auto"
    llm_model: str = ""  # auto-detected if empty
    mock_mode: bool = False

    # Memory
    data_dir: Path = Path("data/second_brain")
    chroma_collection: str = "episodic_memory"
    knowledge_graph_file: str = "knowledge_graph.json"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Cognitive loop
    max_cognitive_loops: int = 5
    critic_enabled: bool = True

    # Tools
    tools_enabled: list = field(
        default_factory=lambda: ["python_interpreter", "web_search", "file_writer"]
    )
    files_output_dir: str = "files"

    @property
    def chroma_persist_dir(self) -> Path:
        """Path to the ChromaDB persistence directory."""
        return self.data_dir / "chroma_db"

    @property
    def kg_path(self) -> Path:
        """Path to the knowledge graph JSON file."""
        return self.data_dir / self.knowledge_graph_file

    @property
    def files_dir(self) -> Path:
        """Path to the file output directory."""
        return self.data_dir / self.files_output_dir

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't already exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
