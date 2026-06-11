"""
Episodic Memory - Vector Store.

Wraps ChromaDB to provide vector-similarity-based episodic memory storage
and retrieval. Falls back to a simple in-memory list-based store when
ChromaDB is not installed.

Author: Aman Singh
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from second_brain.config import SecondBrainConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Attempt to import ChromaDB; set a flag for graceful fallback               #
# --------------------------------------------------------------------------- #
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning(
        "chromadb is not installed. VectorStore will use an in-memory fallback. "
        "Install chromadb for persistent vector storage: pip install chromadb"
    )


# =========================================================================== #
# In-memory fallback store                                                    #
# =========================================================================== #
class _InMemoryStore:
    """Minimal list-based fallback when ChromaDB is unavailable."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata})

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Naive substring search – no real vector similarity."""
        query_lower = query_text.lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._docs:
            text_lower = doc["text"].lower()
            # Simple relevance: ratio of query tokens found in document
            tokens = query_lower.split()
            hits = sum(1 for t in tokens if t in text_lower)
            score = hits / max(len(tokens), 1)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, doc in scored[:n_results]:
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "distance": 1.0 - score,  # lower distance = better match
                }
            )
        return results

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()

    def get_all(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"id": d["id"], "text": d["text"], "metadata": d["metadata"]}
            for d in self._docs[-limit:]
        ]


# =========================================================================== #
# VectorStore – public API                                                    #
# =========================================================================== #
class VectorStore:
    """Episodic memory backed by ChromaDB (or in-memory fallback).

    Stores text documents with metadata and supports vector-similarity
    retrieval.

    Args:
        config: A ``SecondBrainConfig`` instance that supplies persistence
                paths and collection names.
    """

    def __init__(self, config: SecondBrainConfig) -> None:
        self._config = config
        self._collection_name = config.chroma_collection

        if CHROMADB_AVAILABLE:
            config.ensure_dirs()
            persist_dir = str(config.chroma_persist_dir)
            logger.info("Initializing ChromaDB client at %s", persist_dir)
            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_dir,
                    anonymized_telemetry=False,
                )
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            self._fallback: _InMemoryStore | None = None
            logger.info(
                "ChromaDB collection '%s' ready (%d documents).",
                self._collection_name,
                self._collection.count(),
            )
        else:
            logger.info("Using in-memory fallback vector store.")
            self._client = None  # type: ignore[assignment]
            self._collection = None  # type: ignore[assignment]
            self._fallback = _InMemoryStore()

    # ----- public methods -------------------------------------------------- #

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a document to the store.

        Args:
            text: The document text to store.
            metadata: Optional metadata dict. A ``timestamp`` field is
                      automatically added if not present.

        Returns:
            The generated UUID string for the new document.
        """
        doc_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self._fallback is not None:
            self._fallback.add(doc_id, text, metadata)
            logger.debug("Added document %s to in-memory store.", doc_id)
        else:
            self._collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
            )
            logger.debug("Added document %s to ChromaDB.", doc_id)

        return doc_id

    def query(
        self, query_text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Query the store for similar documents.

        Args:
            query_text: The query string to search for.
            n_results: Maximum number of results to return.

        Returns:
            A list of dicts, each with keys ``id``, ``text``, ``metadata``,
            and ``distance``.
        """
        if self._fallback is not None:
            return self._fallback.query(query_text, n_results)

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(self._collection.count(), 1)),
        )

        output: list[dict[str, Any]] = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            for i, doc_id in enumerate(ids):
                output.append(
                    {
                        "id": doc_id,
                        "text": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else None,
                    }
                )
        return output

    def count(self) -> int:
        """Return the number of stored documents."""
        if self._fallback is not None:
            return self._fallback.count()
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents and recreate the collection."""
        if self._fallback is not None:
            self._fallback.clear()
            logger.info("In-memory store cleared.")
        else:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            logger.info("ChromaDB collection '%s' cleared.", self._collection_name)

    def get_all(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent entries from the store.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            A list of dicts with ``id``, ``text``, and ``metadata`` keys.
        """
        if self._fallback is not None:
            return self._fallback.get_all(limit)

        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.get(limit=limit)
        output: list[dict[str, Any]] = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"]):
                output.append(
                    {
                        "id": doc_id,
                        "text": (
                            results["documents"][i]
                            if results.get("documents") and i < len(results["documents"])
                            else ""
                        ),
                        "metadata": (
                            results["metadatas"][i]
                            if results.get("metadatas") and i < len(results["metadatas"])
                            else {}
                        ),
                    }
                )
        return output
