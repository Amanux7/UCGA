"""
Semantic Memory - Knowledge Graph.

Wraps NetworkX to provide a persistent directed knowledge graph for semantic
memory. Entities are stored as nodes and relations as directed edges.
Falls back gracefully when NetworkX is not installed.

Author: Aman Singh
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from second_brain.config import SecondBrainConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Attempt to import NetworkX; set a flag for graceful fallback               #
# --------------------------------------------------------------------------- #
try:
    import networkx as nx
    from networkx.readwrite import json_graph

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning(
        "networkx is not installed. KnowledgeGraph will use a basic in-memory "
        "fallback. Install networkx for full graph capabilities: pip install networkx"
    )


# =========================================================================== #
# In-memory fallback graph                                                    #
# =========================================================================== #
class _InMemoryGraph:
    """Minimal adjacency-dict fallback when NetworkX is unavailable."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

    def add_node(self, name: str, **attrs: Any) -> None:
        if name in self.nodes:
            self.nodes[name].update(attrs)
        else:
            self.nodes[name] = dict(attrs)

    def add_edge(self, source: str, target: str, relation: str, **attrs: Any) -> None:
        # Ensure both nodes exist
        if source not in self.nodes:
            self.nodes[source] = {}
        if target not in self.nodes:
            self.nodes[target] = {}
        self.edges.append(
            {"source": source, "target": target, "relation": relation, **attrs}
        )

    def get_node(self, name: str) -> dict[str, Any] | None:
        return self.nodes.get(name)

    def get_edges_for(self, name: str) -> list[dict[str, Any]]:
        return [
            e for e in self.edges if e["source"] == name or e["target"] == name
        ]

    def neighbors(self, name: str, depth: int = 2) -> list[dict[str, Any]]:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(name, 0)])
        results: list[dict[str, Any]] = []
        while queue:
            current, d = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current != name:
                results.append(
                    {"name": current, "attributes": self.nodes.get(current, {}), "depth": d}
                )
            if d < depth:
                for e in self.edges:
                    if e["source"] == current and e["target"] not in visited:
                        queue.append((e["target"], d + 1))
                    elif e["target"] == current and e["source"] not in visited:
                        queue.append((e["source"], d + 1))
        return results

    def search(self, keyword: str) -> list[str]:
        kw = keyword.lower()
        return [n for n in self.nodes if kw in n.lower()]

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_InMemoryGraph":
        g = cls()
        g.nodes = data.get("nodes", {})
        g.edges = data.get("edges", [])
        return g


# =========================================================================== #
# KnowledgeGraph – public API                                                 #
# =========================================================================== #
class KnowledgeGraph:
    """Semantic memory backed by a NetworkX directed graph.

    Stores entities as nodes and relations as directed edges. The graph
    is serialized to / deserialized from a JSON file on disk.

    Args:
        config: A ``SecondBrainConfig`` instance providing the graph
                file path.
    """

    def __init__(self, config: "SecondBrainConfig") -> None:
        self._config = config
        self._kg_path: Path = config.kg_path
        config.ensure_dirs()

        if NETWORKX_AVAILABLE:
            self._graph: nx.DiGraph = nx.DiGraph()  # type: ignore[name-defined]
            self._fallback: _InMemoryGraph | None = None
        else:
            self._graph = None  # type: ignore[assignment]
            self._fallback = _InMemoryGraph()

        # Attempt to load existing graph from disk
        self.load()

    # ----- public methods -------------------------------------------------- #

    def add_entity(self, name: str, **attributes: Any) -> None:
        """Add or update an entity (node) in the graph.

        Args:
            name: Unique entity identifier.
            **attributes: Arbitrary key-value attributes for the entity.
        """
        if self._fallback is not None:
            self._fallback.add_node(name, **attributes)
        else:
            self._graph.add_node(name, **attributes)
        logger.debug("Entity added/updated: %s", name)
        self.save()

    def add_relation(
        self, source: str, target: str, relation: str, **attributes: Any
    ) -> None:
        """Add a directed relation (edge) between two entities.

        If the source or target entities do not exist yet, they are created
        automatically.

        Args:
            source: Source entity name.
            target: Target entity name.
            relation: Label describing the relationship.
            **attributes: Additional edge attributes.
        """
        if self._fallback is not None:
            self._fallback.add_edge(source, target, relation, **attributes)
        else:
            # Ensure both nodes exist
            if not self._graph.has_node(source):
                self._graph.add_node(source)
            if not self._graph.has_node(target):
                self._graph.add_node(target)
            self._graph.add_edge(source, target, relation=relation, **attributes)
        logger.debug("Relation added: %s -[%s]-> %s", source, relation, target)
        self.save()

    def query_entity(self, name: str) -> dict[str, Any] | None:
        """Query an entity and its relations.

        Args:
            name: The entity name to look up.

        Returns:
            A dict with ``name``, ``attributes``, and ``relations`` keys,
            or ``None`` if the entity does not exist.
        """
        if self._fallback is not None:
            attrs = self._fallback.get_node(name)
            if attrs is None:
                return None
            edges = self._fallback.get_edges_for(name)
            return {"name": name, "attributes": attrs, "relations": edges}

        if not self._graph.has_node(name):
            return None

        attrs = dict(self._graph.nodes[name])
        relations: list[dict[str, Any]] = []

        # Outgoing edges
        for _, target, data in self._graph.out_edges(name, data=True):
            relations.append(
                {"source": name, "target": target, "relation": data.get("relation", ""), **{k: v for k, v in data.items() if k != "relation"}}
            )
        # Incoming edges
        for source, _, data in self._graph.in_edges(name, data=True):
            relations.append(
                {"source": source, "target": name, "relation": data.get("relation", ""), **{k: v for k, v in data.items() if k != "relation"}}
            )

        return {"name": name, "attributes": attrs, "relations": relations}

    def query_neighbors(self, name: str, depth: int = 2) -> list[dict[str, Any]]:
        """BFS traversal returning connected entities up to a given depth.

        Args:
            name: Starting entity name.
            depth: Maximum traversal depth (default 2).

        Returns:
            A list of dicts, each with ``name``, ``attributes``, and
            ``depth`` keys.
        """
        if self._fallback is not None:
            return self._fallback.neighbors(name, depth)

        if not self._graph.has_node(name):
            return []

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(name, 0)])
        results: list[dict[str, Any]] = []

        while queue:
            current, d = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current != name:
                results.append(
                    {
                        "name": current,
                        "attributes": dict(self._graph.nodes[current]),
                        "depth": d,
                    }
                )

            if d < depth:
                # Follow both outgoing and incoming edges for full traversal
                for neighbor in self._graph.successors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))
                for neighbor in self._graph.predecessors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))

        return results

    def search(self, keyword: str) -> list[str]:
        """Case-insensitive search across entity names.

        Args:
            keyword: The search term.

        Returns:
            A list of matching entity names.
        """
        if self._fallback is not None:
            return self._fallback.search(keyword)

        kw = keyword.lower()
        return [n for n in self._graph.nodes if kw in n.lower()]

    def save(self) -> None:
        """Serialize the graph to JSON on disk."""
        try:
            if self._fallback is not None:
                data = self._fallback.to_dict()
            else:
                data = json_graph.node_link_data(self._graph)

            self._kg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._kg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("Knowledge graph saved to %s", self._kg_path)
        except Exception:
            logger.exception("Failed to save knowledge graph to %s", self._kg_path)

    def load(self) -> None:
        """Deserialize the graph from JSON on disk.

        If the file does not exist or is invalid, start with an empty graph.
        """
        if not self._kg_path.exists():
            logger.info(
                "No existing knowledge graph at %s; starting fresh.", self._kg_path
            )
            return

        try:
            with open(self._kg_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if self._fallback is not None:
                self._fallback = _InMemoryGraph.from_dict(data)
            else:
                self._graph = json_graph.node_link_graph(data)
                # Ensure it's a DiGraph
                if not isinstance(self._graph, nx.DiGraph):
                    self._graph = nx.DiGraph(self._graph)

            logger.info(
                "Knowledge graph loaded from %s (%d entities, %d relations).",
                self._kg_path,
                self.get_stats()["entities"],
                self.get_stats()["relations"],
            )
        except Exception:
            logger.exception(
                "Failed to load knowledge graph from %s; starting fresh.",
                self._kg_path,
            )
            if self._fallback is not None:
                self._fallback = _InMemoryGraph()
            else:
                self._graph = nx.DiGraph()

    def get_stats(self) -> dict[str, int]:
        """Return summary statistics for the graph.

        Returns:
            A dict with ``entities`` (node count) and ``relations``
            (edge count) keys.
        """
        if self._fallback is not None:
            return {
                "entities": self._fallback.num_nodes(),
                "relations": self._fallback.num_edges(),
            }
        return {
            "entities": self._graph.number_of_nodes(),
            "relations": self._graph.number_of_edges(),
        }
