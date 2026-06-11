"""
UCGA Second Brain - Central Orchestrator

Implements the 4-step dual-process cognitive loop:
  1. Context Ingestion  — retrieve from episodic + semantic memory
  2. Chain-of-Thought    — structured <thinking> reasoning
  3. Draft + Critique    — draft solution reviewed by CriticAgent
  4. Tool Use / Output   — execute tools or emit <final_output>

Author: Aman Singh
"""

import json
import re
import logging
from typing import Any, Optional

from agents.llm_client import UnifiedLLMClient
from second_brain.config import SecondBrainConfig
from second_brain.memory.vector_store import VectorStore
from second_brain.memory.knowledge_graph import KnowledgeGraph
from second_brain.tools.python_interpreter import PythonInterpreter
from second_brain.tools.web_search import WebSearch
from second_brain.tools.file_writer import FileWriter
from second_brain.agents.critic import CriticAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central Orchestrator implementing the dual-process cognitive loop.

    Coordinates memory retrieval, LLM reasoning, critic review, and tool
    execution in an iterative loop up to ``max_cognitive_loops`` iterations.
    """

    SYSTEM_PROMPT: str = (
        "You are the Central Orchestrator of a dual-process cognitive architecture. "
        "Your primary function is to process user inputs by coordinating external "
        "memory, executing multi-step reasoning, integrating metacognitive feedback, "
        "and triggering external tools.\n\n"
        "You do not guess facts. You rely strictly on retrieved context. You must "
        "expose your reasoning process using specific XML tags.\n\n"
        "## YOUR WORKFLOW\n\n"
        "### STEP 1: CONTEXT INGESTION\n"
        "The system provides you with retrieved memory in <retrieved_memory> tags. "
        "Prioritize this data over your pre-trained weights. If insufficient, state "
        "this explicitly.\n\n"
        "### STEP 2: CHAIN-OF-THOUGHT REASONING\n"
        "Use <thinking> tags to show your step-by-step reasoning. Be explicit about "
        "what you know, what you don't know, and what assumptions you're making.\n\n"
        "### STEP 3: DRAFT SOLUTION\n"
        "After reasoning, produce a <draft_solution> with your proposed answer. "
        "This will be reviewed by a Critic Agent.\n\n"
        "### STEP 4: TOOL USE (if needed)\n"
        "If you need to execute code, search the web, or write files, use "
        "<tool_call> tags:\n"
        "<tool_call>\n"
        '{"tool": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "</tool_call>\n\n"
        "Available tools: python_interpreter (execute Python code), "
        "web_search (search the web), file_writer (write to a file)\n\n"
        "### FINAL OUTPUT\n"
        "When you have your complete, verified answer, wrap it in "
        "<final_output> tags."
    )

    def __init__(self, config: SecondBrainConfig) -> None:
        """Initialize all subsystems from the given configuration.

        Args:
            config: A populated ``SecondBrainConfig`` instance.
        """
        self.config = config
        config.ensure_dirs()

        # ── LLM ─────────────────────────────────────────────────────────
        self.llm = UnifiedLLMClient(
            provider=config.llm_provider if config.llm_provider != "auto" else None,
            model=config.llm_model or None,
            mock=config.mock_mode,
        )

        # ── Memory ──────────────────────────────────────────────────────
        self.vector_store = VectorStore(config)
        self.knowledge_graph = KnowledgeGraph(config)

        # ── Tools ───────────────────────────────────────────────────────
        self.tools: dict[str, Any] = {}
        if "python_interpreter" in config.tools_enabled:
            self.tools["python_interpreter"] = PythonInterpreter()
        if "web_search" in config.tools_enabled:
            self.tools["web_search"] = WebSearch()
        if "file_writer" in config.tools_enabled:
            self.tools["file_writer"] = FileWriter(str(config.files_dir))

        # ── Critic ──────────────────────────────────────────────────────
        self.critic: Optional[CriticAgent] = (
            CriticAgent(self.llm) if config.critic_enabled else None
        )

        # ── Conversation history (multi-turn) ───────────────────────────
        self.conversation_history: list[dict[str, str]] = []

        logger.info(
            "Orchestrator initialised (provider=%s, model=%s, critic=%s, tools=%s)",
            self.llm.provider,
            self.llm.model,
            "ON" if self.critic else "OFF",
            list(self.tools.keys()),
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def process(self, user_query: str) -> str:
        """Process a user query through the full cognitive loop.

        Steps executed per iteration:
        1. Retrieve episodic + semantic memory context.
        2. Build the LLM message sequence.
        3. Loop up to ``max_cognitive_loops`` times:
           a. Call the LLM.
           b. If a ``<tool_call>`` is found → execute and inject result.
           c. If a ``<draft_solution>`` is found and critic is enabled →
              send to CriticAgent; if NEEDS_REVISION, inject feedback.
           d. If a ``<final_output>`` is found → store and return.
           e. Otherwise treat the raw response as final.

        Args:
            user_query: The user's natural-language query.

        Returns:
            The formatted final response string.
        """
        # STEP 1: Retrieve memory context
        retrieved_memory = self._retrieve_context(user_query)

        # Build initial message list
        messages = self._build_messages(user_query, retrieved_memory)

        # ── Cognitive Loop ──────────────────────────────────────────────
        for iteration in range(self.config.max_cognitive_loops):
            logger.debug("Cognitive loop iteration %d", iteration + 1)

            # Get LLM response
            response = self.llm.completion(messages, temperature=0.3)

            # Parse structured XML tags
            parsed = self._parse_response(response)

            # ── Tool call? ──────────────────────────────────────────────
            if parsed.get("tool_call"):
                tool_result = self._execute_tool(parsed["tool_call"])
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"<tool_result>{tool_result}</tool_result>",
                })
                continue

            # ── Draft → Critic review ───────────────────────────────────
            if parsed.get("draft_solution") and self.critic:
                feedback = self.critic.review(
                    user_query, retrieved_memory, parsed["draft_solution"]
                )
                if feedback["status"] == "NEEDS_REVISION":
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"<critic_feedback>{feedback['raw']}</critic_feedback>\n"
                            "Please revise your solution based on this feedback "
                            "and provide a <final_output>."
                        ),
                    })
                    continue

            # ── Final output ────────────────────────────────────────────
            if parsed.get("final_output"):
                final = parsed["final_output"]
                self._store_interaction(user_query, final)
                self.conversation_history.append(
                    {"role": "user", "content": user_query}
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": final}
                )
                return self._format_output(response, parsed)

            # ── Unstructured fallback ───────────────────────────────────
            self._store_interaction(user_query, response)
            self.conversation_history.append(
                {"role": "user", "content": user_query}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": response}
            )
            return response

        return (
            "⚠️ Reached maximum cognitive iterations without producing "
            "a final answer."
        )

    def get_memory_stats(self) -> dict[str, Any]:
        """Return statistics about both memory stores.

        Returns:
            A dict with ``episodic_memory_count`` and ``knowledge_graph``
            sub-dict.
        """
        return {
            "episodic_memory_count": self.vector_store.count(),
            "knowledge_graph": self.knowledge_graph.get_stats(),
        }

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _retrieve_context(self, query: str) -> str:
        """Retrieve context from both vector store and knowledge graph.

        Args:
            query: The search query.

        Returns:
            An XML-formatted ``<retrieved_memory>`` string.
        """
        parts: list[str] = []

        # ── Episodic (vector) memory ────────────────────────────────────
        results = self.vector_store.query(query, n_results=5)
        if results:
            parts.append("<episodic_memory>")
            for r in results:
                distance = r.get("distance", "N/A")
                parts.append(
                    f"  <memory distance='{distance}'>{r['text']}</memory>"
                )
            parts.append("</episodic_memory>")

        # ── Semantic (knowledge graph) memory ───────────────────────────
        kg_entities = self.knowledge_graph.search(query)
        if kg_entities:
            parts.append("<semantic_memory>")
            for entity_name in kg_entities[:5]:
                entity = self.knowledge_graph.query_entity(entity_name)
                if entity:
                    parts.append(
                        f"  <entity name='{entity_name}'>{entity}</entity>"
                    )
            parts.append("</semantic_memory>")

        if not parts:
            return "<retrieved_memory>No relevant memories found.</retrieved_memory>"

        return (
            "<retrieved_memory>\n" + "\n".join(parts) + "\n</retrieved_memory>"
        )

    def _build_messages(
        self, query: str, memory: str
    ) -> list[dict[str, str]]:
        """Build the LLM message list with system prompt, history, and query.

        Args:
            query: The current user query.
            memory: The formatted ``<retrieved_memory>`` XML string.

        Returns:
            A list of chat message dicts.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        # Add recent conversation history (last 6 turns)
        messages.extend(self.conversation_history[-6:])
        # Current query with memory context
        messages.append({
            "role": "user",
            "content": f"{memory}\n\nUser Query: {query}",
        })
        return messages

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse XML tags from the LLM response.

        Extracts ``<thinking>``, ``<draft_solution>``, ``<tool_call>``,
        ``<final_output>``, and ``<critic_feedback>`` blocks.

        Args:
            response: Raw LLM response text.

        Returns:
            A dict mapping tag names to their extracted content.
        """
        parsed: dict[str, Any] = {}

        # <thinking>
        m = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if m:
            parsed["thinking"] = m.group(1).strip()

        # <draft_solution>
        m = re.search(
            r"<draft_solution>(.*?)</draft_solution>", response, re.DOTALL
        )
        if m:
            parsed["draft_solution"] = m.group(1).strip()

        # <tool_call>
        m = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if m:
            try:
                parsed["tool_call"] = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse <tool_call> JSON: %s", m.group(1))
                parsed["tool_call"] = None

        # <final_output>
        m = re.search(
            r"<final_output>(.*?)</final_output>", response, re.DOTALL
        )
        if m:
            parsed["final_output"] = m.group(1).strip()

        # <critic_feedback>
        m = re.search(
            r"<critic_feedback>(.*?)</critic_feedback>", response, re.DOTALL
        )
        if m:
            parsed["critic_feedback"] = m.group(1).strip()

        return parsed

    def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result string.

        Args:
            tool_call: A dict with ``tool`` (name) and ``arguments`` keys.

        Returns:
            The tool output or an error message.
        """
        tool_name = tool_call.get("tool", "")
        args = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            return (
                f"Error: Tool '{tool_name}' not available. "
                f"Available: {list(self.tools.keys())}"
            )

        tool = self.tools[tool_name]
        logger.info("Executing tool '%s' with args %s", tool_name, args)

        try:
            if tool_name == "python_interpreter":
                return tool.execute(args.get("code", ""))
            elif tool_name == "web_search":
                return tool.search(args.get("query", ""), args.get("n_results", 3))
            elif tool_name == "file_writer":
                return tool.write(
                    args.get("filename", "output.txt"),
                    args.get("content", ""),
                )
            else:
                return f"Error: Unknown tool dispatch for '{tool_name}'."
        except Exception as exc:
            logger.error("Tool execution error: %s", exc, exc_info=True)
            return f"Tool execution error: {type(exc).__name__}: {str(exc)}"

    def _store_interaction(self, query: str, answer: str) -> None:
        """Store the query/answer pair in episodic memory.

        Args:
            query: The user's query.
            answer: The final answer (truncated to 500 chars for storage).
        """
        text = f"Q: {query}\nA: {answer[:500]}"
        self.vector_store.add(
            text,
            metadata={"type": "interaction", "query": query[:200]},
        )
        logger.debug("Stored interaction in episodic memory.")

    def _format_output(
        self, raw_response: str, parsed: dict[str, Any]
    ) -> str:
        """Format the output for display, showing the cognitive trace.

        Includes thinking, draft, critic feedback, and final answer sections
        when available.

        Args:
            raw_response: The full raw LLM response.
            parsed: The parsed XML tag dict.

        Returns:
            A human-friendly formatted string.
        """
        parts: list[str] = []

        if parsed.get("thinking"):
            parts.append(f"🧠 Thinking:\n{parsed['thinking']}\n")
        if parsed.get("draft_solution"):
            parts.append(f"📝 Draft:\n{parsed['draft_solution']}\n")
        if parsed.get("critic_feedback"):
            parts.append(f"🔍 Critic:\n{parsed['critic_feedback']}\n")
        if parsed.get("final_output"):
            parts.append(f"✅ Final Answer:\n{parsed['final_output']}")
        else:
            parts.append(raw_response)

        return "\n".join(parts)
