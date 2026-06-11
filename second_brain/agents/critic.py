"""
UCGA Second Brain - Metacognitive Critic Agent

Reviews draft solutions for factual accuracy, logical consistency,
and completeness. Outputs structured XML feedback that the Orchestrator
uses to decide whether to revise or finalize a response.

Author: Aman Singh
"""

import re
import logging
from typing import Any

from agents.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class CriticAgent:
    """Metacognitive Critic Agent that reviews draft solutions.

    The Critic receives the original query, retrieved memory context, and a
    draft solution. It evaluates the draft and returns structured feedback
    indicating whether the draft is APPROVED or NEEDS_REVISION, along with
    specific issues and suggestions.
    """

    def __init__(self, llm_client: UnifiedLLMClient) -> None:
        """Initialize the Critic Agent.

        Args:
            llm_client: A configured UnifiedLLMClient instance for LLM calls.
        """
        self.llm = llm_client
        self.system_prompt = (
            "You are a Metacognitive Critic Agent. Your role is to review draft solutions "
            "for factual accuracy, logical consistency, and completeness.\n\n"
            "You will receive:\n"
            "1. The original user query\n"
            "2. Retrieved memory context\n"
            "3. The draft solution to review\n\n"
            "You must output your feedback in this exact XML format:\n"
            "<critic_feedback>\n"
            "  <status>APPROVED or NEEDS_REVISION</status>\n"
            "  <issues>\n"
            "    <issue>description of problem (if any)</issue>\n"
            "  </issues>\n"
            "  <suggestions>\n"
            "    <suggestion>specific correction or improvement</suggestion>\n"
            "  </suggestions>\n"
            "</critic_feedback>"
        )

    def review(self, query: str, context: str, draft: str) -> dict[str, Any]:
        """Review a draft solution and return structured feedback.

        Sends the query, context, and draft to the LLM with the critic system
        prompt, then parses the XML response into a structured dict.

        Args:
            query: The original user query.
            context: The retrieved memory context provided to the orchestrator.
            draft: The draft solution text to review.

        Returns:
            A dict with keys:
                - status: ``"APPROVED"`` or ``"NEEDS_REVISION"``
                - issues: list of issue description strings
                - suggestions: list of suggestion strings
                - raw: the full raw LLM response text
        """
        user_message = (
            f"## Original Query\n{query}\n\n"
            f"## Retrieved Context\n{context}\n\n"
            f"## Draft Solution\n{draft}\n\n"
            "Please review the draft solution above and provide your feedback "
            "in the required XML format."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info("Critic reviewing draft (query=%s…)", query[:60])
        raw_response = self.llm.completion(messages, temperature=0.2)

        return self._parse_feedback(raw_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_feedback(self, response: str) -> dict[str, Any]:
        """Parse ``<critic_feedback>`` XML from the LLM response.

        If parsing fails (malformed XML, missing tags), the method defaults
        to APPROVED with a note explaining the parse failure.

        Args:
            response: Raw LLM response text.

        Returns:
            Structured feedback dict.
        """
        feedback: dict[str, Any] = {
            "status": "APPROVED",
            "issues": [],
            "suggestions": [],
            "raw": response,
        }

        # Extract the <critic_feedback> block
        block_match = re.search(
            r"<critic_feedback>(.*?)</critic_feedback>",
            response,
            re.DOTALL,
        )

        if not block_match:
            logger.warning(
                "Could not parse <critic_feedback> XML; defaulting to APPROVED."
            )
            feedback["suggestions"].append(
                "Critic response could not be parsed; auto-approved."
            )
            return feedback

        block = block_match.group(1)

        # Extract <status>
        status_match = re.search(r"<status>\s*(.*?)\s*</status>", block, re.DOTALL)
        if status_match:
            raw_status = status_match.group(1).strip().upper()
            feedback["status"] = (
                "NEEDS_REVISION" if "NEEDS_REVISION" in raw_status else "APPROVED"
            )

        # Extract all <issue> entries
        feedback["issues"] = [
            m.strip()
            for m in re.findall(r"<issue>(.*?)</issue>", block, re.DOTALL)
        ]

        # Extract all <suggestion> entries
        feedback["suggestions"] = [
            m.strip()
            for m in re.findall(r"<suggestion>(.*?)</suggestion>", block, re.DOTALL)
        ]

        logger.info(
            "Critic verdict: %s (%d issues, %d suggestions)",
            feedback["status"],
            len(feedback["issues"]),
            len(feedback["suggestions"]),
        )

        return feedback
