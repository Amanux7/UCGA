"""
UCGA Second Brain - Web Search Tool

Provides web search capability via SerpAPI. Falls back gracefully
when the API key is not configured.

Author: Aman Singh
"""

import os
import logging
from typing import List, Dict, Any

import requests

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"


class WebSearch:
    """Searches the web using SerpAPI.

    Requires the ``SERPAPI_KEY`` environment variable to be set.
    If the key is absent, search calls return an informative fallback
    message instead of raising an exception.
    """

    def search(self, query: str, n_results: int = 3) -> str:
        """Search the web for a given query.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return (default 3).

        Returns:
            A formatted numbered list of search results (title, snippet,
            link), or a fallback message if the API key is missing or
            an error occurs.
        """
        api_key: str | None = os.environ.get("SERPAPI_KEY")

        if not api_key:
            logger.warning("SERPAPI_KEY not set; web search is unavailable")
            return (
                "Web search unavailable. Set SERPAPI_KEY env var to enable. "
                f"Query was: {query}"
            )

        logger.info("Searching web for: %s (n_results=%d)", query, n_results)

        try:
            params: Dict[str, Any] = {
                "q": query,
                "api_key": api_key,
                "num": n_results,
                "engine": "google",
            }

            response = requests.get(
                _SERPAPI_ENDPOINT, params=params, timeout=15
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

            organic_results: List[Dict[str, Any]] = data.get(
                "organic_results", []
            )

            if not organic_results:
                logger.info("No results found for query: %s", query)
                return f"No results found for: {query}"

            formatted_lines: List[str] = []
            for idx, result in enumerate(organic_results[:n_results], start=1):
                title: str = result.get("title", "No title")
                snippet: str = result.get("snippet", "No snippet available")
                link: str = result.get("link", "No link")
                formatted_lines.append(
                    f"{idx}. {title}\n   {snippet}\n   {link}"
                )

            return "\n\n".join(formatted_lines)

        except requests.exceptions.RequestException as exc:
            error_msg = f"Error: {type(exc).__name__}: {str(exc)}"
            logger.error("Web search request failed: %s", error_msg)
            return error_msg
        except Exception as exc:
            error_msg = f"Error: {type(exc).__name__}: {str(exc)}"
            logger.error("Unexpected web search error: %s", error_msg)
            return error_msg
