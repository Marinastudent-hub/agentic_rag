
"""Web search tool using DuckDuckGo."""

from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information or facts missing from uploaded documents."""

    if not query.strip():
        return "Search query cannot be empty."

    try:
        results = DDGS().text(
            query,
            max_results=5,
        )

        if not results:
            return "No useful web search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "")

            formatted_results.append(
                f"[{index}] {title}\n"
                f"Information: {body}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Web search failed: {error}"
