
"""Central registry for all Agentic RAG tools."""

from app.tools.retrieval import (
    list_documents,
    search_documents,
    rewrite_query,
)

from app.tools.reasoning import (
    summarize_document,
    compare_documents,
)

from app.tools.compute import (
    calculator,
    analyze_data,
)

from app.tools.web_search import web_search


ALL_TOOLS = [
    list_documents,
    search_documents,
    rewrite_query,
    summarize_document,
    compare_documents,
    calculator,
    analyze_data,
    web_search,
]
