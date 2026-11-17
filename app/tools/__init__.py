"""Agent tools. ALL_TOOLS is the list the LangGraph agent will use in Step 5."""
from app.tools.retrieval import list_documents, search_documents, rewrite_query
from app.tools.reasoning import summarize_document, compare_documents
from app.tools.compute import calculator, analyze_data

ALL_TOOLS = [
    list_documents,
    rewrite_query,
    search_documents,
    summarize_document,
    compare_documents,
    calculator,
    analyze_data,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}