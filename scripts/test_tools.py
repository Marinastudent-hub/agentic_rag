
"""Test all Agentic RAG tools and Groq tool selection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import get_llm
from app.tools import (
    ALL_TOOLS,
    list_documents,
    search_documents,
    rewrite_query,
    summarize_document,
    compare_documents,
    calculator,
    analyze_data,
)
from app.tools.shared import get_store


failures = 0


def run(title, tool, args):
    global failures

    print(f"\n=== {title} ===")
    print(f"args: {args}")

    try:
        result = tool.invoke(args)

        if isinstance(result, tuple):
            print(result[0])
        else:
            print(result)

    except Exception as error:
        failures += 1
        print(f"[ERROR] {error}")


def main():
    global failures

    store = get_store()
    docs = store.list_documents()

    if not docs:
        print("No documents found in the knowledge base.")
        print("Please run the vector store test first.")
        return

    # 1. Retrieval tools
    run("list_documents", list_documents, {})

    run(
        "search_documents (all docs)",
        search_documents,
        {"query": "remote work days per week"},
    )

    run(
        "search_documents (one doc)",
        search_documents,
        {"query": "equipment stipend", "document": "company policy"},
    )

    run(
        "search_documents (irrelevant query)",
        search_documents,
        {"query": "chocolate cake baking temperature"},
    )

    run(
        "search_documents (unknown doc)",
        search_documents,
        {"query": "anything", "document": "does_not_exist.pdf"},
    )

    run(
        "rewrite_query",
        rewrite_query,
        {"question": "How did revenue change from Q2 to Q3 and why did costs go up?"},
    )

    # 2. Reasoning tools
    run(
        "summarize_document",
        summarize_document,
        {"document": "company_policy.md"},
    )

    if len(docs) >= 2:
        run(
            "compare_documents",
            compare_documents,
            {
                "topic": "money and costs",
                "document_a": "company policy",
                "document_b": "Q3 report",
            },
        )

    # 3. Compute tools
    run(
        "calculator (basic)",
        calculator,
        {"expression": "(4.2 - 3.5) / 3.5 * 100"},
    )

    run(
        "calculator (percent)",
        calculator,
        {"expression": "4200000 * 15%"},
    )

    run(
        "calculator (function)",
        calculator,
        {"expression": "sqrt(144) + max(3, 9)"},
    )

    run(
        "calculator (unsafe input)",
        calculator,
        {"expression": "__import__('os').system('dir')"},
    )

    run(
        "calculator (division by zero)",
        calculator,
        {"expression": "5 / 0"},
    )

    run(
        "analyze_data summary",
        analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "summary", "labels": "Q1, Q2, Q3"},
    )

    run(
        "analyze_data growth",
        analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "growth", "labels": "Q1, Q2, Q3"},
    )

    run(
        "analyze_data trend",
        analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "trend"},
    )

    run(
        "analyze_data rank",
        analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "rank", "labels": "Q1, Q2, Q3"},
    )

    # 4. Test Groq tool selection
    print("\n=== Tool selection by the LLM ===")

    test_questions = [
        "Which documents do I have?",
        "What is 15% of 4.2 million?",
        "How many days per week can employees work remotely?",
    ]

    try:
        llm_with_tools = get_llm().bind_tools(ALL_TOOLS)

        for question in test_questions:
            response = llm_with_tools.invoke(question)

            selected_tools = [
                (call["name"], call["args"])
                for call in response.tool_calls
            ]

            print(f"Q: {question}")
            print(f"   -> {selected_tools}")

            if not selected_tools:
                failures += 1
                print("   [WARNING] No tool was selected.")

    except Exception as error:
        failures += 1
        print(f"[ERROR] Tool selection failed: {error}")

    # 5. Final result
    print(
        "\n"
        + (
            "[OK] All tools work. Ready for Step 5!"
            if failures == 0
            else f"[ERROR] {failures} test(s) failed."
        )
    )


if __name__ == "__main__":
    main()