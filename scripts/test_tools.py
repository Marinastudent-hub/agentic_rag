"""Tests every tool, then checks that Groq can choose tools correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app import config
from app.llm import get_llm
from app.tools import ALL_TOOLS
from app.tools import (list_documents, rewrite_query, search_documents,
                       summarize_document, compare_documents, calculator,
                       analyze_data)
from app.tools.shared import get_store

failures = 0


def run(title, tool, args):
    global failures
    print(f"\n=== {title} ===")
    print(f"args: {args}")
    try:
        out = tool.invoke(args)
        text = str(out)
        print(text if len(text) < 800 else text[:800] + "\n... (truncated)")
    except Exception as e:
        failures += 1
        print(f"[FAIL] {type(e).__name__}: {e}")


def main():
    store = get_store()
    if store.count() == 0:
        print("Knowledge base empty. Ingesting data/uploads...")
        for f in sorted(config.UPLOAD_DIR.iterdir()):
            if f.suffix.lower() in config.ALLOWED_EXTENSIONS:
                store.add_file(f)
    docs = [d["source"] for d in store.list_documents()]
    if not docs:
        print("[FAIL] No documents. Run scripts/test_ingestion.py first.")
        return

    # ---- retrieval tools ----
    run("list_documents", list_documents, {})
    run("search_documents (all docs)", search_documents,
        {"query": "remote work days per week"})
    run("search_documents (one doc)", search_documents,
        {"query": "main topic", "document": docs[0]})
    run("search_documents (irrelevant query)", search_documents,
        {"query": "chocolate cake baking temperature"})
    run("search_documents (unknown doc)", search_documents,
        {"query": "anything", "document": "does_not_exist.pdf"})
    run("rewrite_query", rewrite_query,
        {"question": "How did revenue change from Q2 to Q3 and why did costs go up?"})

    # ---- reasoning tools ----
    run("summarize_document", summarize_document, {"document": docs[0]})
    if len(docs) >= 2:
        run("compare_documents", compare_documents,
            {"topic": "key numbers and rules", "document_a": docs[0],
             "document_b": docs[1]})
    else:
        print("\n(Skipping compare_documents: you need at least 2 documents.)")

    # ---- compute tools ----
    run("calculator (basic)", calculator, {"expression": "(4.2 - 3.5) / 3.5 * 100"})
    run("calculator (percent)", calculator, {"expression": "4200000 * 15%"})
    run("calculator (function)", calculator, {"expression": "sqrt(144) + max(3, 9)"})
    run("calculator (unsafe input)", calculator,
        {"expression": "__import__('os').system('dir')"})
    run("calculator (division by zero)", calculator, {"expression": "5 / 0"})
    run("analyze_data summary", analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "summary", "labels": "Q1, Q2, Q3"})
    run("analyze_data growth", analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "growth", "labels": "Q1, Q2, Q3"})
    run("analyze_data trend", analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "trend"})
    run("analyze_data rank", analyze_data,
        {"numbers": "3.5, 4.2, 4.9", "operation": "rank", "labels": "Q1, Q2, Q3"})

    # ---- can Groq choose tools? (preview of Step 5) ----
    print("\n=== Tool selection by the LLM ===")
    llm = get_llm().bind_tools(ALL_TOOLS)
    questions = [
        "Which documents do I have?",
        "What is 15% of 4.2 million?",
        "How many days per week can employees work remotely?",
    ]
    for q in questions:
        try:
            calls = llm.invoke(q).tool_calls
            picked = [(c["name"], c["args"]) for c in calls]
            print(f"Q: {q}\n   -> {picked if picked else 'NO TOOL CHOSEN'}")
            if not picked:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"Q: {q}\n   [FAIL] {type(e).__name__}: {e}")

    print("\n" + ("[OK] All tools work. Ready for Step 5!" if failures == 0
                  else f"[!] {failures} problem(s) above. Paste them to me."))


if __name__ == "__main__":
    main()