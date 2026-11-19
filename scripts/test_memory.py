
"""Test persistent conversation memory."""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent


def main():
    shared_thread = f"memory-test-{uuid4()}"
    separate_thread = f"memory-test-{uuid4()}"

    print("=" * 60)
    print("AGENTIC RAG - CONVERSATION MEMORY TEST")
    print("=" * 60)

    # Test 1: Tell the agent a fact.
    print("\nTEST 1: Save a fact")

    result1 = run_agent(
        "My favorite color is blue. Please remember this.",
        thread_id=shared_thread,
    )
    print(result1["answer"])

    # Test 2: Ask about the previous fact in the same conversation.
    print("\nTEST 2: Recall the fact in the same conversation")

    result2 = run_agent(
        "What is my favorite color?",
        thread_id=shared_thread,
    )
    print(result2["answer"])

    # Test 3: Ask the same question in a different conversation.
    print("\nTEST 3: Check conversation isolation")

    result3 = run_agent(
        "What is my favorite color?",
        thread_id=separate_thread,
    )
    print(result3["answer"])

    print("\n" + "=" * 60)
    print("MEMORY TEST FINISHED")
    print("=" * 60)

    print("\nExpected results:")
    print("- Test 2 should recall: blue")
    print("- Test 3 should not know the previous conversation's fact")


if __name__ == "__main__":
    main()
