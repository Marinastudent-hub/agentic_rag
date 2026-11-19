
"""Test the LangGraph Agent with different question types."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from app.agent import run_agent



TEST_QUESTIONS = [
    "Which documents do I have uploaded?",
    "How many days per week can employees work remotely?",
    "What is 15% of 4.2 million?",
    "Summarize the company remote work policy.",
    (
        "Compare the company's remote work policy with its Q3 business "
        "performance. Calculate the percentage increase in revenue from "
        "Q2 to Q3, identify the main source of growth, and report the "
        "remote work limit and security requirement."
    ),
]


def main():
    print("=" * 60)
    print("AGENTIC RAG - LANGGRAPH AGENT TEST")
    print("=" * 60)

    for index, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n{'=' * 60}")
        print(f"TEST {index}: {question}")
        print("=" * 60)

        try:
            result = run_agent(question)

            print("\nEXECUTION TRACE:")

            if result["trace"]:
                for item in result["trace"]:
                    if item["type"] == "tool_call":
                        print(
                            f"\n[TOOL CALL] {item['tool']}"
                        )
                        print(f"Arguments: {item['arguments']}")

                    elif item["type"] == "tool_result":
                        print(
                            f"[TOOL RESULT] {item['tool']}"
                        )
            else:
                print("No tools were called.")

            print("\nFINAL ANSWER:")
            print(result["answer"])

            if not result["answer"]:
                print("[WARNING] No final answer returned.")

        except Exception as error:
            print(f"\n[ERROR] {type(error).__name__}: {error}")

    print("\n" + "=" * 60)
    print("AGENT TESTING FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
