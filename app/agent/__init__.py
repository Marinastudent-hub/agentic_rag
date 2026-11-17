
"""LangGraph Agent for the Agentic RAG system."""

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.llm import get_llm
from app.tools import ALL_TOOLS


SYSTEM_PROMPT = """
You are an intelligent Agentic RAG assistant.

Your job is to answer user questions accurately using the available tools.

Available capabilities:
- List available documents.
- Search uploaded documents.
- Rewrite and split complex questions into search queries.
- Summarize documents.
- Compare information across documents.
- Perform mathematical calculations.
- Analyze numerical data.

Rules:
1. Decide which tools are needed based on the user's question.
2. You may call multiple tools when necessary.
3. For document-related questions, use document search tools.
4. For summaries, use summarize_document.
5. For comparisons, use compare_documents.
6. For mathematical questions, use calculator.
7. For numerical analysis, use analyze_data.
8. Never invent facts, numbers, or document content.
9. If documents do not contain the answer, say so clearly.
10. Include document names and page numbers when available.
11. Give clear, direct answers.
12. Do not expose private reasoning. Only provide the answer
    and a brief summary of actions when useful.
"""


def build_agent():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


_agent = None


def get_agent():
    global _agent

    if _agent is None:
        _agent = build_agent()

    return _agent


def run_agent(question: str):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    agent = get_agent()

    result = agent.invoke({
        "messages": [HumanMessage(content=question.strip())]
    })

    messages = result["messages"]

    answer = ""
    trace = []

    for message in messages:

        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                trace.append({
                    "type": "tool_call",
                    "tool": call["name"],
                    "arguments": call["args"],
                })

        elif isinstance(message, ToolMessage):
            trace.append({
                "type": "tool_result",
                "tool": message.name,
                "result": message.content,
            })

        elif isinstance(message, AIMessage) and message.content:
            answer = message.content

    return {
        "answer": answer,
        "trace": trace,
    }
