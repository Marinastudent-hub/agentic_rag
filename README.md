# Agentic RAG Assistant

An intelligent document-based AI assistant built with **LangGraph, Flask, Groq, and ChromaDB**. It combines agent-based reasoning with document retrieval to answer questions using uploaded files or reliable general knowledge when document evidence is unavailable.

## Key Features

- **Agentic Workflow:** Uses LangGraph to manage the agent's reasoning and tool execution.
- **Document Q&A:** Upload and ask questions about PDF, DOCX, TXT, and Markdown files.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant document passages using vector search.
- **Source Citations:** Provides document filenames and page numbers when answers are supported by uploaded documents.
- **General Knowledge Fallback:** Answers confidently from general knowledge when retrieved documents do not provide relevant evidence.
- **Persistent Conversations:** Stores chat history and conversation state using SQLite.
- **Interactive Web Interface:** Chat interface built with Flask, HTML, CSS, and JavaScript.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Agent Framework | LangGraph |
| LLM | Groq |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Conversation Persistence | SQLite |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```text
agentic_rag/
├── app/
│   ├── agent/
│   ├── api/
│   ├── ingestion/
│   ├── tools/
│   ├── vectorstore/
│   └── llm.py
├── scripts/
├── static/
│   ├── css/
│   └── js/
├── templates/
├── .env.example
├── .gitignore
├── requirements.txt
└── run.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Marinastudent-hub/agentic_rag.git
cd agentic_rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Replace the placeholder with your own Groq API key. Never commit your `.env` file.

### 5. Run the application

```bash
python run.py
```

Open the local URL displayed in your terminal.

## How It Works

1. Upload documents through the web interface.
2. The ingestion pipeline extracts, cleans, and splits document content into chunks.
3. Embeddings are generated and stored in ChromaDB.
4. The LangGraph agent processes user questions and uses available tools.
5. Relevant document evidence is used to generate answers with source citations.
6. When document evidence is unavailable, the assistant can fall back to general knowledge.
7. Conversation history is persisted for future sessions.

## Future Improvements

- Enhanced retrieval relevance and evaluation.
- More advanced agent planning and tool orchestration.
- Improved document processing and support for additional file formats.

## 📸 Project Screenshots

### Main Interface
![Agentic RAG Interface](Agentic%20Rag.png)

### Document Upload and Chat
![Document Upload](Screenshot%202026-11-18%20013101.png)

### Agent Response
![Agent Response](Screenshot%202026-11-18%20013133.png)

### Agent Trace and Sources
![Agent Trace](Screenshot%202026-11-18%20013146.png)
---

*Built as an AI engineering portfolio project.*
