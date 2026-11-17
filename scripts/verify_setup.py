"""Checks that every core component of the project works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config


def check(name, fn):
    try:
        result = fn()
        print(f"[OK]   {name}" + (f" -> {result}" if result else ""))
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def test_imports():
    import flask, flask_cors, chromadb, langgraph, pypdf, docx, pandas  # noqa
    import sentence_transformers, groq, langchain_groq  # noqa


def test_env():
    if not config.GROQ_API_KEY or "your_groq" in config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")
    return "API key found"


def test_groq():
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    r = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=10,
    )
    return r.choices[0].message.content.strip()


def test_embeddings():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vec = model.encode("hello world")
    return f"vector size {len(vec)}"


def test_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    client.get_or_create_collection("setup_test")
    client.delete_collection("setup_test")
    return "ChromaDB working"


if __name__ == "__main__":
    print("Verifying setup...\n")
    results = [
        check("Package imports", test_imports),
        check("Environment (.env)", test_env),
        check("Groq API", test_groq),
        check("Embedding model (downloads ~90MB first time)", test_embeddings),
        check("ChromaDB", test_chroma),
    ]
    print("\n" + ("All checks passed. Ready for Step 2!" if all(results)
                  else "Some checks failed. Fix them before continuing."))