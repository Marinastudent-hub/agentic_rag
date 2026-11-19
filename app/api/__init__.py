
"""Flask API routes for the Agentic RAG application."""

from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)
from werkzeug.utils import secure_filename

from app import config
from app.agent import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    run_agent,
)
from app.vectorstore.store import VectorStore


PROJECT_DIR = Path(__file__).resolve().parents[2]

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_app():
    app = Flask(
        __name__,
        template_folder=str(PROJECT_DIR / "templates"),
        static_folder=str(PROJECT_DIR / "static"),
    )

    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html")

    # ----------------------------------------
    # Conversations
    # ----------------------------------------

    @app.route("/api/conversations", methods=["GET", "POST"])
    def conversations():
        if request.method == "POST":
            thread_id = create_conversation()

            return jsonify({
                "thread_id": thread_id,
                "title": "New Chat",
            }), 201

        return jsonify({
            "conversations": list_conversations()
        })

    @app.route(
        "/api/conversation/<thread_id>",
        methods=["GET"],
    )
    def conversation_detail(thread_id):
        messages = get_conversation(thread_id)

        return jsonify({
            "thread_id": thread_id,
            "messages": messages,
        })

    @app.route(
        "/api/conversation/<thread_id>",
        methods=["DELETE"],
    )
    def conversation_delete(thread_id):
        try:
            deleted = delete_conversation(thread_id)

            if not deleted:
                return jsonify({
                    "error": "Conversation not found."
                }), 404

            return jsonify({
                "success": True,
                "thread_id": thread_id,
            }), 200

        except Exception as exc:
            app.logger.exception("Conversation deletion failed")

            return jsonify({
                "error": str(exc),
            }), 500

    # ----------------------------------------
    # Chat
    # ----------------------------------------

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}

        question = data.get("message", "").strip()
        thread_id = data.get("thread_id")

        if not question:
            return jsonify({
                "error": "Message cannot be empty."
            }), 400

        try:
            result = run_agent(
                question=question,
                thread_id=thread_id,
            )

            return jsonify({
                "answer": result["answer"],
                "trace": result.get("trace", []),
                "sources": result.get("sources", []),
                "thread_id": result["thread_id"],
            })

        except Exception as exc:
            app.logger.exception("Chat request failed")

            return jsonify({
                "error": str(exc),
            }), 500

    # ----------------------------------------
    # Document Upload
    # ----------------------------------------

    @app.route("/api/upload", methods=["POST"])
    def upload_document():
        if "file" not in request.files:
            return jsonify({
                "error": "No file was provided."
            }), 400

        uploaded_file = request.files["file"]

        if not uploaded_file.filename:
            return jsonify({
                "error": "No file selected."
            }), 400

        filename = secure_filename(uploaded_file.filename)

        if not filename:
            return jsonify({
                "error": "Invalid filename."
            }), 400

        if not allowed_file(filename):
            return jsonify({
                "error": "Unsupported file type. Use PDF, DOCX, TXT, or MD."
            }), 400

        try:
            config.UPLOAD_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_path = config.UPLOAD_DIR / filename

            uploaded_file.save(str(file_path))

            store = VectorStore()
            chunk_count = store.add_file(file_path)

            documents = store.list_documents()

            return jsonify({
                "success": True,
                "filename": filename,
                "chunks": chunk_count,
                "documents": documents,
            }), 200

        except Exception as exc:
            app.logger.exception("Document upload failed")

            return jsonify({
                "error": str(exc),
            }), 500

    # ----------------------------------------
    # Documents
    # ----------------------------------------

    @app.route("/api/documents", methods=["GET"])
    def documents():
        try:
            store = VectorStore()

            return jsonify({
                "documents": store.list_documents()
            })

        except Exception as exc:
            app.logger.exception("Could not list documents")

            return jsonify({
                "error": str(exc),
            }), 500

    @app.route(
        "/api/documents/<path:filename>",
        methods=["DELETE"],
    )
    def delete_document(filename):
        try:
            safe_filename = secure_filename(filename)

            if not safe_filename:
                return jsonify({
                    "error": "Invalid filename."
                }), 400

            store = VectorStore()
            store.delete_document(safe_filename)

            file_path = config.UPLOAD_DIR / safe_filename

            if file_path.exists() and file_path.is_file():
                file_path.unlink()

            return jsonify({
                "success": True,
                "filename": safe_filename,
                "documents": store.list_documents(),
            })

        except Exception as exc:
            app.logger.exception("Document deletion failed")

            return jsonify({
                "error": str(exc),
            }), 500

    return app
