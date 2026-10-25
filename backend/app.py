import sys
import os
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
NLP_ASSISTANT_PATH = BACKEND_DIR / "nlp-assistant"

if not NLP_ASSISTANT_PATH.is_dir():
    raise FileNotFoundError(
        f"NLP assistant directory not found: {NLP_ASSISTANT_PATH}"
    )

# Make nlp-assistant available for Python imports BEFORE
# importing tools, main, or the NLP levels.
sys.path.insert(0, str(NLP_ASSISTANT_PATH))


# ============================================================
# IMPORT NLP ASSISTANT AND RAG TOOLS
# ============================================================

from tools.knowledge_tool import (
    KnowledgeTool,
    DOCUMENT_DIR,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
)

from main import process_message


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

DOCUMENT_DIR = Path(DOCUMENT_DIR)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# RAG / KNOWLEDGE BASE INITIALIZATION
# ============================================================

knowledge_tool = KnowledgeTool()


# ============================================================
# BASIC ROUTES
# ============================================================

@app.route("/")
def home():
    return "Welcome to Flask"


@app.route("/hello")
def hello():
    return "Hi! How are you doing?"


@app.route("/about")
def about():
    return "This is a minor project"


@app.route("/project")
def project():
    return "This is a conversational assistant"


# ============================================================
# API STATUS
# ============================================================

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running",
        "application": "NLP conversational assistant",
        "features": [
            "Four NLP levels",
            "Gemini API integration",
            "RAG document question answering",
        ],
    })


# ============================================================
# API ABOUT
# ============================================================

@app.route("/api/about")
def api_about():
    return jsonify({
        "name": "Minor Project - NLP Conversational Assistant with Levels",
        "description": (
            "A chatbot with multiple NLP levels, "
            "with document-based question answering using RAG."
        ),
        "version": "1.1.0",
    })


# ============================================================
# CHAT API — EXISTING NLP LEVELS
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "No JSON detected",
        }), 400

    message = data.get("message")
    level = data.get("level")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "A non-empty message is required",
        }), 400

    if level is None:
        return jsonify({
            "error": "No NLP level specified",
        }), 400

    try:
        level = int(level)
    except (TypeError, ValueError):
        return jsonify({
            "error": "NLP level must be a number from 1 to 4",
        }), 400

    if level not in (1, 2, 3, 4):
        return jsonify({
            "error": "NLP level must be between 1 and 4",
        }), 400

    try:
        result = process_message(message.strip(), level)

        return jsonify({
            "message": message.strip(),
            **result,
        })

    except Exception:
        app.logger.exception("NLP processing failed")

        return jsonify({
            "error": "NLP processing failed",
        }), 500


# ============================================================
# DOCUMENT UPLOAD HELPERS
# ============================================================

def is_allowed_file(filename):
    """Check whether a file extension is supported."""
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in {
        item.lower().lstrip(".")
        for item in ALLOWED_EXTENSIONS
    }


# ============================================================
# UPLOAD DOCUMENTS
# POST /api/documents
# ============================================================

@app.route("/api/documents", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({
            "error": "No file was provided",
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "No file was selected",
        }), 400

    if not is_allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Unsupported file type",
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        }), 400

    original_filename = secure_filename(uploaded_file.filename)

    if not original_filename:
        return jsonify({
            "error": "Invalid filename",
        }), 400

    extension = original_filename.rsplit(".", 1)[1].lower()

    stored_filename = (
        f"{Path(original_filename).stem}_{os.urandom(8).hex()}.{extension}"
    )

    destination = DOCUMENT_DIR / stored_filename

    try:
        uploaded_file.save(destination)

        # add_document accepts only the file path.
        result = knowledge_tool.add_document(str(destination))

        return jsonify({
            "message": "Document uploaded successfully",
            "filename": original_filename,
            "stored_filename": stored_filename,
            "result": result if result is not None else {},
        }), 201

    except Exception:
        destination.unlink(missing_ok=True)

        app.logger.exception("Document upload/indexing failed")

        return jsonify({
            "error": (
                "The document could not be processed. "
                "Check the backend logs."
            ),
        }), 500


# ============================================================
# LIST UPLOADED DOCUMENTS
# GET /api/documents
# ============================================================

@app.route("/api/documents", methods=["GET"])
def list_documents():
    try:
        # Change this method name if your KnowledgeTool differs.
        documents = knowledge_tool.list_documents()

        return jsonify({
            "documents": documents,
        })

    except Exception:
        app.logger.exception("Could not list documents")

        return jsonify({
            "error": "Could not retrieve the document list",
        }), 500


# ============================================================
# RAG CHAT
# POST /api/rag/chat
# ============================================================

@app.route("/api/rag/chat", methods=["POST"])
def rag_chat():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "A JSON request body is required",
        }), 400

    question = data.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "A non-empty question is required",
        }), 400

    try:
        # The KnowledgeTool should retrieve relevant document
        # chunks and use Gemini to generate a grounded answer.
        # Change this method name if your implementation differs.
        result = knowledge_tool.answer(question.strip())

        if isinstance(result, dict):
            return jsonify({
                "question": question.strip(),
                **result,
            })

        return jsonify({
            "question": question.strip(),
            "answer": str(result),
        })

    except Exception:
        app.logger.exception("RAG question answering failed")

        return jsonify({
            "error": (
                "Could not answer the question using the documents. "
                "Check the backend logs."
            ),
        }), 500


# ============================================================
# FILE SIZE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "The uploaded file exceeds the maximum allowed size",
        "maximum_size_bytes": MAX_FILE_SIZE,
    }), 413


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
    )
