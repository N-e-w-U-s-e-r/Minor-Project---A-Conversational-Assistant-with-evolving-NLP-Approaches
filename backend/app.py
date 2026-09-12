import sys
from pathlib import Path

from flask import Flask, request
from flask_cors import CORS


NLP_ASSISTANT_PATH = Path(__file__).parent / "nlp-assistant"

sys.path.insert(0, str(NLP_ASSISTANT_PATH))

from main import process_message


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


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

    return {
        "status": "running",
        "application": "NLP conversational assistant"
    }


# ============================================================
# API ABOUT
# ============================================================

@app.route("/api/about")
def api_about():

    return {
        "name": "Minor Project - NLP Conversational Assistant with Levels",
        "description": (
            "A chatbot with multiple levels to show "
            "sophistication and development with each "
            "progressive level."
        ),
        "version": "1.0.0"
    }


# ============================================================
# CHAT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():

    data = request.get_json(silent=True)

    if data is None:
        return {
            "error": "No JSON detected"
        }, 400

    message = data.get("message")
    level = data.get("level")

    if message is None:
        return {
            "error": "No message received"
        }, 400

    if level is None:
        return {
            "error": "No NLP level specified"
        }, 400

    try:
        level = int(level)
    except (TypeError, ValueError):
        return {
            "error": "NLP level must be a number from 1 to 4"
        }, 400

    if level not in (1, 2, 3, 4):
        return {
            "error": "NLP level must be between 1 and 4"
        }, 400

    try:

        result = process_message(
            message,
            level
        )

        return {
            "message": message,
            **result
        }

    except Exception as error:

        print("NLP error:", error)

        return {
            "error": "NLP processing failed"
        }, 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )