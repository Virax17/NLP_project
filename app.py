"""
Skin Disease Chatbot - Flask Web Application
=============================================
Conversational interface: understands greetings, questions about skin
conditions and symptom descriptions, and asks follow-up questions when the
model is unsure.

DISCLAIMER: For academic/educational purposes only.
Not a medical diagnosis tool.
"""

from flask import Flask, jsonify, render_template, request

from src.chat_engine import Engine

app = Flask(__name__)
engine = Engine("models")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400
    return jsonify(engine.respond(message, data.get("context")))


if __name__ == "__main__":
    print("Starting Skin Disease Chatbot...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
