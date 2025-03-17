# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from nurse_chat import Nurse2NurseChatbotSummarize, get_session_history

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')
    file_path = data.get('file_path')

    # Validate required fields
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not user_message:
        return jsonify({"error": "message is required"}), 400
    if not file_path:
        return jsonify({"error": "file_path is required"}), 400

    reply = Nurse2NurseChatbotSummarize(file_path, session_id, user_message)
    return jsonify({"reply": reply})

@app.route('/api/history', methods=['GET'])
def history():
    session_id = request.args.get('session_id', 'default')
    history_obj = get_session_history(session_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history_obj.messages]
    return jsonify({"session_id": session_id, "messages": messages})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
