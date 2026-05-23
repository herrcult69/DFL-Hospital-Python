from flask import Flask, request, jsonify
import logging

# Disable werkzeug logging to keep console clean
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)

@app.route("/api/report", methods=["POST"])
def receive_report():
    data = request.json
    print(f"[MOCK DASHBOARD] Received Report: Node {data.get('node_id')} is '{data.get('status')}' (Round {data.get('round')}) at {data.get('timestamp')}")
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    print("Mock Dashboard running on http://127.0.0.1:3000")
    app.run(host="127.0.0.1", port=3000, debug=False)
