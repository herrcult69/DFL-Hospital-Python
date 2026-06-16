from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from flask import Flask, jsonify, request, send_file, render_template

# ── Must patch lib constants before importing round_loop ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser(description="DFL Node")
parser.add_argument("--node-id",       type=int,   required=True)
parser.add_argument("--port",          type=int,   required=True)
parser.add_argument("--peers",         type=str,   nargs="*", default=[])
parser.add_argument("--rounds",        type=int,   default=3)
parser.add_argument("--poll-interval", type=float, default=3.0)
parser.add_argument("--poll-timeout",  type=float, default=600.0)
args = parser.parse_args()

NODE_ID    = args.node_id
PORT       = args.port
OUTPUT_DIR = f"output/p{NODE_ID}_gpt2_lora"

PEER_ADDRESSES: dict[int, str] = {}
for peer_str in (args.peers or []):
    peer_id, peer_addr = peer_str.split(":", 1)
    PEER_ADDRESSES[int(peer_id)] = peer_addr

logging.basicConfig(
    level=logging.INFO,
    format=f"[Node {NODE_ID}] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Patch lib globals before any lib import uses them
import lib.local_trainer as local_trainer
import lib.inference      as inference

local_trainer.NODE_ID      = NODE_ID
local_trainer.OUTPUT_DIR   = OUTPUT_DIR
local_trainer.DATASET_PATH = f"dataset/part{NODE_ID}.jsonl"

from lib.state      import get_state
from lib.round_loop import NodeConfig, run as run_rounds

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/status")
def status():
    return jsonify(get_state())

@app.route("/weights")
def weights():
    path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if not os.path.exists(path):
        return jsonify({"error": "No adapter available yet"}), 404

    requested_round = request.args.get("round", type=int)
    if requested_round is not None:
        current = get_state()
        # Allow peers to fetch weights if we are in the requested round,
        # OR if we have already moved past it and are in an idle/aggregating state
        if current["round"] < requested_round:
            return jsonify({
                "error": f"Node on round {current['round']}, requested {requested_round}"
            }), 404
        if current["round"] > requested_round:
            # We already passed this round. It's technically safe to allow download
            # because the weights on disk represent at least the requested round's completion.
            pass

    return send_file(
        os.path.abspath(path),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"adapter_node_{NODE_ID}.safetensors",
    )

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", "")
    result   = inference.run_inference(symptoms, OUTPUT_DIR)
    return jsonify({"diagnosis": result})

@app.route("/chat")
def chat():
    return render_template("index.html", NODE_ID=NODE_ID)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_flask_ready = threading.Event()

def _run_flask() -> None:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    _flask_ready.set()
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("output",   exist_ok=True)

    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    log.info(f"Waiting for Flask on port {PORT}...")
    _flask_ready.wait()
    log.info(f"Flask ready. Peers: {PEER_ADDRESSES}")

    cfg = NodeConfig(
        node_id        = NODE_ID,
        output_dir     = OUTPUT_DIR,
        peer_addresses = PEER_ADDRESSES,
        total_rounds   = args.rounds,
        poll_interval  = args.poll_interval,
        poll_timeout   = args.poll_timeout,
    )
    run_rounds(cfg)

    flask_thread.join()