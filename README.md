# DFL Clinic — Decentralized Federated Learning

A pure Python, peer-to-peer federated learning system for medical diagnosis.
Three hospital nodes collaboratively fine-tune a shared AI model without ever
sharing raw patient data. No central server. No Java. No raw TCP sockets.

---

## Requirements

- Python 3.10+

```bash
pip install flask transformers peft datasets safetensors torch requests
```

---

## Project Structure

```
dfl/
├── dfl_node.py              ← entry point: CLI args, Flask routes, bootstrap
├── lib/
│   ├── __init__.py
│   ├── state.py            ← shared { phase, round } + threading.Lock
│   ├── round_loop.py       ← sequential round loop + polling barrier
│   ├── local_trainer.py    ← LoRA fine-tuning with HuggingFace Trainer
│   ├── aggregator.py       ← FedIT SVD merge + atomic file write
│   └── inference.py        ← GPT-2 inference, lazy model cache
├── dataset/
│   ├── part1.jsonl         ← Hospital A private data (never leaves node)
│   ├── part2.jsonl         ← Hospital B private data
│   └── part3.jsonl         ← Hospital C private data
├── output/
│   ├── p1_gpt2_lora/
│   ├── p2_gpt2_lora/
│   └── p3_gpt2_lora/
└── templates/
    └── chat.html           ← browser UI for /chat endpoint
```

---

## Dataset Format

Each `partN.jsonl` has one JSON object per line:

```json
{"Question": "Patient presents with...", "Complex_CoT": "Reasoning...", "Response": "Diagnosis: ..."}
```

---

## Running — Single Machine (3 terminals)

```bash
# Terminal 1 — Hospital A
python fl_node.py --node-id 1 --port 5201 \
  --peers 2:127.0.0.1:5202 3:127.0.0.1:5203

# Terminal 2 — Hospital B
python fl_node.py --node-id 2 --port 5202 \
  --peers 1:127.0.0.1:5201 3:127.0.0.1:5203

# Terminal 3 — Hospital C
python fl_node.py --node-id 3 --port 5203 \
  --peers 1:127.0.0.1:5201 2:127.0.0.1:5202
```

Start all three within a few seconds of each other.
Nodes will poll and wait for each other automatically.

---

## Running — LAN Deployment (3 machines)

Replace `127.0.0.1` with each machine's actual LAN IP:

```bash
# Machine A — 192.168.1.10
python fl_node.py --node-id 1 --port 5201 \
  --peers 2:192.168.1.11:5202 3:192.168.1.12:5203

# Machine B — 192.168.1.11
python fl_node.py --node-id 2 --port 5202 \
  --peers 1:192.168.1.10:5201 3:192.168.1.12:5203

# Machine C — 192.168.1.12
python fl_node.py --node-id 3 --port 5203 \
  --peers 1:192.168.1.10:5201 2:192.168.1.11:5202
```

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--node-id` | required | Integer ID for this node (1, 2, or 3) |
| `--port` | required | Port this node's Flask server binds to |
| `--peers` | required | Space-separated list of `ID:IP:PORT` |
| `--rounds` | `3` | Number of FL rounds to run |
| `--poll-interval` | `3.0` | Seconds between peer status polls |
| `--poll-timeout` | `600.0` | Seconds before skipping a stuck peer |

---

## HTTP Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Returns `{"phase": "done", "round": 2}` |
| `GET` | `/weights` | Serves `adapter_model.safetensors` |
| `GET` | `/weights?round=N` | 404 if node's current round ≠ N |
| `POST` | `/predict` | `{"symptoms":"..."}` → `{"diagnosis":"..."}` |
| `GET` | `/chat` | Interactive browser UI |

---

## Testing Manually

```bash
# Node status
curl http://127.0.0.1:5201/status

# Download adapter
curl http://127.0.0.1:5201/weights -o adapter.safetensors

# Run inference
curl -X POST http://127.0.0.1:5201/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever, cough, fatigue"}'

# Browser UI
open http://127.0.0.1:5201/chat
```

---

## Expected Startup Logs

```
[Node 1] 00:00:00 INFO Waiting for Flask on port 5201...
[Node 1] 00:00:00 INFO Flask ready. Peers: {2: '127.0.0.1:5202', 3: '127.0.0.1:5203'}
[Node 1] 00:00:00 INFO === Round 1/3 ===
[Node 1] 00:00:00 INFO State → phase='training'  round=1
[Node 1] 00:00:45 INFO State → phase='done'  round=1
[Node 1] 00:00:45 INFO Local training done. Polling peers...
[Node 1] 00:00:48 INFO Peer 2 done for round 1.
[Node 1] 00:00:54 INFO Peer 3 done for round 1.
[Node 1] 00:00:54 INFO State → phase='aggregating'  round=1
[Node 1] 00:01:10 INFO State → phase='idle'  round=1
[Node 1] 00:01:10 INFO Round 1 complete.
```

> `connection refused` warnings during polling are normal if a peer started
> slightly later — they resolve automatically on the next poll interval.
