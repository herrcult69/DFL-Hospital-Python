# DFL Clinic — Decentralized Federated Learning

A pure Python, peer-to-peer federated learning system for medical diagnosis using Large Language Models (LLMs). The nodes in this network collaborate to fine-tune a shared AI model (using LoRA) on their local private datasets without ever sharing raw patient data.

It features a decentralized ring/graph topology, gRPC-based parameter transfer, a gossip protocol for state synchronization, and a FastAPI-based REST API for node management and status.

---

## Features

- **Peer-to-Peer Architecture:** No single central server for aggregation. Nodes organize into a K-regular graph topology.
- **Federated Fine-Tuning (LoRA):** Low-Rank Adaptation (LoRA) is used to fine-tune the model efficiently.
- **Privacy-Preserving:** Raw patient data never leaves the hospital node. Only model weight updates are transferred securely via gRPC.
- **Gossip Protocol:** Node discovery and status updates are propagated using a lightweight gossip protocol.
- **FastAPI Backend:** Modern, fast REST APIs for dashboard and status monitoring.

---

## Requirements

- Python 3.10+

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You will also need the HuggingFace ML libraries for training and inference:

```bash
pip install transformers peft datasets
```

> **Note:** `transformers`, `peft`, and `datasets` are required by `trainer.py` and `inference.py` but are not listed in `requirements.txt`.

---

## Project Structure

```
dfl-hospital-python/
├── main.py                  ← Entry point for running Bootstrap or Worker nodes
├── requirements.txt         ← Project dependencies
├── ring.proto               ← Protocol Buffers definition for gRPC weight transfer
├── src/
│   ├── config.py            ← Configuration defaults and CLI parsing logic
│   ├── bootstrap.py         ← Bootstrap node API and logic
│   ├── worker.py            ← Worker node API and logic
│   ├── state.py             ← Node state management
│   ├── graph.py             ← K-regular graph topology management
│   ├── gossip.py            ← Gossip protocol implementation
│   ├── trainer.py           ← Local LoRA training logic
│   ├── aggregator.py        ← Model weight aggregation
│   ├── ring_transfer.py     ← gRPC service for weight transfer
│   └── ...
├── models/                  ← Local and merged LoRA adapters
├── dataset/                 ← Local datasets for nodes
└── templates/               ← HTML templates for dashboards

```

---

## Getting Started (For Localhost Single Machine)


### 1. Start the Bootstrap Node

The bootstrap node acts as the entry point to the network.

```bash
python main.py --host 127.0.0.1 --port 8000 --grpc-port 9000
  --bootstrap
  --dataset-path ./dataset/node_1.json
```

### 2. Start Worker Nodes

Start as many worker nodes as needed. They need to point to the bootstrap node URL.

**Worker 1:**
```bash
python main.py --host 127.0.0.1 --port 8001 --grpc-port 9001
  --bootstrap-url 127.0.0.1:8000
  --dataset-path ./dataset/node_2.json
```

**Worker 2:**
```bash
python main.py --host 127.0.0.1 --port 8002 --grpc-port 9002
  --bootstrap-url 127.0.0.1:8000
  --dataset-path ./dataset/node_3.json
```

---

### Running — LAN Deployment (3 Machines)
Replace `127.0.0.1` and `192.168.1.x` with your actual LAN IPs.
**Machine A (Bootstrap Node) — 192.168.1.10:**
```bash
python main.py --host 192.168.1.10 --port 8000 --grpc-port 9000
  --bootstrap
  --dataset-path ./dataset/node_1.json
```
**Machine B (Worker Node 1) — 192.168.1.11:**
```bash
python main.py --host 192.168.1.11 --port 8001 --grpc-port 9001
  --bootstrap-url 192.168.1.10:8000
  --dataset-path ./dataset/node_2.json
```
**Machine C (Worker Node 2) — 192.168.1.12:**
```bash
python main.py --host 192.168.1.12 --port 8002 --grpc-port 9002
  --bootstrap-url 192.168.1.10:8000
  --dataset-path ./dataset/node_3.json
```

---

## CLI Configuration Flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | The host IP for the HTTP and gRPC servers. |
| `--port` | `8000` | The port for the FastAPI HTTP server. |
| `--grpc-port` | `9000` | The port for the gRPC server (used for weight transfer). |
| `--k` | `4` | The degree of the graph topology (number of peers per node). |
| `--bootstrap` | *(flag)* | Run this node as the Bootstrap node. |
| `--bootstrap-url` | `127.0.0.1:8000` | The `IP:PORT` of the Bootstrap node (for workers). |
| `--model-dir` | `./models` | Directory to save local and aggregated LoRA adapters. |
| `--dataset-path` | `""` | Path to the specific JSONL dataset file for this node. |

---

## HTTP Endpoints & Dashboards

Each node exposes a web UI and several REST endpoints for monitoring and administration:

### Bootstrap Node Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | JSON status snapshot (phase, round, table size, etc.) |
| `GET` | `/table` | Full global routing table + node count |
| `GET` | `/graph` | Adjacency list, degrees, K-regularity check |
| `GET` | `/status-page` | **HTML dashboard** — live status, ready sets, heartbeat tracker, predict panel |
| `GET` | `/graph-page` | **HTML graph visualization** — topology rendered with vis.js |
| `POST` | `/join` | Worker registration (returns neighbors, global table, round) |
| `POST` | `/gossip` | Receive gossip rumors |
| `POST` | `/predict` | Run inference (only when `IDLE`) |
| `POST` | `/start-round` | Trigger the next FL round |
| `POST` | `/rewire-evict` | Remove dead nodes from K-graph |

### Worker Node Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | JSON status snapshot |
| `GET` | `/` | **HTML dashboard** — live node status |
| `POST` | `/gossip` | Receive gossip rumors |
| `POST` | `/rewire` | Update neighbor map (called by bootstrap) |
| `POST` | `/predict` | Run inference (only when `IDLE`) |

---

## Testing Manually

```bash
# Check node status
curl http://127.0.0.1:8000/status

# View routing table (bootstrap only)
curl http://127.0.0.1:8000/table

# View graph topology (bootstrap only)
curl http://127.0.0.1:8000/graph

# Run inference on a worker (only works when node is IDLE)
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "headache, nausea, sensitivity to light"}'

# Open dashboards in browser
# Bootstrap status:  http://127.0.0.1:8000/status-page
# Bootstrap graph:   http://127.0.0.1:8000/graph-page  (interactive vis.js)
# Worker dashboard:  http://127.0.0.1:8001/
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `connection refused` on gossip | Peer node hasn't started yet | Normal during startup — resolves on the next heartbeat |
| Phase 3 (aggregation) times out | Exceeds the 600s budget | Increase `phase3_total_budget` in config |
| Phase 4 (training) times out | Training exceeds the 1800s budget | Reduce dataset size or increase `phase4_timeout`; lower epoch count in `trainer.py` |
| `CUDA out of memory` | GPU memory exhausted during training | Reduce batch size or max sequence length in `trainer.py` |
| `/predict` returns `not_idle` | Node is still in a training round | Wait for all rounds to complete (node must be in `IDLE` phase) |
| Adapter not found for inference | No training round has completed yet | Run at least one full FL round before calling `/predict` |

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
