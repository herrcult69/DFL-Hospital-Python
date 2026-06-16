# DFL-Hospital-Python — Legacy Implementation Reference

**Status:** This codebase is now marked as LEGACY. A complete rewrite is in progress.  
**Last Updated:** 2026-06-14  
**Purpose:** This document serves as the authoritative reference for the current implementation, capturing all design decisions, file structures, and architectural choices for use in the new codebase.

---

## Executive Summary

This is a **Decentralized Federated Learning (DFL)** system implementing a 4-phase state machine:

1. **Phase 1 (Control Plane):** Gossip protocol over HTTP for node discovery and consensus
2. **Phase 2 (Data Plane):** Ring all-gather over gRPC for model distribution
3. **Phase 3:** Aggregation placeholder (FedAvg with mmap safetensors)
4. **Phase 4:** Local training placeholder (LoRA fine-tuning)

The system uses a **K-regular gossip graph** (K=4) for control plane communication and a **logical ring** for data plane communication. A **distributed phase barrier** ensures all nodes transition phases synchronously.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.10+ | Core implementation |
| Concurrency | asyncio | Single-threaded async I/O |
| Control Plane | FastAPI / HTTPX | Gossip protocol |
| Data Plane | gRPC / Protobuf | Ring all-gather streaming |
| ML Framework | PyTorch, Safetensors | Model storage and training |
| Graph Theory | Custom GraphManager | K-regular graph maintenance |

---

## File Structure (Legacy)

```
DFL-Hospital-Python/
├── protos/
│   ├── ring_transfer.proto          # gRPC service definition
│   ├── ring_transfer_pb2.py         # Generated (DO NOT EDIT)
│   └── ring_transfer_pb2_grpc.py    # Generated (DO NOT EDIT)
├── src/
│   ├── __init__.py
│   ├── config.py                    # NodeConfig, CLI parsing
│   ├── state.py                     # NodeState, Phase enum
│   ├── graph.py                     # GraphManager (K-regular graph)
│   ├── gossip.py                    # GossipEngine, FastAPI routes
│   ├── bootstrap_server.py          # BootstrapServer (seed node logic)
│   ├── ring_transfer.py             # RingTransferServicer, RingTransferClient
│   └── worker_node.py               # WorkerNode (main entry point)
├── templates/
│   └── index.html                   # Dashboard UI
├── data/                            # Runtime data (gitignored)
│   ├── received_models/             # Incoming model chunks
│   └── adapters/                    # Local model storage
├── generate_proto.sh                # Protobuf compilation script
├── requirements.txt
└── README.md
```

---

## Core Design Decisions

### 1. Node ID Format: `"host:gossip_port:grpc_port"`

**Rationale:** Embeds both ports in the node ID to eliminate the need for a separate gRPC address map.

**Helpers:**
- `split_node_id(node_id) -> (host, gossip_port, grpc_port)`
- `grpc_addr_from_node_id(node_id) -> "host:grpc_port"`
- `gossip_addr_from_node_id(node_id) -> "host:gossip_port"`

### 2. Phase-Based State Machine

**Phases:**
- `PHASE_1_GOSSIP` — Gossip/consensus (Control Plane)
- `PHASE_2_RING` — Ring all-gather (Data Plane)
- `PHASE_3_AGG` — Aggregation placeholder
- `PHASE_4_TRAIN` — Training placeholder
- `PHASE_IDLE` — Terminal state

**Transitions:** All use a **distributed barrier** (READY rumors) except Phase 1→2 which also uses a **3-second quiescence timer**.

### 3. SIR Rumor Model

**SIR Cache:** Set of `"rumor_type:originator:target"` strings.

**Rumor Types:**
- `JOIN` — New node registration
- `SUSPICION` — Suspect node failure
- `DEATH` — Confirmed dead node
- `OFFICIAL_UPDATE` — Edge rewiring instructions
- `READY` — Barrier synchronization

**Prevents infinite loops:** Each rumor is processed exactly once per node.

### 4. K-Regular Graph (Bootstrap Only)

**Graph Invariant:** Every node has exactly K neighbors.

**Bootstrap Algorithm:**
- N < K+1: Clique (fully connected)
- N >= K+1: Circulant graph (each node connects to K nearest neighbors in sorted order)

**Registration:** Edge-breaking algorithm that preserves K-regularity.

### 5. Persistent gRPC Channels

**Servicer:** `RingTransferServicer._fwd_channel` — persistent forward channel
**Client:** `RingTransferClient._client_channel` — persistent outbound channel

**Benefits:** Eliminates TCP handshake overhead per-chunk. Channels are closed and reopened on target change (ring patching).

### 6. Zero-Copy Disk Writes

**Incoming chunks** are written via `loop.run_in_executor(None, _write)` to avoid blocking the asyncio event loop. Files are appended in `data/received_models/{originator}.safetensors` format.

**Phase 3** can mmap these files directly for FedAvg.

### 7. Dynamic TTL

```python
ttl = (total_nodes - 1) - len(dead_nodes_detected)
```

A chunk terminates when `hops_taken >= ttl`. Dead nodes reduce the required hop count.

### 8. Distributed Phase Barrier

**Mechanism:**
1. Node sets `my_target_phase` and calls `record_ready(self, target_phase)`
2. Gossips `READY` rumor with payload `"target_phase:round"`
3. Polls `barrier_ready(target_phase)` every 300ms
4. `barrier_ready()` returns True when all nodes in `global_table` have recorded the same `target_phase`
5. On lift: `clear_ready()`, transition phase

**Integration:** Quiescence loop is paused during barrier waits.

---

## Key Classes and Methods

### `NodeState` (src/state.py)

```python
class NodeState:
    phase: Phase
    round: int
    global_table: List[str]
    gossip_neighbors: List[str]
    left_neighbor: str
    right_neighbor: str
    ready_states: Dict[str, str]
    my_target_phase: str

    def merge_table(incoming: List[str]) -> bool
    def compute_ring_neighbors() -> tuple[str, str]
    def record_ready(node_id: str, target_phase: str)
    def barrier_ready(target_phase: str) -> bool
```

### `GossipEngine` (src/gossip.py)

```python
class GossipEngine:
    _known_rumors: Set[str]
    _paused: bool

    async def receive_rumor(body: GossipBody) -> bool
    async def create_and_spread(rumor_type, target, payload)
    def pause_quiescence()
    def resume_quiescence()
```

### `GraphManager` (src/graph.py)

```python
class GraphManager:
    _adj: Dict[str, Set[str]]  # node_id -> set(neighbors)

    def bootstrap(seed_nodes)
    def register(new_node) -> List[str]
    def evict(dead_node) -> Dict[str, List[str]]
```

### `BootstrapServer` (src/bootstrap_server.py)

```python
class BootstrapServer:
    graph: GraphManager
    state: NodeState

    async def handle_register(req) -> RegisterResponse  # Rejects with 423 if not PHASE_1_GOSSIP
    async def handle_gossip(body: GossipBody) -> dict
```

### `RingTransferServicer` (src/ring_transfer.py)

```python
class RingTransferServicer:
    _fwd_channel: grpc.aio.Channel
    _fwd_stub: RingTransferStub

    async def StreamChunks(request_iterator, context) -> AsyncIterable[Ack]
    async def _forward_chunk(chunk, dead_nodes, override_target) -> bool
```

### `RingTransferClient` (src/ring_transfer.py)

```python
class RingTransferClient:
    _client_channel: grpc.aio.Channel
    _client_stub: RingTransferStub

    async def run_ring_all_gather()
    async def _send_chunk_with_retry(chunk, dead_nodes) -> bool
```

### `WorkerNode` (src/worker_node.py)

```python
class WorkerNode:
    state: NodeState
    gossip_engine: GossipEngine
    ring_servicer: RingTransferServicer
    ring_client: RingTransferClient
    bootstrap: Optional[BootstrapServer]

    async def start()
    async def shutdown()
    async def _state_machine_loop()
```

---

## Configuration (src/config.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--node-id` | REQUIRED | Node identifier (auto-built if not provided) |
| `--gossip-port` | REQUIRED | FastAPI server port |
| `--grpc-port` | REQUIRED | gRPC server port |
| `--bootstrap` | `""` | Bootstrap URL (empty = this IS bootstrap) |
| `--host` | `0.0.0.0` | Bind address |
| `--data-dir` | `data` | Root data directory |
| `--model` | `openai-community/gpt2` | Model name |
| `--rounds` | `3` | Number of FL rounds |
| `--quiescence` | `3.0` | Silence window (seconds) |
| `--chunk-size` | `4MB` | Chunk size for splitting |

---

## Known Issues (Legacy)

1. **Bootstrap's WorkerNode state is separate from BootstrapServer's state** — requires `_sync_bootstrap_state_loop()` to reconcile.
2. **Quiescence loop keeps firing during barrier waits** — mitigated by `_paused` flag.
3. **No persistent storage for global table** — lost on restart.
4. **Dashboard shows stale data during fast transitions** — no real-time update throttling.
5. **No encryption/authentication** — all communication is plaintext.

---

## Testing the Legacy System

```bash
# Start 4-node cluster
python -m src.worker_node --node-id "127.0.0.1:5400" --gossip-port 5400 --grpc-port 5500 &
python -m src.worker_node --node-id "127.0.0.1:5401" --gossip-port 5401 --grpc-port 5501 --bootstrap "http://127.0.0.1:5400" &
python -m src.worker_node --node-id "127.0.0.1:5402" --gossip-port 5402 --grpc-port 5502 --bootstrap "http://127.0.0.1:5400" &
python -m src.worker_node --node-id "127.0.0.1:5403" --gossip-port 5403 --grpc-port 5503 --bootstrap "http://127.0.0.1:5400" &

# View dashboard
open http://127.0.0.1:5400/dashboard
```

---

## Contact

For questions about this legacy implementation, refer to the git history or the original developer.

**End of Legacy Document**