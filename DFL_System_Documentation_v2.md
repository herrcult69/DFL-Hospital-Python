# DFL Hospital Symptom Checker — System Documentation

---

# Section 1: System Overview & Core Terminology

## The Problem We Are Solving

Traditional AI requires pooling all training data into one central server. In healthcare, sharing raw patient data violates strict privacy laws (like HIPAA and GDPR).

Federated Learning (FL) solves this by sending the *AI model* to the hospitals instead of sending the *patient data* to the server. However, traditional FL still relies on a "Central Master Server" to orchestrate everything, creating a massive bandwidth bottleneck and a single point of failure.

## Our Solution: Decentralized Federated Learning (DFL)

We are building a **SEMI-DECENTRALIZED** network. There is no central master. Every hospital is an ALMOST equal participant. They train the AI on their private data, and then mathematically share what they learned directly with each other.

---

## 📖 Core Concepts & Terminology

### 1. The Node (A Hospital)

A single computer running our Python software. It holds:

- Its own private patient data (never leaves this machine)
- A PyTorch AI model for local training
- A FastAPI web server (HTTP) for lightweight control messages
- A gRPC server for high-speed model weight streaming

---

### 2. The Bootstrap / Seed Node (The Welcome Desk)

If there is no central master, how does a new hospital know who to connect to?

We designate the hospital that has redundancy and hardening features to act as the Bootstrap. It is a completely normal hospital node that trains the AI and merges models just like everyone else. However, it has two extra responsibilities:

- It acts as the network's **Welcome Desk**: when a new node turns on, that node registers with the Bootstrap and receives its neighbours' addresses and the Global Table (a list of all active nodes — their IPs, Public Keys, and IDs).
- The Bootstrap holds the **Master Adjacency List**: the authoritative blueprint of who is connected to whom in the K-Regular gossip graph, so it can mathematically rewire the graph when a new node joins or when a node dies.

Once the join handshake is complete, the Bootstrap has no elevated role for the rest of the round.

> 📝 **IMPLEMENTATION NOTE:** `JoinResponse` now also includes the current `round` integer. The worker node sets `state.round = data["round"]` immediately on join. Without this, all its gossip rumors carry `round=0` and are discarded by everyone on round mismatch.

---

### 3. The Gossip Protocol (Heartbeats & New Joins)

Instead of relying on a central server to manage liveness and network updates, hospitals spread messages node-to-node — exactly like spreading a rumor. Gossip handles two specific events:

**New Joins (Birth):** When a new hospital registers with the Bootstrap, the Bootstrap injects a "Node X Joined" rumor into the gossip network. That message hops from neighbor to neighbor until every hospital has added Node X to their Global Table. The Bootstrap also sends signed HTTP rewire instructions to the specific existing nodes whose neighbor connections changed (see Section 1.5 for why this must be Bootstrap-signed).

**Heartbeats (Liveness):** To confirm that peers are still online, each node periodically gossips an `"I am alive"` message at the start of Phase 1. Every node maintains a liveness checklist as these heartbeats spread. The heartbeat mechanism only needs to catch nodes that died **after** Phase 2 ended (during Phases 3 or 4) — nodes that died **during** Phase 2 are already known to everyone via piggyback metadata in the ring packet (see Phase 2). If a node's heartbeat is missing after the stability timer, unaffected nodes independently drop it from their Global Table. Affected nodes (those who had the dead node as a K-neighbor) additionally send a rewire request to the Bootstrap to repair their Neighbor Map.

> 📝 **IMPLEMENTATION NOTE (Heartbeat):** The heartbeat loop is a background asyncio task (`_heartbeat_loop`) that runs continuously while `state.phase == Phase.PHASE_1`. It fires `originate_heartbeat()` every `heartbeat_interval` seconds (default: 2s). The loop stops automatically when the phase advances to `PHASE_2`.

#### Rumor Message Schema

Every gossip rumor carries these base fields. The `round` field on every message is the core safety mechanism — any message whose round does not match the receiver's current round is silently discarded, preventing stale or replayed messages from corrupting state. The `ttl` field ensures rumors die naturally after reaching the whole network rather than circulating forever.

| Field | Type | Present On | Notes |
|:------|:-----|:-----------|:------|
| `type` | string | All | `HEARTBEAT`, `JOIN`, `READY`, `DONE` |
| `originator_id` | string | All | Node ID of whoever created the rumor |
| `round` | int | All | Current round — mismatched round messages are discarded |
| `rumor_id` | string | All | Deduplication key — format varies by type (see below) |
| `ttl` | int | All | Hops remaining. Each node decrements by 1 before forwarding. Rumor is dropped when TTL hits 0. Default: 10 (dynamic `ceil(log2(N)) + 2` deferred) |
| `payload` | dict | Varies | `JOIN` carries `{"node_id": "..."}`, `READY` carries `{"target_phase": "PHASE_2"}`, others empty |
| `target_phase` | string | READY (in payload) | The phase this node is ready to enter e.g. `PHASE_2` — distinguishes which barrier transition this signal belongs to |

> 📝 **IMPLEMENTATION NOTE (`rumor_id` format):** The original design used `type:originator_id:round` for all types. This was changed for HEARTBEAT and READY to include a `time.time()` timestamp suffix:
>
> | Type | `rumor_id` Format |
> |:-----|:-----------------|
> | `JOIN` | `JOIN:{originator_id}:{round}` |
> | `HEARTBEAT` | `HEARTBEAT:{originator_id}:{round}:{timestamp}` |
> | `READY` | `READY:{originator_id}:{round}:{timestamp}` |
> | `DONE` | `DONE:{originator_id}:{round}` |
>
> **Why timestamp for HEARTBEAT?** Each beat must be a unique rumor so multiple beats propagate independently through the K-graph for redundancy. Without the timestamp, all beats in a round share the same `rumor_id` and beats 2, 3, 4... are silently dropped by SIR dedup. The `heartbeat_seen` set only stores `originator_id` (not `rumor_id`), so multiple beats still produce only one liveness entry — correctness is preserved.
>
> **Why timestamp for READY?** Prevents dedup collision across barrier retries in the same round.

> All rumor types propagate via full SIR gossip over the K-graph. Each node forwards to all K neighbors and deduplicates via `rumor_id`. There are no direct 1-hop-only messages.

---

### 4. What Each Node Stores in Memory

| Structure | Description | Who Holds It |
|:----------|:------------|:-------------|
| **Global Table** | Complete sorted list of every active node (IP, ID). Should look identical on every node. Locked at end of Phase 1 stability timer. | Every node |
| **Neighbor Map** | Exactly K=4 direct gossip neighbors on the K-Regular Graph. Used for gossip only. Persists across rounds. | Every node |
| **Ring Map** | Left and right neighbors for Phase 2 streaming. Derived at end of Phase 1 by sorting the locked Global Table alphabetically by Node ID. | Every node (computed locally) |
| **Master Adjacency List** | The authoritative blueprint of all K-graph connections. Used to compute edge-breaking when a new node joins. | Bootstrap only |
| **heartbeat_seen** | Set of node IDs from whom a HEARTBEAT was received this Phase 1. Used for dead node detection. | Every node |
| **ready_set** | Set of node IDs from whom READY(PHASE_2) was received. Used for the READY barrier. | Every node |
| **seen_rumors** | Set of `rumor_id` strings already processed this round. SIR dedup. Reset at start of each round. | Every node |

> **Key distinction:** The Neighbor Map is persistent topology infrastructure. The Ring Map is computed fresh and independently by every node at the **end** of each Phase 1 — it is ephemeral and round-specific.

> 📝 **IMPLEMENTATION NOTE:** `NodeState` now carries the following additional fields implemented this iteration:
> ```
> phase:                  Phase      = Phase.PHASE_1
> ring_left:              str | None = None
> ring_right:             str | None = None
> table_locked:           bool       = False
> last_table_change_time: float      = time.time()
> ready_set:              set[str]   = set()
> ready_timeout:          float      = 0.0
> heartbeat_seen:         set[str]   = set()
> ```
> `reset_phase1()` clears `heartbeat_seen`, `seen_rumors`, `ready_set`, resets `table_locked = False` and `last_table_change_time`. Called at Phase 4→1 transition.

---

## 🕸️ The Dual-Topology Design (The Shape of the Network)

Our network morphs into two completely different shapes depending on what it is trying to do.

### Topology A: The Control Plane (The K-Regular Graph)

- **Used for:** Spreading lightweight gossip messages (heartbeats, join announcements, READY signals).
- **Shape:** Every node maintains exactly K=4 neighbors. The connections are calculated by the Bootstrap during the join handshake to maintain mathematical regularity.
- **Why K=4?** If 100 hospitals all talked to all other hospitals simultaneously, the network would be flooded with 10,000 messages per second. By restricting every node to exactly 4 neighbors, a rumor still spreads exponentially fast while keeping bandwidth under control.

### Topology B: The Data Plane (The Directed Ring)

- **Used for:** Transferring large machine learning model weight files (potentially several GB).
- **Shape:** Every node sorts the Global Table alphabetically by Node ID and forms a ring: Node A → Node B → Node C → ... → Node A. This ring is recomputed from scratch at the end of every Phase 1.
- **Why a ring?** You cannot use the gossip graph to send multi-GB files — the network would instantly saturate. The ring topology passes heavy files in a continuous pipeline, like passing buckets of water down a line of firefighters.

---

## 🔗 The Join Handshake & Edge-Breaking

When a new node joins, the Bootstrap must wire it into the K-Regular Graph without breaking the mathematical regularity (every node must still have exactly K=4 connections). The algorithm works as follows:

1. New node sends an HTTP `POST /join` to the Bootstrap.
2. Bootstrap checks `table_locked` — if `True`, immediately returns **HTTP 423 (Locked)**. New node must wait and retry.
3. Bootstrap replies with the full Global Table, current `round`, and the new node's assigned neighbors.
4. Bootstrap finds K/2 = 2 disjoint edges (pairs of nodes that are currently neighbors and share no endpoints with each other) in the Master Adjacency List.
5. It breaks both edges, freeing exactly K = 4 endpoints, and wires all 4 to the new node.
6. Bootstrap sends HTTP rewire instructions to the 4 affected existing nodes to update their Neighbor Maps.
7. Bootstrap injects a "Node X Joined" gossip rumor so all other nodes update their Global Tables.

> **Why K/2 edges?** Each broken edge frees 2 endpoints. Breaking K/2 = 2 edges frees 2 × 2 = 4 endpoints — exactly K new connections for the incoming node.

> 📝 **IMPLEMENTATION NOTE:** The 423 check happens **before** `graph.register()` is called. This is critical — if the check happened after, the adjacency graph would be mutated but `global_table` would reject the add (due to lock), causing permanent desync between `adj` and `global_table`.

### Why Rewire Instructions Must Come From Bootstrap Only

Rewire messages are the highest-privilege messages in the network because they change the physical topology. Every node's rewire endpoint (`POST /rewire`) accepts only messages from Bootstrap. Any rewire request from any other source is rejected outright.

This is not just a convenience rule — it prevents a compromised node from impersonating rewire authority: telling a peer "disconnect from your current neighbor and connect to me instead" in order to position itself as a man-in-the-middle on Phase 2 model streams. Because no normal node has any legitimate need to rewire another node, the rule has zero false positives.

> 📝 **IMPLEMENTATION NOTE:** HMAC/signatures on rewire messages are **deferred**. Currently rewire accepts any caller — signature verification is a hardening task for a future iteration.

---

# Section 2: The 4-Phase Loop

Because there is no central master to tell hospitals when to start or stop, the entire network runs on a strict, self-governing 4-phase loop. Every hospital independently advances through the phases using local conditions and gossip signals — no coordination required.

## The Phase State Machine

Every node independently runs a local state machine with two variables:
- A `phase` enum (`PHASE_1 → PHASE_2 → PHASE_3 → PHASE_4`)
- A `round` integer that increments every time the network completes a full cycle

Every incoming message — gossip rumors, gRPC chunks, rewire instructions — is validated against the receiver's current phase and round number. A message from the wrong phase or a stale round is silently discarded. This is what prevents a slow node from injecting Phase 2 gRPC streams into a network that has already moved to Phase 3, and what blocks a resurrected node from re-entering mid-round without a fresh join handshake.

When a new node joins, the Bootstrap's response includes the current `round` number so the new node is immediately synchronized before it sends its first heartbeat.

---

## The READY Barrier (Phase Sync)

Forming the Ring is not enough to start Phase 2 — a fast node cannot begin gRPC streaming while slow nodes are still finishing gossip. To synchronize the transition without a central boss, every node uses a **READY Barrier**.

When a node has locked its Global Table and computed its Ring Map, it does not immediately switch to Phase 2. Instead it gossips a `READY(target=PHASE_2, round=R)` rumor over the K-Regular graph and waits. As other nodes finish Phase 1 they each gossip their own `READY` signal. Each node collects these signals and only flips its phase enum to Phase 2 once every node in its Global Table has sent `READY(PHASE_2, R)`.

> READY rumors travel over the K-graph for all transitions — it is faster than the ring at O(log N) vs O(N), always available, and unaffected by any ring patching that occurred during Phase 2.

**How READY signals propagate:** A READY rumor travels via full SIR gossip — each receiving node decrements the TTL and forwards it to all K neighbors, deduplicating via `rumor_id`. This ensures the signal reaches every node in the network via exponential fan-out, not just the sender's direct neighbors. Each node maintains a local `ready_set` — a set of originator IDs who have sent `READY(PHASE_2, R)`. A node advances to Phase 2 when `ready_set` equals the full cleaned Global Table, or when the barrier timeout drops unresponsive nodes from the working set.

If a node does not send `READY` before the barrier timeout expires, it is treated as dead for this round — dropped from the local working set — and the barrier resolves with the remaining nodes. The network does not restart Phase 1; it proceeds with a reduced but valid participant set. The only exception is if the surviving count drops below the minimum viable network size, in which case the node holds and waits.

> This same READY Barrier fires at every phase transition: Phase 1→2, Phase 2→3, Phase 3→4, and Phase 4→1.

> **Note:** The relationship between the `"I am done, Round R"` push and the READY Barrier for the Phase 4→1 transition is an open implementation decision — whether they are unified into one message or kept as two separate signals will be determined during the rewrite.

> 📝 **IMPLEMENTATION NOTE (`_process()` READY handler):** The handler checks `target == "PHASE_2"` — **NOT** `target == state.phase.value`. This is a critical distinction: when a READY rumor arrives, the receiving node is still in PHASE_1 (that's the whole point of the barrier). Comparing against `state.phase.value` would yield `"PHASE_1" != "PHASE_2"` and silently drop every READY rumor. Additionally, the handler checks `originator_id in state.global_table` as a sanity check to reject signals from unknown or already-evicted nodes.

> 📝 **IMPLEMENTATION NOTE (`_wait_ready_barrier()` return type):** The function returns `bool`. `True` = all nodes ready, advance. `False` = not enough nodes remain after timeout, hold. The phase flip in `_end_phase1()` is **gated on this return value** — `if not advanced: return` before `state.phase = Phase.PHASE_2`. Returning `None` (e.g. using `break` instead of `return True`) is falsy and causes the barrier to permanently hold even on success.

---

## Phase 1: Consensus & Ring Formation

> ✅ **STATUS: FULLY IMPLEMENTED**

**What happens:** Phase 1 runs in a strict sequence. Every node executes these steps independently in this exact order:

1. **Start gossip** — broadcast `HEARTBEAT(round=R)` to K neighbors every `heartbeat_interval` seconds, begin forwarding incoming rumors
2. **Wait for stability** — background `_stability_timer` task checks every 1 second until both conditions are true:
   - (a) Minimum floor elapsed (`phase1_floor`, default: 5s)
   - (b) Global Table has not changed for `stability_window` seconds (default: 10s) — tracked via `last_table_change_time`, updated by `add_node()`
3. **Lock the Global Table** — `state.table_locked = True`. `add_node()` silently rejects further additions. `/join` returns HTTP 423.
4. **Dead Node Cleanup** — any node absent from `heartbeat_seen` is declared dead:
   - Remove from local Global Table
   - *(Deferred: K-neighbor rewire-request to Bootstrap — not yet implemented)*
5. **Minimum viable network check** — if fewer than 2 nodes remain after cleanup: unlock table, reset `last_table_change_time`, return without advancing. Stability timer keeps watching.
6. **Form the Ring Map** — `global_table` is already sorted (maintained by `add_node()`). Derive: `ring_left = global_table[(idx-1) % n]`, `ring_right = global_table[(idx+1) % n]`
7. **Gossip READY** — originate `READY(target=PHASE_2, round=R)`, add self to `ready_set`, mark rumor seen locally
8. **Wait for barrier** — `_wait_ready_barrier()` polls every 0.5s:
   - Advances when `ready_set >= set(global_table)`
   - On timeout: evict missing nodes from `global_table`. If fewer than 2 remain → return `False` (hold). Else → return `True`.
9. **Flip phase** — only if barrier returns `True`: `state.phase = Phase.PHASE_2`. This also stops the heartbeat loop.

### Minimum Viable Network

> 📝 **IMPLEMENTATION NOTE:** The original doc used K+1 as the minimum viable threshold. The current implementation uses **2 nodes** as the minimum — sufficient for the 3-4 PC demo. A 2-node ring (A→B→A) is mathematically valid: each node's `ring_left` and `ring_right` both point to the other node, and aggregation works with 2 parameter files.

If the surviving node count drops below 2, the node:
- Unlocks the table
- Resets the stability window clock
- Stays in Phase 1 — heartbeat loop keeps running
- Stability timer keeps watching — fires again once enough nodes join and table re-stabilizes

The hold breaks naturally when a new node joins via Bootstrap — the injected JOIN rumor updates the Global Table and raises the count.

### Dead Node Cleanup (Detail)

Once the stability timer fires and the Global Table is locked, any node absent from the `heartbeat_seen` checklist is officially declared dead for this round. Unaffected nodes silently remove it from their local Global Table.

> 📝 **IMPLEMENTATION NOTE (Deferred):** The original design also requires affected nodes (those who listed the dead node as a K-graph neighbor) to send a `POST /rewire-request` to Bootstrap and wait for a signed rewire response before continuing. **This is not yet implemented.** Currently all dead-node evictions are silent table removals only. K-graph repair after dead node eviction is a future task.

> 📝 **IMPLEMENTATION NOTE (Deferred):** A **minimum 2-beat requirement** for `heartbeat_seen` is also deferred. A node that beats once and then dies before Phase 2 still appears in `heartbeat_seen` and passes liveness. Phase 2 ring patching catches these cases at runtime. The 2-beat hardening can be added by changing `heartbeat_seen` from `set[str]` to `dict[str, int]` and requiring count ≥ 2.

> A node that crashed and restarted cannot re-enter at this point — its phase/round mismatch causes it to be rejected; it must perform a full re-join via Bootstrap starting at Round R+1.

### Forming the Ring

Every node independently sorts the locked Global Table alphabetically by Node ID. From this sorted list, it derives its Ring Map — its immediate Left Neighbor (`ring_left`) and Right Neighbor (`ring_right`) for Phase 2. Because every node performs the same deterministic sort on the same locked data, all Ring Maps across the network are consistent with zero coordination needed.

> 📝 **IMPLEMENTATION NOTE:** Plain Python string sort (lexicographic) is used — no hashing. This is fully deterministic across all machines given identical input. Python's `hash()` is randomized per-process (PYTHONHASHSEED) and must NOT be used for sorting.

---

## Phase 2: Parameter Transfer (The Ring All-Gather)

> 🔲 **STATUS: NOT YET IMPLEMENTED**

**What happens:** Hospitals share the AI knowledge (model weights) they learned in the previous training round.

**Inputs available from Phase 1 when this starts:**
- `state.ring_left` — node to receive gRPC stream from
- `state.ring_right` — node to send gRPC stream to
- `state.global_table` — locked, clean, sorted list of all participants
- `state.round` — current round number
- `state.phase` — will be `Phase.PHASE_2`

**The bucket brigade:** Using the Directed Ring derived in Phase 1, each node takes its local model parameters and streams them to its Right Neighbor over gRPC (using safetensors for efficient zero-copy transfer). When it receives parameters from its Left Neighbor, it saves them to disk and immediately forwards them onward to its Right Neighbor.

### Dead Node Detection (Piggyback Metadata)

If a node cannot reach its right neighbor during streaming, it records that node in a local `dead_this_round` set and piggybacks this information onto the packet metadata for every downstream node to read. It skips to the next alive node and dynamically recalculates the remaining hop count using:

```
TTL = (N - 1) - len(dead_this_round)
```

This ensures the chunk still reaches every surviving node exactly once. By the time Phase 2 completes, every node has an identical `dead_this_round` set from the metadata. This set is used in Phase 3 to know exactly how many parameter files to expect in the merge — preventing an off-by-one in the aggregation math.

> The Global Table itself is not modified at this point. Official eviction only happens in Phase 1 when the dead node fails to heartbeat.

Each node stores this as a local `dead_this_round` set — separate from the Global Table — which Phase 3 reads to know the exact number of parameter files to aggregate.

---

### The "Zero-Copy" Data Pipeline

If you have 10 nodes sharing 5GB models, loading all 10 models into RAM would crash the computer (Out-of-Memory error). We solve this using **Safetensors** and **Memory Mapping (mmap)**. Models are sent across the Ring as continuous byte-streams using gRPC. When Node B receives a piece of a model from Node A, it does not load it into RAM — it immediately appends the raw bytes directly to a file on its hard drive. Later, PyTorch reads the model straight from the hard drive layer by layer, skipping the RAM bottleneck completely.

---

### The Scatter-Gather Transfer

Instead of sending the whole model at once, each node splits its model into equal chunks. Every node simultaneously streams a chunk to their Right neighbor and receives a chunk from their Left neighbor. As a node receives a chunk, it writes it to disk and immediately streams it out to the Right neighbor again. Every chunk tracks how many hops it has taken. Because this is a Ring, a chunk will reach every single node in exactly N−1 hops (where N is the total number of nodes). Once a chunk hits N−1 hops, the transfer stops perfectly.

---

### Dynamic Ring Patching (Fault Tolerance)

What happens if a node crashes right in the middle of Phase 2?

If Node C loses power, Node B's gRPC stream will timeout. Node B looks at its Phase 1 Ring Map, skips dead Node C, and instantly opens a new connection to Node D. Node B adds "Node C is dead" to the metadata of the stream. This piggyback annotation travels with every subsequent hop — so by the time the ring completes, every node including the Bootstrap has seen the dead list. Each downstream node dynamically recalculates the remaining hop count. The ring heals itself instantly and the transfer continues without failing.

> **The result:** After N−1 hops around the ring, every single hospital holds the complete set of model parameters from all N nodes — without any central server ever touching the data.

---

## Phase 3: Aggregation (The Brain Merge)

> 🔲 **STATUS: NOT YET IMPLEMENTED**

**What happens:** This phase requires zero network messages. It happens entirely inside each hospital's local computer.

**The math:** Each hospital takes the full set of parameter files it collected in Phase 2 and runs **FedIT SVD** — a mathematical merging function that produces a single unified model. The number of files to aggregate = `len(global_table) - len(dead_this_round)`.

> **The result:** Because every node started with the exact same parameter files and runs the exact same deterministic math, every hospital independently produces the exact same merged global model — with no communication required.

---

## Phase 4: Local Training (The Homework)

> 🔲 **STATUS: NOT YET IMPLEMENTED**

**What happens:** Each hospital loads the freshly merged global model into its PyTorch environment and fine-tunes it using **LoRA** on its own private, local patient data.

> **The result:** The model becomes incrementally smarter on local patterns. The newly trained local adapter is saved to disk, ready to be shared in the next Phase 2.

> The patient data never leaves the hospital at any point in this process.

---

## Phase 4 → Phase 1 Transition: "I Am Done" Push

> 🔲 **STATUS: NOT YET IMPLEMENTED**

When a node finishes Phase 4, it gossips an `"I am done, Round R"` (`DONE`) rumor over the full K-graph via SIR propagation — not just a direct push to K neighbors. This ensures every node in the network eventually receives the signal, not just the sender's 4 direct neighbors.

Each node maintains a local `done_set` — a set of originator IDs who have sent `DONE(round=R)`. A node advances to Phase 1 when `done_set` equals the full Global Table, or when unresponsive nodes exceed the timeout and are dropped from the working set. The `round` field on the DONE message ensures a late-arriving DONE from Round R−1 is discarded and never counted toward the current round's barrier.

On transition: call `state.reset_phase1()` and increment `state.round`.

> **Note:** The relationship between the `DONE` push and the READY Barrier for the Phase 4→1 transition is an open implementation decision — whether they are unified into one message or kept as two separate signals will be determined during the rewrite.

### Why "I Am Done" (Push) is Better Than "Are You Done?" (Pull)

|  | Pull Polling (old design) | Push Gossip (current design) |
|:--|:--------------------------|:-----------------------------|
| Network traffic | O(K × N) redundant HTTP requests every poll interval | Only N push messages total, sent once on completion |
| Latency | Depends on poll interval, not actual finish time | Triggers immediately when training completes |
| Architecture feel | Each node acts like a mini-coordinator | Fully decentralized, reuses existing gossip layer |
| Infrastructure | Requires a separate polling loop | No extra infrastructure needed |

---

# Section 3: Byzantine Fault Tolerance

## Byzantine Problem in Neighbor Discovery

### 1. A malicious node can create many fake identities
- Affects the membership of the ring connection
- **Solution:** `signature = Sign(authority_private_key, IP || port || public_key of node)`
- A specific authority signs the IP:Port and public key of a specific node, so when in Phase 1, IP:Port, node public key and the CA signature is sent by gossiping

### 2. A malicious node may refuse to forward gossip messages
- There are other paths that a gossip message of a legitimate node can go through
- A malicious node that refuses to send its own gossip message → it is then excluded in the membership for ring connection

### 3. Suppress the information but later releases it
- The nodes go to the next phase
- New information is ignored

---

## Byzantine Problem in Ring Topology

### 1. A node may refuse to exchange (forward) parameters
- Treat it as a disabled node

### 2. A node may change the message sent from another node
- Each frame sent from another node must have: ID of that node, signature of that node, and the parameters
- The receiving node checks the signature based on the `public_key` from the neighbor discovery phase

### 3. A node itself may send false parameters (add an additional layer for parameters processing)
- **Median**
- **Trim out outliers**
- **Krum or Multi-Krum** — looks at the *entire model file* as a single vector. It calculates the geometric distance between all models and simply **selects the one single model that is most similar to the rest**, discarding all others.

---

# Section 4: Implementation State & Deferred Items

## Current Implementation Status

| Component | Status |
|:----------|:-------|
| Bootstrap join handshake | ✅ Done |
| K-regular graph / edge-breaking | ✅ Done |
| SIR gossip engine (spread, receive, dedup) | ✅ Done |
| JOIN rumor propagation | ✅ Done |
| Heartbeat loop (originate + receive) | ✅ Done |
| Stability timer (floor + stability window) | ✅ Done |
| Table lock + 423 on /join | ✅ Done |
| Dead node cleanup (heartbeat-based eviction) | ✅ Done |
| Ring Map formation (ring_left, ring_right) | ✅ Done |
| READY barrier (originate + collect + timeout) | ✅ Done |
| Phase flip to PHASE_2 | ✅ Done |
| round sync on join | ✅ Done |
| Phase 2 gRPC ring transfer | 🔲 Not started |
| Phase 3 FedIT SVD aggregation | 🔲 Not started |
| Phase 4 LoRA local training | 🔲 Not started |
| Phase 4→1 DONE rumor + round increment | 🔲 Not started |
| K-neighbor rewire-request after dead node eviction | 🔲 Deferred |
| HMAC / Bootstrap signature on rewires | 🔲 Deferred |
| Minimum 2-beat heartbeat hardening | 🔲 Deferred |
| Dynamic gossip_ttl = ceil(log2(N)) + 2 | 🔲 Deferred |
| Worker 423 retry loop | 🔲 Deferred |
| BaseNode OOP refactor (extract shared logic) | 🔲 Deferred (after Phase 2) |
| `_wait_for_quorum()` shared helper | 🔲 Deferred (after Phase 2) |

## NetworkConfig Reference

```python
@dataclass
class NetworkConfig:
    k:                  int   = 4       # gossip graph degree
    http_timeout:       float = 5.0
    host:               str   = "127.0.0.1"
    port:               int   = 8000
    grpc_port:          int   = 9000
    bootstrap_url:      str   = "127.0.0.1:8000"
    gossip_ttl:         int   = 10      # tune to ceil(log2(N)) + 2 later
    heartbeat_interval: float = 2.0     # seconds between heartbeat beats
    phase1_floor:       float = 5.0     # minimum Phase 1 duration before lock
    stability_window:   float = 10.0    # global_table must be stable this long
    ready_timeout:      float = 10.0    # max wait for READY barrier
```

## File Structure

```
dfl/
├── src/
│   ├── __init__.py
│   ├── config.py       — NetworkConfig dataclass
│   ├── models.py       — Rumor, RumorType, JoinRequest/Response, StatusResponse, etc.
│   ├── state.py        — NodeState dataclass + Phase enum
│   ├── graph.py        — GraphManager (Bootstrap only): K-regular graph, edge-breaking
│   ├── gossip.py       — GossipEngine: spread(), receive(), _process(), originate_heartbeat()
│   ├── bootstrap.py    — Bootstrap FastAPI app
│   └── worker.py       — Worker FastAPI app
├── templates/
│   ├── status.html
│   └── graph.html
└── main.py
```

## Planned Refactor: BaseNode OOP

Bootstrap and Worker currently duplicate: `_heartbeat_loop`, `_stability_timer`, `_end_phase1`, `_wait_ready_barrier`. After Phase 2 is complete, these will be extracted into a `BaseNode` parent class. The quorum-wait pattern (collect a set until it equals `global_table` or timeout) also appears in HEARTBEAT cleanup, READY barrier, and the future DONE barrier — this will be extracted as `_wait_for_quorum(seen_set, timeout, label) -> bool`.
