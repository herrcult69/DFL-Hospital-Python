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

---

### 3. The Gossip Protocol (Heartbeats & New Joins)

Instead of relying on a central server to manage liveness and network updates, hospitals spread messages node-to-node — exactly like spreading a rumor. Gossip handles two specific events:

**New Joins (Birth):** When a new hospital registers with the Bootstrap, the Bootstrap injects a "Node X Joined" rumor into the gossip network. That message hops from neighbor to neighbor until every hospital has added Node X to their Global Table. The Bootstrap also sends signed HTTP rewire instructions to the specific existing nodes whose neighbor connections changed (see Section 1.5 for why this must be Bootstrap-signed).

**Heartbeats (Liveness):** To confirm that peers are still online, each node periodically gossips an `"I am alive"` message at the start of Phase 1. Every node maintains a liveness checklist as these heartbeats spread. The heartbeat mechanism only needs to catch nodes that died **after** Phase 2 ended (during Phases 3 or 4) — nodes that died **during** Phase 2 are already known to everyone via piggyback metadata in the ring packet (see Phase 2). If a node's heartbeat is missing after the stability timer, unaffected nodes independently drop it from their Global Table. Affected nodes (those who had the dead node as a K-neighbor) additionally send a rewire request to the Bootstrap to repair their Neighbor Map.

#### Rumor Message Schema

Every gossip rumor carries these base fields. The `round` field on every message is the core safety mechanism — any message whose round does not match the receiver's current round is silently discarded, preventing stale or replayed messages from corrupting state. The `ttl` field ensures rumors die naturally after reaching the whole network rather than circulating forever.

| Field | Type | Present On | Notes |
|:------|:-----|:-----------|:------|
| `type` | string | All | `HEARTBEAT`, `JOIN`, `READY`, `DONE` |
| `originator_id` | string | All | Node ID of whoever created the rumor |
| `round` | int | All | Current round — mismatched round messages are discarded |
| `rumor_id` | string | All | `type:originator_id:round` — SIR deduplication key |
| `ttl` | int | All | Hops remaining. Each node decrements by 1 before forwarding. Rumor is dropped when TTL hits 0. Recommended starting value: `ceil(log2(N)) + 2` |
| `target_phase` | string | READY only | The phase this node is ready to enter e.g. `PHASE_2` — distinguishes which barrier transition this signal belongs to |

> All rumor types propagate via full SIR gossip over the K-graph. Each node forwards to all K neighbors and deduplicates via `rumor_id`. There are no direct 1-hop-only messages.

---

### 4. What Each Node Stores in Memory

| Structure | Description | Who Holds It |
|:----------|:------------|:-------------|
| **Global Table** | Complete list of every active node (IP, ID, Public Key). Should look identical on every node. | Every node |
| **Neighbor Map** | Exactly K=4 direct gossip neighbors on the K-Regular Graph. Used for gossip only. Persists across rounds. | Every node |
| **Ring Map** | Left and right neighbors for Phase 2 streaming. Derived fresh each round by sorting the Global Table alphabetically by Node ID. | Every node (computed locally) |
| **Master Adjacency List** | The authoritative blueprint of all K-graph connections. Used to compute edge-breaking when a new node joins. | Bootstrap only |

> **Key distinction:** The Neighbor Map is persistent topology infrastructure. The Ring Map is computed fresh and independently by every node at the start of each Phase 1 — it is ephemeral and round-specific.

---

## 🕸️ The Dual-Topology Design (The Shape of the Network)

Our network morphs into two completely different shapes depending on what it is trying to do.

### Topology A: The Control Plane (The K-Regular Graph)

- **Used for:** Spreading lightweight gossip messages (heartbeats, join announcements).
- **Shape:** Every node maintains exactly K=4 neighbors. The connections are calculated by the Bootstrap during the join handshake to maintain mathematical regularity.
- **Why K=4?** If 100 hospitals all talked to all other hospitals simultaneously, the network would be flooded with 10,000 messages per second. By restricting every node to exactly 4 neighbors, a rumor still spreads exponentially fast while keeping bandwidth under control.

### Topology B: The Data Plane (The Directed Ring)

- **Used for:** Transferring large machine learning model weight files (potentially several GB).
- **Shape:** Every node sorts the Global Table alphabetically by Node ID and forms a ring: Node A → Node B → Node C → ... → Node A. This ring is recomputed from scratch at the start of every round.
- **Why a ring?** You cannot use the gossip graph to send multi-GB files — the network would instantly saturate. The ring topology passes heavy files in a continuous pipeline, like passing buckets of water down a line of firefighters.

---

## 🔗 The Join Handshake & Edge-Breaking

When a new node joins, the Bootstrap must wire it into the K-Regular Graph without breaking the mathematical regularity (every node must still have exactly K=4 connections). The algorithm works as follows:

1. New node sends an HTTP `POST /join` to the Bootstrap.
2. Bootstrap replies with the full Global Table.
3. Bootstrap finds K/2 = 2 disjoint edges (pairs of nodes that are currently neighbors and share no endpoints with each other) in the Master Adjacency List.
4. It breaks both edges, freeing exactly K = 4 endpoints, and wires all 4 to the new node.
5. Bootstrap sends Bootstrap-signed HTTP rewire instructions to the 4 affected existing nodes to update their Neighbor Maps.
6. Bootstrap injects a "Node X Joined" gossip rumor so all other nodes update their Global Tables.

> **Why K/2 edges?** Each broken edge frees 2 endpoints. Breaking K/2 = 2 edges frees 2 × 2 = 4 endpoints — exactly K new connections for the incoming node.

> **Note:** The Bootstrap rejects `POST /join` requests with HTTP 423 (Locked) if the network is not currently in Phase 1. New nodes must wait and retry until the next Phase 1 window opens.

### Why Rewire Instructions Must Come From Bootstrap Only

Rewire messages are the highest-privilege messages in the network because they change the physical topology. Every node's rewire endpoint (`POST /update-neighbors`) accepts only messages signed with the Bootstrap's private key. Any rewire request from any other source is rejected outright.

This is not just a convenience rule — it prevents a compromised node from impersonating rewire authority: telling a peer "disconnect from your current neighbor and connect to me instead" in order to position itself as a man-in-the-middle on Phase 2 model streams. Because no normal node has any legitimate need to rewire another node, the rule has zero false positives.

---

# Section 2: The 4-Phase Loop

Because there is no central master to tell hospitals when to start or stop, the entire network runs on a strict, self-governing 4-phase loop. Every hospital independently advances through the phases using local conditions and gossip signals — no coordination required.

## The Phase State Machine

Every node independently runs a local state machine with two variables:
- A `phase` enum (`PHASE_1 → PHASE_2 → PHASE_3 → PHASE_4`)
- A `round` integer that increments every time the network completes a full cycle

Every incoming message — gossip rumors, gRPC chunks, rewire instructions — is validated against the receiver's current phase and round number. A message from the wrong phase or a stale round is silently discarded. This is what prevents a slow node from injecting Phase 2 gRPC streams into a network that has already moved to Phase 3, and what blocks a resurrected node from re-entering mid-round without a fresh join handshake.

When a new node joins, the Bootstrap's response includes the current `round` number and current `phase` so the new node is immediately synchronized before it sends its first heartbeat.

---

## The READY Barrier (Phase Sync)

Forming the Ring is not enough to start Phase 2 — a fast node cannot begin gRPC streaming while slow nodes are still finishing gossip. To synchronize the transition without a central boss, every node uses a **READY Barrier**.

When a node has locked its Global Table and computed its Ring Map, it does not immediately switch to Phase 2. Instead it gossips a `READY(target=PHASE_2, round=R)` rumor over the K-Regular graph and waits. As other nodes finish Phase 1 they each gossip their own `READY` signal. Each node collects these signals and only flips its phase enum to Phase 2 once every node in its Global Table has sent `READY(PHASE_2, R)`.

> READY rumors travel over the K-graph for all transitions — it is faster than the ring at O(log N) vs O(N), always available, and unaffected by any ring patching that occurred during Phase 2.

**How READY signals propagate:** A READY rumor travels via full SIR gossip — each receiving node decrements the TTL and forwards it to all K neighbors, deduplicating via `rumor_id`. This ensures the signal reaches every node in the network via exponential fan-out, not just the sender's direct neighbors. Each node maintains a local `ready_set` — a set of originator IDs who have sent `READY(PHASE_2, R)`. A node advances to Phase 2 when `ready_set` equals the full cleaned Global Table, or when the barrier timeout drops unresponsive nodes from the working set.

If a node does not send `READY` before the barrier timeout expires, it is treated as dead for this round — dropped from the local working set — and the barrier resolves with the remaining nodes. The network does not restart Phase 1; it proceeds with a reduced but valid participant set. The only exception is if the surviving count drops below the minimum viable network size (K+1 nodes), in which case the node holds and waits.

> This same READY Barrier fires at every phase transition: Phase 1→2, Phase 2→3, Phase 3→4, and Phase 4→1.

> **Note:** The relationship between the `"I am done, Round R"` push and the READY Barrier for the Phase 4→1 transition is an open implementation decision — whether they are unified into one message or kept as two separate signals will be determined during the rewrite.

---

## Phase 1: Consensus & Ring Formation

**What happens:** Phase 1 runs in a strict sequence. Every node executes these steps independently in this exact order:

1. **Start gossip** — broadcast `HEARTBEAT(round=R)` to K neighbors, begin forwarding incoming rumors
2. **Wait for stability** — hold until both conditions are true:
   - (a) Minimum floor elapsed (e.g. 3 seconds)
   - (b) Global Table has not changed for X seconds (e.g. 5 seconds)
3. **Lock the Global Table** — no further changes accepted
4. **Dead Node Cleanup** — any node absent from the heartbeat checklist is declared dead:
   - Remove from local Global Table
   - If it was a K-neighbor → send `POST /rewire-request` to Bootstrap
   - Wait for Bootstrap's signed rewire response before continuing
5. **Form the Ring Map** — sort the now-cleaned Global Table alphabetically → derive left and right neighbors
6. **Gossip READY** — broadcast `READY(target=PHASE_2, round=R)` over the K-graph
7. **Wait for barrier** — collect `READY(PHASE_2, R)` from every node in the cleaned Global Table:
   - If a node times out → drop from working set, continue
   - If working set drops below K+1 → hold and wait (see minimum viable network below)
8. **Flip phase** — set local phase enum to `PHASE_2`

### Minimum Viable Network

If the surviving node count drops below K+1, the network cannot form a valid ring or maintain K-graph redundancy. The node holds, re-runs heartbeat gossip every 10 seconds, and waits. The hold breaks naturally when a new node joins via Bootstrap — the injected JOIN rumor updates the Global Table and the count rises above the threshold.

### Dead Node Cleanup (Detail)

Once the stability timer fires and the Global Table is locked, any node absent from the heartbeat checklist is officially declared dead for this round. Unaffected nodes silently remove it from their local Global Table. Affected nodes — those who listed the dead node as a K-graph neighbor — send a `POST /rewire-request` to the Bootstrap. The Bootstrap runs the edge-breaking algorithm and sends signed rewire instructions back before the READY Barrier fires, ensuring the K-graph is fully repaired before Phase 2 begins.

> A node that crashed and restarted cannot re-enter at this point — its phase/round mismatch causes it to be rejected; it must perform a full re-join via Bootstrap starting at Round R+1.

### Forming the Ring

Every node independently sorts the locked Global Table alphabetically by Node ID. From this sorted list, it derives its Ring Map — its immediate Left Neighbor and Right Neighbor for Phase 2. Because every node performs the same deterministic sort on the same locked data, all Ring Maps across the network are consistent with zero coordination needed.

---

## Phase 2: Parameter Transfer (The Ring All-Gather)

**What happens:** Hospitals share the AI knowledge (model weights) they learned in the previous training round.

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

**What happens:** This phase requires zero network messages. It happens entirely inside each hospital's local computer.

**The math:** Each hospital takes the full set of parameter files it collected in Phase 2 and runs **FedIT SVD** — a mathematical merging function that produces a single unified model.

> **The result:** Because every node started with the exact same parameter files and runs the exact same deterministic math, every hospital independently produces the exact same merged global model — with no communication required.

---

## Phase 4: Local Training (The Homework)

**What happens:** Each hospital loads the freshly merged global model into its PyTorch environment and fine-tunes it using **LoRA** on its own private, local patient data.

> **The result:** The model becomes incrementally smarter on local patterns. The newly trained local adapter is saved to disk, ready to be shared in the next Phase 2.

> The patient data never leaves the hospital at any point in this process.

---

## Phase 4 → Phase 1 Transition: "I Am Done" Push

When a node finishes Phase 4, it gossips an `"I am done, Round R"` (`DONE`) rumor over the full K-graph via SIR propagation — not just a direct push to K neighbors. This ensures every node in the network eventually receives the signal, not just the sender's 4 direct neighbors.

Each node maintains a local `done_set` — a set of originator IDs who have sent `DONE(round=R)`. A node advances to Phase 1 when `done_set` equals the full Global Table, or when unresponsive nodes exceed the timeout and are dropped from the working set. The `round` field on the DONE message ensures a late-arriving DONE from Round R−1 is discarded and never counted toward the current round's barrier.

> **Note:** The relationship between the `DONE` push and the READY Barrier for the Phase 4→1 transition is an open implementation decision — whether they are unified into one message or kept as two separate signals will be determined during the rewrite.

### Why "I Am Done" (Push) is Better Than "Are You Done?" (Pull)

|  | Pull Polling (old design) | Push Gossip (current design) |
|:--|:--------------------------|:-----------------------------|
| Network traffic | O(K × N) redundant HTTP requests every poll interval | Only N push messages total, sent once on completion |
| Latency | Depends on poll interval, not actual finish time | Triggers immediately when training completes |
| Architecture feel | Each node acts like a mini-coordinator | Fully decentralized, reuses existing gossip layer |
| Infrastructure | Requires a separate polling loop | No extra infrastructure needed |

---

# Section 3: Byzantine Fault Tolerance !! ONLY FOR CONTEXT NO IMPLEMENTATION"

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
