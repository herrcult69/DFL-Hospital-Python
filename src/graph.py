"""
K-Regular Graph Manager (K=4).

Maintains the master adjacency list for the physical K-regular graph overlay.
Used by the Bootstrap Server to manage node registration, edge rewiring, and
dead-node eviction.

Graph Invariant:
  Every node has exactly K = 4 neighbors at all times.
  The graph is stored as an undirected adjacency dict:
    { node_id: set(neighbor_id, ...) }
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Set

log = logging.getLogger(__name__)

# Default degree for the regular graph
K = 4


class GraphManager:
    """Thread-safe K-regular graph manager (K=4) for the gossip overlay."""

    def __init__(self, k: int = K) -> None:
        self._k = k
        self._adj: Dict[str, Set[str]] = {}  # node_id -> set of neighbor_ids

    # ── Public accessors ──────────────────────────────────────────────────

    @property
    def adj(self) -> Dict[str, Set[str]]:
        """Return a copy of the adjacency dict (safe for iteration)."""
        return {n: set(v) for n, v in self._adj.items()}

    @property
    def node_count(self) -> int:
        return len(self._adj)

    def neighbors(self, node_id: str) -> Set[str]:
        return set(self._adj.get(node_id, set()))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._adj

    def all_nodes(self) -> List[str]:
        return list(self._adj.keys())

    # ── Bootstrap a new graph ─────────────────────────────────────────────

    def bootstrap(self, seed_nodes: List[str]) -> None:
        """Initialize a K-regular graph from a list of seed node IDs.

        Uses a circulant graph construction: each node i connects to
        nodes (i-2, i-1, i+1, i+2) mod N. This guarantees perfect
        K-regularity for any N >= K+1.
        """
        self._adj = {}
        n = len(seed_nodes)

        if n < self._k + 1:
            # Fully connect all nodes (clique)
            for nid in seed_nodes:
                self._adj[nid] = {o for o in seed_nodes if o != nid}
            log.info(
                "Bootstrapped clique graph (%d nodes, < K+1)",
                n,
            )
            return

        # Circulant graph: each node connects to K nearest neighbors in the ring
        # Offsets: for K=4, connect to i-2, i-1, i+1, i+2
        offsets = list(range(-(self._k // 2), 0)) + list(range(1, self._k // 2 + 1))

        for idx, nid in enumerate(seed_nodes):
            self._adj[nid] = set()
            for off in offsets:
                nbr = seed_nodes[(idx + off) % n]
                self._adj[nid].add(nbr)

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, new_node: str) -> List[str]:
        """
        Register a new node into the K-regular graph.

        The algorithm:
          1. Pick (K/2) existing disjoint edges to break.
          2. Connect each freed endpoint to the new node.
        This perfectly preserves degree K for all nodes.

        Returns the list of K neighbors assigned to the new node.
        """
        if new_node in self._adj:
            log.warning("Node %s already registered", new_node)
            return list(self._adj[new_node])

        existing = [n for n in self._adj if len(self._adj[n]) == self._k]

        # Calculate how many edges to break: K/2 (e.g. K=4 → 2, K=2 → 1)
        edges_to_break = self._k // 2
        # Need at least edges_to_break * 2 saturated nodes to find that many disjoint edges
        if len(existing) < edges_to_break * 2:
            # Not enough saturated nodes — assign from unsaturated ones
            return self._register_sparse(new_node)

        edges = self._sample_disjoint_edges(existing, num_edges=edges_to_break)

        assigned: List[str] = []
        self._adj[new_node] = set()
        for a, b in edges:
            # Break A-B
            self._adj[a].discard(b)
            self._adj[b].discard(a)
            # Wire A to new_node
            self._adj[a].add(new_node)
            self._adj[new_node].add(a)
            # Wire B to new_node
            self._adj[b].add(new_node)
            self._adj[new_node].add(b)
            assigned.extend([a, b])

        log.info(
            "Registered node %s — broke %d edges, assigned %s",
            new_node,
            len(edges),
            assigned,
        )
        return assigned

    def _register_sparse(self, new_node: str) -> List[str]:
        """Fallback when not enough saturated nodes exist."""
        self._adj[new_node] = set()
        existing = [n for n in self._adj if n != new_node]
        random.shuffle(existing)
        count = 0
        for peer in existing:
            if count >= self._k:
                break
            if len(self._adj[peer]) < self._k:
                self._adj[peer].add(new_node)
                self._adj[new_node].add(peer)
                count += 1
        log.info(
            "Registered node %s (sparse fallback) — assigned %d neighbors",
            new_node,
            count,
        )
        return list(self._adj[new_node])

    # ── Dead node eviction ────────────────────────────────────────────────

    def evict(self, dead_node: str) -> Dict[str, List[str]]:
        """
        Remove a dead node and rewire its former neighbors.

        Returns a map: { surviving_node_id: [new_neighbors_to_add] }
        so the Bootstrap can gossip an "Official Update" with explicit
        edge-rewiring instructions.
        """
        if dead_node not in self._adj:
            return {}

        former_neighbors = list(self._adj[dead_node])
        del self._adj[dead_node]

        rewired: Dict[str, List[str]] = {}

        for fn in former_neighbors:
            if fn in self._adj:
                self._adj[fn].discard(dead_node)
                # Replenish fn's missing edge(s)
                missing = self._k - len(self._adj[fn])
                if missing > 0:
                    new_edges = self._find_fresh_neighbors(fn, missing)
                    for ne in new_edges:
                        self._adj[fn].add(ne)
                        self._adj[ne].add(fn)
                    rewired[fn] = new_edges

        log.info(
            "Evicted node %s — rewired %d former neighbors",
            dead_node,
            len(rewired),
        )
        return rewired

    def _find_fresh_neighbors(
        self, node_id: str, needed: int
    ) -> List[str]:
        """Find `needed` nodes that are not already neighbors of `node_id`."""
        pool = [
            n
            for n in self._adj
            if n != node_id
            and n not in self._adj[node_id]
            and len(self._adj[n]) < self._k
        ]
        random.shuffle(pool)
        return pool[:needed]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _sample_disjoint_edges(
        self, candidates: List[str], num_edges: int = 2
    ) -> List[tuple[str, str]]:
        """Pick `num_edges` disjoint edges (A-B, C-D, ...) from the graph."""
        random.shuffle(candidates)
        edges: List[tuple[str, str]] = []
        used: Set[str] = set()

        for nid in candidates:
            if nid in used:
                continue
            nbrs = [nb for nb in self._adj[nid] if nb not in used]
            if nbrs:
                nb = random.choice(nbrs)
                edges.append((nid, nb))
                used.add(nid)
                used.add(nb)
                if len(edges) == num_edges:
                    break

        # If we couldn't find enough disjoint edges, fall back to overlapping
        if len(edges) < num_edges:
            log.warning(
                "Could not find %d disjoint edges (got %d); using overlapping fallback",
                num_edges,
                len(edges),
            )
            return self._sample_overlapping_edges(candidates, num_edges=num_edges)

        return edges

    def _sample_overlapping_edges(
        self, candidates: List[str], num_edges: int = 2
    ) -> List[tuple[str, str]]:
        """Fallback: pick `num_edges` edges even if they may share a node."""
        edges: List[tuple[str, str]] = []
        seen_nodes: Set[str] = set()

        for nid in candidates:
            nbrs = list(self._adj[nid])
            if nbrs:
                nb = random.choice(nbrs)
                edges.append((nid, nb))
                seen_nodes.add(nid)
                seen_nodes.add(nb)
                if len(edges) == num_edges:
                    break
        return edges