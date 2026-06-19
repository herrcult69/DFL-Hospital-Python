"""
GraphManager — Bootstrap only.
Maintains the Master Adjacency List for the K-Regular graph.
Pure math
"""
import random
import logging

log = logging.getLogger(__name__)


class GraphManager:

    def __init__(self, k: int = 4):
        self.k = k
        self.adj: dict[str, set[str]] = {}

    # ── Seed ────────────────────────────────────────────────────────────────

    def seed(self, node_id: str) -> None:
        """Register the Bootstrap itself as the first node. No edges yet."""
        self.adj[node_id] = set()
        log.info(f"Graph seeded with bootstrap node: {node_id}")

    # ── Register ─────────────────────────────────────────────────────────────

    def register(self, new_node: str) -> dict[str, list[str]]:
        """
        Wire new_node into the graph.

        Small network (N <= K):  connect new_node to every existing node.
        Full network  (N >  K):  edge-breaking — find K/2 disjoint edges,
                                 break them, wire freed endpoints to new_node.

        Returns {affected_node_id: [updated_neighbor_list], ...}
        Bootstrap HTTP-notifies each affected node to update its Neighbor Map.
        """
        self.adj[new_node] = set()
        
        # Build list of existing nodes (excluding the new node)
        existing = []
        for n in self.adj:
            if n != new_node:
                existing.append(n)

        if not existing:
            return {}  # first node, nothing to connect to

        # ── Small network: connect to everyone ───────────────────────────────
        if len(existing) <= self.k:
            for node in existing:
                self.adj[new_node].add(node)
                self.adj[node].add(new_node)
            
            # Build result dictionary with updated neighbor lists
            result = {}
            for node in existing:
                result[node] = list(self.adj[node])
            return result

        # ── Full network: edge-breaking ───────────────────────────────────────
        edges = self._sample_disjoint_edges(self.k // 2)

        if not edges:## This only when k = 2 and N = 3 triangle, no disjoint edge
            raise RuntimeError(
                f"Could not find {self.k // 2} disjoint edges at N={len(self.adj)}. "
                f"This should never happen for K={self.k}."
    )

        affected: dict[str, list[str]] = {}
        for (a, b) in edges:
            # Step 1: Break the edge between a and b
            self.adj[a].discard(b)  # Remove b from a's neighbors
            self.adj[b].discard(a)  # Remove a from b's neighbors
            
            # Step 2: Connect both a and b to the new_node
            self.adj[a].add(new_node)    # a now connected to new_node
            self.adj[b].add(new_node)    # b now connected to new_node
            self.adj[new_node].add(a)    # new_node connected to a
            self.adj[new_node].add(b)    # new_node connected to b
            
            # Step 3: Record the updated neighbor lists
            affected[a] = list(self.adj[a])  # Save a's new neighbors
            affected[b] = list(self.adj[b])  # Save b's new neighbors

        return affected

    # ── Evict ────────────────────────────────────────────────────────────────

    def evict(self, dead_node: str) -> dict[str, list[str]]:
        """
        Remove dead_node from the graph and re-wire its former neighbors
        so the graph stays as close to K-regular as possible.

        Strategy:
          - Remove dead_node and all its edges.
          - Its former neighbors now have degree K-1 (or less).
          - For each under-degree survivor, try to wire it to another
            under-degree survivor it's not already connected to.
          - Returns {affected_node_id: [new_neighbor_list]}.
        """
        if dead_node not in self.adj:
            return {}

        former_neighbors = list(self.adj[dead_node])
        affected: dict[str, list[str]] = {}

        # Step 1: Remove dead_node from all neighbors' adj lists
        for nbr in former_neighbors:
            self.adj[nbr].discard(dead_node)

        # Step 2: Remove dead_node itself
        del self.adj[dead_node]

        # Step 3: Pair up under-degree nodes greedily
        under = [n for n in self.adj if len(self.adj[n]) < self.k]
        random.shuffle(under)

        i = 0
        while i < len(under) - 1:
            a = under[i]
            for j in range(i + 1, len(under)):
                b = under[j]
                if b not in self.adj[a] and a not in self.adj[b]:
                    self.adj[a].add(b)
                    self.adj[b].add(a)
                    affected[a] = list(self.adj[a])
                    affected[b] = list(self.adj[b])
                    under.pop(j)
                    break
            i += 1

        # Mark any nodes that lost edges as affected (even if not re-wired)
        for nbr in former_neighbors:
            if nbr in self.adj and nbr not in affected:
                affected[nbr] = list(self.adj[nbr])

        log.info(f"Evicted {dead_node} | re-wired: {list(affected.keys())}")
        return affected

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return the list of neighbors for a given node. Returns empty list if node doesn't exist."""
        neighbors = self.adj.get(node_id, set())
        return list(neighbors)

    def _sample_disjoint_edges(self, count: int) -> list[tuple[str, str]]:
        """Pick `count` edges that share no vertices."""
        # Build list of all unique edges (avoid duplicates in undirected graph)
        all_edges = []
        for node_a in self.adj:
            for node_b in self.adj[node_a]:
                # Only add edge once: when node_a comes before node_b alphabetically (A, B) = -(B, A)- -> A < B so (A, B) is chosen  
                if node_a < node_b:
                    all_edges.append((node_a, node_b))
        
        # Randomize order for fair sampling
        random.shuffle(all_edges)
        
        # Greedily select edges that don't share vertices
        selected = []
        used_nodes = set()
        
        for node_a, node_b in all_edges:
            if len(selected) >= count:
                break
            # Only pick this edge if neither node is already used
            if node_a not in used_nodes and node_b not in used_nodes:
                selected.append((node_a, node_b))
                used_nodes.add(node_a)
                used_nodes.add(node_b)
        
        return selected
    # ── Debug ─────────────────────────────────────────────────────────────────

    def degrees(self) -> dict[str, int]:
        """Return a dictionary mapping each node to its degree (number of neighbors)."""
        result = {}
        for node, neighbors in self.adj.items():
            result[node] = len(neighbors)
        return result

    def is_k_regular(self) -> bool:
        """Check if the graph is k-regular (all nodes have exactly k neighbors)."""
        # Small networks are considered k-regular by best-effort
        if len(self.adj) <= self.k:
            return True
        
        # Check that every node has exactly k neighbors
        for neighbors in self.adj.values():
            if len(neighbors) != self.k:
                return False
        return True

    def dump(self) -> dict[str, list[str]]:
        """Return a dictionary with sorted neighbor lists for each node."""
        result = {}
        for node, neighbors in self.adj.items():
            result[node] = sorted(neighbors)
        return result