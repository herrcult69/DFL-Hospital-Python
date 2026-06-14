"""
Shared state machine for a DFL node.

Phases:
  PHASE_1_GOSSIP  — Gossip / consensus (Control Plane)
  PHASE_2_RING    — Ring all-gather (Data Plane)
  PHASE_3_AGG     — FedAvg aggregation (disk-based mmap)
  PHASE_4_TRAIN   — Local LoRA fine-tuning

Transitions:
  PHASE_1_GOSSIP → (quiescence timer + barrier) → PHASE_2_RING
  PHASE_2_RING   → (barrier) → PHASE_3_AGG
  PHASE_3_AGG    → (barrier) → PHASE_4_TRAIN
  PHASE_4_TRAIN  → (barrier) → PHASE_1_GOSSIP
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class Phase(str, Enum):
    PHASE_1_GOSSIP = "PHASE_1_GOSSIP"
    PHASE_2_RING = "PHASE_2_RING"
    PHASE_3_AGG = "PHASE_3_AGG"
    PHASE_4_TRAIN = "PHASE_4_TRAIN"
    PHASE_IDLE = "PHASE_IDLE"


def split_node_id(node_id: str) -> tuple[str, int, int]:
    """Parse a node ID of the form 'host:gossip_port:grpc_port'.

    Returns (host, gossip_port, grpc_port).
    For backward compat with 'host:port' (no gRPC port), grpc_port defaults
    to gossip_port + 100.
    """
    parts = node_id.split(":")
    host = parts[0]
    gossip_port = int(parts[1])
    grpc_port = int(parts[2]) if len(parts) >= 3 else gossip_port + 100
    return host, gossip_port, grpc_port


def grpc_addr_from_node_id(node_id: str) -> str:
    """Extract the 'host:grpc_port' address from a node ID."""
    host, _, grpc_port = split_node_id(node_id)
    return f"{host}:{grpc_port}"


def gossip_addr_from_node_id(node_id: str) -> str:
    """Extract the 'host:gossip_port' address from a node ID."""
    host, gossip_port, _ = split_node_id(node_id)
    return f"{host}:{gossip_port}"


class NodeState:
    """Not thread-safe — runs inside a single asyncio event loop."""

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._phase: Phase = Phase.PHASE_1_GOSSIP
        self._round: int = 0
        # Global active-node table: sorted list of "host:gossip:grpc" strings
        self._global_table: List[str] = []
        # Consensus hash used to verify table sync across the ring
        self._table_hash: str = ""
        # Left / right ring neighbors (computed from sorted global table)
        self._left_neighbor: str = ""
        self._right_neighbor: str = ""
        # K gossip neighbors (assigned by Bootstrap)
        self._gossip_neighbors: List[str] = []
        # Distributed phase barrier: node_id -> target_phase
        self._ready_states: Dict[str, str] = {}
        # The phase this node is waiting to enter ("" if not waiting)
        self._my_target_phase: str = ""

    # ── Phase / Round ─────────────────────────────────────────────────────

    @property
    def phase(self) -> Phase:
        return self._phase

    @phase.setter
    def phase(self, value: Phase) -> None:
        self._phase = value

    @property
    def round(self) -> int:
        return self._round

    @round.setter
    def round(self, value: int) -> None:
        self._round = value

    # ── Global table ──────────────────────────────────────────────────────

    @property
    def global_table(self) -> List[str]:
        return list(self._global_table)

    @global_table.setter
    def global_table(self, value: List[str]) -> None:
        self._global_table = sorted(value)

    def merge_table(self, incoming: List[str]) -> bool:
        """Merge an incoming table into ours. Returns True if changed."""
        before = len(self._global_table)
        merged = set(self._global_table) | set(incoming)
        self._global_table = sorted(merged)
        return len(self._global_table) != before

    @property
    def table_hash(self) -> str:
        return self._table_hash

    @table_hash.setter
    def table_hash(self, value: str) -> None:
        self._table_hash = value

    # ── Ring neighbors (computed from sorted global table) ────────────────

    @property
    def left_neighbor(self) -> str:
        return self._left_neighbor

    @property
    def right_neighbor(self) -> str:
        return self._right_neighbor

    def compute_ring_neighbors(self) -> tuple[str, str]:
        """
        Sort the global table alphanumerically and determine left and right
        ring neighbors. Updates internal state and returns (left, right).
        """
        if not self._global_table:
            self._left_neighbor = ""
            self._right_neighbor = ""
            return ("", "")

        table = sorted(set(self._global_table))
        n = len(table)

        try:
            idx = table.index(self._node_id)
        except ValueError:
            table.append(self._node_id)
            table.sort()
            idx = table.index(self._node_id)
            self._global_table = table

        self._left_neighbor = table[(idx - 1) % n]
        self._right_neighbor = table[(idx + 1) % n]
        return (self._left_neighbor, self._right_neighbor)

    # ── Gossip neighbors (K graph) ───────────────────────────────────────

    @property
    def gossip_neighbors(self) -> List[str]:
        return list(self._gossip_neighbors)

    @gossip_neighbors.setter
    def gossip_neighbors(self, value: List[str]) -> None:
        self._gossip_neighbors = list(value)

    # ── Distributed phase barrier ─────────────────────────────────────────

    @property
    def my_target_phase(self) -> str:
        return self._my_target_phase

    @my_target_phase.setter
    def my_target_phase(self, value: str) -> None:
        self._my_target_phase = value

    @property
    def ready_states(self) -> Dict[str, str]:
        return dict(self._ready_states)

    def record_ready(self, node_id: str, target_phase: str) -> None:
        """Record that `node_id` is ready for `target_phase`."""
        self._ready_states[node_id] = target_phase

    def clear_ready(self) -> None:
        """Wipe the barrier after a transition completes."""
        self._ready_states.clear()
        self._my_target_phase = ""

    def barrier_ready(self, target_phase: str) -> bool:
        """Check whether ALL known nodes are ready for `target_phase`."""
        if not self._ready_states or self._my_target_phase != target_phase:
            return False
        if any(v != target_phase for v in self._ready_states.values()):
            return False
        known = set(self._ready_states.keys()) | {self._node_id}
        expected = set(self._global_table)
        return known >= expected

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "node_id": self._node_id,
            "phase": self._phase.value,
            "round": self._round,
            "global_table": list(self._global_table),
            "n_nodes": len(self._global_table),
            "left_neighbor": self._left_neighbor,
            "right_neighbor": self._right_neighbor,
            "gossip_neighbors": list(self._gossip_neighbors),
        }