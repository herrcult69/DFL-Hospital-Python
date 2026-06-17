from dataclasses import dataclass, field
from enum import Enum
import time, logging

log = logging.getLogger(__name__)


class Phase(Enum):
    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"
    PHASE_3 = "PHASE_3"
    PHASE_4 = "PHASE_4"


@dataclass
class NodeState:
    node_id:                str
    is_bootstrap:           bool       = False
    global_table:           list[str]  = field(default_factory=list)
    neighbor_map:           set[str]   = field(default_factory=set)
    round:                  int        = 0
    phase:                  Phase      = Phase.PHASE_1
    ring_left:              str | None = None
    ring_right:             str | None = None
    seen_rumors:            set[str]   = field(default_factory=set)
    heartbeat_seen:         set[str]   = field(default_factory=set)
    table_locked:           bool       = False
    last_table_change_time: float      = field(default_factory=time.time)
    ready_set:              set[str]   = field(default_factory=set)
    ready_timeout:          float      = 0.0
    dead_this_round:        set[str]   = field(default_factory=set) 
    ready_set_p3:            set[str]   = field(default_factory=set) 

    def add_node(self, node_id: str) -> bool:
        """Add a node to the global table. Returns True if it was new."""
        if self.table_locked:   # reject if locked
            log.debug(f"Table locked, ignoring add_node({node_id})")
            return False
        if node_id not in self.global_table:
            self.global_table.append(node_id)
            self.global_table.sort()
            self.last_table_change_time = time.time()
            return True
        return False

    def is_seen(self, rumor_id: str) -> bool:
        return rumor_id in self.seen_rumors

    def mark_seen(self, rumor_id: str) -> None:
        self.seen_rumors.add(rumor_id)

    def reset_phase1(self) -> None:
        """Called at Phase 4→1 transition — clear all per-round state."""
        self.heartbeat_seen         = set()
        self.seen_rumors            = set()
        self.phase                  = Phase.PHASE_1
        self.table_locked           = False
        self.dead_this_round        = set()   
        self.ready_set_p3           = set()
        self.last_table_change_time = time.time()

    def snapshot(self) -> dict:
        def get_timestamp(rumor_id: str) -> float:
            parts = rumor_id.split(":")
            try:
                return float(parts[-1])
            except (ValueError, IndexError):
                return 0.0

        sorted_rumors = sorted(self.seen_rumors, key=get_timestamp)
        return {
            "node_id":        self.node_id,
            "is_bootstrap":   self.is_bootstrap,
            "global_table":   list(self.global_table),
            "neighbor_map":   list(self.neighbor_map),
            "ring_left":      self.ring_left,
            "ring_right":     self.ring_right,
            "table_locked":   self.table_locked,
            "ready_set":      list(self.ready_set),
            "current_round":  self.round,
            "phase":          self.phase.value,
            "seen_rumors":    sorted_rumors[-10:],
            "heartbeat_seen": list(self.heartbeat_seen),
        }