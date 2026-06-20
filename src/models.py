from pydantic import BaseModel
from enum import Enum
import time

class JoinRequest(BaseModel):
    node_id: str  # "host:gossip_port:grpc_port" + later public key and certs 


class JoinResponse(BaseModel):
    neighbors:    list[str]
    global_table: list[str]
    round:        int
    message: str


class StatusResponse(BaseModel):
    node_id:          str
    global_table:     list[str]
    neighbor_map:     list[str]
    ring_left:        str | None
    ring_right:       str | None
    table_locked:     bool
    ready_set:        list[str]
    ready_set_p3:     list[str]
    ready_set_p4:     list[str]
    no_model_set:     list[str]
    dead_this_round:  list[str]
    is_bootstrap:     bool
    current_round:    int
    phase:            str
    seen_rumors:      list[str]
    rumor_log:        list[dict]
    heartbeat_seen:   list[str]

class RewireRequest(BaseModel):
    new_neighbors: list[str]
    
class RewireResponse(BaseModel):
    status: str
    
class RumorType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    JOIN      = "JOIN"
    READY     = "READY" # Ready for phase 2
    NO_MODEL  = "NO_MODEL"
    DONE      = "DONE" # Phases 2 3 4 done?


class Rumor(BaseModel):
    type:          RumorType
    originator_id: str
    round:         int
    rumor_id:      str        # "{type}:{originator_id}:{round}"
    ttl:           int
    payload:       dict = {}  # JOIN carries {"node_id": "..."}, others empty

    @staticmethod
    def build(type: RumorType, originator_id: str, round: int, ttl: int, payload: dict = {}) -> "Rumor":
        ts = time.time()
        if type == RumorType.JOIN:
            joining_node = payload.get("node_id", originator_id)
            rumor_id = f"{type}:{joining_node}:{round}:{ts}"
        elif type == RumorType.READY:
            target_phase = payload.get("target_phase", "")
            rumor_id = f"{type}:{originator_id}:{round}:{target_phase}:{ts}"
        else:
            rumor_id = f"{type}:{originator_id}:{round}:{ts}"
        return Rumor(
            type=type,
            originator_id=originator_id,
            round=round,
            rumor_id=rumor_id,
            ttl=ttl,
            payload=payload,
        )
        
class EvictRequest(BaseModel):
    dead_nodes: list[str]