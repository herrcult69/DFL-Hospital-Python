from pydantic import BaseModel
from enum import Enum

class JoinRequest(BaseModel):
    node_id: str  # "host:gossip_port:grpc_port" + later public key and certs 


class JoinResponse(BaseModel):
    neighbors:    list[str]
    global_table: list[str]
    message: str


class StatusResponse(BaseModel):
    node_id:      str
    global_table: list[str]
    neighbor_map: list[str]
    is_bootstrap: bool

class RewireRequest(BaseModel):
    new_neighbors: list[str]
    
class RewireResponse(BaseModel):
    status: str
    
class RumorType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    JOIN      = "JOIN"
    READY     = "READY" # Ready for phase 2
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
        rumor_id = f"{type}:{originator_id}:{round}"
        return Rumor(
            type=type,
            originator_id=originator_id,
            round=round,
            rumor_id=rumor_id,
            ttl=ttl,
            payload=payload,
        )