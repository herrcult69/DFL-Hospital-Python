from pydantic import BaseModel


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