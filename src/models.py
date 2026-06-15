from pydantic import BaseModel


class JoinRequest(BaseModel):
    node_id: str  # "host:gossip_port:grpc_port"


class JoinResponse(BaseModel):
    global_table: list[str]
    message: str


class StatusResponse(BaseModel):
    node_id:      str
    global_table: list[str]
    is_bootstrap: bool