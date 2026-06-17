from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetworkConfig: # Default settings
    k:             int   = 4
    http_timeout:  float = 5.0
    host:          str   = "127.0.0.1"
    port:          int   = 8000
    grpc_port:     int   = 9000
    bootstrap_url: str   = "127.0.0.1:8000"
    gossip_ttl: int = 10  # safe default; tune to ceil(log2(N)) + 2 later
    heartbeat_interval: float = 2.0
    phase1_floor: float = 5.0 # minimal second that the phase one gonna be
    stability_window: float = 10.0 # stability timer -> global table not gonna for current round 
    ready_timeout: float = 10.0  # seconds to wait for all READY signals
    model_dir:     str   = "./models"  # directory for local LoRA adapters
    
    @property
    def node_id(self) -> str:
        return f"{self.host}:{self.port}:{self.grpc_port}"
    
    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)