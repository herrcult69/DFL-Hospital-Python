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
    heartbeat_interval: float = 1.0
    phase1_floor: float = 7.0 # minimal second that the phase one gonna be
    stability_window: float = 6.0 # stability timer -> global table not gonna for current round 
    ready_timeout: float = 8.0  # seconds to wait for all READY signals
    model_dir:     str   = "./models"  # directory for local LoRA adapters
    dataset_path:  str   = "./data/dataset.jsonl"  # this node's local dataset
    phase3_total_budget: float = 600.0
    phase4_timeout: float = 1800.0
    total_rounds: int = 5  # 0 = infinite rounds
    
    @property
    def node_id(self) -> str:
        return f"{self.host}:{self.port}:{self.grpc_port}"
    
    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)