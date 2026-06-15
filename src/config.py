from dataclasses import dataclass


@dataclass
class NetworkConfig: # Default settings
    k:             int   = 4
    http_timeout:  float = 5.0
    host:          str   = "127.0.0.1"
    port:          int   = 8000
    grpc_port:     int   = 9000
    bootstrap_url: str   = "127.0.0.1:8000"

    @property
    def node_id(self) -> str:
        return f"{self.host}:{self.port}:{self.grpc_port}"