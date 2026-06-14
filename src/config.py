"""
Node configuration — parsed from CLI args or environment.

⚠️  **LEGACY CODE** — This file is part of the legacy DFL-Hospital-Python implementation.
    A complete rewrite is in progress. Do not modify this file.
    See LEGACY_REFERENCE.md for the authoritative design documentation.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class NodeConfig:
    # Identity
    node_id: str = ""
    host: str = "0.0.0.0"
    gossip_port: int = 0  # FastAPI (Phase 1)
    grpc_port: int = 0  # gRPC (Phase 2)
    data_dir: str = "data"

    # Bootstrap info
    bootstrap_url: str = ""  # e.g. "http://192.168.1.1:5400"

    # Gossip
    k_neighbors: int = 2
    quiescence_seconds: float = 3.0
    gossip_timeout_seconds: float = 2.0

    # Ring
    chunk_size_bytes: int = 4 * 1024 * 1024  # 4 MB default
    grpc_timeout_seconds: float = 10.0

    # ML
    model_name: str = "openai-community/gpt2"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    total_rounds: int = 3

    # Paths (set during init)
    received_models_dir: str = ""
    output_adapter_dir: str = ""

    def __post_init__(self) -> None:
        if not self.received_models_dir:
            self.received_models_dir = os.path.join(self.data_dir, "received_models")
        if not self.output_adapter_dir:
            self.output_adapter_dir = os.path.join(self.data_dir, "adapters")


def parse_cli_args() -> NodeConfig:
    parser = argparse.ArgumentParser(
        description="DFL Node — Decentralized Federated Learning P2P Node"
    )
    parser.add_argument("--node-id", required=True, help="Unique node ID (e.g. '192.168.1.10:5201:5301')")
    parser.add_argument("--gossip-port", type=int, required=True, help="Port for FastAPI gossip server")
    parser.add_argument("--grpc-port", type=int, required=True, help="Port for gRPC ring server")
    parser.add_argument("--bootstrap", default="", help="Bootstrap server URL (empty = this node IS the bootstrap)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quiescence", type=float, default=3.0)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)

    args = parser.parse_args()

    # Build node_id as "host:gossip_port:grpc_port" so the gRPC address is
    # embedded directly — no separate address map needed.
    node_id = f"{args.host}:{args.gossip_port}:{args.grpc_port}"
    if args.host == "0.0.0.0":
        node_id = f"127.0.0.1:{args.gossip_port}:{args.grpc_port}"

    return NodeConfig(
        node_id=node_id,
        host=args.host,
        gossip_port=args.gossip_port,
        grpc_port=args.grpc_port,
        bootstrap_url=args.bootstrap,
        data_dir=args.data_dir,
        model_name=args.model,
        total_rounds=args.rounds,
        quiescence_seconds=args.quiescence,
        chunk_size_bytes=args.chunk_size,
        received_models_dir=os.path.join(args.data_dir, "received_models"),
        output_adapter_dir=os.path.join(args.data_dir, "adapters"),
    )