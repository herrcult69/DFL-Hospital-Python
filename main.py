"""
DFL Node entry point.

Bootstrap:
    python main.py --host 127.0.0.1 --port 8000 --grpc-port 9000 --bootstrap

Worker:
    python main.py --host 127.0.0.1 --port 8001 --grpc-port 9001 --bootstrap-url 127.0.0.1:8000
"""
import argparse
import logging
import uvicorn

from src.config import NetworkConfig
from src.state import NodeState
from src.graph import GraphManager
from src.bootstrap import create_bootstrap_app
from src.worker import create_worker_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def parse_args():
    p = argparse.ArgumentParser(description="DFL Node")
    p.add_argument("--host",          default="127.0.0.1")
    p.add_argument("--port",          type=int, default=8000)
    p.add_argument("--grpc-port",     type=int, default=9000)
    p.add_argument("--k",             type=int, default=4)
    p.add_argument("--bootstrap",     action="store_true")
    p.add_argument("--bootstrap-url", default="127.0.0.1:8000")
    return p.parse_args()


def main():
    args = parse_args()

    config = NetworkConfig(
        k=args.k,
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
    )

    if args.bootstrap:
        state = NodeState(node_id=config.node_id, is_bootstrap=True)
        state.add_node(config.node_id)   # Bootstrap registers itself
        graph = GraphManager(k=config.k)
        graph.seed(config.node_id)
        app = create_bootstrap_app(state=state, graph=graph, config=config)

        print(f"\n[BOOTSTRAP] {config.node_id}")
        print(f"  Status    : http://{args.host}:{args.port}/status")
        print(f"  Table     : http://{args.host}:{args.port}/table")
        print(f"  Graph     : http://{args.host}:{args.port}/graph")
        print(f"  Status UI : http://{args.host}:{args.port}/status-page")
        print(f"  Graph UI  : http://{args.host}:{args.port}/graph-page\n")

    else:
        config.bootstrap_url = args.bootstrap_url
        state = NodeState(node_id=config.node_id, is_bootstrap=False)
        app = create_worker_app(state=state, config=config)

        print(f"\n[NODE] {config.node_id}")
        print(f"  Bootstrap : {args.bootstrap_url}")
        print(f"  Status    : http://{args.host}:{args.port}/status")
        print(f"  Dashboard : http://{args.host}:{args.port}/\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()