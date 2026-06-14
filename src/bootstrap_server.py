"""
Bootstrap / Seed Node (FastAPI).

The Bootstrap is a standard peer in the ring but additionally manages the
master K-regular graph adjacency list. Responsibilities:

  1. Accept POST /register — assign 4 neighbors via edge-breaking.
  2. Accept POST /gossip — inspect "Suspicion" rumors, ping the suspect,
     rewire the graph if dead, and gossip an "Official Update".
  3. Accept GET /status — health check.
  4. Expose GET /global_table — return the full O(N) active node list.

The Bootstrap runs on its own gossip_port and can be started as:
  python src/bootstrap_server.py --gossip-port 5400
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.gossip import GossipBody
from src.graph import GraphManager
from src.state import NodeState, Phase, split_node_id

log = logging.getLogger(__name__)

# ── Pydantic models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    node_id: str


class RegisterResponse(BaseModel):
    node_id: str
    neighbors: List[str]
    global_table: List[str]
    phase: str
    round: int


class StatusResponse(BaseModel):
    node_id: str
    phase: str
    round: int
    n_nodes: int
    gossip_neighbors: List[str]


# ── Bootstrap Server ───────────────────────────────────────────────────────────

class BootstrapServer:
    """FastAPI application logic for the Bootstrap seed node."""

    def __init__(
        self,
        config: "NodeConfig",  # type: ignore
    ) -> None:
        self.config = config
        self.node_id = config.node_id
        self.gossip_port = config.gossip_port
        self.grpc_port = config.grpc_port
        self.graph = GraphManager(k=config.k_neighbors)
        self.state = NodeState(config.node_id)
        self._http_client: Optional[httpx.AsyncClient] = None

        # Ensure data directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.received_models_dir, exist_ok=True)
        os.makedirs(config.output_adapter_dir, exist_ok=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(timeout=5.0)
        # Bootstrap itself into the graph
        self.graph.bootstrap([self.node_id])
        self.state.gossip_neighbors = []
        self.state.global_table = [self.node_id]

    async def shutdown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()

    def _get_node_addr(self, node_id: str) -> str:
        """The gossip address is just 'host:gossip_port' from the node_id."""
        from src.state import gossip_addr_from_node_id
        return gossip_addr_from_node_id(node_id)

    # ── Routes (called from FastAPI) ──────────────────────────────────────

    async def handle_register(self, req: RegisterRequest) -> RegisterResponse:
        """Register a new node into the K-regular graph."""
        # Reject if the network has already moved past gossip phase
        if self.state.phase != Phase.PHASE_1_GOSSIP:
            raise HTTPException(
                status_code=423,
                detail=f"Network is in {self.state.phase.value}. Retry later.",
            )

        if self.graph.has_node(req.node_id):
            nbrs = list(self.graph.neighbors(req.node_id))
        else:
            nbrs = self.graph.register(req.node_id)
            self.state.merge_table([req.node_id])
            await self._gossip_join_rumor(req.node_id)

        log.info(
            "Registered %s — assigned neighbors: %s (graph has %d nodes)",
            req.node_id,
            sorted(nbrs),
            self.graph.node_count,
        )

        return RegisterResponse(
            node_id=req.node_id,
            neighbors=sorted(nbrs),
            global_table=self.state.global_table,
            phase=self.state.phase.value,
            round=self.state.round,
        )

    async def handle_gossip(self, body: GossipBody) -> dict:
        """
        Incoming gossip rumor. If SUSPICION, ping the suspect.
        If DEATH confirmed, rewire and broadcast OFFICIAL_UPDATE.
        """
        log.debug("Gossip received: %s", body)

        if body.rumor_type == "SUSPICION":
            await self._handle_suspicion(body)
        elif body.rumor_type == "DEATH":
            await self._handle_death(body)
        elif body.rumor_type == "JOIN":
            # Propagate the join to our neighbors (SIR will prevent loops)
            self.state.merge_table([body.target or body.originator])

        return {"status": "ok"}

    async def handle_status(self) -> StatusResponse:
        snap = self.state.snapshot()
        return StatusResponse(
            node_id=snap["node_id"],
            phase=snap["phase"],
            round=snap["round"],
            n_nodes=snap["n_nodes"],
            gossip_neighbors=snap["gossip_neighbors"],
        )

    async def handle_global_table(self) -> dict:
        return {
            "node_id": self.node_id,
            "global_table": self.state.global_table,
            "n_nodes": len(self.state.global_table),
        }

    # ── Suspicion handling ────────────────────────────────────────────────

    async def _handle_suspicion(self, body: GossipBody) -> None:
        """On SUSPICION: directly ping the suspect. If dead, evict + update."""
        suspect = body.target
        if not suspect:
            log.warning("SUSPICION rumor missing target: %s", body)
            return

        suspect_addr = self._get_node_addr(suspect)
        if not suspect_addr:
            log.warning("SUSPICION for unknown node %s", suspect)
            return

        # Directly ping suspect
        alive = await self._ping_node(suspect_addr)

        if not alive:
            log.warning("Confirmed dead: %s — evicting", suspect)
            rewired = self.graph.evict(suspect)
            self.state.merge_table([])  # Re-sync (no new nodes, just triggers internal refresh)

            # Gossip OFFICIAL_UPDATE with rewire instructions
            await self._gossip_official_update(suspect, rewired)

    async def _handle_death(self, body: GossipBody) -> None:
        """Direct DEATH rumor (e.g. from Phase 2 gRPC metadata)."""
        dead = body.target
        if not dead or not self.graph.has_node(dead):
            return
        rewired = self.graph.evict(dead)
        self.state.merge_table([])
        await self._gossip_official_update(dead, rewired)

    async def _ping_node(self, addr: str) -> bool:
        """Ping a node's /status endpoint. Returns True if alive."""
        assert self._http_client is not None
        try:
            url = f"http://{addr}/status"
            resp = await self._http_client.get(url, timeout=3.0)
            return resp.status_code == 200
        except (httpx.TimeoutException, httpx.ConnectError):
            return False

    # ── Gossip propagation ────────────────────────────────────────────────

    async def _gossip_join_rumor(self, new_node: str) -> None:
        """Asynchronously inform all existing nodes about the new node."""
        tasks = []
        for peer_id in self.graph.all_nodes():
            if peer_id == self.node_id or peer_id == new_node:
                continue
            addr = self._get_node_addr(peer_id)
            if addr:
                tasks.append(self._send_gossip(addr, GossipBody(
                    rumor_type="JOIN",
                    originator=self.node_id,
                    target=new_node,
                    payload="",
                )))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _gossip_official_update(
        self, dead_node: str, rewired: Dict[str, List[str]]
    ) -> None:
        """Gossip an OFFICIAL_UPDATE about a dead node and new edges."""
        tasks = []
        items = []
        for k, v in rewired.items():
            items.append(f"{k}:{','.join(v)}")
        payload = f"{dead_node}|{','.join(items)}"
        for peer_id in self.graph.all_nodes():
            if peer_id == self.node_id:
                continue
            addr = self._get_node_addr(peer_id)
            if addr:
                tasks.append(self._send_gossip(addr, GossipBody(
                    rumor_type="OFFICIAL_UPDATE",
                    originator=self.node_id,
                    target=dead_node,
                    payload=payload,
                )))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_gossip(self, addr: str, body: GossipBody) -> None:
        """Send a single gossip POST to a peer."""
        assert self._http_client is not None
        try:
            url = f"http://{addr}/gossip"
            await self._http_client.post(url, json=body.model_dump(), timeout=2.0)
        except Exception as e:
            log.debug("Gossip to %s failed: %s", addr, e)


# ── FastAPI app factory ────────────────────────────────────────────────────────

def create_app(server: BootstrapServer) -> FastAPI:
    app = FastAPI(title="DFL Bootstrap Server")

    @app.post("/register")
    async def register(body: RegisterRequest):
        return await server.handle_register(body)

    @app.post("/gossip")
    async def gossip(body: GossipBody):
        return await server.handle_gossip(body)

    @app.get("/status")
    async def status():
        return await server.handle_status()

    @app.get("/global_table")
    async def global_table():
        return await server.handle_global_table()

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

async def run_bootstrap(server: BootstrapServer) -> None:
    await server.start()
    app = create_app(server)
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=server.gossip_port,
        log_level="info",
    )
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", default="bootstrap:5400")
    parser.add_argument("--gossip-port", type=int, default=5400)
    parser.add_argument("--grpc-port", type=int, default=5500)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.config import NodeConfig
    cfg = NodeConfig(
        node_id=args.node_id,
        gossip_port=args.gossip_port,
        grpc_port=args.grpc_port,
        data_dir=args.data_dir,
    )
    server = BootstrapServer(config=cfg)
    asyncio.run(run_bootstrap(server))