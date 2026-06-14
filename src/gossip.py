"""
SIR Rumor-Mongering Gossip Protocol (Phase 1).

Each node runs a FastAPI server with two gossip endpoints:
  - GET /sync:   Return the O(N) global table + phase state (for new nodes).
  - POST /gossip: Receive a rumor. SIR-model cache prevents infinite loops.

Background gossip client:
  - Forwards new rumors to all K=4 neighbors asynchronously.
  - Runs a Quiescence Timer: resets on each new rumor.
  - After 3 seconds of silence: sorts the global table, computes ring
    neighbors, and triggers the Phase 1 → Phase 2 transition.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Callable, List, Optional, Set

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import NodeConfig
from src.state import NodeState, Phase, gossip_addr_from_node_id

log = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────────

class SyncResponse(BaseModel):
    node_id: str
    global_table: list[str]
    phase: str
    round: int
    table_hash: str


class GossipBody(BaseModel):
    rumor_type: str  # "JOIN" | "SUSPICION" | "DEATH" | "OFFICIAL_UPDATE"
    originator: str  # node that created the rumor
    target: Optional[str] = None
    payload: str = ""


class GossipResponse(BaseModel):
    status: str
    known: bool


# ── Quiescence event ──────────────────────────────────────────────────────────

QuiescenceCallback = Callable[[], None]


# ── Gossip Engine ─────────────────────────────────────────────────────────────

class GossipEngine:
    """
    Manages rumor reception, SIR cache, async forwarding, and the
    quiescence timer that triggers Phase 2.
    """

    def __init__(
        self,
        node_state: NodeState,
        config: NodeConfig,
        on_quiescence: Optional[QuiescenceCallback] = None,
    ) -> None:
        self.state = node_state
        self.config = config
        self.on_quiescence = on_quiescence

        # SIR cache: set of "rumor_type:originator:target" strings
        self._known_rumors: Set[str] = set()

        # HTTP client for forwarding
        self._client: Optional[httpx.AsyncClient] = None

        # Quiescence timer task
        self._quiescence_task: Optional[asyncio.Task] = None
        self._last_rumor_time: float = 0.0
        self._paused: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=self.config.gossip_timeout_seconds,
            limits=httpx.Limits(max_connections=50),
        )
        self._reset_quiescence_timer()

    async def shutdown(self) -> None:
        if self._quiescence_task:
            self._quiescence_task.cancel()
        if self._client:
            await self._client.aclose()

    # ── Rumor handling ────────────────────────────────────────────────────

    def _rumor_key(self, body: GossipBody) -> str:
        return f"{body.rumor_type}:{body.originator}:{body.target or ''}"

    def is_known(self, body: GossipBody) -> bool:
        return self._rumor_key(body) in self._known_rumors

    async def receive_rumor(self, body: GossipBody) -> bool:
        """
        Process an incoming rumor.
        Returns True if it was new (and was accepted), False if already known.
        """
        key = self._rumor_key(body)
        if key in self._known_rumors:
            return False  # SIR: already infected, drop silently

        self._known_rumors.add(key)

        log.debug("Accepted rumor: %s", key)
        self._rumor_received()

        # Handle JOIN / DEATH / SUSPICION — merge the target node into table
        if body.rumor_type in ("JOIN", "DEATH", "SUSPICION"):
            self.state.merge_table([body.target or body.originator])

        # Handle READY rumor — record the peer's readiness for the barrier
        if body.rumor_type == "READY":
            parts = body.payload.split(":")
            if len(parts) >= 2:
                target_phase = parts[0]
                self.state.record_ready(body.originator, target_phase)

        # Handle OFFICIAL_UPDATE payload (edge rewire instructions)
        if body.rumor_type == "OFFICIAL_UPDATE" and body.payload:
            await self._apply_official_update(body)

        # Async-forward to all gossip neighbors
        asyncio.create_task(self._forward_rumor(body))

        return True

    async def create_and_spread(
        self, rumor_type: str, target: Optional[str] = None, payload: str = ""
    ) -> None:
        """Create a rumor as this node and spread it to neighbors."""
        body = GossipBody(
            rumor_type=rumor_type,
            originator=self.state.snapshot()["node_id"],
            target=target,
            payload=payload,
        )
        # Add to our own SIR cache
        self._known_rumors.add(self._rumor_key(body))
        # Forward to neighbors
        await self._forward_rumor(body)

    async def _forward_rumor(self, body: GossipBody) -> None:
        """Forward a rumor to all K=4 gossip neighbors."""
        assert self._client is not None
        neighbors = self.state.gossip_neighbors
        if not neighbors:
            return

        tasks = []
        for nbr in neighbors:
            tasks.append(self._send_gossip(nbr, body))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_gossip(self, addr: str, body: GossipBody) -> None:
        """POST /gossip to a single neighbor. `addr` is a node_id in
        'host:gossip_port:grpc_port' format — extract the gossip address."""
        assert self._client is not None
        gossip_addr = gossip_addr_from_node_id(addr)
        try:
            url = f"http://{gossip_addr}/gossip"
            await self._client.post(url, json=body.model_dump(), timeout=2.0)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            log.debug("Gossip forward to %s failed: %s", addr, e)
            # Generate SUSPICION rumor using the full node_id
            my_id = self.state.snapshot()["node_id"]
            suspicion = GossipBody(
                rumor_type="SUSPICION",
                originator=my_id,
                target=addr.split(":")[0] if ":" in addr else addr,
            )
            # Only spread if we haven't already
            suspicion_key = self._rumor_key(suspicion)
            if suspicion_key not in self._known_rumors:
                self._known_rumors.add(suspicion_key)
                asyncio.create_task(self._forward_rumor(suspicion))

    async def _apply_official_update(self, body: GossipBody) -> None:
        """
        Parse OFFICIAL_UPDATE payload and update local state.
        Payload format: "dead_node|neighbor1:new1,new2|neighbor2:new3,new4"
        """
        try:
            parts = body.payload.split("|")
            if len(parts) < 2:
                return
            dead_node = parts[0]
            # Remove dead node from global table
            current_table = self.state.global_table
            if dead_node in current_table:
                current_table.remove(dead_node)
                self.state.global_table = current_table

            # Process rewire instructions (if we are an affected neighbor)
            for entry in parts[1:]:
                if ":" not in entry:
                    continue
                nid, new_nbrs_str = entry.split(":", 1)
                new_nbrs = new_nbrs_str.split(",") if new_nbrs_str else []
                if nid == self.state.snapshot()["node_id"]:
                    current_gossip = self.state.gossip_neighbors
                    # Remove the dead node
                    if dead_node in current_gossip:
                        current_gossip.remove(dead_node)
                    # Add new neighbors
                    for nn in new_nbrs:
                        if nn not in current_gossip:
                            current_gossip.append(nn)
                    self.state.gossip_neighbors = current_gossip
        except Exception as e:
            log.warning("Failed to apply OFFICIAL_UPDATE: %s", e)

    # ── Quiescence Timer ──────────────────────────────────────────────────

    def _rumor_received(self) -> None:
        """Called whenever a new rumor is accepted. Resets the timer."""
        self._last_rumor_time = time.monotonic()
        self._reset_quiescence_timer()

    def pause_quiescence(self) -> None:
        """Pause the quiescence loop. No-op if already paused."""
        self._paused = True

    def resume_quiescence(self) -> None:
        """Resume the quiescence loop and reset its timer."""
        self._paused = False
        self._last_rumor_time = time.monotonic()
        self._reset_quiescence_timer()

    def _reset_quiescence_timer(self) -> None:
        if self._quiescence_task and not self._quiescence_task.done():
            self._quiescence_task.cancel()
        self._quiescence_task = asyncio.create_task(self._quiescence_loop())

    async def _quiescence_loop(self) -> None:
        """Continuously check for quiescence every `quiescence_seconds`.
        Skips cycles while paused (e.g. during a distributed barrier).
        """
        try:
            while True:
                await asyncio.sleep(self.config.quiescence_seconds)
                if self._paused:
                    continue
                elapsed = time.monotonic() - self._last_rumor_time
                if elapsed < self.config.quiescence_seconds:
                    # A rumor arrived during sleep — skip this cycle
                    continue

                log.info(
                    "Quiescence reached (%.1f s)",
                    elapsed,
                )
                # Sort global table
                sorted_table = sorted(self.state.global_table)
                self.state.global_table = sorted_table

                # Compute hash
                raw = ",".join(sorted_table).encode("utf-8")
                self.state.table_hash = hashlib.sha256(raw).hexdigest()[:16]

                if self.on_quiescence:
                    self.on_quiescence()
        except asyncio.CancelledError:
            pass

    # ── SIR cache access for dashboard ───────────────────────────────────

    @property
    def known_rumors(self) -> List[str]:
        """Return the list of known rumor keys (SIR cache)."""
        return sorted(self._known_rumors)

    # ── Sync handler ──────────────────────────────────────────────────────

    async def handle_sync(self) -> SyncResponse:
        """GET /sync — return current global table for new-node bootstrapping."""
        snap = self.state.snapshot()
        return SyncResponse(
            node_id=snap["node_id"],
            global_table=self.state.global_table,
            phase=snap["phase"],
            round=snap["round"],
            table_hash=self.state.table_hash,
        )


# ── FastAPI router factory ─────────────────────────────────────────────────────

def create_gossip_router(engine: GossipEngine) -> FastAPI:
    """Create a FastAPI app with gossip endpoints. Mount this as a sub-app
    or use the routes directly."""

    app = FastAPI(title="DFL Gossip Node")

    @app.get("/sync")
    async def sync():
        return await engine.handle_sync()

    @app.post("/gossip")
    async def gossip(body: GossipBody):
        new_rumor = await engine.receive_rumor(body)
        return GossipResponse(status="ok", known=not new_rumor)

    @app.get("/status")
    async def status():
        snap = engine.state.snapshot()
        snap["known_rumors"] = engine.known_rumors
        snap["table_hash"] = engine.state.table_hash
        return snap

    @app.get("/global_table")
    async def global_table():
        return {
            "node_id": engine.state.snapshot()["node_id"],
            "global_table": engine.state.global_table,
            "n_nodes": len(engine.state.global_table),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        html = _dashboard_html()
        return html

    return app


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")

def _dashboard_html() -> str:
    """Read the dashboard HTML from templates/index.html."""
    path = os.path.join(_TEMPLATE_DIR, "index.html")
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return "<html><body><h1>Dashboard template not found</h1></body></html>"