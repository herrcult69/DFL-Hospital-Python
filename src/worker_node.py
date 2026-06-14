"""
DFL Worker Node — Main Entry Point.

Ties together:
  - Phase 1: Gossip (FastAPI) with SIR cache and quiescence timer.
  - Phase 2: Ring All-Gather (gRPC) with zero-copy disk writes.
  - Bootstrap registration and state sync.

Usage:
  # Bootstrap/seed node:
  python src/worker_node.py --node-id "192.168.1.1:5400" --gossip-port 5400 --grpc-port 5500

  # Regular node:
  python src/worker_node.py --node-id "192.168.1.2:5401" --gossip-port 5401 --grpc-port 5501 \
    --bootstrap "http://192.168.1.1:5400"
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import grpc
import httpx
import uvicorn
from fastapi import FastAPI

from src.bootstrap_server import (
    BootstrapServer,
    GossipBody,
    RegisterRequest,
    create_app as create_bootstrap_app,
)
from src.config import NodeConfig, parse_cli_args
from src.gossip import GossipEngine, create_gossip_router
from src.ring_transfer import RingTransferClient, RingTransferServicer
from src.state import NodeState, Phase, gossip_addr_from_node_id

import protos.ring_transfer_pb2_grpc as pb2_grpc

log = logging.getLogger(__name__)


class WorkerNode:
    """
    Complete DFL worker node.

    Manages:
      - FastAPI server for gossip (Phase 1).
      - gRPC server for ring transfer (Phase 2).
      - State machine transitions (Phase 1 → 2 → 3 → 4 → 1).
      - Bootstrap registration and periodic sync.
    """

    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self.state = NodeState(config.node_id)
        self.state.gossip_neighbors = []

        # Bootstrap reference (only used if this node IS the bootstrap)
        self.bootstrap: Optional[BootstrapServer] = None

        # Gossip engine (Phase 1)
        self.gossip_engine = GossipEngine(
            node_state=self.state,
            config=config,
            on_quiescence=self._on_quiescence,
        )

        # Ring transfer (Phase 2)
        self.ring_servicer = RingTransferServicer(self.state, config)
        self.ring_client = RingTransferClient(self.state, config, self.ring_servicer)

        # gRPC server
        self._grpc_server: Optional[grpc.aio.Server] = None

        # HTTP client for bootstrap / outbound gossip
        self._http: Optional[httpx.AsyncClient] = None

        # Background task handles
        self._background_tasks: list[asyncio.Task] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all node services."""
        log.info("Starting DFL node: %s", self.config.node_id)

        self._http = httpx.AsyncClient(timeout=5.0)

        # Ensure data directories
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.received_models_dir, exist_ok=True)
        os.makedirs(self.config.output_adapter_dir, exist_ok=True)

        # Start gossip engine
        await self.gossip_engine.start()

        # Start ring servicer
        await self.ring_servicer.start()

        # Bootstrap registration
        if self.config.bootstrap_url:
            await self._register_with_bootstrap()
        else:
            # This node IS the bootstrap
            await self._initialize_as_bootstrap()

        # Start gRPC server (ring transfer)
        await self._start_grpc_server()

        # Start FastAPI server (gossip)
        # (uvicorn runs in a background task)
        gossip_app = self._build_gossip_app()
        self._background_tasks.append(
            asyncio.create_task(self._run_fastapi(gossip_app))
        )

        # Start state machine loop
        self._background_tasks.append(
            asyncio.create_task(self._state_machine_loop())
        )

        log.info("Node %s started — gossip on %d, gRPC on %d",
                 self.config.node_id, self.config.gossip_port, self.config.grpc_port)

    async def shutdown(self) -> None:
        """Graceful shutdown of all services."""
        log.info("Shutting down node %s", self.config.node_id)

        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        if self._grpc_server:
            await self._grpc_server.stop(grace=5.0)

        if self._http:
            await self._http.aclose()
        await self.gossip_engine.shutdown()
        await self.ring_servicer.stop()

        if self.bootstrap:
            await self.bootstrap.shutdown()

    # ── Bootstrap registration / init ─────────────────────────────────────

    async def _register_with_bootstrap(self) -> None:
        """Register this node with the bootstrap server."""
        assert self._http is not None
        try:
            url = f"{self.config.bootstrap_url}/register"
            resp = await self._http.post(
                url,
                json={"node_id": self.config.node_id},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()

            # Set gossip neighbors from bootstrap response
            self.state.gossip_neighbors = data.get("neighbors", [])

            # Merge global table
            incoming_table = data.get("global_table", [])
            self.state.merge_table(incoming_table)

            # Ensure self is in the table
            if self.config.node_id not in self.state.global_table:
                self.state.merge_table([self.config.node_id])

            log.info(
                "Registered with bootstrap. Neighbors: %s, N=%d",
                self.state.gossip_neighbors,
                len(self.state.global_table),
            )

            # Sync with neighbors (GET /sync)
            await self._sync_with_neighbors()

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            log.error("Failed to register with bootstrap: %s", e)
            raise

    async def _sync_with_neighbors(self) -> None:
        """After registration, sync global table from all neighbors."""
        assert self._http is not None
        tasks = []
        for nbr in self.state.gossip_neighbors:
            tasks.append(self._sync_from_neighbor(nbr))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict) and "global_table" in result:
                self.state.merge_table(result["global_table"])

        log.info("Sync complete — global table has %d nodes", len(self.state.global_table))

    async def _sync_from_neighbor(self, addr: str) -> Optional[dict]:
        """GET /sync from a single neighbor.
        `addr` is a node_id in 'host:gossip:grpc' format — extract gossip addr."""
        assert self._http is not None
        gossip_addr = gossip_addr_from_node_id(addr)
        try:
            url = f"http://{gossip_addr}/sync"
            resp = await self._http.get(url, timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.debug("Sync from %s failed: %s", addr, e)
            return None

    async def _initialize_as_bootstrap(self) -> None:
        """Initialize this node as the bootstrap server."""
        # Bootstrap itself into the graph
        self.state.global_table = [self.config.node_id]
        self.state.gossip_neighbors = []

        # Start the bootstrap manager (will run the FastAPI with register routes)
        self.bootstrap = BootstrapServer(config=self.config)
        await self.bootstrap.start()

        # Periodically sync global table from the BootstrapServer's state
        # (the bootstrap never receives its own JOIN rumors)
        self._background_tasks.append(
            asyncio.create_task(self._sync_bootstrap_state_loop())
        )

        log.info("Initialized as bootstrap node — waiting for peers")

    async def _sync_bootstrap_state_loop(self) -> None:
        """Periodically sync global table from the BootstrapServer's state."""
        try:
            while True:
                await asyncio.sleep(2.0)
                if self.bootstrap:
                    self.state.merge_table(self.bootstrap.state.global_table)
        except asyncio.CancelledError:
            pass

    # ── gRPC server ───────────────────────────────────────────────────────

    async def _start_grpc_server(self) -> None:
        """Start the gRPC server for ring transfer."""
        self._grpc_server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100 MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100 MB
            ]
        )
        pb2_grpc.add_RingTransferServicer_to_server(self.ring_servicer, self._grpc_server)

        listen_addr = f"0.0.0.0:{self.config.grpc_port}"
        self._grpc_server.add_insecure_port(listen_addr)
        await self._grpc_server.start()
        log.info("gRPC server listening on %s", listen_addr)

    # ── FastAPI gossip server ─────────────────────────────────────────────

    def _build_gossip_app(self) -> FastAPI:
        """Build the FastAPI app with gossip routes and optionally bootstrap routes."""
        gossip_app = create_gossip_router(self.gossip_engine)

        # If this node is also the bootstrap, mount bootstrap routes
        if self.bootstrap:
            bootstrap_app = create_bootstrap_app(self.bootstrap)
            # Mount bootstrap routes under /bootstrap prefix to avoid route conflicts
            # Actually, let's merge them directly
            @gossip_app.post("/register")
            async def register(body: RegisterRequest):
                return await self.bootstrap.handle_register(body)

            @gossip_app.get("/global_table")
            async def global_table():
                return await self.bootstrap.handle_global_table()

        return gossip_app

    async def _run_fastapi(self, app: FastAPI) -> None:
        """Run uvicorn FastAPI server in background."""
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.gossip_port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()

    # ── State Machine Loop ────────────────────────────────────────────────

    async def _state_machine_loop(self) -> None:
        """
        Main state machine loop.
        Transitions: PHASE_1_GOSSIP → PHASE_2_RING → PHASE_3_AGG → PHASE_4_TRAIN → ...
        """
        while True:
            try:
                current_phase = self.state.phase
                # If we're waiting in a barrier, don't re-enter phase handlers
                if self.state.my_target_phase:
                    await asyncio.sleep(0.5)
                    continue
                if current_phase == Phase.PHASE_IDLE:
                    await asyncio.sleep(5)
                    continue
                if current_phase == Phase.PHASE_1_GOSSIP:
                    await self._phase_1_gossip()
                elif current_phase == Phase.PHASE_2_RING:
                    await self._phase_2_ring()
                elif current_phase == Phase.PHASE_3_AGG:
                    await self._phase_3_aggregation()
                elif current_phase == Phase.PHASE_4_TRAIN:
                    await self._phase_4_train()
                else:
                    await asyncio.sleep(1)

                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("State machine error: %s", e, exc_info=True)
                await asyncio.sleep(1)

    async def _phase_1_gossip(self) -> None:
        """
        Phase 1: Gossip / Consensus.

        This phase runs passively — the gossip engine handles rumors.
        The quiescence timer will call _on_quiescence() which transitions
        to Phase 2.
        """
        # Just wait — the quiescence callback fires the transition
        await asyncio.sleep(1)

    def _on_quiescence(self) -> None:
        """
        Called by the gossip engine when quiescence is reached
        (no new rumors for 3 seconds). Transitions to Phase 2.
        """
        # Only respond to quiescence when actively in Phase 1
        if self.state.phase != Phase.PHASE_1_GOSSIP:
            return

        # Already waiting in a barrier — skip
        if self.state.my_target_phase:
            return

        # Minimum ring threshold: need at least 3 nodes for a ring
        n_nodes = len(self.state.global_table)
        if n_nodes < 3:
            log.info(
                "Not enough nodes to form a ring (N=%d < 3). Waiting for peers...",
                n_nodes,
            )
            return

        # Compute ring neighbors now that we have enough nodes
        left, right = self.state.compute_ring_neighbors()
        log.info("Ring neighbors — left: %s, right: %s", left, right)

        # Check total_rounds cap
        if self.state.round >= self.config.total_rounds:
            log.info("All %d rounds complete — staying idle", self.config.total_rounds)
            return

        # Enter the distributed barrier instead of transitioning directly
        self.state.round += 1
        target_phase = Phase.PHASE_2_RING.value
        log.info(
            "Quiescence reached — entering barrier for Round %d → %s",
            self.state.round,
            target_phase,
        )
        asyncio.create_task(self._enter_barrier(target_phase))

    async def _enter_barrier(self, target_phase: str) -> None:
        """Distributed phase barrier.

        Sets the local target, gossips a READY rumor, then polls until
        ALL nodes in the global table have reported the same target_phase.
        Once the barrier lifts, transitions the state machine.
        Pauses the quiescence timer while waiting.
        """
        self.state.my_target_phase = target_phase
        self.state.record_ready(self.config.node_id, target_phase)
        self.gossip_engine.pause_quiescence()

        # Gossip READY to all peers (fire-and-forget via the engine)
        payload = f"{target_phase}:{self.state.round}"
        asyncio.create_task(
            self.gossip_engine.create_and_spread("READY", target=target_phase, payload=payload)
        )

        log.info(
            "Barrier: waiting for all peers to be READY for %s (round %d)",
            target_phase,
            self.state.round,
        )

        # Poll until barrier is satisfied
        while not self.state.barrier_ready(target_phase):
            await asyncio.sleep(0.3)
            if self.state.phase == Phase.PHASE_IDLE:
                self.gossip_engine.resume_quiescence()
                return

        self.state.clear_ready()
        self.gossip_engine.resume_quiescence()
        log.info("Barrier: all peers READY — transitioning to %s", target_phase)
        self.state.phase = Phase(target_phase)

    async def _phase_2_ring(self) -> None:
        """Phase 2: Execute the all-gather ring transfer."""
        log.info("=== Phase 2: Ring All-Gather (Round %d) ===", self.state.round)
        try:
            await self.ring_client.run_ring_all_gather()
            log.info("Phase 2 complete — entering barrier for Phase 3")
            await self._enter_barrier(Phase.PHASE_3_AGG.value)
        except Exception as e:
            log.error("Phase 2 failed: %s", e, exc_info=True)
            self.state.phase = Phase.PHASE_1_GOSSIP

    async def _phase_3_aggregation(self) -> None:
        """
        Phase 3: Aggregation placeholder.

        In the full system, this reads the gathered chunks via safetensors mmap
        and runs FedAvg. For now, just transition to Phase 4.
        """
        log.info("=== Phase 3: Aggregation (Round %d) ===", self.state.round)
        # TODO: Implement FedAvg with mmap safetensors
        await self._enter_barrier(Phase.PHASE_4_TRAIN.value)

    async def _phase_4_train(self) -> None:
        """
        Phase 4: Local training placeholder.

        In the full system, this trains the aggregated model on local data.
        After training, transition back to Phase 1 for the next round.
        """
        log.info("=== Phase 4: Local Training (Round %d) ===", self.state.round)

        if self.state.round >= self.config.total_rounds:
            log.info("All %d rounds complete — entering idle", self.config.total_rounds)
            self.state.phase = Phase.PHASE_IDLE
            return

        # TODO: Implement local LoRA fine-tuning
        # For now, simulate with a delay
        await asyncio.sleep(2)

        log.info("Training complete — entering barrier for Phase 1")
        await self._enter_barrier(Phase.PHASE_1_GOSSIP.value)


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main() -> None:
    config = parse_cli_args()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{config.node_id}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    node = WorkerNode(config)
    try:
        await node.start()
        # Keep alive
        while True:
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    finally:
        await node.shutdown()


if __name__ == "__main__":
    asyncio.run(main())