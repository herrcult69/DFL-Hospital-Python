"""
Phase 2 — gRPC ring parameter transfer.

Each node:
  1. Starts a gRPC server to receive LoRA models from ring_left.
  2. Sends its own LoRA file to ring_right in one unary call.
  3. On receive: writes file to disk, forwards to ring_right if TTL > 0.
  4. Dead node detection: failed call → add to dead_this_round → skip to next alive node.
  5. After all N-1 files received → READY barrier → PHASE_3.

Entry point: await RingPhase(state, config, gossip).run()
"""

import asyncio
import logging
import time
import os
from pathlib import Path

import grpc
from grpc import aio as grpc_aio

from .config import NetworkConfig
from .state  import NodeState, Phase
from .gossip import GossipEngine
from .models import Rumor, RumorType
from . import ring_pb2
from . import ring_pb2_grpc

log       = logging.getLogger(__name__)



# gRPC message size limit — 64 MB, well above worst-case 30 MB LoRA file
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length",    64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _grpc_addr(node_id: str) -> str:
    """'host:gossip_port:grpc_port' → 'host:grpc_port'"""
    host, _, grpc_port = node_id.split(":")
    return f"{host}:{grpc_port}"


def _model_path(round_: int, node_id: str, chunk_dir: Path) -> Path:
    safe = node_id.replace(":", "_")
    return chunk_dir / f"round{round_}_{safe}.safetensors"


def _local_model_path(round_: int, node_id: str, model_dir: Path) -> Path:
    """Path where this node's own trained LoRA adapter lives."""
    safe = node_id.replace(":", "_")
    return model_dir / f"round{round_}_{safe}.safetensors"


def _find_alive_target(
    start: str,
    dead_this_round: set,
    ring: list[str],
    self_id: str,
) -> str | None:
    """
    Walk ring rightward from `start` until we find a node that is
    not dead and not ourselves. Returns None if the whole ring is dead.
    """
    visited = set()
    target  = start
    while target in dead_this_round or target == self_id:
        if target in visited:
            return None   # full loop — everyone is dead
        visited.add(target)
        idx    = ring.index(target)
        target = ring[(idx + 1) % len(ring)]
    return target


# ── gRPC servicer ─────────────────────────────────────────────────────────────

class RingTransferServicer(ring_pb2_grpc.RingTransferServicer):

    def __init__(self, state: NodeState, config: NetworkConfig, dead_this_round: set, chunk_dir: Path):
        self.state           = state
        self.config          = config
        self.dead_this_round = dead_this_round
        self.chunk_dir       = chunk_dir   # shared reference with RingPhase

    async def SendModel(self, request, context):
        payload = request

        # ── 1. write to disk immediately ─────────────────────────────────────
        dest = _model_path(payload.round, payload.originator_id, self.chunk_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload.data)
        log.info(
            f"Received model from {payload.originator_id} "
            f"({len(payload.data) / 1024:.1f} KB) hop={payload.hop} ttl={payload.ttl}"
        )

        # ── 2. merge piggybacked dead list ────────────────────────────────────
        for dead in payload.dead_nodes:
            if dead not in self.dead_this_round:
                self.dead_this_round.add(dead)
                log.warning(f"Piggybacked dead node learned: {dead}")

        # ── 3. forward if TTL allows ──────────────────────────────────────────
        ttl = payload.ttl - 1

        if ttl > 0:
            asyncio.create_task(
                _forward_model(
                    payload         = ring_pb2.ModelPayload(
                        originator_id = payload.originator_id,
                        round         = payload.round,
                        hop           = payload.hop + 1,
                        ttl           = ttl,
                        data          = payload.data,
                        dead_nodes    = list(self.dead_this_round),
                    ),
                    start_target    = self.state.ring_right,
                    dead_this_round = self.dead_this_round,
                    state           = self.state,
                    config          = self.config,
                )
            )

        return ring_pb2.ModelAck(
            receiver_id = self.state.node_id,
            status      = "ok",
        )


# ── forward helper ────────────────────────────────────────────────────────────

async def _forward_model(
    payload:         ring_pb2.ModelPayload,
    start_target:    str | None,
    dead_this_round: set,
    state:           NodeState,
    config:          NetworkConfig,
) -> None:
    """Send an already-built ModelPayload to the next alive node in the ring."""
    if start_target is None:
        log.warning("ring_right is None — nowhere to forward")
        return

    target = _find_alive_target(
        start           = start_target,
        dead_this_round = dead_this_round,
        ring            = state.global_table,
        self_id         = state.node_id,
    )
    if target is None:
        log.warning("All downstream nodes dead — forward aborted")
        return

    try:
        async with grpc_aio.insecure_channel(_grpc_addr(target), options=_GRPC_OPTIONS) as ch:
            stub = ring_pb2_grpc.RingTransferStub(ch)
            await stub.SendModel(payload)
            log.info(f"Forwarded {payload.originator_id}'s model to {target}")

    except grpc.RpcError as e:
        log.warning(f"{target} unreachable ({e.code()}) — marking dead, retrying forward")
        dead_this_round.add(target)
        # retry once with the next alive node
        await _forward_model(payload, target, dead_this_round, state, config)


# ── local model sender ────────────────────────────────────────────────────────

async def _send_local_model(
    state:           NodeState,
    config:          NetworkConfig,
    dead_this_round: set,
) -> None:
    """Read local LoRA adapter and send it to ring_right in one call."""
    model_path = _local_model_path(state.round, state.node_id, config.model_path)

    if not model_path.exists():
        log.warning(f"Local model not found at {model_path} — nothing to send")
        return

    data = model_path.read_bytes()
    n    = len(state.global_table)
    ttl  = (n - 1) - len(dead_this_round)

    if ttl <= 0:
        log.info("TTL=0 — sole survivor, no one to send to")
        return

    target = _find_alive_target(
        start           = state.ring_right,
        dead_this_round = dead_this_round,
        ring            = state.global_table,
        self_id         = state.node_id,
    )
    if target is None:
        log.warning("No alive ring_right found — skipping local model send")
        return

    payload = ring_pb2.ModelPayload(
        originator_id = state.node_id,
        round         = state.round,
        hop           = 0,
        ttl           = ttl,
        data          = data,
        dead_nodes    = list(dead_this_round),
    )

    try:
        async with grpc_aio.insecure_channel(_grpc_addr(target), options=_GRPC_OPTIONS) as ch:
            stub = ring_pb2_grpc.RingTransferStub(ch)
            await stub.SendModel(payload)
            log.info(
                f"Local model sent to {target} "
                f"({len(data) / 1024:.1f} KB) ttl={ttl}"
            )
    except grpc.RpcError as e:
        log.warning(f"ring_right {target} unreachable ({e.code()}) — marking dead, retrying")
        dead_this_round.add(target)
        await _send_local_model(state, config, dead_this_round)


# ── Phase 2 orchestrator ──────────────────────────────────────────────────────

class RingPhase:

    def __init__(self, state: NodeState, config: NetworkConfig, gossip: GossipEngine):
        self.state           = state
        self.config          = config
        self.gossip          = gossip
        self.dead_this_round: set[str] = set()
        # stable unique dir per node — safe across restarts
        safe = state.node_id.replace(":", "_")
        self.chunk_dir = Path(f"./chunks_{safe}")
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        log.info("=== Phase 2 started — LoRA ring transfer ===")

        # ── 1. start gRPC server ──────────────────────────────────────────────
        server   = grpc_aio.server(options=_GRPC_OPTIONS)
        servicer = RingTransferServicer(self.state, self.config, self.dead_this_round, self.chunk_dir)
        ring_pb2_grpc.add_RingTransferServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{self.config.grpc_port}")
        await server.start()
        await asyncio.sleep(1.0)   # wait for all nodes' gRPC servers to be ready
        log.info(f"gRPC server up on port {self.config.grpc_port}")

        # ── 2. send our own LoRA adapter to ring_right ────────────────────────
        await _send_local_model(
            state           = self.state,
            config          = self.config,
            dead_this_round = self.dead_this_round,
        )

        # ── 3. wait until N-1 foreign files are on disk ───────────────────────
        await self._wait_all_models()

        # ── 4. stop gRPC server ───────────────────────────────────────────────
        await server.stop(grace=5)
        log.info("gRPC server stopped")

        # ── 5. READY barrier for Phase 2→3 ───────────────────────────────────
        advanced = await self._ready_barrier_phase3()
        if advanced:
            self.state.phase = Phase.PHASE_3
            log.info(f"=== Phase flipped to PHASE_3 | participants: {self.state.global_table} ===")
        else:
            log.warning("Phase 2→3 barrier failed — holding")

    # ── internal helpers ──────────────────────────────────────────────────────

    async def _wait_all_models(self) -> None:
        """
        Poll disk until a .safetensors file exists for every expected node.
        Expected = global_table minus self minus dead_this_round.
        Timeout → add missing to dead_this_round (clean drop, Phase 3 won't count them).
        """
        deadline = time.time() + self.config.ready_timeout * 6

        while True:
            expected = (
                set(self.state.global_table)
                - {self.state.node_id}
                - self.dead_this_round
            )

            received = {
                p.name
                 .removeprefix(f"round{self.state.round}_")
                 .removesuffix(".safetensors")
                 .replace("_", ":", 2)
                for p in self.chunk_dir.glob(f"round{self.state.round}_*.safetensors")
            }

            if expected <= received:
                log.info(f"All {len(expected)} foreign LoRA files confirmed on disk")
                return

            if time.time() > deadline:
                missing = expected - received
                log.warning(f"Wait timeout — treating as dead: {missing}")
                self.dead_this_round.update(missing)
                return

            await asyncio.sleep(0.5)

    async def _ready_barrier_phase3(self) -> bool:
        """READY barrier targeting PHASE_3 — uses dedicated ready_set_p3."""
        # Do NOT reset ready_set — that belongs to Phase 1→2 barrier.
        # ready_set_p3 accumulates from the moment gossip _process() starts
        # seeing PHASE_3 signals — even before this barrier is entered.
        self.state.ready_set_p3.add(self.state.node_id)
        self.state.ready_timeout = time.time()

        ts = time.time()
        rumor = Rumor(
            type          = RumorType.READY,
            originator_id = self.state.node_id,
            round         = self.state.round,
            rumor_id      = f"READY:{self.state.node_id}:{self.state.round}:{ts}",
            ttl           = self.config.gossip_ttl,
            payload       = {"target_phase": "PHASE_3"},
        )
        self.state.mark_seen(rumor.rumor_id)
        await self.gossip.spread(rumor)

        while True:
            await asyncio.sleep(0.5)

            if self.state.ready_set_p3 >= set(self.state.global_table):
                log.info("Phase 2→3 READY barrier cleared")
                return True

            if time.time() - self.state.ready_timeout >= self.config.ready_timeout:
                missing = set(self.state.global_table) - self.state.ready_set_p3
                log.warning(f"Phase 2→3 timeout — dropping: {missing}")
                for node in missing:
                    self.state.global_table.remove(node)
                if len(self.state.global_table) < 2:
                    self.state.table_locked           = False
                    self.state.ready_set_p3           = set()
                    self.state.last_table_change_time = time.time()
                    return False
                return True