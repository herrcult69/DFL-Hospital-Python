"""
Worker node FastAPI app.
Registers with Bootstrap on startup, then stays alive as a real server.
"""
import logging, time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, Response

from .config import NetworkConfig
from .models import JoinRequest, RewireRequest, RewireResponse, StatusResponse, PredictRequest, PredictResponse
from .state import NodeState, Phase

from .gossip import GossipEngine
from .models import Rumor, RumorType
from .ring_transfer import RingPhase

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
def _handle_task_exception(loop, context):
    msg = context.get("exception", context["message"])
    log.error(f"Unhandled async task exception: {msg}", exc_info=context.get("exception"))

asyncio.get_event_loop().set_exception_handler(_handle_task_exception)


def create_worker_app(state: NodeState, config: NetworkConfig) -> FastAPI:
    gossip = GossipEngine(state=state, config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=config.http_timeout) as client:
            try:
                r = await client.post(
                    f"http://{config.bootstrap_url}/join",
                    json=JoinRequest(node_id=state.node_id).model_dump(),
                )
                r.raise_for_status()
                data = r.json()
                for n in data["global_table"]:
                    state.add_node(n)
                state.neighbor_map = set(data["neighbors"])
                state.round = data["round"]
                log.info(
                    f"Joined. Neighbors: {state.neighbor_map} | "
                    f"Table size: {len(state.global_table)}"
                )
            except Exception as e:
                log.error(f"Could not reach Bootstrap at {config.bootstrap_url}: {e}")
                
        async def _heartbeat_loop():
            while True:
                if state.phase == Phase.PHASE_1:
                    await gossip.originate_heartbeat()
                await asyncio.sleep(config.heartbeat_interval)
        

        phase1_start = time.time()

        async def _stability_timer():
            while True:
                await asyncio.sleep(1.0)

                # Only active during PHASE_1
                if state.phase != Phase.PHASE_1:
                    await asyncio.sleep(2.0)   # idle poll — wait for reset
                    continue

                now              = time.time()
                floor_met        = (now - state.last_table_change_time) >= config.phase1_floor
                table_stable     = (now - state.last_table_change_time) >= config.stability_window

                if floor_met and table_stable and not state.table_locked:
                    log.info(f"Stability timer fired — round {state.round}, locking table")
                    await _end_phase1()

        async def _end_phase1():
            # Step 1: Lock the table
            state.table_locked = True
            log.info(f"Table locked: {state.global_table}")

            # Step 2: Dead node cleanup
            dead = [n for n in state.global_table if n not in state.heartbeat_seen]
            for node in dead:
                log.warning(f"Dead node detected (no heartbeat): {node}")
                state.global_table.remove(node)
                
            # Minimum viable network check
            if len(state.global_table) < 2:              # ← adjust threshold as needed
                log.warning("Not enough nodes to proceed — holding, unlocking table")
                state.table_locked = False               # unlock so new joins can come in
                state.last_table_change_time = time.time()  # reset stability window
                # heartbeat loop keeps running since phase never flipped
                return                                   # timer task exits, but heartbeat continues

            # Step 3: Form Ring Map — sort locked table, derive left/right neighbors
            ring = state.global_table  # already sorted (add_node keeps it sorted)
            idx  = ring.index(state.node_id)
            n    = len(ring)
            state.ring_left  = ring[(idx - 1) % n]
            state.ring_right = ring[(idx + 1) % n]
            log.info(f"Ring formed — left: {state.ring_left} | right: {state.ring_right}")

            # Step 4: originate READY
            state.ready_set.add(state.node_id)   # count self
            state.ready_timeout = time.time()
            ts = time.time()
            ready_rumor = Rumor(
                type=RumorType.READY,
                originator_id=state.node_id,
                round=state.round,
                rumor_id=f"READY:{state.node_id}:{state.round}:{ts}",
                ttl=config.gossip_ttl,
                payload={"target_phase": "PHASE_2", "phase2_start_ts": ts},
            )
            state.mark_seen(ready_rumor.rumor_id, ready_rumor.model_dump())
            await gossip.spread(ready_rumor)

            # Step 5: wait for barrier
            advanced = await _wait_ready_barrier()
            if not advanced:
                return   # hold, stability timer will retry

            # Step 6: flip phase only if barrier passed
            state.phase = Phase.PHASE_2
            log.info(f"Phase flipped to PHASE_2 | participants: {state.global_table}")

        async def _wait_ready_barrier() -> bool:   # ← return bool
            while True:
                await asyncio.sleep(0.5)

                if state.ready_set >= set(state.global_table):
                    log.info("READY barrier cleared — all nodes ready")
                    return True                    # ← success

                if time.time() - state.ready_timeout >= config.ready_timeout:
                    missing = set(state.global_table) - state.ready_set
                    log.warning(f"READY barrier timeout — dropping: {missing}")
                    for node in missing:
                        state.global_table.remove(node)
                    if len(state.global_table) < 2:
                        log.warning("Not enough nodes after timeout — holding")
                        state.table_locked           = False
                        state.ready_set              = set()
                        state.last_table_change_time = time.time()
                        return False               # ← failed, don't flip
                    return True                    # ← reduced set, still proceed
        async def _phase2_loop():
            while True:
                while state.phase == Phase.PHASE_1:
                    await asyncio.sleep(0.5)

                if state.phase == Phase.PHASE_2:
                    ring_phase = RingPhase(state=state, config=config, gossip=gossip)
                    await ring_phase.run()

                elif state.phase == Phase.PHASE_3:
                    from .phase3 import AggregationPhase
                    safe      = state.node_id.replace(":", "_")
                    chunk_dir = Path(f"./chunks_{safe}")
                    agg_phase = AggregationPhase(state=state, config=config, gossip=gossip)
                    await agg_phase.run(chunk_dir=chunk_dir)

                elif state.phase == Phase.PHASE_4:
                    from .phase4 import RoundCompletionPhase
                    safe      = state.node_id.replace(":", "_")
                    chunk_dir = Path(f"./chunks_{safe}")
                    p4        = RoundCompletionPhase(state=state, config=config, gossip=gossip)
                    await p4.run(chunk_dir=chunk_dir)
                    # reset_phase1() sets phase → PHASE_1, outer while True loops back

                if config.total_rounds and state.round >= config.total_rounds:
                    log.info(f"All {config.total_rounds} rounds complete — node going IDLE")
                    state.phase = Phase.IDLE
                    return

        timer_task    = asyncio.create_task(_stability_timer())
        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        phase2_task    = asyncio.create_task(_phase2_loop())
        yield
        heartbeat_task.cancel()
        timer_task.cancel()
        phase2_task.cancel()

    app = FastAPI(title="DFL Worker Node", lifespan=lifespan)

    @app.post("/gossip")
    async def gossip_receive(rumor: Rumor, request: Request):
        sender_id = request.headers.get("X-Sender-Id", "")
        await gossip.receive(rumor, sender_id=sender_id)
        return {"status": "ok"}
    
    @app.post("/rewire", response_model=RewireResponse)
    async def rewire(req: RewireRequest):
        state.neighbor_map = set(req.new_neighbors)
        log.info(f"Rewired. New neighbors: {req.new_neighbors}")
        for node_id in req.new_neighbors:
            if node_id not in state.global_table:
                state.add_node(node_id)
        return RewireResponse(status="ok")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        if state.phase != Phase.IDLE:
            return PredictResponse(
                response="Node is not idle — training still in progress.",
                node_id=state.node_id,
                round=state.round,
                status="not_idle",
            )

        loop = asyncio.get_event_loop()

        def _infer():
            # ── TESTING MODE (comment out for real) ──────────────────────────
            return f"[STUB] Echo from {state.node_id} round {state.round}: {req.message}"
            # ── REAL IMPLEMENTATION (uncomment for production) ────────────────
            # from .lib.inference import run_inference
            # return run_inference(req.message, str(config.model_path))

        try:
            result = await loop.run_in_executor(None, _infer)
            return PredictResponse(
                response=result,
                node_id=state.node_id,
                round=state.round,
                status="ok",
            )
        except Exception as e:
            log.error(f"Inference failed: {e}")
            return PredictResponse(
                response=f"Inference error: {e}",
                node_id=state.node_id,
                round=state.round,
                status="error",
            )

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return FileResponse(TEMPLATES_DIR / "status.html")

    return app