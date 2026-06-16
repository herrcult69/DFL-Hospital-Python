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
from .models import JoinRequest, RewireRequest, RewireResponse, StatusResponse
from .state import NodeState, Phase

from .gossip import GossipEngine
from .models import Rumor, RumorType

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


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
            while state.phase == Phase.PHASE_1:
                await gossip.originate_heartbeat()
                await asyncio.sleep(config.heartbeat_interval)
        

        phase1_start = time.time()

        async def _stability_timer():
            while True:
                await asyncio.sleep(1.0)  # check every second

                now              = time.time()
                floor_met        = (now - phase1_start) >= config.phase1_floor
                table_stable     = (now - state.last_table_change_time) >= config.stability_window

                if floor_met and table_stable:
                    log.info("Stability timer fired — locking table")
                    await _end_phase1()
                    if state.phase == Phase.PHASE_2:   # only exit if we actually advanced
                        return  # task is done

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
                payload={"target_phase": "PHASE_2"},
            )
            state.mark_seen(ready_rumor.rumor_id)
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

    
                
        timer_task    = asyncio.create_task(_stability_timer())
        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        yield
        heartbeat_task.cancel()
        timer_task.cancel()

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

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return FileResponse(TEMPLATES_DIR / "status.html")

    return app