import logging, time
import asyncio
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, Response

from .config import NetworkConfig
from .models import JoinRequest, JoinResponse, RewireRequest, RewireResponse, StatusResponse, Rumor, RumorType
from .state import NodeState, Phase
from .graph import GraphManager
from .gossip import GossipEngine

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def create_bootstrap_app(state: NodeState, graph: GraphManager, config: NetworkConfig) -> FastAPI:

    http_client: httpx.AsyncClient | None = None
    gossip = GossipEngine(state=state, config=config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI): # Fast API Startup code
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=config.http_timeout)
        async def _heartbeat_loop():
            while state.phase == Phase.PHASE_1:   # stops automatically when phase advances
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
                return  
            # Step 6: flip phase
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
        await http_client.aclose() # Fast API Close code

    app = FastAPI(title="DFL Bootstrap Node", lifespan=lifespan)
    

    def _gossip_addr(node_id: str) -> str:
        host, gossip_port, _ = node_id.split(":")
        return f"http://{host}:{gossip_port}"

    async def _send_rewire(node_id: str, new_neighbors: list[str]) -> None:
        body = RewireRequest(new_neighbors=new_neighbors)
        try:
            r = await http_client.post(
                f"{_gossip_addr(node_id)}/rewire",
                json=body.model_dump()
            )
            r.raise_for_status()
            log.info(f"Rewired {node_id} → {new_neighbors}")
        except Exception as e:
            log.warning(f"Rewire failed for {node_id}: {e}")

    @app.post("/join", response_model=JoinResponse)
    async def join(req: JoinRequest):
        if state.table_locked:                      
            return Response(status_code=423)            # Locked — Phase 1 is closing

        rewire_map = graph.register(req.node_id)
        state.add_node(req.node_id)
        
        for affected_node, new_neighbors in rewire_map.items():
            if affected_node == state.node_id:
                # Bootstrap updates its own neighbor map directly — no HTTP call
                state.neighbor_map = set(new_neighbors)
                log.info(f"Bootstrap self-rewired → {new_neighbors}")
            else:
                await _send_rewire(affected_node, new_neighbors)
        join_rumor = Rumor.build(
            type=RumorType.JOIN,
            originator_id=state.node_id,
            round=state.round,
            ttl=config.gossip_ttl,
            payload={"node_id": req.node_id},
        )
        await gossip.spread(join_rumor)
        neighbors = graph.get_neighbors(req.node_id)
        log.info(f"Node joined: {req.node_id} | neighbors: {neighbors} | total: {len(state.global_table)}")

        return JoinResponse(
            neighbors=neighbors,
            global_table=list(state.global_table),
            message=f"Welcome! Network has {len(state.global_table)} node(s).",
            round=state.round,  
        )
    @app.post("/gossip")
    async def gossip_receive(rumor: Rumor, request: Request):
        sender_id = request.headers.get("X-Sender-Id", "")
        await gossip.receive(rumor, sender_id=sender_id)
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    @app.get("/table")
    async def table():
        return {"global_table": state.global_table, "count": len(state.global_table)}

    @app.get("/graph")
    async def graph_dump():
        return {
            "adjacency":    graph.dump(),
            "degrees":      graph.degrees(),
            "is_k_regular": graph.is_k_regular(),
        }



    @app.get("/status-page", response_class=HTMLResponse)
    async def status_page():
        return FileResponse(TEMPLATES_DIR / "status.html")

    @app.get("/graph-page", response_class=HTMLResponse)
    async def graph_page():
        return FileResponse(TEMPLATES_DIR / "graph.html")

    return app