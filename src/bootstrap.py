import logging, time
import asyncio
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, Response

from .config import NetworkConfig
from .models import JoinRequest, JoinResponse, RewireRequest, RewireResponse, StatusResponse, PredictRequest, PredictResponse, Rumor, RumorType, EvictRequest
from .state import NodeState, Phase
from .graph import GraphManager
from .gossip import GossipEngine
from .ring_transfer import RingPhase

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
def _handle_task_exception(loop, context):
    msg = context.get("exception", context["message"])
    log.error(f"Unhandled async task exception: {msg}", exc_info=context.get("exception"))

asyncio.get_event_loop().set_exception_handler(_handle_task_exception)

def create_bootstrap_app(state: NodeState, graph: GraphManager, config: NetworkConfig) -> FastAPI:

    http_client: httpx.AsyncClient | None = None
    gossip = GossipEngine(state=state, config=config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI): # Fast API Startup code
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=config.http_timeout)
        
        async def _heartbeat_loop():
            while True:
                if state.phase == Phase.PHASE_1:
                    await gossip.originate_heartbeat()
                await asyncio.sleep(config.heartbeat_interval)

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

            # Step 3 -> REwiring the network
            for node in dead:
                if node in graph.adj:
                    affected = graph.evict(node)
                    for affected_node, new_neighbors in affected.items():
                        if affected_node == state.node_id:
                            state.neighbor_map = set(new_neighbors)
                        else:
                            await _send_rewire(affected_node, new_neighbors)
                
            # Minimum viable network check
            if len(state.global_table) < 2:              # ← adjust threshold as needed
                log.warning("Not enough nodes to proceed — holding, unlocking table")
                state.table_locked = False               # unlock so new joins can come in
                state.last_table_change_time = time.time()  # reset stability window
                # heartbeat loop keeps running since phase never flipped
                return                                   # timer task exits, but heartbeat continues

            
            # Step 3: Form Ring Map — sort locked table, derive left/right neighbors
            ring = state.global_table  # already sorted (add_node keeps it sorted) """" Duplicated CODED SKIPIING""""
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
                    # recompute ring with the cleaned table
                    idx = state.global_table.index(state.node_id)
                    n   = len(state.global_table)
                    state.ring_left  = state.global_table[(idx - 1) % n]
                    state.ring_right = state.global_table[(idx + 1) % n]
                    log.info(f"Ring recomputed after READY eviction — left: {state.ring_left} | right: {state.ring_right}")
                    return True                    # ← reduced set, still proceed

        async def _phase2_loop():
            while True:   # outer loop drives all rounds
                # Wait for Phase 1 to complete
                while state.phase == Phase.PHASE_1:
                    await asyncio.sleep(0.5)

                if state.phase == Phase.PHASE_2:
                    ring_phase = RingPhase(state=state, config=config, gossip=gossip)
                    await ring_phase.run()

                elif state.phase == Phase.PHASE_3:
                    from .phase3 import AggregationPhase
                    safe       = state.node_id.replace(":", "_")
                    chunk_dir  = Path(f"./chunks_{safe}")
                    agg_phase  = AggregationPhase(state=state, config=config, gossip=gossip)
                    await agg_phase.run(chunk_dir=chunk_dir)

                elif state.phase == Phase.PHASE_4:
                    from .phase4 import RoundCompletionPhase
                    safe       = state.node_id.replace(":", "_")
                    chunk_dir  = Path(f"./chunks_{safe}")
                    p4         = RoundCompletionPhase(state=state, config=config, gossip=gossip)
                    await p4.run(chunk_dir=chunk_dir)
                    # p4.run() calls reset_phase1() → state.phase == PHASE_1 again
                    # → outer while True loops back, re-enters Phase 1 wait

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
            # return f"[STUB] Echo from {state.node_id} round {state.round}: {req.message}"
            # ── REAL IMPLEMENTATION (uncomment for production) ────────────────
            from .inference import run_inference
            return run_inference(req.message, str(config.model_path))

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



    @app.post("/rewire-evict")
    async def rewire_evict(req: EvictRequest):
        """
        Called at the start of each round by bootstrap itself to remove
        dead nodes from the K-graph and re-wire survivors.
        """
        rewired: dict[str, list[str]] = {}
        for dead in req.dead_nodes:
            if dead not in graph.adj:
                continue
            affected = graph.evict(dead)
            rewired.update(affected)

        # Notify all affected alive nodes of their new neighbor lists
        for node_id, new_neighbors in rewired.items():
            if node_id == state.node_id:
                state.neighbor_map = set(new_neighbors)
            else:
                await _send_rewire(node_id, new_neighbors)

        return {"rewired": rewired, "dead_removed": req.dead_nodes}


    @app.get("/status-page", response_class=HTMLResponse)
    async def status_page():
        return FileResponse(TEMPLATES_DIR / "status.html")

    @app.get("/graph-page", response_class=HTMLResponse)
    async def graph_page():
        return FileResponse(TEMPLATES_DIR / "graph.html")

    return app