import logging
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse

from .config import NetworkConfig
from .models import JoinRequest, JoinResponse, RewireRequest, RewireResponse, StatusResponse
from .state import NodeState
from .graph import GraphManager

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def create_bootstrap_app(state: NodeState, graph: GraphManager, config: NetworkConfig) -> FastAPI:

    http_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI): # Fast API Startup code
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=config.http_timeout)
        yield
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
        rewire_map = graph.register(req.node_id)
        state.add_node(req.node_id)

        for affected_node, new_neighbors in rewire_map.items():
            if affected_node == state.node_id:
                # Bootstrap updates its own neighbor map directly — no HTTP call
                state.neighbor_map = set(new_neighbors)
                log.info(f"Bootstrap self-rewired → {new_neighbors}")
            else:
                await _send_rewire(affected_node, new_neighbors)

        neighbors = graph.get_neighbors(req.node_id)
        log.info(f"Node joined: {req.node_id} | neighbors: {neighbors} | total: {len(state.global_table)}")

        return JoinResponse(
            neighbors=neighbors,
            global_table=list(state.global_table),
            message=f"Welcome! Network has {len(state.global_table)} node(s).",
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



    @app.get("/status-page", response_class=HTMLResponse)
    async def status_page():
        return FileResponse(TEMPLATES_DIR / "status.html")

    @app.get("/graph-page", response_class=HTMLResponse)
    async def graph_page():
        return FileResponse(TEMPLATES_DIR / "graph.html")

    return app