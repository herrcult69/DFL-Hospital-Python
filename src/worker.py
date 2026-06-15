"""
Worker node FastAPI app.
Registers with Bootstrap on startup, then stays alive as a real server.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse

from .config import NetworkConfig
from .models import JoinRequest, RewireRequest, RewireResponse, StatusResponse
from .state import NodeState

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def create_worker_app(state: NodeState, config: NetworkConfig) -> FastAPI:

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
                log.info(
                    f"Joined. Neighbors: {state.neighbor_map} | "
                    f"Table size: {len(state.global_table)}"
                )
            except Exception as e:
                log.error(f"Could not reach Bootstrap at {config.bootstrap_url}: {e}")
        yield

    app = FastAPI(title="DFL Worker Node", lifespan=lifespan)

    @app.post("/rewire", response_model=RewireResponse)
    async def rewire(req: RewireRequest):
        state.neighbor_map = set(req.new_neighbors)
        log.info(f"Rewired. New neighbors: {req.new_neighbors}")
        return RewireResponse(status="ok")

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return FileResponse(TEMPLATES_DIR / "status.html")

    return app