"""
Worker node FastAPI app.
Registers with Bootstrap on startup, then stays alive as a real server.
"""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .models import JoinRequest, JoinResponse, StatusResponse
from .state import NodeState

log = logging.getLogger(__name__)


def create_worker_app(state: NodeState, bootstrap_url: str) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup: register with Bootstrap before serving ──────────────────
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.post(
                    f"{bootstrap_url}/join",
                    json=JoinRequest(node_id=state.node_id).model_dump(),
                )
                r.raise_for_status()
                data = r.json()
                for n in data["global_table"]:
                    state.add_node(n)
                log.info(
                    f"Registered with Bootstrap. "
                    f"Table: {state.global_table}"
                )
            except Exception as e:
                log.error(f"Could not reach Bootstrap: {e}")

        yield   # server is now live and serving requests
        # ── Shutdown (nothing to clean up yet) ───────────────────────────────

    app = FastAPI(title="DFL Worker Node", lifespan=lifespan)

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    return app