"""
Bootstrap node FastAPI app.
"""
import logging
from fastapi import FastAPI
from .models import JoinRequest, JoinResponse, StatusResponse
from .state import NodeState

log = logging.getLogger(__name__)


def create_bootstrap_app(state: NodeState) -> FastAPI:
    app = FastAPI(title="DFL Bootstrap Node")

    @app.post("/join", response_model=JoinResponse)
    async def join(req: JoinRequest):
        is_new = state.add_node(req.node_id)
        if is_new:
            log.info(f"New node joined: {req.node_id}  (total: {len(state.global_table)})")
        return JoinResponse(
            global_table=list(state.global_table),  
            message=f"Welcome! Network has {len(state.global_table)} node(s).",
        )

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(**state.snapshot())

    @app.get("/table")
    async def table():
        return {"global_table": state.global_table, "count": len(state.global_table)}

    return app