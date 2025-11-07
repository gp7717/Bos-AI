"""FastAPI server exposing the multi-agent orchestrator."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from Agents.core.logging import configure_logging
from Agents.core.models import AgentRequest, OrchestratorResponse

from .service import arun


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Agentic Orchestrator", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=OrchestratorResponse)
    async def run_query(request: AgentRequest) -> OrchestratorResponse:
        try:
            return await arun(request)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


__all__ = ["app", "create_app"]


