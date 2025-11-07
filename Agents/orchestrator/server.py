"""FastAPI server exposing the multi-agent orchestrator."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from Agents.core.logging import configure_logging
from Agents.core.models import AgentRequest, OrchestratorResponse

from .service import arun, astream


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Agentic Orchestrator", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=OrchestratorResponse)
    async def run_query(request: AgentRequest, raw_request: Request) -> OrchestratorResponse:
        logger.info(
            "Received orchestrator request",
            extra={
                "client": raw_request.client.host if raw_request.client else None,
                "path": str(raw_request.url),
                "payload": request.model_dump(exclude_none=True),
            },
        )
        try:
            response = await arun(request)
            logger.info(
                "Returning orchestrator response",
                extra={"answer_preview": response.answer[:200], "metadata_keys": list(response.metadata.keys())},
            )
            return response
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unhandled exception while processing orchestrator request")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query/stream")
    async def stream_query(request: AgentRequest, raw_request: Request) -> StreamingResponse:
        logger.info(
            "Received orchestrator stream request",
            extra={
                "client": raw_request.client.host if raw_request.client else None,
                "path": str(raw_request.url),
                "payload": request.model_dump(exclude_none=True),
            },
        )

        async def event_generator():
            try:
                async for chunk in astream(request):
                    yield chunk
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Streamed orchestrator execution failed")
                yield json.dumps({"error": str(exc)}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/json")

    return app


app = create_app()


__all__ = ["app", "create_app"]


