"""Runtime helpers for executing the multi-agent orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Optional

from Agents.core.logging import configure_logging
from Agents.core.models import AgentRequest, OrchestratorResponse

from .workflow import OrchestratorState, compile_orchestrator

logger = logging.getLogger(__name__)

_COMPILED_GRAPH: Optional[Any] = None


def _get_compiled_graph() -> Any:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        logger.debug("No compiled orchestrator graph present; compiling")
        configure_logging()
        try:
            graph = compile_orchestrator()
            _COMPILED_GRAPH = graph.compile()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to compile orchestrator graph")
            raise RuntimeError(f"Unable to compile orchestrator graph: {exc}") from exc
        logger.info("Orchestrator graph compiled successfully")
    return _COMPILED_GRAPH


async def arun(request: AgentRequest) -> OrchestratorResponse:
    graph = _get_compiled_graph()
    initial_state: OrchestratorState = {
        "request": request,
    }
    logger.info("Executing orchestrator", extra={"request": request.model_dump(exclude_none=True)})
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Async orchestrator execution failed")
        raise RuntimeError(f"Orchestrator execution failed: {exc}") from exc

    response = final_state.get("response")
    if response is None:
        logger.error("Orchestrator completed without producing a response", extra={"final_state_keys": list(final_state.keys())})
        raise RuntimeError("Orchestrator completed without producing a response")

    logger.info("Orchestrator execution succeeded", extra={"response": response.model_dump(exclude_none=True)})
    return response


async def astream(request: AgentRequest) -> AsyncIterator[str]:
    graph = _get_compiled_graph()
    initial_state: OrchestratorState = {
        "request": request,
    }
    logger.info("Streaming orchestrator execution", extra={"request": request.model_dump(exclude_none=True)})
    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        yield json.dumps(_serialise_chunk(chunk)) + "\n"


def _serialise_chunk(chunk: Any) -> Any:
    if isinstance(chunk, dict):
        return {str(key): _serialise_chunk(value) for key, value in chunk.items()}
    if isinstance(chunk, (list, tuple)):
        return [_serialise_chunk(item) for item in chunk]
    if isinstance(chunk, datetime):
        return chunk.isoformat()
    if isinstance(chunk, date):
        return chunk.isoformat()
    if isinstance(chunk, Decimal):
        return float(chunk)
    if hasattr(chunk, "model_dump") and callable(chunk.model_dump):
        return _serialise_chunk(chunk.model_dump(exclude_none=True))
    if hasattr(chunk, "dict") and callable(chunk.dict):
        return _serialise_chunk(chunk.dict())
    if isinstance(chunk, (str, int, float, bool)) or chunk is None:
        return chunk
    return str(chunk)


def run(request: AgentRequest) -> OrchestratorResponse:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        logger.debug("Event loop already running; scheduling orchestrator execution")
        future = asyncio.run_coroutine_threadsafe(arun(request), loop)
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Synchronous orchestrator execution failed via running loop")
            raise

    logger.debug("Event loop idle; running orchestrator synchronously")
    return loop.run_until_complete(arun(request))


__all__ = ["run", "arun", "astream"]


