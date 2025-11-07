"""Runtime helpers for executing the multi-agent orchestrator."""

from __future__ import annotations

import asyncio
from typing import Optional

from langgraph.graph import CompiledStateGraph

from Agents.core.logging import configure_logging
from Agents.core.models import AgentRequest, OrchestratorResponse

from .workflow import OrchestratorState, compile_orchestrator

_COMPILED_GRAPH: Optional[CompiledStateGraph] = None


def _get_compiled_graph() -> CompiledStateGraph:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        configure_logging()
        graph = compile_orchestrator()
        _COMPILED_GRAPH = graph.compile()
    return _COMPILED_GRAPH


async def arun(request: AgentRequest) -> OrchestratorResponse:
    graph = _get_compiled_graph()
    initial_state: OrchestratorState = {
        "request": request,
    }
    final_state = await graph.ainvoke(initial_state)
    response = final_state.get("response")
    if response is None:
        raise RuntimeError("Orchestrator completed without producing a response")
    return response


def run(request: AgentRequest) -> OrchestratorResponse:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        return asyncio.run_coroutine_threadsafe(arun(request), loop).result()
    return loop.run_until_complete(arun(request))


__all__ = ["run", "arun"]


