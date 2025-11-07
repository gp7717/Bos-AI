"""LangGraph orchestrator wiring planner, SQL agent, computation agent, and composer."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from Agents.ComputationAgent import ComputationAgent
from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentName,
    AgentRequest,
    AgentResult,
    OrchestratorResponse,
    PlannerDecision,
    TabularResult,
    TraceEvent,
    TraceEventType,
)
from Agents.QueryAgent.config import ConfigurationError
from Agents.QueryAgent.sql_agent import compile_sql_agent
from Agents.QueryAgent.state import SQLAgentState

from .composer import Composer
from .planner import Planner


class OrchestratorState(TypedDict, total=False):
    request: AgentRequest
    planner: PlannerDecision
    pending_agents: List[AgentName]
    agent_results: List[AgentResult]
    trace: List[TraceEvent]
    response: OrchestratorResponse


_SQL_GRAPH = compile_sql_agent()
_COMPUTATION_AGENT = ComputationAgent()
_COMPOSER = Composer()
_PLANNER = Planner()


def plan_node(state: OrchestratorState) -> OrchestratorState:
    request = state["request"]
    decision = _PLANNER.plan(
        question=request.question,
        prefer=request.prefer_agents,
        disable=request.disable_agents,
        context=request.context,
    )

    trace = list(state.get("trace", []))
    trace.append(
        TraceEvent(
            event_type=TraceEventType.DECISION,
            agent="planner",  # type: ignore[assignment]
            message="Planner produced execution order",
            data={
                "agents": list(decision.chosen_agents),
                "confidence": decision.confidence,
            },
        )
    )

    return {
        "request": request,
        "planner": decision,
        "pending_agents": list(decision.chosen_agents),
        "agent_results": state.get("agent_results", []),
        "trace": trace,
    }


def _route_after_plan(state: OrchestratorState) -> str:
    if state.get("pending_agents"):
        return "execute_agent"
    return "compose"


def execute_agent(state: OrchestratorState) -> OrchestratorState:
    pending = list(state.get("pending_agents", []))
    if not pending:
        return {"pending_agents": []}

    agent = pending.pop(0)
    request = state["request"]
    trace = list(state.get("trace", []))

    if agent == "sql":
        result = _run_sql_agent(request.question, request.context)
    elif agent == "computation":
        result = _COMPUTATION_AGENT.invoke(request.question, context=request.context)
    else:
        result = AgentResult(
            agent=agent,
            status=AgentExecutionStatus.skipped,
            error=AgentError(message=f"Agent '{agent}' not implemented"),
        )

    # Merge traces while keeping chronological order
    if result.trace:
        trace.extend(result.trace)

    existing_results = list(state.get("agent_results", []))
    existing_results.append(result)

    return {
        "request": request,
        "pending_agents": pending,
        "agent_results": existing_results,
        "trace": trace,
    }


def _route_after_execute(state: OrchestratorState) -> str:
    if state.get("pending_agents"):
        return "execute_agent"
    return "compose"


def compose_node(state: OrchestratorState) -> OrchestratorState:
    request = state["request"]
    planner = state.get("planner")
    agent_results = state.get("agent_results", [])
    trace = list(state.get("trace", []))

    response = _COMPOSER.compose(
        question=request.question,
        planner_rationale=planner.rationale if planner else "",
        agent_results=agent_results,
        metadata={
            "confidence": planner.confidence if planner else None,
            "agents": [result.agent for result in agent_results],
        },
    )

    trace.append(
        TraceEvent(
            event_type=TraceEventType.RESULT,
            agent="composer",  # type: ignore[assignment]
            message="Composer generated final answer",
        )
    )

    return {
        "request": request,
        "response": response,
        "trace": trace,
    }


def compile_orchestrator() -> StateGraph:
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_agent", execute_agent)
    workflow.add_node("compose", compose_node)

    workflow.set_entry_point("plan")
    workflow.add_edge(START, "plan")
    workflow.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"execute_agent": "execute_agent", "compose": "compose"},
    )
    workflow.add_conditional_edges(
        "execute_agent",
        _route_after_execute,
        {"execute_agent": "execute_agent", "compose": "compose"},
    )
    workflow.add_edge("compose", END)

    return workflow


def _run_sql_agent(question: str, context: Optional[Dict[str, Any]]) -> AgentResult:
    start = time.perf_counter()
    initial_state: SQLAgentState = {
        "messages": [HumanMessage(content=question)],
        "metadata": {"context": context or {}},
    }

    try:
        final_state = _SQL_GRAPH.invoke(initial_state)
    except ConfigurationError as exc:
        latency = (time.perf_counter() - start) * 1000
        error = AgentError(
            message="SQL agent configuration error",
            type="ConfigurationError",
            details={"reason": str(exc)},
        )
        return AgentResult(
            agent="sql",  # type: ignore[assignment]
            status=AgentExecutionStatus.failed,
            error=error,
            trace=[
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="sql",  # type: ignore[assignment]
                    message=str(exc),
                )
            ],
            latency_ms=latency,
        )
    except Exception as exc:  # pragma: no cover - defensive catch
        latency = (time.perf_counter() - start) * 1000
        error = AgentError(
            message="SQL agent execution error",
            type=type(exc).__name__,
            details={"exception": str(exc)},
        )
        return AgentResult(
            agent="sql",  # type: ignore[assignment]
            status=AgentExecutionStatus.failed,
            error=error,
            trace=[
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="sql",  # type: ignore[assignment]
                    message=str(exc),
                )
            ],
            latency_ms=latency,
        )

    latency = (time.perf_counter() - start) * 1000
    result_payload = final_state.get("last_query_result")
    answer = final_state.get("final_answer")

    tabular = None
    if isinstance(result_payload, dict) and result_payload.get("success"):
        tabular = TabularResult(
            columns=[str(col) for col in result_payload.get("columns", [])],
            rows=result_payload.get("data", []),
            row_count=result_payload.get("row_count", 0),
        )

    trace = [
        TraceEvent(
            event_type=TraceEventType.MESSAGE,
            agent="sql",  # type: ignore[assignment]
            message="SQL agent completed execution",
            data={
                "query": final_state.get("last_query"),
                "row_count": result_payload.get("row_count") if isinstance(result_payload, dict) else None,
            },
        )
    ]

    status = (
        AgentExecutionStatus.succeeded
        if isinstance(result_payload, dict) and result_payload.get("success")
        else AgentExecutionStatus.failed
    )

    error = None
    if status == AgentExecutionStatus.failed:
        error_message = None
        if isinstance(result_payload, dict):
            error_message = result_payload.get("error") or "SQL query failed"
        error = AgentError(message=str(error_message or "SQL agent failed"), type="QueryError")

    return AgentResult(
        agent="sql",  # type: ignore[assignment]
        status=status,
        answer=answer,
        tabular=tabular,
        error=error,
        trace=trace,
        latency_ms=latency,
    )


__all__ = ["compile_orchestrator", "OrchestratorState"]


