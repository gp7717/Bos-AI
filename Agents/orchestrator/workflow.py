"""LangGraph orchestrator wiring planner, SQL agent, computation agent, and composer."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from Agents.ApiDocsAgent import ApiDocsAgent
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
_API_DOCS_AGENT = ApiDocsAgent()
_COMPOSER = Composer()
_PLANNER = Planner()

_MAX_SQL_AGENT_RETRIES = 3


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
    elif agent == "api_docs":
        result = _run_api_docs_agent(request.question, request.context)
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

    if request.trace:
        response = response.model_copy(update={"trace": trace})

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
    metadata = {"context": context or {}}
    messages: List[BaseMessage] = [HumanMessage(content=question)]
    attempt_history: List[Dict[str, Any]] = []
    final_state: Optional[SQLAgentState] = None

    for attempt in range(_MAX_SQL_AGENT_RETRIES + 1):
        attempt_start = time.perf_counter()
        initial_state: SQLAgentState = {
            "messages": list(messages),
            "metadata": dict(metadata),
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
                        data={"attempt": attempt + 1},
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
                        data={"attempt": attempt + 1},
                    )
                ],
                latency_ms=latency,
            )

        attempt_latency = (time.perf_counter() - attempt_start) * 1000
        result_payload = final_state.get("last_query_result")
        success = isinstance(result_payload, dict) and result_payload.get("success")
        query = final_state.get("last_query")
        row_count = (
            result_payload.get("row_count") if isinstance(result_payload, dict) else None
        )
        error_text = None if success else _extract_sql_error(result_payload, final_state)

        attempt_history.append(
            {
                "attempt": attempt + 1,
                "query": query,
                "row_count": row_count,
                "success": bool(success),
                "error": error_text,
                "latency_ms": attempt_latency,
            }
        )

        if success:
            return _build_sql_agent_success(final_state, result_payload, attempt_history, start)

        if attempt == _MAX_SQL_AGENT_RETRIES:
            break

        messages = list(final_state.get("messages", []))
        feedback = _build_retry_feedback(query, error_text)
        messages.append(HumanMessage(content=feedback))

    return _build_sql_agent_failure(final_state, attempt_history, start)


def _extract_sql_error(result_payload: Optional[Dict[str, Any]], state: Optional[SQLAgentState]) -> str:
    if isinstance(result_payload, dict):
        error_text = result_payload.get("error")
        if error_text:
            return str(error_text)
    if state:
        summary = state.get("final_answer")
        if summary:
            return str(summary)
    return "SQL agent failed"


def _build_retry_feedback(query: Optional[str], error_text: Optional[str]) -> str:
    parts = ["The previous SQL query failed to execute."]
    if error_text:
        parts.append(f"Error details: {error_text}")
    if query:
        parts.append(f"Failing query:\n{query}")
    parts.append(
        "Generate a corrected read-only SQL query that addresses the error and try again using only the authorised tables."
    )
    return "\n\n".join(parts)


def _build_trace_events(attempt_history: List[Dict[str, Any]]) -> List[TraceEvent]:
    events: List[TraceEvent] = []
    for entry in attempt_history:
        data = {
            "attempt": entry.get("attempt"),
            "query": entry.get("query"),
            "row_count": entry.get("row_count"),
            "latency_ms": entry.get("latency_ms"),
        }
        if not entry.get("success"):
            data["error"] = entry.get("error")
        filtered_data = {k: v for k, v in data.items() if v is not None}
        event_type = (
            TraceEventType.MESSAGE if entry.get("success") else TraceEventType.ERROR
        )
        message = (
            "SQL agent attempt succeeded"
            if entry.get("success")
            else "SQL agent attempt failed"
        )
        events.append(
            TraceEvent(
                event_type=event_type,
                agent="sql",  # type: ignore[assignment]
                message=message,
                data=filtered_data,
            )
        )
    return events


def _build_sql_agent_success(
    final_state: SQLAgentState,
    result_payload: Dict[str, Any],
    attempt_history: List[Dict[str, Any]],
    overall_start: float,
) -> AgentResult:
    latency = (time.perf_counter() - overall_start) * 1000
    answer = final_state.get("final_answer")
    tabular = TabularResult(
        columns=[str(col) for col in result_payload.get("columns", [])],
        rows=result_payload.get("data", []),
        row_count=result_payload.get("row_count", 0),
    )
    trace = _build_trace_events(attempt_history)

    return AgentResult(
        agent="sql",  # type: ignore[assignment]
        status=AgentExecutionStatus.succeeded,
        answer=answer,
        tabular=tabular,
        trace=trace,
        latency_ms=latency,
    )


def _build_sql_agent_failure(
    final_state: Optional[SQLAgentState],
    attempt_history: List[Dict[str, Any]],
    overall_start: float,
) -> AgentResult:
    latency = (time.perf_counter() - overall_start) * 1000
    trace = _build_trace_events(attempt_history)
    error_message = attempt_history[-1]["error"] if attempt_history else "SQL agent failed"
    query = attempt_history[-1]["query"] if attempt_history else None
    error = AgentError(
        message=str(error_message or "SQL agent failed"),
        type="QueryError",
        details={"query": query, "attempts": len(attempt_history)},
    )
    answer = final_state.get("final_answer") if final_state else None

    return AgentResult(
        agent="sql",  # type: ignore[assignment]
        status=AgentExecutionStatus.failed,
        answer=answer,
        error=error,
        trace=trace,
        latency_ms=latency,
    )


def _run_api_docs_agent(question: str, context: Optional[Dict[str, Any]]) -> AgentResult:
    start = time.perf_counter()
    try:
        result = _API_DOCS_AGENT.invoke(question, context=context or {})
    except Exception as exc:  # pragma: no cover - defensive
        latency = (time.perf_counter() - start) * 1000
        error = AgentError(message=str(exc), type=type(exc).__name__)
        return AgentResult(
            agent="api_docs",
            status=AgentExecutionStatus.failed,
            error=error,
            trace=[
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="api_docs",  # type: ignore[assignment]
                    message="ApiDocsAgent invocation raised an exception",
                    data={"exception": str(exc)},
                )
            ],
            latency_ms=latency,
        )

    result.latency_ms = (time.perf_counter() - start) * 1000
    return result


__all__ = ["compile_orchestrator", "OrchestratorState"]


