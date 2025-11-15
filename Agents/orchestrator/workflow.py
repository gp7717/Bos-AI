"""LangGraph orchestrator wiring planner, SQL agent, computation agent, and composer."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from Agents.ApiDocsAgent import ApiDocsAgent
from Agents.ComputationAgent import ComputationAgent
from Agents.GraphAgent import GraphAgent
from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentName,
    AgentRequest,
    AgentResult,
    DecomposedQuery,
    OrchestratorResponse,
    PlannerDecision,
    RouterDecision,
    Scratchpad,
    TabularResult,
    TraceEvent,
    TraceEventType,
)
from Agents.QueryAgent.config import ConfigurationError
from Agents.QueryAgent.sql_agent import compile_sql_agent
from Agents.QueryAgent.state import SQLAgentState

from .composer import Composer
from .decomposer import QueryDecomposer
from .parallel_executor import ParallelExecutor
from .planner import Planner
from .router import Router


class OrchestratorState(TypedDict, total=False):
    request: AgentRequest
    router: RouterDecision
    decomposed_query: DecomposedQuery
    planner: PlannerDecision
    pending_agents: List[AgentName]
    agent_results: List[AgentResult]
    trace: List[TraceEvent]
    response: OrchestratorResponse
    scratchpad: Scratchpad


_SQL_GRAPH = compile_sql_agent()
_COMPUTATION_AGENT = ComputationAgent()
_API_DOCS_AGENT = ApiDocsAgent()
_GRAPH_AGENT = GraphAgent()
_COMPOSER = Composer()
_PLANNER = Planner()
_ROUTER = Router()
_DECOMPOSER = QueryDecomposer()
_PARALLEL_EXECUTOR = ParallelExecutor(max_workers=4)

_MAX_SQL_AGENT_RETRIES = 3


def _attach_temporal_context(request: AgentRequest) -> AgentRequest:
    base_context = dict(request.context or {})
    now_utc = datetime.now(timezone.utc)

    updated = False
    if "current_datetime_utc" not in base_context:
        base_context["current_datetime_utc"] = now_utc.isoformat()
        updated = True
    if "current_date" not in base_context:
        iso_date = now_utc.date().isoformat()
        base_context["current_date"] = iso_date
        base_context.setdefault("current_date_start_hour", f"{iso_date} 00")
        base_context.setdefault("current_date_end_hour", f"{iso_date} 23")
        updated = True

    if not updated:
        return request
    return request.model_copy(update={"context": base_context})


def router_node(state: OrchestratorState) -> OrchestratorState:
    """Route query to determine if agents are needed."""
    request = _attach_temporal_context(state["request"])
    decision = _ROUTER.route(
        question=request.question,
        context=request.context,
    )

    trace = list(state.get("trace", []))
    trace.append(
        TraceEvent(
            event_type=TraceEventType.DECISION,
            agent="router",
            message="Router classified query",
            data={
                "route_type": decision.route_type,
                "confidence": decision.confidence,
            },
        )
    )

    # Initialize scratchpad if not present
    scratchpad = state.get("scratchpad")
    if scratchpad is None:
        scratchpad = Scratchpad()

    return {
        "request": request,
        "router": decision,
        "trace": trace,
        "scratchpad": scratchpad,
    }


def _route_after_router(state: OrchestratorState) -> str:
    """Route after router decision: simple_response -> compose, needs_agents -> decompose."""
    router = state.get("router")
    if router and router.route_type == "simple_response":
        return "compose"
    return "decompose"


def decompose_node(state: OrchestratorState) -> OrchestratorState:
    """Decompose natural language query into specific sub-queries."""
    request = _attach_temporal_context(state["request"])

    # Decompose query
    decomposed = _DECOMPOSER.decompose(
        question=request.question,
        context=request.context,
    )

    trace = list(state.get("trace", []))
    trace.append(
        TraceEvent(
            event_type=TraceEventType.DECISION,
            agent=None,  # Decomposer is not an agent, it's an orchestrator component
            message=f"Decomposed query into {len(decomposed.sub_queries)} sub-queries",
            data={
                "interpreted_query": decomposed.interpreted_query,
                "sub_query_count": len(decomposed.sub_queries),
                "intent": decomposed.intent.primary_intent,
                "confidence": decomposed.confidence,
                "inferred_metrics": decomposed.inferred_metrics,
                "component": "decomposer",  # Store component name in data instead
            },
        )
    )

    scratchpad = state.get("scratchpad") or Scratchpad()
    scratchpad.add(
        agent="router",  # Use router as orchestrator component (decomposer is not a valid agent)
        content=f"[Decomposer] Query interpreted as: {decomposed.interpreted_query}. Decomposed into {len(decomposed.sub_queries)} sub-queries.",
        category="context",
        metadata={
            "intent": decomposed.intent.primary_intent,
            "metrics": decomposed.inferred_metrics,
        },
    )

    return {
        "request": request,
        "decomposed_query": decomposed,
        "trace": trace,
        "scratchpad": scratchpad,
    }


def _should_decompose(state: OrchestratorState) -> str:
    """Determine if query needs decomposition or can proceed to planning."""
    router = state.get("router")
    if router and router.route_type == "simple_response":
        return "compose"  # Skip decomposition for simple queries

    # Always proceed to plan after decomposition
    return "plan"


def plan_node(state: OrchestratorState) -> OrchestratorState:
    request = _attach_temporal_context(state["request"])
    decomposed = state.get("decomposed_query")
    
    decision = _PLANNER.plan(
        question=request.question,
        decomposed_query=decomposed,  # Pass decomposed query if available
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

    scratchpad = state.get("scratchpad") or Scratchpad()
    # Add planner decision to scratchpad
    scratchpad.add(
        agent="planner",
        content=f"Planned execution order: {', '.join(decision.chosen_agents)}. Rationale: {decision.rationale}",
        category="context",
    )

    return {
        "request": request,
        "planner": decision,
        "pending_agents": list(decision.chosen_agents),
        "agent_results": state.get("agent_results", []),
        "trace": trace,
        "scratchpad": scratchpad,
    }


def _route_after_plan(state: OrchestratorState) -> str:
    """Route after plan: check if parallel execution is possible."""
    pending = state.get("pending_agents", [])
    if not pending:
        return "compose"
    
    # Check if we can use parallel execution
    decomposed = state.get("decomposed_query")
    if decomposed and len(decomposed.sub_queries) > 1:
        # Check if there are independent agents
        # For now, use parallel execution if we have multiple agents
        if len(pending) > 1:
            return "execute_agents_parallel"
    
    return "execute_agent"


def _extract_data_summary(tabular: Optional[TabularResult]) -> str:
    """Extract a concise summary of tabular data for scratchpad with key insights."""
    if not tabular or not tabular.rows:
        return "No data returned."
    
    row_count = tabular.row_count or len(tabular.rows)
    columns = tabular.columns or []
    
    summary_parts = [f"Retrieved {row_count} rows"]
    if columns:
        summary_parts.append(f"with columns: {', '.join(columns[:5])}")
        if len(columns) > 5:
            summary_parts.append(f"(+{len(columns) - 5} more)")
    
    # Extract key insights from numeric columns
    if tabular.rows and len(tabular.rows) > 0 and isinstance(tabular.rows[0], dict):
        numeric_insights = []
        
        # Look for common numeric columns and calculate insights
        for col in columns:
            col_lower = col.lower()
            values = []
            for row in tabular.rows:
                if isinstance(row, dict):
                    val = row.get(col)
                    if val is not None:
                        try:
                            # Try to convert to float
                            num_val = float(str(val).replace(",", ""))
                            values.append(num_val)
                        except (ValueError, AttributeError):
                            pass
            
            if values:
                if "revenue" in col_lower or "amount" in col_lower or "total" in col_lower:
                    total = sum(values)
                    avg = total / len(values) if values else 0
                    max_val = max(values)
                    min_val = min(values)
                    numeric_insights.append(
                        f"{col}: total={total:,.0f}, avg={avg:,.0f}, max={max_val:,.0f}, min={min_val:,.0f}"
                    )
                elif "count" in col_lower or "orders" in col_lower:
                    total = sum(values)
                    avg = total / len(values) if values else 0
                    max_val = max(values)
                    numeric_insights.append(
                        f"{col}: total={total:,.0f}, avg={avg:.1f}, max={max_val:,.0f}"
                    )
                elif "date" in col_lower or "time" in col_lower:
                    # Extract date range (first and last row)
                    if len(tabular.rows) > 0:
                        first_row = tabular.rows[0]
                        last_row = tabular.rows[-1]
                        if isinstance(first_row, dict) and isinstance(last_row, dict):
                            first_date = str(first_row.get(col, ""))
                            last_date = str(last_row.get(col, ""))
                            if first_date and last_date:
                                if first_date != last_date:
                                    numeric_insights.append(f"{col}: range from {first_date} to {last_date}")
                                else:
                                    numeric_insights.append(f"{col}: {first_date}")
        
        if numeric_insights:
            summary_parts.append("Key metrics: " + "; ".join(numeric_insights[:3]))  # Limit to 3 insights
    
    # Add sample values if available
    if tabular.rows and len(tabular.rows) > 0:
        first_row = tabular.rows[0]
        if isinstance(first_row, dict) and columns:
            sample_values = []
            for col in columns[:2]:  # First 2 columns for sample
                val = first_row.get(col)
                if val is not None:
                    sample_values.append(f"{col}={val}")
            if sample_values:
                summary_parts.append(f"Sample row: {', '.join(sample_values)}")
    
    return ". ".join(summary_parts) + "."


def execute_agent(state: OrchestratorState) -> OrchestratorState:
    pending = list(state.get("pending_agents", []))
    if not pending:
        return {"pending_agents": []}

    agent = pending.pop(0)
    request = state["request"]
    trace = list(state.get("trace", []))
    existing_results = list(state.get("agent_results", []))
    scratchpad = state.get("scratchpad") or Scratchpad()

    # Prepare context for agents - include scratchpad and previous results
    context = dict(request.context or {})
    context["scratchpad"] = scratchpad  # Pass scratchpad to agents
    
    # Check for decomposed query and use sub-query context
    decomposed = state.get("decomposed_query")
    agent_question = request.question
    
    if decomposed and len(decomposed.sub_queries) > 0:
        # Find relevant sub-queries for current agent
        relevant_sub_queries = [
            sq for sq in decomposed.sub_queries
            if agent in sq.required_agents
        ]
        
        if relevant_sub_queries:
            # Use detailed queries from sub-queries instead of original
            sub_query_texts = [sq.detailed_query for sq in relevant_sub_queries]
            # Combine sub-queries into a focused query for this agent
            if len(sub_query_texts) > 1:
                focused_query = " | ".join(sub_query_texts)
            else:
                focused_query = sub_query_texts[0] if sub_query_texts else request.question
            
            # Add sub-query context
            context["sub_queries"] = [sq.model_dump() for sq in relevant_sub_queries]
            context["focused_query"] = focused_query
            context["decomposed_context"] = True
            context["selected_tools"] = [
                tool_id
                for sq in relevant_sub_queries
                for tool_id in sq.selected_tools
            ]
            
            # Use focused query for agent execution
            agent_question = focused_query
    
    # Prepare context for graph agent - include tabular data from previous agents
    if agent == "graph":
        # Find tabular data from previous agents
        for prev_result in existing_results:
            if prev_result.tabular:
                context["tabular_data"] = prev_result.tabular
                break
        # Also pass agent_results for reference
        context["agent_results"] = existing_results

    # Execute agent with focused question
    if agent == "sql":
        result = _run_sql_agent(agent_question, context)
    elif agent == "computation":
        result = _COMPUTATION_AGENT.invoke(agent_question, context=context)
    elif agent == "api_docs":
        result = _run_api_docs_agent(agent_question, context)
    elif agent == "graph":
        result = _GRAPH_AGENT.invoke(agent_question, context=context)
    else:
        result = AgentResult(
            agent=agent,
            status=AgentExecutionStatus.skipped,
            error=AgentError(message=f"Agent '{agent}' not implemented"),
        )

    # Extract findings and add to scratchpad
    if result.status == AgentExecutionStatus.succeeded:
        # Add agent answer as finding if available
        if result.answer:
            scratchpad.add(
                agent=agent,
                content=result.answer,
                category="finding",
                metadata={"status": "succeeded"},
            )
        
        # Add data summary if tabular data exists
        if result.tabular:
            data_summary = _extract_data_summary(result.tabular)
            scratchpad.add(
                agent=agent,
                content=data_summary,
                category="data_summary",
                metadata={
                    "row_count": result.tabular.row_count,
                    "column_count": len(result.tabular.columns) if result.tabular.columns else 0,
                },
            )
        
        # Add graph insights if available
        if result.graph and result.graph.trend_analysis:
            scratchpad.add(
                agent=agent,
                content=result.graph.trend_analysis,
                category="insight",
                metadata={"chart_type": result.graph.chart_type},
            )
    elif result.status == AgentExecutionStatus.failed and result.error:
        scratchpad.add(
            agent=agent,
            content=f"Error: {result.error.message}",
            category="error",
            metadata={"error_type": result.error.type or "Unknown"},
        )

    # Merge traces while keeping chronological order
    if result.trace:
        trace.extend(result.trace)

    existing_results.append(result)

    return {
        "request": request,
        "pending_agents": pending,
        "agent_results": existing_results,
        "trace": trace,
        "scratchpad": scratchpad,
    }


def _route_after_execute(state: OrchestratorState) -> str:
    """Route after execute: continue with more agents or compose."""
    pending = state.get("pending_agents", [])
    if pending:
        # Check if we should use parallel execution
        decomposed = state.get("decomposed_query")
        if decomposed and len(pending) > 1:
            return "execute_agents_parallel"
        return "execute_agent"
    return "compose"


def execute_agents_parallel(state: OrchestratorState) -> OrchestratorState:
    """Execute independent agents in parallel."""
    pending = list(state.get("pending_agents", []))
    if not pending:
        return {"pending_agents": []}

    request = state["request"]
    trace = list(state.get("trace", []))
    existing_results = list(state.get("agent_results", []))
    scratchpad = state.get("scratchpad") or Scratchpad()
    decomposed = state.get("decomposed_query")

    # Build dependencies from decomposed query
    dependencies: Dict[str, List[str]] = {}
    if decomposed:
        for sq in decomposed.sub_queries:
            for agent in sq.required_agents:
                if sq.dependencies:
                    dep_agents = []
                    for dep_id in sq.dependencies:
                        dep_sq = next(
                            (dsq for dsq in decomposed.sub_queries if dsq.id == dep_id),
                            None
                        )
                        if dep_sq:
                            dep_agents.extend(dep_sq.required_agents)
                    if dep_agents:
                        dependencies[agent] = dep_agents

    # Prepare context
    context = dict(request.context or {})
    context["scratchpad"] = scratchpad
    context["agent_results"] = existing_results

    # Build agent invokers with sub-query context handling
    def sql_invoker(q: str, ctx: Dict[str, Any]) -> AgentResult:
        # Use focused query if available
        focused_q = ctx.get("focused_query", q)
        return _run_sql_agent(focused_q, ctx)

    def computation_invoker(q: str, ctx: Dict[str, Any]) -> AgentResult:
        focused_q = ctx.get("focused_query", q)
        return _COMPUTATION_AGENT.invoke(focused_q, context=ctx)

    def api_docs_invoker(q: str, ctx: Dict[str, Any]) -> AgentResult:
        focused_q = ctx.get("focused_query", q)
        return _run_api_docs_agent(focused_q, ctx)

    def graph_invoker(q: str, ctx: Dict[str, Any]) -> AgentResult:
        # Find tabular data from previous results
        for prev_result in existing_results:
            if prev_result.tabular:
                ctx["tabular_data"] = prev_result.tabular
                break
        ctx["agent_results"] = existing_results
        focused_q = ctx.get("focused_query", q)
        return _GRAPH_AGENT.invoke(focused_q, context=ctx)
    
    # Prepare context for each agent with sub-query information
    agent_contexts: Dict[AgentName, Dict[str, Any]] = {}
    if decomposed:
        for agent in pending:
            agent_ctx = dict(context)
            # Find relevant sub-queries for this agent
            relevant_sub_queries = [
                sq for sq in decomposed.sub_queries
                if agent in sq.required_agents
            ]
            
            if relevant_sub_queries:
                sub_query_texts = [sq.detailed_query for sq in relevant_sub_queries]
                if len(sub_query_texts) > 1:
                    focused_query = " | ".join(sub_query_texts)
                else:
                    focused_query = sub_query_texts[0] if sub_query_texts else request.question
                
                agent_ctx["sub_queries"] = [sq.model_dump() for sq in relevant_sub_queries]
                agent_ctx["focused_query"] = focused_query
                agent_ctx["decomposed_context"] = True
                agent_ctx["selected_tools"] = [
                    tool_id
                    for sq in relevant_sub_queries
                    for tool_id in sq.selected_tools
                ]
            else:
                agent_ctx = dict(context)
            
            agent_contexts[agent] = agent_ctx
    else:
        # No decomposition, use same context for all
        for agent in pending:
            agent_contexts[agent] = dict(context)

    agent_invokers = {
        "sql": sql_invoker,
        "computation": computation_invoker,
        "api_docs": api_docs_invoker,
        "graph": graph_invoker,
    }
    
    # Create wrapper invokers that use agent-specific context
    def create_agent_invoker(agent_name: AgentName, base_invoker):
        def wrapped_invoker(question: str, base_context: Dict[str, Any]) -> AgentResult:
            # Use agent-specific context if available
            agent_ctx = agent_contexts.get(agent_name, base_context)
            return base_invoker(question, agent_ctx)
        return wrapped_invoker
    
    wrapped_invokers = {
        agent: create_agent_invoker(agent, agent_invokers.get(agent, sql_invoker))
        for agent in pending
    }

    # Use asyncio to run parallel execution
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If loop is running, use run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(
            _PARALLEL_EXECUTOR.execute_agents_parallel(
                pending, request.question, context, dependencies, wrapped_invokers
            ),
            loop,
        )
        new_results = future.result()
    else:
        new_results = loop.run_until_complete(
            _PARALLEL_EXECUTOR.execute_agents_parallel(
                pending, request.question, context, dependencies, wrapped_invokers
            )
        )

    # Merge results
    existing_results.extend(new_results)

    # Update scratchpad with new results
    for result in new_results:
        if result.status == AgentExecutionStatus.succeeded:
            if result.answer:
                scratchpad.add(agent=result.agent, content=result.answer, category="finding")
            if result.tabular:
                data_summary = _extract_data_summary(result.tabular)
                scratchpad.add(agent=result.agent, content=data_summary, category="data_summary")
        elif result.status == AgentExecutionStatus.failed and result.error:
            scratchpad.add(
                agent=result.agent,
                content=f"Error: {result.error.message}",
                category="error",
                metadata={"error_type": result.error.type or "Unknown"},
            )

    # Merge traces
    for result in new_results:
        if result.trace:
            trace.extend(result.trace)

    return {
        "request": request,
        "pending_agents": [],  # All executed
        "agent_results": existing_results,
        "trace": trace,
        "scratchpad": scratchpad,
    }


def compose_node(state: OrchestratorState) -> OrchestratorState:
    request = state["request"]
    router = state.get("router")
    planner = state.get("planner")
    agent_results = state.get("agent_results", [])
    trace = list(state.get("trace", []))

    # Check if this is a simple response (no agents needed)
    if router and router.route_type == "simple_response":
        response = _COMPOSER.compose_simple_response(
            question=request.question,
            router_rationale=router.rationale,
            context=request.context,
            metadata={
                "confidence": router.confidence,
                "route_type": router.route_type,
            },
        )
    else:
        # Regular composer with agent results
        scratchpad = state.get("scratchpad")
        response = _COMPOSER.compose(
            question=request.question,
            planner_rationale=planner.rationale if planner else "",
            agent_results=agent_results,
            context=request.context,
            metadata={
                "confidence": planner.confidence if planner else None,
                "agents": [result.agent for result in agent_results],
            },
            scratchpad=scratchpad,
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
    workflow.add_node("route", router_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_agent", execute_agent)
    workflow.add_node("execute_agents_parallel", execute_agents_parallel)
    workflow.add_node("compose", compose_node)

    workflow.set_entry_point("route")
    workflow.add_edge(START, "route")
    workflow.add_conditional_edges(
        "route",
        _route_after_router,
        {"compose": "compose", "decompose": "decompose"},
    )
    workflow.add_conditional_edges(
        "decompose",
        _should_decompose,
        {"compose": "compose", "plan": "plan"},
    )
    workflow.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "execute_agent": "execute_agent",
            "execute_agents_parallel": "execute_agents_parallel",
            "compose": "compose",
        },
    )
    workflow.add_conditional_edges(
        "execute_agent",
        _route_after_execute,
        {
            "execute_agent": "execute_agent",
            "execute_agents_parallel": "execute_agents_parallel",
            "compose": "compose",
        },
    )
    workflow.add_conditional_edges(
        "execute_agents_parallel",
        _route_after_execute,
        {
            "execute_agent": "execute_agent",
            "execute_agents_parallel": "execute_agents_parallel",
            "compose": "compose",
        },
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


