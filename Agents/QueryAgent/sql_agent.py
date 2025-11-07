"""Assembly of the LangGraph SQL agent workflow."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    call_get_schema,
    check_query,
    generate_query,
    get_schema,
    list_tables,
    run_query,
    summarize_result,
)
from .state import SQLAgentState, ensure_messages

logger = logging.getLogger(__name__)


def _should_continue(state: SQLAgentState) -> Literal["check_query", END]:
    """Route execution depending on whether a tool call was produced."""

    messages = ensure_messages(state)
    if not messages:
        logger.debug("No messages present after generate_query; ending graph")
        return END
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        logger.debug("No tool calls detected; ending graph")
        return END
    logger.debug("Tool call detected; routing to check_query")
    return "check_query"


def build_sql_agent() -> StateGraph:
    """Create the uncompiled LangGraph for the SQL agent."""

    logger.info("Building SQL agent state graph")
    workflow = StateGraph(SQLAgentState)

    workflow.add_node("list_tables", list_tables)
    workflow.add_node("call_get_schema", call_get_schema)
    workflow.add_node("get_schema", get_schema)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("check_query", check_query)
    workflow.add_node("run_query", run_query)
    workflow.add_node("summarize_result", summarize_result)

    workflow.set_entry_point("list_tables")
    workflow.add_edge(START, "list_tables")
    workflow.add_edge("list_tables", "call_get_schema")
    workflow.add_edge("call_get_schema", "get_schema")
    workflow.add_edge("get_schema", "generate_query")
    workflow.add_conditional_edges("generate_query", _should_continue)
    workflow.add_edge("check_query", "run_query")
    workflow.add_edge("run_query", "summarize_result")

    logger.debug("SQL agent graph constructed")
    return workflow


def compile_sql_agent():
    """Compile the SQL agent workflow into an executable graph."""

    logger.info("Compiling SQL agent graph")
    graph = build_sql_agent().compile()
    logger.debug("SQL agent graph compiled")
    return graph


__all__ = [
    "build_sql_agent",
    "compile_sql_agent",
]


