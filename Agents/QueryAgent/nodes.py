"""LangGraph node implementations for the custom SQL agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Iterable, Optional, Tuple, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .config import ConfigurationError, TableContext, get_resources, get_tool
from .state import SQLAgentState, ensure_messages

_resources = get_resources()
_LLM = _resources.llm
_ALLOWED_TABLES = _resources.allowed_tables
_TABLE_CONTEXT = _resources.table_context
_DIALECT = _resources.dialect

logger = logging.getLogger(__name__)

_GET_SCHEMA_TOOL = get_tool("mcp_db_schema")
_RUN_QUERY_TOOL = get_tool("mcp_db_query")

_FORBIDDEN_SQL_PATTERNS = [
    r"\bCREATE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bMERGE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
]

_ALLOWED_TABLES_SET = set(_ALLOWED_TABLES)


def _format_table_overview(context: Dict[str, TableContext]) -> str:
    lines = []
    for table in _ALLOWED_TABLES:
        ctx = context.get(table)
        if ctx is None:
            lines.append(f"- {table} (no additional context)")
            continue
        column_preview = ", ".join(ctx.columns[:6]) if ctx.columns else ""
        if ctx.description and column_preview:
            lines.append(f"- {table}: {ctx.description} | key columns: {column_preview}")
        elif ctx.description:
            lines.append(f"- {table}: {ctx.description}")
        elif column_preview:
            lines.append(f"- {table}: key columns: {column_preview}")
        else:
            lines.append(f"- {table}")
    return "\n".join(lines)


_TABLE_OVERVIEW_TEXT = _format_table_overview(_TABLE_CONTEXT)

_GENERATE_QUERY_PROMPT = (
    "You are an agent that specialises in analysing a {dialect} database.\n"
    "You can only query the following tables:\n"
    "{table_overview}\n\n"
    "Given a user request, craft a syntactically correct SQL query.\n"
    "Do not mutate data (no INSERT/UPDATE/DELETE/etc.) and prefer readable SQL."
).format(dialect=_DIALECT, table_overview=_TABLE_OVERVIEW_TEXT)

_CHECK_QUERY_PROMPT = (
    "You are a meticulous SQL reviewer for a {dialect} database.\n"
    "Verify that the query is safe, only references the authorised tables, and avoids common mistakes such as incorrect joins, unsafe NULL handling, data type mismatches, or misuse of DISTINCT/UNION.\n"
    "If corrections are required, return a revised query. Otherwise, reproduce the original query verbatim."
).format(dialect=_DIALECT)


def list_tables(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Provide the model with the curated list of tables and context."""

    logger.info("Emitting curated table overview")
    overview_message = AIMessage(content=f"Available tables:\n{_TABLE_OVERVIEW_TEXT}")
    return {
        "messages": [overview_message],
        "metadata": {"tables_listed": True},
        "table_context": {table: ctx.__dict__ for table, ctx in _TABLE_CONTEXT.items()},
    }


def call_get_schema(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Allow the LLM to request schema details for specific tables."""

    messages = ensure_messages(state)
    llm_with_tools = _LLM.bind_tools([_GET_SCHEMA_TOOL], tool_choice="any")
    logger.info("Requesting schema information via LLM tool call")
    response = llm_with_tools.invoke(messages, config=config)

    tool_messages = []
    if getattr(response, "tool_calls", None):
        temp_state = cast(SQLAgentState, dict(state))
        temp_state["messages"] = messages + [response]
        execution = _execute_tool_messages(
            temp_state,
            _GET_SCHEMA_TOOL.name,
            config,
            allowed_tools=[_GET_SCHEMA_TOOL.name],
        )
        tool_messages = execution.get("messages", [])

    return {"messages": [response] + tool_messages}


def generate_query(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Generate a SQL query for the user's task."""

    messages = ensure_messages(state)
    system_message = {"role": "system", "content": _GENERATE_QUERY_PROMPT}
    llm_with_tools = _LLM.bind_tools([_RUN_QUERY_TOOL])
    logger.info("Generating SQL query from context")
    response = llm_with_tools.invoke([system_message] + messages, config=config)

    query = None
    if getattr(response, "tool_calls", None):
        try:
            query = response.tool_calls[0]["args"].get("query")
        except (IndexError, KeyError, AttributeError):
            query = None

    payload: Dict[str, object] = {"messages": [response]}
    if query:
        logger.debug("Generated SQL query", extra={"query": query})
        is_safe, reason = _is_safe_query(query)
        if not is_safe:
            logger.warning("Blocked unsafe SQL query", extra={"query": query, "reason": reason})
            warning_message = AIMessage(
                content=(
                    "The generated SQL was blocked because it attempted a potentially destructive or unauthorised "
                    f"operation ({reason}). Please rephrase your request to perform a read-only analysis."
                )
            )
            return {"messages": [warning_message], "error": reason}
        payload["last_query"] = query
    return payload


def check_query(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Ask the model to double-check the generated SQL before execution."""

    messages = ensure_messages(state)
    if not messages:
        raise ConfigurationError("No messages available for query checking.")

    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        raise ConfigurationError("Expected a tool call with a SQL query to validate.")

    tool_call = last_message.tool_calls[0]
    query = tool_call.get("args", {}).get("query")
    if not query:
        raise ConfigurationError("Query payload missing from tool call arguments.")

    system_message = {"role": "system", "content": _CHECK_QUERY_PROMPT}
    user_message = {"role": "user", "content": query}

    llm_with_tools = _LLM.bind_tools([_RUN_QUERY_TOOL], tool_choice="any")
    logger.info("Running LLM-based SQL quality check")
    response = llm_with_tools.invoke([system_message, user_message], config=config)

    response.id = last_message.id

    checked_query = None
    if getattr(response, "tool_calls", None):
        try:
            checked_query = response.tool_calls[0]["args"].get("query")
        except (IndexError, KeyError, AttributeError):
            checked_query = None

    payload: Dict[str, object] = {"messages": [response]}
    if checked_query:
        logger.debug("Checked SQL query", extra={"query": checked_query})
        is_safe, reason = _is_safe_query(checked_query)
        if not is_safe:
            logger.warning(
                "Blocked unsafe SQL query during validation",
                extra={"query": checked_query, "reason": reason},
            )
            warning_message = AIMessage(
                content=(
                    "The proposed SQL query was rejected because it includes a potentially destructive or "
                    f"unauthorised operation ({reason}). Only read-only SELECT statements against the approved tables are permitted."
                )
            )
            return {"messages": [warning_message], "error": reason}
        payload["last_query"] = checked_query
    return payload


def _execute_tool_messages(
    state: SQLAgentState,
    tool_name: str,
    config: Optional[RunnableConfig] = None,
    *,
    allowed_tools: Iterable[str] | None = None,
):
    messages = ensure_messages(state)
    if not messages:
        raise ConfigurationError("No messages available for tool execution.")

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        raise ConfigurationError("Expected tool call instructions from the LLM.")

    rendered_messages = []
    extra: Dict[str, object] = {}
    for tool_call in tool_calls:
        name = tool_call.get("name")
        if allowed_tools is not None and name not in set(allowed_tools):
            logger.debug("Ignoring tool call", extra={"tool_call": name})
            continue

        tool = get_tool(name)
        if name == _RUN_QUERY_TOOL.name:
            sql_text = tool_call.get("args", {}).get("query")
            is_safe, reason = _is_safe_query(sql_text)
            if not is_safe:
                logger.warning("Blocked unsafe SQL execution", extra={"query": sql_text, "reason": reason})
                raise ConfigurationError(
                    f"Unsafe SQL detected ({reason}). Only read-only SELECT queries over approved tables are allowed."
                )
        try:
            result = tool.invoke(tool_call.get("args", {}), config=config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Tool execution failed", extra={"tool": name})
            result = f"Error executing {name}: {exc}"

        structured_result = None
        if isinstance(result, ToolMessage):
            rendered = result
        else:
            if not isinstance(result, str):
                try:
                    content = json.dumps(result, default=str)
                except TypeError:
                    content = str(result)
            else:
                content = result

            if name == _RUN_QUERY_TOOL.name:
                try:
                    structured_result = json.loads(content)
                except json.JSONDecodeError:
                    structured_result = None

            rendered = ToolMessage(
                content=content,
                tool_call_id=tool_call.get("id", ""),
                name=name,
            )

        if structured_result is not None:
            extra["last_query_result"] = structured_result

        rendered_messages.append(rendered)

    if not rendered_messages:
        raise ConfigurationError(f"No matching tool calls for tool '{tool_name}'.")

    extra["messages"] = rendered_messages
    return extra


def get_schema(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    logger.info("Executing schema retrieval tool")
    messages = ensure_messages(state)
    if not messages or not getattr(messages[-1], "tool_calls", None):
        logger.debug("No schema tool calls to execute; skipping")
        return {"messages": []}

    return _execute_tool_messages(
        state,
        _GET_SCHEMA_TOOL.name,
        config,
        allowed_tools=[_GET_SCHEMA_TOOL.name],
    )


def run_query(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    logger.info("Executing SQL query tool")
    execution = _execute_tool_messages(
        state,
        _RUN_QUERY_TOOL.name,
        config,
        allowed_tools=[_RUN_QUERY_TOOL.name],
    )
    payload: Dict[str, object] = {"messages": execution.get("messages", [])}
    if "last_query_result" in execution:
        payload["last_query_result"] = execution["last_query_result"]
    return payload


def summarize_result(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Summarise the query result into a final AI message."""

    result = state.get("last_query_result")
    query = state.get("last_query")

    if not result:
        message = AIMessage(content="No SQL query was executed; unable to provide an answer.")
        return {"messages": [message], "final_answer": message.content}

    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        message = AIMessage(content=f"The SQL query failed to execute: {error}")
        return {"messages": [message], "final_answer": message.content}

    rows = result.get("data", [])
    columns = result.get("columns", [])
    row_count = result.get("row_count", len(rows))

    if not rows:
        message = AIMessage(content="The query executed successfully but returned no rows for the requested timeframe.")
        return {"messages": [message], "final_answer": message.content}

    preview_rows = rows[:5]
    preview_lines = _format_preview(preview_rows, columns)
    summary = [
        "Query executed successfully.",
        f"Rows returned: {row_count}.",
    ]
    if query:
        summary.append(f"SQL used: {query}")
    summary.append("Preview:\n" + preview_lines)

    message = AIMessage(content="\n".join(summary))
    return {"messages": [message], "final_answer": message.content}


def _format_preview(rows, columns) -> str:
    header = " | ".join(columns) if columns else ""
    lines = [header] if header else []
    for row in rows:
        if isinstance(row, dict):
            values = [str(row.get(col, "")) for col in columns]
        else:
            values = [str(value) for value in row]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def _extract_table_names(query: str) -> Iterable[str]:
    pattern = re.compile(r"(?:FROM|JOIN)\s+([\w\.]+)", re.IGNORECASE)
    for match in pattern.findall(query):
        table = match.strip().strip(",")
        yield table if "." in table else f"public.{table}"


def _is_safe_query(query: Optional[str]) -> Tuple[bool, str]:
    if not query:
        return False, "empty query"

    normalized = query.strip()
    if not normalized.upper().startswith("SELECT"):
        return False, "non-SELECT statement"

    upper_query = normalized.upper()
    for pattern in _FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, upper_query):
            return False, f"contains forbidden keyword matching '{pattern}'"

    non_whitelisted = [
        table for table in _extract_table_names(normalized)
        if table not in _ALLOWED_TABLES_SET
    ]
    if non_whitelisted:
        return False, "references disallowed tables: " + ", ".join(sorted(set(non_whitelisted)))

    return True, ""


__all__ = [
    "call_get_schema",
    "check_query",
    "generate_query",
    "get_schema",
    "list_tables",
    "run_query",
    "summarize_result",
]


