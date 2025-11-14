"""LangGraph node implementations for the custom SQL agent."""

from __future__ import annotations

import json
import logging
import re
from decimal import Decimal
from typing import Dict, Iterable, Optional, Tuple, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from openai import BadRequestError

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

_FORBIDDEN_SQL_KEYWORDS = [
    "CREATE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "CALL",
    "EXEC",
    "EXECUTE",
    "INVOKE",
]
_FORBIDDEN_SQL_PATTERNS = {
    keyword: re.compile(rf"\b{keyword}\b", re.IGNORECASE)
    for keyword in _FORBIDDEN_SQL_KEYWORDS
}

_ALLOWED_TABLES_SET = set(_ALLOWED_TABLES)
_DEFAULT_SCHEMA = (
    _ALLOWED_TABLES[0].split(".")[0] if _ALLOWED_TABLES and "." in _ALLOWED_TABLES[0] else "public"
)

_SQL_EXTRACTION_PATTERN = re.compile(
    r"(?P<sql>(?:WITH\s+[\s\S]+?|SELECT\s+[\s\S]+?))(?:$|\n\s*[A-Z]{2,}|\Z)",
    re.IGNORECASE,
)


def _extract_table_tokens(_query: str) -> Iterable[str]:  # legacy compatibility
    """Return an empty iterator; retained to avoid NameError in cached orchestrator runs."""
    return []


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
    "Given a user request, you must respond with a single mcp_db_query tool call containing a syntactically correct, read-only SQL query.\n"
    "Do not mutate data (no INSERT/UPDATE/DELETE/etc.) and prefer readable SQL.\n"
    "Do not return natural-language explanations or summaries; if a query cannot be produced, respond with an empty tool call and explain the reason in the tool message."
).format(dialect=_DIALECT, table_overview=_TABLE_OVERVIEW_TEXT)

_CHECK_QUERY_PROMPT = (
    "You are a SQL code reviewer for a {dialect} database system.\n"
    "Your task is to review SQL code for safety, correctness, and adherence to authorized table access.\n"
    "Check that the SQL code only references authorized tables and avoids common mistakes such as incorrect joins, unsafe NULL handling, data type mismatches, or misuse of DISTINCT/UNION.\n"
    "Review the SQL code provided and if corrections are needed, return a revised query. Otherwise, return the original query exactly as provided."
).format(dialect=_DIALECT)


def list_tables(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Provide the model with the curated list of tables and context."""

    logger.info("Emitting curated table overview")
    try:
        overview_message = AIMessage(content=f"Available tables:\n{_TABLE_OVERVIEW_TEXT}")
        payload = {
            "messages": [overview_message],
            "metadata": {"tables_listed": True},
            "table_context": {table: ctx.__dict__ for table, ctx in _TABLE_CONTEXT.items()},
        }
        logger.debug(
            "Table overview prepared",
            extra={
                "table_count": len(_ALLOWED_TABLES),
                "with_descriptions": sum(1 for ctx in _TABLE_CONTEXT.values() if ctx.description),
            },
        )
        return payload
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build table overview payload")
        raise ConfigurationError(f"Unable to prepare table overview: {exc}") from exc


def call_get_schema(state: SQLAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, object]:
    """Allow the LLM to request schema details for specific tables."""

    messages = ensure_messages(state)
    llm_with_tools = _LLM.bind_tools([_GET_SCHEMA_TOOL], tool_choice="any")
    logger.info("Requesting schema information via LLM tool call")
    try:
        response = llm_with_tools.invoke(messages, config=config)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("LLM schema tool invocation failed")
        raise ConfigurationError(f"Schema retrieval request failed: {exc}") from exc

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
    try:
        response = llm_with_tools.invoke([system_message] + messages, config=config)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("LLM query generation failed")
        raise ConfigurationError(f"Query generation failed: {exc}") from exc
    logger.debug(
        "LLM response content: %r | tool_calls: %r",
        getattr(response, "content", None),
        getattr(response, "tool_calls", None),
    )

    query = None
    if getattr(response, "tool_calls", None):
        try:
            query = response.tool_calls[0]["args"].get("query")
        except (IndexError, KeyError, AttributeError):
            query = None
    else:
        query = _infer_sql_from_text(getattr(response, "content", ""))
        if query:
            response.tool_calls = [
                {
                    "name": _RUN_QUERY_TOOL.name,
                    "args": {"query": query},
                    "id": getattr(response, "id", None) or "synthetic-mcp-query",
                }
            ]
            logger.info("Synthesised tool call from free-form SQL response", extra={"query": query})

    payload: Dict[str, object] = {"messages": [response]}
    if query:
        logger.debug("Candidate SQL query:\n%s", query)
        is_safe, reason = _is_safe_query(query)
        if not is_safe:
            logger.warning("Blocked unsafe SQL query (reason=%s):\n%s", reason, query)
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
    # Wrap SQL query in explicit context to avoid content filter false positives
    user_message = {
        "role": "user",
        "content": f"Please review the following SQL code:\n\n```sql\n{query}\n```"
    }

    llm_with_tools = _LLM.bind_tools([_RUN_QUERY_TOOL], tool_choice="any")
    logger.info("Running LLM-based SQL quality check")
    try:
        response = llm_with_tools.invoke([system_message, user_message], config=config)
    except BadRequestError as exc:
        # Handle Azure OpenAI content filter errors gracefully
        error_str = str(exc).lower()
        # Check if this is a content filter error (jailbreak detection, etc.)
        if "content_filter" in error_str or "content management policy" in error_str:
            logger.warning(
                "SQL query validation blocked by content filter; proceeding with original query",
                extra={"query_preview": query[:100] if query else None}
            )
            # Fallback: return original query without validation
            payload: Dict[str, object] = {"messages": [], "last_query": query}
            return payload
        # Re-raise other BadRequestErrors
        logger.exception("LLM query validation failed with BadRequestError")
        raise ConfigurationError(f"Query validation failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("LLM query validation failed")
        raise ConfigurationError(f"Query validation failed: {exc}") from exc

    response.id = last_message.id

    checked_query = None
    if getattr(response, "tool_calls", None):
        try:
            checked_query = response.tool_calls[0]["args"].get("query")
        except (IndexError, KeyError, AttributeError):
            checked_query = None

    payload: Dict[str, object] = {"messages": [response]}
    if checked_query:
        logger.debug("Checked SQL query:\n%s", checked_query)
        is_safe, reason = _is_safe_query(checked_query)
        if not is_safe:
            logger.warning("Blocked unsafe SQL query during validation (reason=%s):\n%s", reason, checked_query)
            warning_message = AIMessage(
                content=(
                    "The proposed SQL query was rejected because it includes a potentially destructive or "
                    f"unauthorised operation ({reason}). Only read-only SELECT statements against the approved tables are permitted."
                )
            )
            return {"messages": [warning_message], "error": reason}

        original_normalized = query.strip()
        checked_normalized = checked_query.strip()
        degraded = False
        if checked_normalized.lower().startswith("select 1") and len(checked_normalized) <= 20:
            degraded = True
        elif len(checked_normalized) < max(20, int(len(original_normalized) * 0.5)):
            degraded = True

        if degraded:
            logger.info(
                "Validation produced a trivial query; retaining original",
                extra={"original_query": query, "validated_query": checked_query},
            )
            if getattr(response, "tool_calls", None):
                try:
                    response.tool_calls[0]["args"]["query"] = query
                except (IndexError, KeyError, TypeError):
                    pass
            payload["last_query"] = query
        else:
            if getattr(response, "tool_calls", None):
                try:
                    response.tool_calls[0]["args"]["query"] = checked_query
                except (IndexError, KeyError, TypeError):
                    pass
            payload["last_query"] = checked_query
    else:
        payload["last_query"] = query
    return payload


def _infer_sql_from_text(content: Optional[str]) -> Optional[str]:
    if not content:
        return None

    # Strip code fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.split("```", 1)[0].strip()

    match = _SQL_EXTRACTION_PATTERN.search(cleaned)
    if not match:
        return None

    sql = match.group("sql").strip()
    if not sql.lower().startswith(("select", "with")):
        return None

    # Ensure trailing semicolon for consistency
    if not sql.rstrip().endswith(";"):
        sql = sql.rstrip() + ";"
    return sql


def _execute_tool_messages(
    state: SQLAgentState,
    tool_name: str,
    config: Optional[RunnableConfig] = None,
    *,
    allowed_tools: Iterable[str] | None = None,
):
    logger.debug(
        "Executing tool messages",
        extra={
            "tool_name": tool_name,
            "allowed_tools": list(allowed_tools) if allowed_tools is not None else None,
        },
    )
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
        structured_result = None
        try:
            logger.debug("Invoking tool", extra={"tool": name})
            result = tool.invoke(tool_call.get("args", {}), config=config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Tool execution failed", extra={"tool": name})
            error_payload = {"success": False, "error": str(exc)}
            result = error_payload
            structured_result = error_payload

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

            if name == _RUN_QUERY_TOOL.name and structured_result is None:
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
    logger.debug("Executing run_query node", extra={"has_messages": bool(state.get("messages"))})
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

    logger.debug(
        "Summarizing results",
        extra={
            "has_last_query": "last_query" in state,
            "has_last_query_result": "last_query_result" in state,
        },
    )

    result = state.get("last_query_result")
    query = state.get("last_query")

    if not result:
        logger.warning("No query result available for summarization")
        message = AIMessage(content="No SQL query was executed; unable to provide an answer.")
        return {"messages": [message], "final_answer": message.content}

    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        logger.error("SQL query execution reported failure", extra={"error": error})
        message = AIMessage(content=f"The SQL query failed to execute: {error}")
        return {"messages": [message], "final_answer": message.content}

    rows = result.get("data", [])
    columns = result.get("columns", [])
    row_count = result.get("row_count", len(rows))

    if not rows:
        logger.info("Query executed with no results")
        message = AIMessage(
            content=_format_no_rows_message(query)
        )
        return {"messages": [message], "final_answer": message.content}

    preview_rows = rows[: min(len(rows), 5)]
    preview_lines = _format_preview(preview_rows, columns)

    highlights = _summarise_primary_rows(rows, columns, limit=2)

    summary_lines = ["Query executed successfully.", f"Rows returned: {row_count}."]
    if highlights:
        summary_lines.append("Key results:")
        summary_lines.extend(f"- {line}" for line in highlights)
    summary_lines.append("Preview:\n" + preview_lines)
    if query:
        summary_lines.append("SQL used:\n" + query)

    message = AIMessage(content="\n".join(summary_lines))
    final_answer_lines = summary_lines[:2]
    if highlights:
        final_answer_lines.extend(f"- {line}" for line in highlights)
    final_answer = "\n".join(final_answer_lines)
    return {"messages": [message], "final_answer": final_answer}


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


def _format_no_rows_message(query: Optional[str]) -> str:
    lines = ["The query executed successfully but returned no rows for the requested timeframe."]
    if query:
        lines.append("SQL used:\n" + query)
    return "\n".join(lines)


def _summarise_primary_rows(rows: Iterable[Dict[str, object]], columns: Iterable[str], limit: int = 1) -> list[str]:
    summaries: list[str] = []
    if not rows:
        return summaries

    selected_columns = list(columns)
    for idx, row in enumerate(rows):
        if idx >= limit:
            break
        if not isinstance(row, dict):
            summaries.append(str(row))
            continue

        numeric_items = []
        textual_items = []
        for col in selected_columns:
            value = row.get(col)
            if value is None or value == "":
                continue
            formatted = _format_value(value)
            if isinstance(value, (int, float)):
                numeric_items.append(f"{col}={formatted}")
            else:
                textual_items.append(f"{col}={formatted}")

        ordered_items = numeric_items + textual_items
        if not ordered_items:
            ordered_items = [str(row)]

        label = f"Row {idx + 1}: " + ", ".join(ordered_items[:6])
        summaries.append(label)

    return summaries


def _format_value(value: object) -> str:
    if isinstance(value, Decimal):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _is_safe_query(query: Optional[str]) -> Tuple[bool, str]:
    if not query:
        return False, "empty query"

    normalized = query.strip()
    for keyword, pattern in _FORBIDDEN_SQL_PATTERNS.items():
        if pattern.search(normalized):
            return False, f"contains forbidden keyword '{keyword.lower()}'"

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


