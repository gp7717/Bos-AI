"""Runtime configuration and resource initialisation for the SQL agent."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.settings import load_settings

from .mcp_client import LangchainMCPDatabaseClient

logger = logging.getLogger(__name__)

# Ensure .env variables are available even when running the package standalone.
_ENV_PATHS = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
]
for env_path in _ENV_PATHS:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


try:  # pragma: no cover - optional dependency on the main application
    from Assistant.app.config.settings import settings as app_settings  # type: ignore
except Exception:  # pylint: disable=broad-except
    app_settings = None


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration for the agent is missing."""


@dataclass(frozen=True)
class TableContext:
    """Descriptive context for a database table."""

    table: str
    description: str
    columns: List[str]


@dataclass(frozen=True)
class SQLAgentResources:
    """Container for shared SQL agent resources."""

    llm: AzureChatOpenAI
    mcp_client: LangchainMCPDatabaseClient
    tools: Dict[str, BaseTool]
    allowed_tables: List[str]
    table_context: Dict[str, TableContext]
    dialect: str


def _resolve_azure_config() -> Dict[str, str]:
    """Gather Azure OpenAI configuration parameters."""

    settings = load_settings()
    azure_settings = settings.azure_openai

    endpoint = (
        (str(azure_settings.endpoint) if azure_settings else None)
        or os.getenv("QUERY_AGENT_AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or (getattr(app_settings, "azure_openai_endpoint", None) if app_settings else None)
    )
    api_key = (
        (azure_settings.api_key if azure_settings else None)
        or os.getenv("QUERY_AGENT_AZURE_OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or (getattr(app_settings, "azure_openai_api_key", None) if app_settings else None)
    )
    deployment = (
        (azure_settings.deployment if azure_settings else None)
        or os.getenv("QUERY_AGENT_AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or (getattr(app_settings, "azure_openai_deployment_name", None) if app_settings else None)
    )
    api_version = (
        (azure_settings.api_version if azure_settings else None)
        or os.getenv("QUERY_AGENT_AZURE_OPENAI_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or (getattr(app_settings, "azure_openai_api_version", None) if app_settings else None)
    )

    missing = {
        "QUERY_AGENT_AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_ENDPOINT": endpoint,
        "QUERY_AGENT_AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_KEY": api_key,
        "QUERY_AGENT_AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_DEPLOYMENT_NAME": deployment,
        "QUERY_AGENT_AZURE_OPENAI_API_VERSION / AZURE_OPENAI_API_VERSION": api_version,
    }

    missing_keys = [key for key, value in missing.items() if not value]
    if missing_keys:
        raise ConfigurationError(
            "Azure OpenAI configuration incomplete. Missing: " + ", ".join(missing_keys)
        )

    logger.debug(
        "Azure OpenAI configuration resolved",
        extra={"endpoint": endpoint, "deployment": deployment},
    )

    return {
        "azure_endpoint": endpoint,
        "api_key": api_key,
        "azure_deployment": deployment,
        "api_version": api_version,
    }


def _run_sync(coro):
    """Execute an async coroutine synchronously."""

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():  # pragma: no cover - unexpected when running via CLI
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    return loop.run_until_complete(coro)


def _load_table_contexts(
    client: LangchainMCPDatabaseClient,
) -> Tuple[List[str], Dict[str, TableContext]]:
    """Load allowed tables and contextual metadata."""

    context_path_env = os.getenv("QUERY_AGENT_TABLE_CONTEXT_FILE")
    default_context_path = Path(__file__).resolve().parent / "table_context.yaml"
    context_path = Path(context_path_env) if context_path_env else default_context_path

    settings = load_settings()

    allowed_env = os.getenv("QUERY_AGENT_ALLOWED_TABLES")
    allowed_from_env = [
        table.strip()
        for table in (allowed_env.split(",") if allowed_env else [])
        if table.strip()
    ]
    allowed_from_settings = [table.strip() for table in settings.allowed_tables if table.strip()]

    table_context: Dict[str, TableContext] = {}

    if context_path.exists():
        logger.debug("Loading table context from %s", context_path)
        with context_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
        tables_section = payload.get("tables", {})
        for table_name, info in tables_section.items():
            description = info.get("description", "")
            columns = info.get("columns", []) or []
            table_context[table_name] = TableContext(
                table=table_name,
                description=description,
                columns=columns,
            )

    allowed_tables = allowed_from_env or allowed_from_settings or list(table_context.keys())

    if not allowed_tables:
        logger.debug("Discovering tables via MCP client")
        tables_response = _run_sync(client.list_tables())
        if tables_response.get("success"):
            allowed_tables = [t["full_name"] for t in tables_response.get("tables", [])]
        else:
            raise ConfigurationError(
                "Failed to discover tables from MCP server and no explicit table list provided"
            )

    normalised_allowed: List[str] = []
    default_schema = settings.default_schema or "public"
    for table in allowed_tables:
        if "." in table:
            normalised_allowed.append(table)
        else:
            normalised_allowed.append(f"{default_schema}.{table}")

    missing_context_tables = [
        table for table in normalised_allowed if table not in table_context
    ]
    if missing_context_tables:
        schema_info = _run_sync(client.get_schema_for_tables(missing_context_tables))
        for info in schema_info:
            full_name = info.get("full_name") or f"{info.get('schema')}.{info.get('table')}"
            columns = [col.get("name") for col in info.get("columns", [])]
            table_context[full_name] = TableContext(
                table=full_name,
                description="",
                columns=columns,
            )

    return normalised_allowed, table_context


def _build_tools(
    client: LangchainMCPDatabaseClient,
    allowed_tables: List[str],
    table_context: Dict[str, TableContext],
) -> Dict[str, BaseTool]:
    allowed_set = set(allowed_tables)

    class SchemaArgs(BaseModel):
        table_names: List[str] = Field(
            description="List of tables to inspect. Tables must be schema-qualified (schema.table)."
        )

    class QueryArgs(BaseModel):
        query: str = Field(description="SQL SELECT query to execute")

    @tool("mcp_db_schema", args_schema=SchemaArgs)
    def mcp_db_schema(table_names: List[str]) -> str:  # type: ignore[override]
        """Return schema metadata and contextual notes for the requested tables."""
        filtered = []
        for table in table_names:
            formatted = table if "." in table else f"public.{table}"
            if formatted not in allowed_set:
                raise ConfigurationError(
                    f"Table '{formatted}' is not permitted for this agent."
                )
            filtered.append(formatted)

        schemas = _run_sync(client.get_schema_for_tables(filtered))

        chunks: List[str] = []
        for schema in schemas:
            full_name = schema.get("full_name") or f"{schema.get('schema')}.{schema.get('table')}"
            columns = schema.get("columns", [])
            column_lines = [
                f"- {col.get('name')} ({col.get('type')})" for col in columns
            ]
            description = table_context.get(full_name, TableContext(full_name, "", [])).description
            if description:
                chunks.append(
                    f"Table: {full_name}\nDescription: {description}\nColumns:\n" + "\n".join(column_lines)
                )
            else:
                chunks.append(
                    f"Table: {full_name}\nColumns:\n" + "\n".join(column_lines)
                )

        return "\n\n".join(chunks) if chunks else "No schema information available."

    @tool("mcp_db_query", args_schema=QueryArgs)
    def mcp_db_query(query: str) -> str:  # type: ignore[override]
        """Execute a read-only SQL query against the configured database."""
        result = _run_sync(client.execute_query(query, params=None))
        return json.dumps(result, default=str)

    return {
        "mcp_db_schema": mcp_db_schema,
        "mcp_db_query": mcp_db_query,
    }


@lru_cache(maxsize=1)
def get_resources() -> SQLAgentResources:
    """Create or return cached SQL agent resources."""

    azure_cfg = _resolve_azure_config()

    logger.info("Initialising SQL agent resources")

    temperature_str = os.getenv("QUERY_AGENT_AZURE_OPENAI_TEMPERATURE")
    if temperature_str:
        try:
            temperature = float(temperature_str)
        except ValueError:
            logger.warning(
                "Invalid QUERY_AGENT_AZURE_OPENAI_TEMPERATURE '%s'; defaulting to 1.0",
                temperature_str,
            )
            temperature = 1.0
    else:
        temperature = 1.0

    llm = AzureChatOpenAI(
        **azure_cfg,
        temperature=temperature,
    )
    mcp_client = _resolve_mcp_client()

    allowed_tables, table_context = _load_table_contexts(mcp_client)
    tools = _build_tools(mcp_client, allowed_tables, table_context)

    settings = load_settings()
    dialect = (
        os.getenv("QUERY_AGENT_SQL_DIALECT")
        or settings.sql_dialect
    )
    if not dialect and app_settings is not None:
        dialect = getattr(app_settings, "database_dialect", None)
    dialect = dialect or "PostgreSQL"

    logger.info(
        "SQL agent resources ready",
        extra={"tool_names": sorted(tools), "allowed_tables": allowed_tables},
    )

    return SQLAgentResources(
        llm=llm,
        mcp_client=mcp_client,
        tools=tools,
        allowed_tables=allowed_tables,
        table_context=table_context,
        dialect=dialect,
    )


def get_tool(name: str) -> BaseTool:
    """Return a toolkit tool by name, raising :class:`ConfigurationError` if missing."""

    resources = get_resources()
    tool = resources.tools.get(name)
    if tool is None:
        raise ConfigurationError(f"Tool '{name}' is not available in the SQL agent configuration")
    logger.debug("Accessed toolkit tool", extra={"tool": name})
    return tool


__all__ = [
    "ConfigurationError",
    "TableContext",
    "SQLAgentResources",
    "get_resources",
    "get_tool",
]


def _resolve_database_url() -> str:
    settings = load_settings()
    candidates = [
        os.getenv("QUERY_AGENT_DATABASE_URL"),
        os.getenv("DATABASE_URL"),
        settings.database_url,
    ]

    if app_settings is not None:
        candidates.append(getattr(app_settings, "database_url", None))

    for value in candidates:
        if value:
            return value

    raise ConfigurationError(
        "Database URL not configured. Set QUERY_AGENT_DATABASE_URL or DATABASE_URL, "
        "or ensure a settings.database_url is available."
    )


def _resolve_mcp_client() -> LangchainMCPDatabaseClient:
    database_url = _resolve_database_url()

    module_path = os.getenv("QUERY_AGENT_MCP_CLIENT_MODULE")
    if module_path:
        factory_name = os.getenv("QUERY_AGENT_MCP_CLIENT_FACTORY", "get_mcp_client")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ConfigurationError(
                f"Failed to import module '{module_path}' for MCP client: {exc}"
            ) from exc

        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise ConfigurationError(
                f"Factory '{factory_name}' not found or not callable in '{module_path}'."
            )

        client = factory()
        if not isinstance(client, LangchainMCPDatabaseClient):
            raise ConfigurationError(
                "Custom MCP client factory must return a LangchainMCPDatabaseClient instance."
            )
        return client

    return LangchainMCPDatabaseClient(database_url)


