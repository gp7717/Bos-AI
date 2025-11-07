"""LangChain-backed MCP database client for the QueryAgent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "MERGE",
    "GRANT",
    "REVOKE",
}


class LangchainMCPDatabaseClient:
    """Thin wrapper that provides MCP-style async methods using LangChain primitives."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.database = SQLDatabase.from_uri(database_url)
        self.engine: Engine = self.database._engine  # type: ignore[attr-defined]
        logger.info("Langchain MCP client initialised", extra={"dialect": self.database.dialect})

    async def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            return self._error("Only SELECT statements are allowed.")

        for keyword in _FORBIDDEN_KEYWORDS:
            if keyword in sql_upper:
                return self._error(f"Query contains forbidden keyword: {keyword}")

        def _run_query():
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = list(result.keys())
            return rows, columns

        rows, columns = await asyncio.to_thread(_run_query)
        data = [dict(zip(columns, row)) for row in rows]
        return {
            "success": True,
            "data": data,
            "row_count": len(data),
            "columns": columns,
        }

    async def list_tables(self, schema: Optional[str] = None) -> Dict[str, Any]:
        inspector = inspect(self.engine)
        schemas = [schema] if schema else [self.database._schema or "public"]
        tables: List[Dict[str, str]] = []
        for schema_name in schemas:
            for table in inspector.get_table_names(schema=schema_name):
                tables.append(
                    {
                        "schema": schema_name,
                        "table": table,
                        "full_name": f"{schema_name}.{table}",
                    }
                )
        return {"success": True, "tables": tables, "count": len(tables)}

    async def get_schema_for_tables(self, table_names: List[str]) -> List[Dict[str, Any]]:
        inspector = inspect(self.engine)
        results: List[Dict[str, Any]] = []
        for full_name in table_names:
            schema_name, table = full_name.split(".", 1) if "." in full_name else ("public", full_name)
            columns = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                }
                for col in inspector.get_columns(table, schema=schema_name)
            ]
            results.append(
                {
                    "schema": schema_name,
                    "table": table,
                    "full_name": f"{schema_name}.{table}",
                    "columns": columns,
                }
            )
        return results

    async def get_table_schema(self, schema: str, table: str) -> Dict[str, Any]:
        result = await self.get_schema_for_tables([f"{schema}.{table}"])
        return {"success": True, "schema": result[0] if result else None}

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "data": [],
            "row_count": 0,
            "columns": [],
        }


__all__ = ["LangchainMCPDatabaseClient"]

