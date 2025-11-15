"""Multi-source executor for handling queries across multiple data sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from Agents.core.models import AgentError, AgentExecutionStatus, AgentResult

logger = logging.getLogger(__name__)


class MultiSourceExecutor:
    """Handles queries across multiple data sources (databases, APIs, etc.)."""

    def __init__(self):
        """Initialize multi-source executor."""
        self.sources: Dict[str, Any] = {}  # source_name -> client/connection

    def register_source(self, name: str, client: Any) -> None:
        """
        Register a data source.

        Args:
            name: Source identifier (e.g., "database_a", "api_backend")
            client: Client/connection object for this source
        """
        self.sources[name] = client
        logger.info(f"Registered data source: {name}")

    async def query_multiple_sources(
        self,
        queries: Dict[str, str],  # source_name -> query
        context: Dict[str, Any],
        query_executor: Any,  # Function to execute query: (source, query, context) -> AgentResult
    ) -> Dict[str, AgentResult]:
        """
        Execute queries across multiple sources in parallel.

        Args:
            queries: Dictionary mapping source names to queries
            context: Shared context
            query_executor: Function to execute query for a source

        Returns:
            Dictionary mapping source names to AgentResult objects
        """
        tasks = {}
        for source_name, query in queries.items():
            if source_name not in self.sources:
                logger.warning(f"Source '{source_name}' not registered, skipping")
                continue

            client = self.sources[source_name]
            # Create task for this source
            tasks[source_name] = self._execute_source_query(
                source_name, client, query, context, query_executor
            )

        # Execute all in parallel
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            return {
                source: (
                    result
                    if not isinstance(result, Exception)
                    else AgentResult(
                        agent="sql",
                        status=AgentExecutionStatus.failed,
                        error=AgentError(
                            message=str(result), type=type(result).__name__
                        ),
                    )
                )
                for source, result in zip(queries.keys(), results)
            }

        return {}

    async def _execute_source_query(
        self,
        source_name: str,
        client: Any,
        query: str,
        context: Dict[str, Any],
        query_executor: Any,
    ) -> AgentResult:
        """Execute a query for a specific source."""
        loop = asyncio.get_event_loop()

        # Prepare source-specific context
        source_context = dict(context)
        source_context["source_name"] = source_name
        source_context["source_client"] = client

        try:
            # Run query executor in thread pool
            result = await loop.run_in_executor(
                None, query_executor, source_name, query, source_context
            )
            return result
        except Exception as exc:
            logger.error(f"Source '{source_name}' query failed: {exc}", exc_info=exc)
            return AgentResult(
                agent="sql",
                status=AgentExecutionStatus.failed,
                error=AgentError(message=str(exc), type=type(exc).__name__),
            )

    def list_sources(self) -> List[str]:
        """Get list of registered source names."""
        return list(self.sources.keys())

    def get_source(self, name: str) -> Optional[Any]:
        """Get client for a specific source."""
        return self.sources.get(name)


__all__ = ["MultiSourceExecutor"]

