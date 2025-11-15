"""Parallel execution engine for independent agents and queries."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from Agents.core.models import AgentError, AgentExecutionStatus, AgentName, AgentResult

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Executes independent agents/queries in parallel."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute_agents_parallel(
        self,
        agents: List[AgentName],
        question: str,
        context: Dict[str, Any],
        dependencies: Optional[Dict[str, List[str]]] = None,
        agent_invokers: Optional[Dict[AgentName, Any]] = None,
    ) -> List[AgentResult]:
        """
        Execute agents in parallel where possible, respecting dependencies.

        Args:
            agents: List of agents to execute
            question: Original question (may be modified per agent)
            context: Shared context
            dependencies: Dict mapping agent -> list of agents it depends on
            agent_invokers: Dict mapping agent name to invoker function

        Returns:
            List of AgentResult in execution order
        """
        # Build dependency graph
        dependency_graph = dependencies or {}

        # Identify independent agents (no dependencies)
        independent = [
            a for a in agents if a not in dependency_graph or not dependency_graph.get(a)
        ]
        dependent = [a for a in agents if a in dependency_graph and dependency_graph.get(a)]

        results: Dict[AgentName, AgentResult] = {}

        # Phase 1: Execute independent agents in parallel
        if independent:
            logger.info(f"Executing {len(independent)} independent agents in parallel")
            tasks = [
                self._execute_agent_async(agent, question, context, agent_invokers)
                for agent in independent
            ]
            independent_results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent, result in zip(independent, independent_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent} failed: {result}", exc_info=result)
                    results[agent] = AgentResult(
                        agent=agent,
                        status=AgentExecutionStatus.failed,
                        error=AgentError(message=str(result)),
                    )
                else:
                    results[agent] = result

        # Phase 2: Execute dependent agents sequentially (or in batches if possible)
        for agent in dependent:
            deps = dependency_graph.get(agent, [])
            # Wait for dependencies
            for dep in deps:
                if dep not in results:
                    logger.warning(f"Agent {agent} depends on {dep} which hasn't executed")

            # Execute dependent agent
            result = await self._execute_agent_async(agent, question, context, agent_invokers, results)
            results[agent] = result

        # Return in original order
        return [results[a] for a in agents if a in results]

    async def _execute_agent_async(
        self,
        agent: AgentName,
        question: str,
        context: Dict[str, Any],
        agent_invokers: Optional[Dict[AgentName, Any]] = None,
        previous_results: Optional[Dict[AgentName, AgentResult]] = None,
    ) -> AgentResult:
        """Execute a single agent asynchronously."""
        loop = asyncio.get_event_loop()

        # Prepare context with previous results
        agent_context = dict(context)
        if previous_results:
            agent_context["previous_results"] = previous_results

        # Get invoker function
        invokers = agent_invokers or {}
        invoker = invokers.get(agent)

        if not invoker:
            # Default invoker - this should be provided by caller
            logger.warning(f"No invoker provided for agent {agent}, returning error")
            return AgentResult(
                agent=agent,
                status=AgentExecutionStatus.failed,
                error=AgentError(message=f"No invoker provided for agent {agent}"),
            )

        # Run agent in thread pool (since agents are sync)
        try:
            result = await loop.run_in_executor(self.executor, invoker, question, agent_context)
            return result
        except Exception as exc:
            logger.error(f"Agent {agent} execution failed: {exc}", exc_info=exc)
            return AgentResult(
                agent=agent,
                status=AgentExecutionStatus.failed,
                error=AgentError(message=str(exc), type=type(exc).__name__),
            )

    def execute_queries_parallel(
        self,
        queries: List[str],
        agent_type: AgentName,
        context: Dict[str, Any],
        query_invoker: Any,
    ) -> List[AgentResult]:
        """
        Execute multiple queries in parallel using the same agent type.
        Useful for: multiple SQL queries, multiple API calls, etc.

        Args:
            queries: List of queries to execute
            agent_type: Type of agent to use
            context: Shared context
            query_invoker: Function to invoke for each query (question, context) -> AgentResult

        Returns:
            List of AgentResult objects
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(query_invoker, query, context): query for query in queries
            }

            results = []
            for future in as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error(f"Query '{query[:50]}...' failed: {exc}", exc_info=exc)
                    results.append(
                        AgentResult(
                            agent=agent_type,
                            status=AgentExecutionStatus.failed,
                            error=AgentError(message=str(exc), type=type(exc).__name__),
                        )
                    )
            return results

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


__all__ = ["ParallelExecutor"]

