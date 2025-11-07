"""LangChain-powered computation agent with safe Python execution."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentName,
    AgentResult,
    TabularResult,
    TraceEvent,
    TraceEventType,
)
from Agents.QueryAgent.config import get_resources

from .sandbox import SafeComputationSandbox, SandboxViolation


class _ComputationPlan(BaseModel):
    reasoning: str = Field(..., description="High-level reasoning prior to computation")
    python: str = Field(..., description="Pure Python code that assigns to a 'result' variable")
    explain: str = Field(
        default="Use the computed result to answer the question",
        description="Guidance for the final explanation",
    )


class _ComputationSummary(BaseModel):
    answer: str
    highlights: str


class ComputationAgent:
    """Co-ordinates planning, execution, and summarisation for computations."""

    def __init__(
        self,
        *,
        llm: Optional[AzureChatOpenAI] = None,
        sandbox: Optional[SafeComputationSandbox] = None,
    ) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self.sandbox = sandbox or SafeComputationSandbox()
        self._plan_parser = PydanticOutputParser(pydantic_object=_ComputationPlan)
        self._summary_parser = PydanticOutputParser(pydantic_object=_ComputationSummary)
        self._plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise analytical assistant. "
                    "Write a short Python snippet to compute numeric or tabular answers. "
                    "You must only use the provided safe sandbox with built-in math/statistics utilities. "
                    "Always assign your final numeric or structured answer to a variable named 'result'. "
                    "Return JSON following the given format instructions.",
                ),
                (
                    "user",
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Format instructions: {format_instructions}",
                ),
            ]
        )
        self._summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You turn computation outputs into concise answers. "
                    "Explain the result clearly and include key figures."
                    "Return JSON following the format instructions.",
                ),
                (
                    "user",
                    "Question: {question}\n"
                    "Computed result (repr): {result_repr}\n"
                    "Python code executed:\n{python}\n"
                    "stdout:\n{stdout}\n"
                    "Planning rationale: {reasoning}\n"
                    "Guidance: {guidance}\n"
                    "Format instructions: {format_instructions}",
                ),
            ]
        )

    def invoke(self, question: str, *, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        trace: list[TraceEvent] = []
        start_time = time.perf_counter()

        plan_event = TraceEvent(
            event_type=TraceEventType.DECISION,
            agent=cast_agent("computation"),
            message="Planner started for computation agent",
        )
        trace.append(plan_event)

        plan_chain = (
            self._plan_prompt.partial(
                format_instructions=self._plan_parser.get_format_instructions()
            )
            | self.llm
            | self._plan_parser
        )

        plan: _ComputationPlan
        try:
            plan = plan_chain.invoke({"question": question, "context": context or {}})
        except Exception as exc:  # pragma: no cover - defensive error capture
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(message="Failed to generate computation plan", type=type(exc).__name__)
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("computation"),
                    message=str(exc),
                    data={"stage": "planning"},
                )
            )
            return AgentResult(
                agent=cast_agent("computation"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        trace.append(
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent=cast_agent("computation"),
                message="Computation plan generated",
                data=plan.model_dump(),
            )
        )

        try:
            execution = self.sandbox.execute(plan.python, context=context)
        except SandboxViolation as exc:
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="Sandbox rejected computation",
                type="SandboxViolation",
                details={"reason": str(exc)},
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("computation"),
                    message=str(exc),
                    data={"stage": "execution"},
                )
            )
            return AgentResult(
                agent=cast_agent("computation"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="Execution error in computation sandbox",
                type=type(exc).__name__,
                details={"python": plan.python},
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("computation"),
                    message=str(exc),
                    data={"stage": "execution"},
                )
            )
            return AgentResult(
                agent=cast_agent("computation"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        trace.append(
            TraceEvent(
                event_type=TraceEventType.RESULT,
                agent=cast_agent("computation"),
                message="Sandbox execution complete",
                data={
                    "stdout": execution.stdout,
                    "locals": {k: repr(v) for k, v in execution.locals.items()},
                },
            )
        )

        summary_chain = (
            self._summary_prompt.partial(
                format_instructions=self._summary_parser.get_format_instructions()
            )
            | self.llm
            | self._summary_parser
        )

        summary: _ComputationSummary
        try:
            summary = summary_chain.invoke(
                {
                    "question": question,
                    "result_repr": repr(execution.result),
                    "python": plan.python,
                    "stdout": execution.stdout or "<no output>",
                    "reasoning": plan.reasoning,
                    "guidance": plan.explain,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="Failed to summarise computation result",
                type=type(exc).__name__,
                details={"python": plan.python, "result": repr(execution.result)},
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("computation"),
                    message=str(exc),
                    data={"stage": "summary"},
                )
            )
            return AgentResult(
                agent=cast_agent("computation"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        tabular_payload = execution.as_tabular()
        tabular = TabularResult(**tabular_payload) if tabular_payload else None

        latency = (time.perf_counter() - start_time) * 1000
        trace.append(
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent=cast_agent("computation"),
                message="Computation agent completed",
                data={"latency_ms": latency},
            )
        )

        return AgentResult(
            agent=cast_agent("computation"),
            status=AgentExecutionStatus.succeeded,
            answer=summary.answer,
            tabular=tabular,
            trace=trace,
            latency_ms=latency,
        )


def compile_computation_agent() -> RunnableLambda:
    """Return a runnable computation agent for orchestrator integration."""

    agent = ComputationAgent()
    return RunnableLambda(lambda payload: agent.invoke(payload["question"], context=payload.get("context")))


def cast_agent(name: str) -> AgentName:
    return name.lower()  # type: ignore[return-value]


__all__ = ["ComputationAgent", "compile_computation_agent"]


