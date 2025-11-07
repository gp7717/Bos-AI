"""Shared Pydantic models for the Agentic multi-agent orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator


AgentName = Literal["sql", "computation", "planner", "composer"]


class AgentExecutionStatus(str, Enum):
    """Execution lifecycle state for an individual agent."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class TraceEventType(str, Enum):
    """Categorisation for orchestrator trace events."""

    MESSAGE = "message"
    TOOL = "tool"
    DECISION = "decision"
    RESULT = "result"
    ERROR = "error"


class TabularResult(BaseModel):
    """Structured representation of tabular data returned by an agent."""

    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0

    @field_validator("row_count", mode="before")
    @classmethod
    def _default_row_count(cls, value: int | None, values: Dict[str, Any]) -> int:
        if value is None:
            rows = values.get("rows")
            if isinstance(rows, Sequence):
                return len(rows)
            return 0
        return value


class AgentError(BaseModel):
    """Standard error envelope for agent executions."""

    message: str
    type: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Captures planner/orchestrator timeline for debugging and analytics."""

    event_type: TraceEventType
    agent: Optional[AgentName] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentExecutionStep(BaseModel):
    """Intermediate execution step emitted by an individual agent."""

    agent: AgentName
    status: AgentExecutionStatus = AgentExecutionStatus.pending
    output: Optional[str] = None
    tabular: Optional[TabularResult] = None
    error: Optional[AgentError] = None
    latency_ms: Optional[float] = None
    trace: List[TraceEvent] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    """Planner output describing which agents should run next."""

    rationale: str
    chosen_agents: Tuple[AgentName, ...]
    confidence: Optional[float] = None
    guardrails: Dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Canonical request payload accepted by the orchestrator."""

    question: str = Field(..., min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)
    prefer_agents: Tuple[AgentName, ...] = Field(default_factory=tuple)
    disable_agents: Tuple[AgentName, ...] = Field(default_factory=tuple)
    max_turns: Optional[int] = None
    include_data: bool = True
    trace: bool = False
    system_prompt: Optional[str] = None

    @field_validator("prefer_agents", "disable_agents", mode="before")
    @classmethod
    def _normalise_agent_lists(
        cls, value: Optional[Iterable[AgentName | str]]
    ) -> Tuple[AgentName, ...]:
        if value is None:
            return ()
        normalised: List[AgentName] = []
        for item in value:
            if item is None:
                continue
            normalised.append(cls._coerce_agent(item))
        return tuple(normalised)

    @staticmethod
    def _coerce_agent(value: AgentName | str) -> AgentName:
        lookup = str(value).strip().lower()
        if lookup not in {"sql", "computation", "planner", "composer"}:
            raise ValueError(f"Unsupported agent identifier: {value}")
        return lookup  # type: ignore[return-value]


class AgentResult(BaseModel):
    """Final outcome for a specific agent execution."""

    agent: AgentName
    status: AgentExecutionStatus
    answer: Optional[str] = None
    tabular: Optional[TabularResult] = None
    error: Optional[AgentError] = None
    trace: List[TraceEvent] = Field(default_factory=list)
    latency_ms: Optional[float] = None


class OrchestratorResponse(BaseModel):
    """Structured response returned by the multi-agent orchestrator."""

    answer: str
    data: Optional[TabularResult] = None
    agent_results: List[AgentResult] = Field(default_factory=list)
    trace: List[TraceEvent] = Field(default_factory=list)
    planner: Optional[PlannerDecision] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AgentError",
    "AgentExecutionStep",
    "AgentExecutionStatus",
    "AgentName",
    "AgentRequest",
    "AgentResult",
    "OrchestratorResponse",
    "PlannerDecision",
    "TabularResult",
    "TraceEvent",
    "TraceEventType",
]


