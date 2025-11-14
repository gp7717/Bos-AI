"""Shared Pydantic models for the Agentic multi-agent orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator


AgentName = Literal["sql", "computation", "planner", "composer", "api_docs", "router", "graph"]


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


class GraphResult(BaseModel):
    """Structured representation of graph/chart data returned by an agent."""

    chart_type: Literal["line", "bar", "pie", "scatter", "area"] = Field(
        ..., description="Type of chart to render"
    )
    x_axis: str = Field(..., description="Column name or label for x-axis")
    y_axis: str | List[str] = Field(..., description="Column name(s) or label(s) for y-axis")
    data: List[Dict[str, Any]] = Field(
        ..., description="Simplified/aggregated data points for the graph"
    )
    title: Optional[str] = Field(None, description="Chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")
    aggregation: Optional[str] = Field(
        None, description="Aggregation method applied (e.g., 'daily', 'monthly', 'sum', 'average')"
    )
    trend_analysis: Optional[str] = Field(None, description="LLM-generated trend insights")


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


class MemoryEntry(BaseModel):
    """A single entry in the shared memory/scratchpad."""

    agent: AgentName
    category: Literal["finding", "data_summary", "insight", "context", "error"] = Field(
        default="finding", description="Category of memory entry"
    )
    content: str = Field(..., description="The actual memory content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Scratchpad(BaseModel):
    """Shared memory/context window for agents to read/write during execution."""

    entries: List[MemoryEntry] = Field(default_factory=list, description="Chronological memory entries")
    max_entries: int = Field(
        default=50, description="Maximum number of entries to keep (FIFO when exceeded)"
    )

    def add(
        self,
        agent: AgentName,
        content: str,
        category: Literal["finding", "data_summary", "insight", "context", "error"] = "finding",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new memory entry."""
        entry = MemoryEntry(
            agent=agent,
            category=category,
            content=content,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        # Maintain max_entries limit (FIFO)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def get_by_agent(self, agent: AgentName) -> List[MemoryEntry]:
        """Get all entries from a specific agent."""
        return [entry for entry in self.entries if entry.agent == agent]

    def get_by_category(
        self, category: Literal["finding", "data_summary", "insight", "context", "error"]
    ) -> List[MemoryEntry]:
        """Get all entries of a specific category."""
        return [entry for entry in self.entries if entry.category == category]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent entries."""
        return self.entries[-limit:] if len(self.entries) > limit else self.entries

    def get_summary(self) -> str:
        """Get a formatted summary of all entries for LLM consumption."""
        if not self.entries:
            return "No memory entries available."

        lines = ["=== Shared Memory/Scratchpad ==="]
        for entry in self.entries[-20:]:  # Last 20 entries
            timestamp_str = entry.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{timestamp_str}] {entry.agent.upper()} ({entry.category}): {entry.content}")
        return "\n".join(lines)

    def get_data_summaries(self) -> Dict[str, str]:
        """Get all data summaries keyed by agent."""
        summaries = {}
        for entry in self.entries:
            if entry.category == "data_summary":
                summaries[entry.agent] = entry.content
        return summaries

    def get_findings(self) -> List[str]:
        """Get all findings across all agents."""
        return [entry.content for entry in self.entries if entry.category == "finding"]


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


class RouterDecision(BaseModel):
    """Router output describing whether agents are needed for the query."""

    route_type: Literal["simple_response", "needs_agents"]
    rationale: str
    confidence: Optional[float] = None


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
        if lookup not in {"sql", "computation", "planner", "composer", "api_docs", "graph"}:
            raise ValueError(f"Unsupported agent identifier: {value}")
        return lookup  # type: ignore[return-value]


class AgentResult(BaseModel):
    """Final outcome for a specific agent execution."""

    agent: AgentName
    status: AgentExecutionStatus
    answer: Optional[str] = None
    tabular: Optional[TabularResult] = None
    graph: Optional[GraphResult] = None
    error: Optional[AgentError] = None
    trace: List[TraceEvent] = Field(default_factory=list)
    latency_ms: Optional[float] = None


class OrchestratorResponse(BaseModel):
    """Structured response returned by the multi-agent orchestrator."""

    answer: str
    data: Optional[TabularResult] = None
    graph: Optional[GraphResult] = None
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
    "GraphResult",
    "MemoryEntry",
    "OrchestratorResponse",
    "PlannerDecision",
    "RouterDecision",
    "Scratchpad",
    "TabularResult",
    "TraceEvent",
    "TraceEventType",
]


