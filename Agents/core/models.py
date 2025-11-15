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
    """Structured representation of graph/chart data returned by an agent.
    
    Follows API_RESPONSE_FORMAT.md specification:
    - y_axis should always be an array (recommended format)
    - chart_type must be one of: line, bar, pie, donut, scatter, area
    - data must contain objects with x_axis and all y_axis keys
    """

    chart_type: Literal["line", "bar", "pie", "donut", "scatter", "area"] = Field(
        ..., description="Type of chart to render. Valid: line, bar, pie, donut, scatter, area"
    )
    x_axis: str = Field(..., description="Column name or label for x-axis (must exist in data objects)")
    y_axis: List[str] = Field(
        ..., 
        description="Y-axis column name(s) as array. RECOMMENDED: Always use array format even for single metrics. Must exist in data objects."
    )
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="Array of data objects. Each object must contain x_axis key and all y_axis keys with numeric values."
    )
    title: Optional[str] = Field(None, description="Chart title displayed above the chart")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label (only used if single y-axis metric)")
    aggregation: Optional[str] = Field(
        None, description="Aggregation method applied (e.g., 'daily', 'monthly', 'sum', 'average')"
    )
    trend_analysis: Optional[str] = Field(None, description="Optional text analysis displayed below the chart")

    @field_validator("y_axis", mode="before")
    @classmethod
    def _normalize_y_axis(cls, value: str | List[str] | None) -> List[str]:
        """Normalize y_axis to always be an array (recommended format per API spec)."""
        if value is None:
            return []
        if isinstance(value, list):
            # Ensure all items are strings and filter out empty values
            return [str(item).strip() for item in value if item]
        if isinstance(value, str):
            # Handle comma-separated string
            if "," in value:
                return [col.strip() for col in value.split(",") if col.strip()]
            # Single string -> convert to array (recommended format)
            return [value.strip()] if value.strip() else []
        # Fallback: convert to string then array
        return [str(value)] if value else []

    @field_validator("chart_type", mode="before")
    @classmethod
    def _normalize_chart_type(cls, value: str) -> str:
        """Normalize chart type to valid values."""
        if isinstance(value, str):
            normalized = value.lower().strip()
            valid_types = ["line", "bar", "pie", "donut", "scatter", "area"]
            if normalized in valid_types:
                return normalized
            # Fallback to line for invalid types
            return "line"
        return "line"


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


class Tool(BaseModel):
    """Represents a specific tool/endpoint with capabilities."""

    id: str = Field(..., description="Unique identifier for the tool")
    name: str = Field(..., description="Tool name (e.g., 'GET /api/net_profit')")
    type: Literal["api_endpoint", "sql_query", "python_function", "mcp_tool"] = Field(
        ..., description="Type of tool"
    )
    agent: AgentName = Field(..., description="Which agent owns this tool")
    capabilities: List[str] = Field(
        default_factory=list, description="List of what this tool can do"
    )
    metrics: List[str] = Field(
        default_factory=list, description="List of metrics this tool provides"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Required/optional parameters"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context (performance, availability, etc.)"
    )


class Skill(BaseModel):
    """Represents a capability of an agent."""

    id: str = Field(..., description="Unique identifier for the skill")
    name: str = Field(..., description="Skill name (e.g., 'data_retrieval', 'metric_calculation')")
    agent: AgentName = Field(..., description="Which agent has this skill")
    description: str = Field(..., description="What the skill does")
    tools: List[str] = Field(
        default_factory=list, description="List of tool IDs that implement this skill"
    )
    use_cases: List[str] = Field(
        default_factory=list, description="When to use this skill"
    )
    preferences: List[str] = Field(
        default_factory=list, description="When to prefer this skill over alternatives"
    )


class QueryPattern(BaseModel):
    """Query pattern that indicates when an agent should be used."""
    
    keywords: List[str] = Field(..., description="Keywords that trigger this pattern")
    boost_score: int = Field(default=0, description="Score boost when pattern matches")
    reason: str = Field(..., description="Why this pattern indicates the agent")


class AgentCapability(BaseModel):
    """Complete capability profile for an agent."""

    agent_name: AgentName = Field(..., description="Agent identifier")
    description: str = Field(..., description="What this agent does")
    skills: List[Skill] = Field(default_factory=list, description="List of skills")
    tools: List[Tool] = Field(default_factory=list, description="List of tools")
    strengths: List[str] = Field(
        default_factory=list, description="What this agent is best at"
    )
    limitations: List[str] = Field(
        default_factory=list, description="What this agent cannot do"
    )
    preferred_for: List[str] = Field(
        default_factory=list, description="When to prefer this agent"
    )
    requires: List[str] = Field(
        default_factory=list, description="Prerequisites (other agents, data, etc.)"
    )
    query_patterns: Dict[str, QueryPattern] = Field(
        default_factory=dict, description="Query patterns that indicate this agent should be used"
    )


class QueryIntent(BaseModel):
    """Identified intent from user query."""

    primary_intent: Literal[
        "data_retrieval",
        "analysis",
        "comparison",
        "trend_analysis",
        "forecast",
        "visualization",
        "computation",
        "api_inquiry",
    ] = Field(..., description="Primary intent of the query")
    secondary_intents: List[str] = Field(
        default_factory=list, description="Additional intents if query is multi-faceted"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in intent detection")


class SubQuery(BaseModel):
    """A specific, actionable sub-query derived from user query."""

    id: str = Field(..., description="Unique identifier")
    original_phrase: str = Field(
        ..., description="Original phrase from user query that triggered this sub-query"
    )
    detailed_query: str = Field(..., description="Detailed, specific query text")
    intent: Literal[
        "data_retrieval", "computation", "api_call", "visualization", "analysis"
    ] = Field(..., description="Intent category of this sub-query")
    required_agents: List[AgentName] = Field(
        default_factory=list, description="Agents needed for this sub-query"
    )
    selected_tools: List[str] = Field(
        default_factory=list, description="Tool IDs to use (from capability registry)"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of sub-queries this depends on"
    )
    priority: int = Field(default=0, description="Execution priority (higher = first)")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context (timeframes, metrics, filters)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecomposedQuery(BaseModel):
    """Result of natural language query decomposition."""

    original_query: str = Field(..., description="Original user query")
    interpreted_query: str = Field(
        ..., description="LLM's interpretation of what user wants"
    )
    intent: QueryIntent = Field(..., description="Detected intent")
    sub_queries: List[SubQuery] = Field(
        default_factory=list, description="List of specific sub-queries"
    )
    inferred_metrics: List[str] = Field(
        default_factory=list, description="Metrics inferred from query"
    )
    inferred_timeframes: Dict[str, str] = Field(
        default_factory=dict, description="Timeframes inferred (e.g., {'start': '2024-01-01'})"
    )
    inferred_filters: Dict[str, Any] = Field(
        default_factory=dict, description="Filters inferred (e.g., {'region': 'US'})"
    )
    capability_matches: Dict[str, Any] = Field(
        default_factory=dict, description="Matched tools/agents from registry"
    )
    decomposition_rationale: str = Field(..., description="Explanation of decomposition")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decomposition")


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
    graphs: List[GraphResult] = Field(
        default_factory=list, description="Multiple graphs (new format)"
    )
    agent_results: List[AgentResult] = Field(default_factory=list)
    trace: List[TraceEvent] = Field(default_factory=list)
    planner: Optional[PlannerDecision] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def all_graphs(self) -> List[GraphResult]:
        """Get all graphs (backward compatible)."""
        graphs = list(self.graphs)
        if self.graph and self.graph not in graphs:
            graphs.insert(0, self.graph)
        return graphs


__all__ = [
    "AgentCapability",
    "AgentError",
    "AgentExecutionStep",
    "AgentExecutionStatus",
    "AgentName",
    "AgentRequest",
    "AgentResult",
    "DecomposedQuery",
    "GraphResult",
    "MemoryEntry",
    "OrchestratorResponse",
    "PlannerDecision",
    "QueryIntent",
    "RouterDecision",
    "Scratchpad",
    "Skill",
    "SubQuery",
    "TabularResult",
    "Tool",
    "TraceEvent",
    "TraceEventType",
]


