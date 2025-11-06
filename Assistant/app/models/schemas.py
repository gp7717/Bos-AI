"""Pydantic models for request/response schemas."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, date
from enum import Enum


class Intent(str, Enum):
    """Intent classification."""
    ANALYTICS_QUERY = "analytics.query"
    DATA_EXPORT = "data.export"
    DIAGNOSTICS = "diagnostics"
    META_HELP = "meta.help"
    UNKNOWN = "unknown"


class Channel(str, Enum):
    """Advertising channels."""
    SP = "SP"  # Sponsored Products
    SB = "SB"  # Sponsored Brands
    SD = "SD"  # Sponsored Display
    META = "META"
    GOOGLE = "GOOGLE"
    ORGANIC = "ORGANIC"


class TimeRange(BaseModel):
    """Time range specification."""
    start: Optional[date] = None
    end: Optional[date] = None
    range: Optional[str] = None  # "last_week", "last_month", etc.
    tz: str = "Asia/Kolkata"


class TaskSpec(BaseModel):
    """Task specification from Router Agent."""
    intent: Intent
    metrics: List[str] = []
    entities: Dict[str, Any] = {}
    time: TimeRange
    filters: List[Dict[str, Any]] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ToolCapability(BaseModel):
    """Tool capability definition."""
    name: str
    inputs_schema_ref: Optional[str] = None
    metrics: List[str] = []
    channels: List[str] = []
    description: Optional[str] = None


class ToolDefinition(BaseModel):
    """Tool definition in Tool Registry."""
    tool_id: str
    kind: Literal["api", "sql", "compute"]
    capabilities: List[ToolCapability] = []
    auth: Optional[str] = None
    quotas: Dict[str, Any] = {}
    latency_hint_ms: Optional[int] = None
    retries: Dict[str, Any] = {}
    observability_tags: List[str] = []
    connection_ref: Optional[str] = None
    schemas: List[str] = []
    join_keys: List[Dict[str, str]] = []
    rls: Dict[str, Any] = {}


class PlanStep(BaseModel):
    """Single step in execution plan."""
    id: str
    tool: str
    inputs: Dict[str, Any]
    depends_on: List[str] = []
    output_key: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Execution plan DAG from Planner Agent."""
    steps: List[PlanStep]
    outputs: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    """User query request."""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Query response."""
    answer: str
    data: Optional[Dict[str, Any]] = None
    table: Optional[List[Dict[str, Any]]] = None
    chart_spec: Optional[Dict[str, Any]] = None
    reasoning_trace: Optional[str] = None
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Guardrail validation result."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []


class MetricDefinition(BaseModel):
    """Metric definition in Metric Dictionary."""
    metric_id: str
    name: str
    formula: str
    description: str
    unit: str
    category: str
    dependencies: List[str] = []  # Required metrics/columns
    computation_type: Literal["simple", "aggregation", "ratio", "delta"] = "simple"


class SchemaColumn(BaseModel):
    """Database column definition."""
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None
    description: Optional[str] = None


class SchemaTable(BaseModel):
    """Database table definition."""
    schema_name: str = Field(alias="schema")
    table: str
    columns: List[SchemaColumn]
    primary_keys: List[str] = []
    foreign_keys: List[Dict[str, str]] = []
    indexes: List[str] = []
    rls_enabled: bool = False
    
    class Config:
        populate_by_name = True


class AgentResponse(BaseModel):
    """Generic agent response."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# ============================================================================
# Agent Input/Output Schemas - Typed Contracts for Agent Communication
# ============================================================================

class RouterAgentInput(BaseModel):
    """Input schema for Router Agent."""
    query: str = Field(description="User's natural language query")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RouterAgentOutput(BaseModel):
    """Output schema for Router Agent."""
    success: bool
    task_spec: Optional[TaskSpec] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerAgentInput(BaseModel):
    """Input schema for Planner Agent."""
    task_spec: TaskSpec = Field(description="Task specification from router")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PlannerAgentOutput(BaseModel):
    """Output schema for Planner Agent."""
    success: bool
    plan: Optional[ExecutionPlan] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GuardrailAgentInput(BaseModel):
    """Input schema for Guardrail Agent."""
    task_spec: Optional[TaskSpec] = None
    plan: Optional[ExecutionPlan] = None
    validation_type: Literal["task", "plan"] = Field(description="Type of validation to perform")


class GuardrailAgentOutput(BaseModel):
    """Output schema for Guardrail Agent."""
    success: bool
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    corrected_plan: Optional[ExecutionPlan] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataAccessAgentInput(BaseModel):
    """Input schema for Data Access Agents."""
    sql: Optional[str] = Field(None, description="LLM-generated SQL query")
    params: Dict[str, Any] = Field(default_factory=dict, description="SQL parameters")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Template-based query inputs")


class DataAccessAgentOutput(BaseModel):
    """Output schema for Data Access Agents."""
    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query results as list of records")
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComputationAgentInput(BaseModel):
    """Input schema for Computation Agent."""
    operation: Literal["compute", "aggregate", "join", "calculate", "top_n"] = Field(description="Operation type")
    metric_id: Optional[str] = Field(None, description="Metric ID for compute operation (must be from metric dictionary)")
    data: Optional[Any] = Field(None, description="Input data (DataFrame or list of dicts)")
    left_data: Optional[Any] = Field(None, description="Left data for join operation")
    right_data: Optional[Any] = Field(None, description="Right data for join operation")
    group_by: Optional[List[str]] = Field(None, description="Columns to group by for aggregation")
    aggregations: Optional[Dict[str, str]] = Field(None, description="Aggregation functions")
    join_keys: Optional[List[str]] = Field(None, description="Join keys for join operation")
    how: str = Field("left", description="Join type")
    sort_by: Optional[List[str]] = Field(None, description="Columns to sort by for top_n operation")
    ascending: Optional[bool] = Field(True, description="Sort order for top_n operation")
    limit: Optional[int] = Field(None, description="Limit for top_n operation")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class ComputationAgentOutput(BaseModel):
    """Output schema for Computation Agent."""
    success: bool
    data: Optional[Any] = Field(None, description="Result data (DataFrame)")
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComposerAgentInput(BaseModel):
    """Input schema for Composer Agent."""
    task_spec: TaskSpec = Field(description="Original task specification")
    results: Dict[str, Any] = Field(description="Execution results")
    execution_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ComposerAgentOutput(BaseModel):
    """Output schema for Composer Agent."""
    success: bool
    response: Optional[QueryResponse] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanStepExecutionInput(BaseModel):
    """Input schema for plan step execution."""
    step: PlanStep = Field(description="Plan step to execute")
    step_results: Dict[str, Any] = Field(description="Results from previous steps")


class PlanStepExecutionOutput(BaseModel):
    """Output schema for plan step execution."""
    success: bool
    data: Optional[Any] = Field(None, description="Step execution result (DataFrame)")
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

