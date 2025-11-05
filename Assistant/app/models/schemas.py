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

