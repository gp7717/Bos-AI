"""LangChain-powered graph agent that analyzes table data and generates graph-ready structures."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, cast

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentName,
    AgentResult,
    GraphResult,
    TabularResult,
    TraceEvent,
    TraceEventType,
)
from Agents.QueryAgent.config import get_resources


class _GraphAnalysis(BaseModel):
    """LLM analysis of table data structure and chart requirements."""

    reasoning: str = Field(..., description="Analysis of the data structure and user requirements")
    chart_type: str = Field(..., description="Recommended chart type: line, bar, pie, donut, scatter, or area")
    x_axis_column: str = Field(..., description="Column name to use for x-axis")
    y_axis_columns: List[str] = Field(
        ..., 
        description="Column name(s) to use for y-axis as array. For multiple metrics (e.g., revenue and orders), include all in array."
    )
    needs_aggregation: bool = Field(default=False, description="Whether data needs aggregation/simplification")
    aggregation_method: Optional[str] = Field(
        None, description="Aggregation method if needed: daily, monthly, sum, average, count, etc."
    )
    title: Optional[str] = Field(None, description="Suggested chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label (only used if single y-axis metric)")


class _GraphData(BaseModel):
    """Final graph data structure with simplified/aggregated data points.
    
    IMPORTANT: Follow API_RESPONSE_FORMAT.md specification:
    - y_axis MUST be an array format: ["metric1", "metric2"] even for single metrics
    - Each data object must contain x_axis key and all y_axis keys
    - y_axis values must be numeric (strings will be parsed)
    """

    chart_type: str = Field(..., description="Chart type: line, bar, pie, donut, scatter, or area")
    x_axis: str = Field(..., description="X-axis column name or label (must exist in data objects)")
    y_axis: List[str] = Field(
        ..., 
        description="Y-axis column name(s) as ARRAY. RECOMMENDED: Always use array format even for single metrics. Example: ['revenue'] or ['revenue', 'orders']"
    )
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="Array of data objects. Each object MUST contain x_axis key and all y_axis keys with numeric values. Can include additional keys for tooltips."
    )
    title: Optional[str] = Field(None, description="Chart title displayed above the chart")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label (only used if single y-axis metric)")
    aggregation: Optional[str] = Field(None, description="Aggregation method applied")
    trend_analysis: Optional[str] = Field(None, description="Optional text analysis displayed below the chart")


def cast_agent(name: str) -> AgentName:
    """Cast agent name string to AgentName type."""
    return name.lower()  # type: ignore[return-value]


class GraphAgent:
    """Agent that analyzes table data and generates graph-ready data structures."""

    def __init__(self, *, llm: Optional[AzureChatOpenAI] = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._analysis_parser = PydanticOutputParser(pydantic_object=_GraphAnalysis)
        self._data_parser = PydanticOutputParser(pydantic_object=_GraphData)
        self._analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a data visualization expert. Analyze the provided table data structure "
                        "and user question to determine the best chart type and data organization. "
                        "Consider:\n"
                        "- Data types (numeric, categorical, temporal)\n"
                        "- Number of rows (if >100, suggest aggregation)\n"
                        "- User intent from the question\n"
                        "- Appropriate chart type (line for trends over time, bar for categories, "
                        "pie/donut for proportions, scatter for correlations, area for cumulative trends)\n"
                        "- For multiple metrics (e.g., revenue AND orders), include ALL in y_axis_columns array\n"
                        "- CRITICAL: y_axis_columns MUST be an array, even for single metrics: ['metric'] not 'metric'\n"
                        "Return JSON following the format instructions."
                    ),
                ),
                (
                    "user",
                    (
                        "Question: {question}\n"
                        "Table columns: {columns}\n"
                        "Row count: {row_count}\n"
                        "Sample data (first 5 rows):\n{sample_data}\n"
                        "Format instructions: {format_instructions}"
                    ),
                ),
            ]
        )
        self._data_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a data processing expert. Based on the analysis, process the table data "
                        "to create graph-ready data points following API_RESPONSE_FORMAT.md specification.\n"
                        "CRITICAL REQUIREMENTS:\n"
                        "- y_axis MUST be an array format: ['metric'] even for single metrics, ['metric1', 'metric2'] for multiple\n"
                        "- Each data object MUST contain the x_axis key and ALL y_axis keys\n"
                        "- y_axis values must be numeric (convert strings to numbers)\n"
                        "- You can include additional keys in data objects (they'll be available in tooltips)\n"
                        "Processing rules:\n"
                        "- If aggregation is needed, apply it intelligently\n"
                        "- For time-series data, group by appropriate time periods\n"
                        "- For categorical data, aggregate numeric values appropriately\n"
                        "- Return simplified data points that clearly show trends and patterns\n"
                        "- Include trend analysis insights\n"
                        "Return JSON following the format instructions."
                    ),
                ),
                (
                    "user",
                    (
                        "Question: {question}\n"
                        "Analysis: {analysis}\n"
                        "Full table data:\n{table_data}\n"
                        "Format instructions: {format_instructions}"
                    ),
                ),
            ]
        )

    def invoke(self, question: str, *, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Analyze table data and generate graph-ready structure."""
        trace: List[TraceEvent] = []
        start_time = time.perf_counter()

        # Extract tabular data from context (from previous agents)
        tabular_data: Optional[TabularResult] = None
        if context:
            # Check if tabular_data is passed directly
            if "tabular_data" in context:
                tab_data = context["tabular_data"]
                # Handle both TabularResult object and dict representation
                if isinstance(tab_data, TabularResult):
                    tabular_data = tab_data
                elif isinstance(tab_data, dict):
                    tabular_data = TabularResult(**tab_data)
            # Check if it's in agent_results (from previous agents)
            elif "agent_results" in context:
                agent_results = context["agent_results"]
                if isinstance(agent_results, list):
                    for result in agent_results:
                        # Handle both AgentResult object and dict representation
                        if isinstance(result, dict):
                            if result.get("tabular"):
                                tab_data = result["tabular"]
                                tabular_data = (
                                    tab_data
                                    if isinstance(tab_data, TabularResult)
                                    else TabularResult(**tab_data)
                                )
                                break
                        elif hasattr(result, "tabular") and result.tabular:
                            tabular_data = result.tabular
                            break

        if not tabular_data or not tabular_data.rows:
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="No tabular data available for graph generation. Graph agent requires table data from previous agents (SQL/Computation) or explicit tabular_data in context.",
                type="MissingData",
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("graph"),
                    message="No tabular data found",
                    data={"context_keys": list(context.keys()) if context else []},
                )
            )
            return AgentResult(
                agent=cast_agent("graph"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        trace.append(
            TraceEvent(
                event_type=TraceEventType.DECISION,
                agent=cast_agent("graph"),
                message="Starting graph data analysis",
                data={"row_count": tabular_data.row_count, "column_count": len(tabular_data.columns)},
            )
        )

        # Prepare sample data for analysis (first 5 rows)
        sample_rows = tabular_data.rows[:5] if tabular_data.rows else []
        sample_data_str = json.dumps(sample_rows, default=str, indent=2)

        # Phase 1: Analyze data structure and determine chart requirements
        analysis_chain = (
            self._analysis_prompt.partial(
                format_instructions=self._analysis_parser.get_format_instructions()
            )
            | self.llm
            | self._analysis_parser
        )

        analysis: _GraphAnalysis
        try:
            analysis = analysis_chain.invoke(
                {
                    "question": question,
                    "columns": ", ".join(tabular_data.columns),
                    "row_count": tabular_data.row_count,
                    "sample_data": sample_data_str,
                }
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="Failed to analyze table data structure", type=type(exc).__name__
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("graph"),
                    message=str(exc),
                    data={"stage": "analysis"},
                )
            )
            return AgentResult(
                agent=cast_agent("graph"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        trace.append(
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent=cast_agent("graph"),
                message="Data analysis complete",
                data={
                    "chart_type": analysis.chart_type,
                    "x_axis": analysis.x_axis_column,
                    "y_axis": analysis.y_axis_columns,
                    "needs_aggregation": analysis.needs_aggregation,
                },
            )
        )

        # Phase 2: Process data and generate graph-ready structure
        # Convert full table data to JSON string for LLM processing
        full_data_str = json.dumps(tabular_data.rows, default=str)

        data_chain = (
            self._data_prompt.partial(format_instructions=self._data_parser.get_format_instructions())
            | self.llm
            | self._data_parser
        )

        graph_data: _GraphData
        try:
            graph_data = data_chain.invoke(
                {
                    "question": question,
                    "analysis": json.dumps(analysis.model_dump(), indent=2),
                    "table_data": full_data_str,
                }
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="Failed to generate graph data structure", type=type(exc).__name__
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("graph"),
                    message=str(exc),
                    data={"stage": "data_generation"},
                )
            )
            return AgentResult(
                agent=cast_agent("graph"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        # Validate and normalize chart_type (GraphResult model will handle normalization)
        chart_type = graph_data.chart_type.lower().strip()
        valid_chart_types = ["line", "bar", "pie", "donut", "scatter", "area"]
        if chart_type not in valid_chart_types:
            chart_type = "line"  # Default fallback

        # Normalize y_axis to always be an array (per API spec recommendation)
        # GraphResult model will also normalize, but we do it here for clarity
        y_axis_normalized: List[str]
        if isinstance(graph_data.y_axis, list):
            # Filter out empty values and ensure strings
            y_axis_normalized = [str(item).strip() for item in graph_data.y_axis if item]
        elif isinstance(graph_data.y_axis, str):
            # Handle comma-separated string
            if "," in graph_data.y_axis:
                y_axis_normalized = [col.strip() for col in graph_data.y_axis.split(",") if col.strip()]
            else:
                # Single string -> convert to array (recommended format)
                y_axis_normalized = [graph_data.y_axis.strip()] if graph_data.y_axis.strip() else []
        else:
            # Fallback: convert to string then array
            y_axis_normalized = [str(graph_data.y_axis)] if graph_data.y_axis else []

        # Validate data structure: ensure each data object has x_axis and all y_axis keys
        validated_data = []
        for idx, data_point in enumerate(graph_data.data):
            if not isinstance(data_point, dict):
                continue  # Skip invalid entries
            
            # Ensure x_axis key exists
            if graph_data.x_axis not in data_point:
                continue  # Skip entries missing x_axis
            
            # Ensure all y_axis keys exist (create with 0 if missing)
            validated_point = dict(data_point)
            for y_key in y_axis_normalized:
                if y_key not in validated_point:
                    validated_point[y_key] = 0
                else:
                    # Ensure numeric value (convert string to number)
                    val = validated_point[y_key]
                    try:
                        if isinstance(val, str):
                            # Try to convert string to number
                            validated_point[y_key] = float(val.replace(",", ""))
                        elif not isinstance(val, (int, float)):
                            validated_point[y_key] = 0
                    except (ValueError, AttributeError):
                        validated_point[y_key] = 0
            
            validated_data.append(validated_point)

        if not validated_data:
            latency = (time.perf_counter() - start_time) * 1000
            error = AgentError(
                message="No valid data points generated after validation. Each data point must contain x_axis and all y_axis keys.",
                type="DataValidationError",
            )
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent=cast_agent("graph"),
                    message="Data validation failed",
                    data={"x_axis": graph_data.x_axis, "y_axis": y_axis_normalized},
                )
            )
            return AgentResult(
                agent=cast_agent("graph"),
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency,
            )

        # Build GraphResult (y_axis will be normalized to array by model validator)
        graph_result = GraphResult(
            chart_type=chart_type,  # type: ignore[arg-type]
            x_axis=graph_data.x_axis,
            y_axis=y_axis_normalized,  # Already normalized to array
            data=validated_data,
            title=graph_data.title,
            x_label=graph_data.x_label,
            y_label=graph_data.y_label,
            aggregation=graph_data.aggregation,
            trend_analysis=graph_data.trend_analysis,
        )

        # Generate answer summary
        answer_parts = [f"Generated {chart_type} chart"]
        if graph_result.title:
            answer_parts.append(f"titled '{graph_result.title}'")
        if graph_result.aggregation:
            answer_parts.append(f"with {graph_result.aggregation} aggregation")
        if graph_result.trend_analysis:
            answer_parts.append(f"\n\nTrend Analysis: {graph_result.trend_analysis}")

        answer = ". ".join(answer_parts) + "."

        latency = (time.perf_counter() - start_time) * 1000
        trace.append(
            TraceEvent(
                event_type=TraceEventType.RESULT,
                agent=cast_agent("graph"),
                message="Graph data generation complete",
                data={
                    "chart_type": chart_type,
                    "data_points": len(graph_data.data),
                    "latency_ms": latency,
                },
            )
        )

        return AgentResult(
            agent=cast_agent("graph"),
            status=AgentExecutionStatus.succeeded,
            answer=answer,
            graph=graph_result,
            trace=trace,
            latency_ms=latency,
        )


__all__ = ["GraphAgent"]

