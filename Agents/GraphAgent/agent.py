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
    chart_type: str = Field(..., description="Recommended chart type: line, bar, pie, scatter, or area")
    x_axis_column: str = Field(..., description="Column name to use for x-axis")
    y_axis_columns: List[str] = Field(..., description="Column name(s) to use for y-axis")
    needs_aggregation: bool = Field(default=False, description="Whether data needs aggregation/simplification")
    aggregation_method: Optional[str] = Field(
        None, description="Aggregation method if needed: daily, monthly, sum, average, count, etc."
    )
    title: Optional[str] = Field(None, description="Suggested chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")


class _GraphData(BaseModel):
    """Final graph data structure with simplified/aggregated data points."""

    chart_type: str = Field(..., description="Chart type: line, bar, pie, scatter, or area")
    x_axis: str = Field(..., description="X-axis column name or label")
    y_axis: str | List[str] = Field(..., description="Y-axis column name(s) or label(s)")
    data: List[Dict[str, Any]] = Field(..., description="Simplified/aggregated data points")
    title: Optional[str] = Field(None, description="Chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")
    aggregation: Optional[str] = Field(None, description="Aggregation method applied")
    trend_analysis: Optional[str] = Field(None, description="Key trends and insights identified")


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
                        "pie for proportions, scatter for correlations, area for cumulative trends)\n"
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
                        "to create graph-ready data points. If aggregation is needed, apply it intelligently. "
                        "For time-series data, group by appropriate time periods. For categorical data, "
                        "aggregate numeric values appropriately. Return simplified data points that clearly "
                        "show trends and patterns. Include trend analysis insights. "
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

        # Validate chart_type
        valid_chart_types = ["line", "bar", "pie", "scatter", "area"]
        chart_type = graph_data.chart_type.lower()
        if chart_type not in valid_chart_types:
            chart_type = "line"  # Default fallback

        # Normalize y_axis to handle both string and list formats
        y_axis_normalized: str | List[str]
        if isinstance(graph_data.y_axis, list):
            y_axis_normalized = graph_data.y_axis
        elif isinstance(graph_data.y_axis, str):
            # If it's a comma-separated string, split it; otherwise use as single string
            if "," in graph_data.y_axis:
                y_axis_normalized = [col.strip() for col in graph_data.y_axis.split(",")]
            else:
                y_axis_normalized = graph_data.y_axis
        else:
            y_axis_normalized = str(graph_data.y_axis)

        # Build GraphResult
        graph_result = GraphResult(
            chart_type=chart_type,  # type: ignore[arg-type]
            x_axis=graph_data.x_axis,
            y_axis=y_axis_normalized,
            data=graph_data.data,
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

