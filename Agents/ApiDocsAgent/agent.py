"""
Execution-focused agent that plans and performs real HTTP API calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import httpx
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError, field_validator

from Agents.QueryAgent.config import get_resources
from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentResult,
    TraceEvent,
    TraceEventType,
)
from Agents.ApiDocsAgent.mcp_client import ApiMCPClient, EndpointDefinition

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTEXT_PATH = Path(__file__).resolve().parents[2] / "Docs" / "api_docs_context.yaml"
DEFAULT_ENDPOINTS_PATH = Path(__file__).parent / "endpoints.json"
DEFAULT_BASE_URL = "https://dashbackend-a3cbagbzg0hydhen.centralindia-01.azurewebsites.net"
DEFAULT_TIMEOUT_SECONDS = 30.0
FIREBASE_ID_ENDPOINT = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
TOKEN_REFRESH_GRACE_SECONDS = 120
MAX_API_CALL_ATTEMPTS = 3
_TOKEN_CACHE: Dict[str, Dict[str, Any]] = {"token": None, "expires_at": None}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@dataclass(frozen=True)
class ApiEndpointSpec:
    """Lightweight representation of a documented API endpoint."""

    id: str
    title: str
    method: str
    path: str
    description: str
    source: str
    tokens: List[str]

    @classmethod
    def from_line(cls, section_id: str, title: str, source: str, line: str) -> ApiEndpointSpec | None:
        line_stripped = line.strip()
        if not line_stripped:
            return None
        
        # Only process lines that look like endpoint definitions
        # Must be either:
        # 1. Markdown header: ## GET /api/path
        # 2. Direct format: GET /api/path (with optional colon and description)
        line_lower = line_stripped.lower()
        
        # Skip lines that are clearly not endpoint definitions
        # (SQL queries, code snippets, descriptions, bullet points)
        if line_stripped.startswith("- ") or line_stripped.startswith("* "):
            # This is a bullet point, likely a description
            return None
        
        if not (line_stripped.startswith("##") or re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+/", line_stripped, re.IGNORECASE)):
            # Not a markdown header and not a direct endpoint definition
            # Check if it contains SQL/code keywords (likely a description line)
            if any(skip in line_lower for skip in ["select ", "insert ", "update ", "delete ", "executes ", "joins ", "calculates ", "margin =", "returns ", "filters "]):
                return None
        
        # Try to extract from markdown header format first (e.g., `## GET /api/net_profit`)
        # This is the most common format in the YAML
        markdown_match = re.search(r"^##\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(/[\w\-_/:?=&]+)", line_stripped, re.IGNORECASE)
        if markdown_match:
            method_match = markdown_match
            path = markdown_match.group(2).strip()
        else:
            # Try direct format: GET /api/path: description
            direct_match = re.search(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(/[\w\-_/:?=&]+)", line_stripped, re.IGNORECASE)
            if direct_match:
                method_match = direct_match
                path = direct_match.group(2).strip()
            else:
                # Try to extract path from backticks as last resort (e.g., `GET `/api/net_profit``)
                path_match = re.search(r"`([^`]+)`", line_stripped)
                if path_match:
                    candidate_path = path_match.group(1).strip()
                    if candidate_path.startswith("/") or candidate_path.startswith("http"):
                        path = candidate_path
                        method_match = re.search(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", line_stripped, re.IGNORECASE)
                    else:
                        return None
                else:
                    return None
        
        if not path or not path.startswith("/"):
            return None

        method = method_match.group(1).upper() if method_match else "GET"
        
        # Clean up path - remove trailing description after colon
        # Handle cases like "GET /api/net_profit: description"
        if ":" in path:
            # Split on colon, but only if it's not part of the path itself (e.g., /api/:id)
            # Check if colon is followed by space or end of string (description)
            colon_idx = path.find(":")
            if colon_idx > 0 and (colon_idx == len(path) - 1 or path[colon_idx + 1] == " "):
                path = path[:colon_idx].strip()
        
        # Final validation: path must be a valid HTTP path
        if not path.startswith("/") or len(path) < 2:
            return None
        
        # Store the path (query params included if present)
        description = line.replace("`", "").strip(" -")

        tokens = _tokenize(" ".join([title, method, path, description]))
        endpoint_id = f"{section_id}:{method}:{path}"
        return cls(
            id=endpoint_id,
            title=title,
            method=method,
            path=path,
            description=description,
            source=source,
            tokens=tokens,
        )


@dataclass(frozen=True)
class DocumentationSnippet:
    """Structured snippet providing additional endpoint context."""

    id: str
    title: str
    source: str
    content: str
    tokens: List[str]

    @classmethod
    def from_mapping(cls, mapping: dict) -> DocumentationSnippet:
        content = mapping.get("content", "") or ""
        title = mapping.get("title", "") or ""
        tokens = _tokenize(f"{title}\n{content}")
        return cls(
            id=mapping.get("id", "") or "",
            title=title,
            source=mapping.get("source", "") or "",
            content=content,
            tokens=tokens,
        )
@dataclass(frozen=True)
class FirebaseAuthConfig:
    """Configuration used to acquire Firebase ID tokens."""

    direct_token: Optional[str]
    token_file: Optional[Path]
    web_api_key: Optional[str]
    email: Optional[str]
    password: Optional[str]
    context_headers: Dict[str, str]


class ApiCallPlan(BaseModel):
    """Structured instruction for a single HTTP API call."""

    make_request: bool = Field(
        description="Set to true when the agent should proceed with the HTTP request. "
        "If false, failure_reason must explain why the call cannot be made."
    )
    method: str = Field(default="GET", description="HTTP verb to execute.")
    path: str = Field(description="Relative request path, e.g. /api/orders.")
    path_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values to substitute into templated path segments such as :id or {id}."
    )
    query_params: Dict[str, Any] = Field(
        default_factory=dict, description="Key-value query string parameters."
    )
    json_body: Dict[str, Any] | None = Field(
        default=None, description="JSON body for POST/PUT/PATCH requests."
    )
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Additional request headers specific to this call."
    )
    reason: str = Field(description="Short justification for the selected request.")
    failure_reason: str | None = Field(
        default=None, description="Explanation when make_request is false."
    )

    @field_validator("method")
    @classmethod
    def _uppercase_method(cls, value: str) -> str:
        return value.upper()


class ApiDocsAgent:
    """
    Agent that plans an HTTP request using documented endpoints and executes it against the live API.
    """

    def __init__(
        self,
        *,
        llm: Optional[AzureChatOpenAI] = None,
        context_path: Path | str | None = None,
        endpoints_path: Path | str | None = None,
        top_k: int = 5,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        resources = get_resources()
        self.llm: AzureChatOpenAI = llm or resources.llm
        self._context_path = Path(context_path) if context_path else DEFAULT_CONTEXT_PATH
        self._top_k = top_k
        self._http_client = http_client
        
        # Initialize MCP client for endpoint definitions - this is the primary source
        endpoints_path = Path(endpoints_path) if endpoints_path else DEFAULT_ENDPOINTS_PATH
        self._mcp_client = ApiMCPClient(endpoints_path)
        
        # Get base context from MCP (endpoints.json) - this is the source of truth
        self._mcp_context = self._mcp_client.get_all_endpoints_context()
        
        # Legacy YAML is now only for additional documentation snippets
        # Endpoint definitions come from endpoints.json via MCP
        self._endpoints, self._doc_contexts = self._load_context_sources(self._context_path)

        self._planner_parser = PydanticOutputParser(pydantic_object=ApiCallPlan)
        self._planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an integration specialist tasked with planning exactly one HTTP API call. "
                        "Choose an endpoint ONLY from the provided candidates list. "
                        "CRITICAL: Use the EXACT endpoint path from the candidates list. Do NOT construct or invent paths. "
                        "Do NOT use SQL queries, database commands, or any non-HTTP paths. "
                        "The path must match exactly one of the provided endpoint paths. "
                        "\n\n"
                        "PARAMETER USAGE RULES (CRITICAL - READ CAREFULLY):"
                        "\n1. Use ONLY the parameters documented for the selected endpoint (shown in the candidate list)."
                        "\n2. Do NOT invent or add parameters that are not documented."
                        "\n3. IMPORTANT: If the candidate list shows 'Parameters: n (Number of days; default 7)', "
                        "   then the endpoint ONLY accepts 'n' as a query parameter. "
                        "   DO NOT use startDate, endDate, start_date, end_date, granularity, or any other parameters."
                        "\n4. If the endpoint accepts 'n' (number of days):"
                        "   - Calculate 'n' from the requested date range"
                        "   - Use ONLY the 'n' parameter in query_params"
                        "   - Example: For 'last 30 days', use query_params = {{'n': 30}}"
                        "   - Example: For date range 2025-10-16 to 2025-11-14, calculate n = 30 days"
                        "\n5. If the endpoint accepts 'startDate' and 'endDate':"
                        "   - Use them in YYYY-MM-DD format"
                        "   - Example: query_params = {{'startDate': '2025-10-16', 'endDate': '2025-11-14'}}"
                        "\n6. NEVER mix parameter types - if endpoint uses 'n', don't use startDate/endDate"
                        "\n7. Check the 'Additional context' section for detailed parameter documentation."
                        "\n8. If the question provides explicit parameter calculation instructions, follow them exactly."
                        "\n\n"
                        "Do not invent payload fields. "
                        "If you lack mandatory parameters or cannot determine the correct parameters from documentation, set make_request to false and explain why. "
                        "Respond strictly in JSON matching the format instructions."
                    ),
                ),
                (
                    "user",
                    (
                        "Question: {question}\n"
                        "Candidate endpoints:\n{endpoints}\n"
                        "Additional context:\n{context}\n"
                        "Format instructions: {format_instructions}"
                    ),
                ),
            ]
        )

        self._summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You summarise HTTP API responses for internal stakeholders. "
                        "Provide a concise answer focused on the user question, "
                        "note the status code, mention relevant fields, and call out errors or follow-up steps."
                    ),
                ),
                (
                    "user",
                    (
                        "Question: {question}\n"
                        "HTTP request: {method} {url}\n"
                        "Status code: {status_code}\n"
                        "Planner reasoning: {reason}\n"
                        "Response body snippet:\n{body_snippet}\n"
                        "JSON payload snippet:\n{json_snippet}"
                    ),
                ),
            ]
        )

    def invoke(self, question: str, *, context: Optional[dict] = None) -> AgentResult:
        request_context = context or {}
        start_time = time.perf_counter()
        trace: List[TraceEvent] = []

        if not self._endpoints:
            error_message = "API endpoint catalogue is empty; cannot plan API call."
            LOGGER.error(error_message)
            error = AgentError(message=error_message, type="MissingContext")
            return AgentResult(
                agent="api_docs",  # type: ignore[assignment]
                status=AgentExecutionStatus.failed,
                error=error,
                trace=[
                    TraceEvent(
                        event_type=TraceEventType.ERROR,
                        agent="api_docs",  # type: ignore[assignment]
                        message=error_message,
                    )
                ],
            )

        try:
            plan = self._plan_api_call(question, request_context)
        except AgentError as planning_error:
            latency_ms = (time.perf_counter() - start_time) * 1000
            planning_event = TraceEvent(
                event_type=TraceEventType.ERROR,
                agent="api_docs",  # type: ignore[assignment]
                message=planning_error.message,
                data={"type": planning_error.type},
            )
            trace.append(planning_event)
            return AgentResult(
                agent="api_docs",  # type: ignore[assignment]
                status=AgentExecutionStatus.failed,
                error=planning_error,
                trace=trace,
                latency_ms=latency_ms,
            )

        trace.append(
            TraceEvent(
                event_type=TraceEventType.DECISION,
                agent="api_docs",  # type: ignore[assignment]
                message="Planned API call",
                data=plan.model_dump(),
            )
        )

        if not plan.make_request and self._should_force_attempt(plan):
            forced_plan = plan.model_copy(update={"make_request": True, "failure_reason": None})
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="api_docs",  # type: ignore[assignment]
                    message="Planner declined but proceeding with API attempt to gather runtime feedback",
                    data={"original_failure_reason": plan.failure_reason},
                )
            )
            plan = forced_plan

        if not plan.make_request:
            error_reason = plan.failure_reason or "Planner declined to execute the API request."
            latency_ms = (time.perf_counter() - start_time) * 1000
            error = AgentError(message=error_reason, type="PlanningDeclined")
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="api_docs",  # type: ignore[assignment]
                    message=error_reason,
                )
            )
            return AgentResult(
                agent="api_docs",  # type: ignore[assignment]
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
                latency_ms=latency_ms,
            )

        try:
            execution, plan, extra_events = self._execute_plan(plan, request_context)
        except AgentError as execution_error:
            latency_ms = (time.perf_counter() - start_time) * 1000
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="api_docs",  # type: ignore[assignment]
                    message=execution_error.message,
                    data=execution_error.details,
                )
            )
            return AgentResult(
                agent="api_docs",  # type: ignore[assignment]
                status=AgentExecutionStatus.failed,
                error=execution_error,
                trace=trace,
                latency_ms=latency_ms,
            )

        if extra_events:
            trace.extend(extra_events)

        trace.append(
            TraceEvent(
                event_type=TraceEventType.TOOL,
                agent="api_docs",  # type: ignore[assignment]
                message="Executed HTTP request",
                data={
                    "url": execution["url"],
                    "method": execution["method"],
                    "status_code": execution["status_code"],
                    "elapsed_ms": execution.get("elapsed_ms"),
                },
            )
        )

        answer = self._summarise_execution(question, plan, execution)
        trace.append(
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent="api_docs",  # type: ignore[assignment]
                message="Summarised API response",
            )
        )

        status = (
            AgentExecutionStatus.succeeded
            if 200 <= execution["status_code"] < 400
            else AgentExecutionStatus.failed
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        error = None
        if status is AgentExecutionStatus.failed:
            error = AgentError(
                message=f"HTTP {execution['status_code']} response returned.",
                type="HttpError",
                details={
                    "url": execution["url"],
                    "status_code": execution["status_code"],
                    "body": execution.get("body_snippet"),
                },
            )

        return AgentResult(
            agent="api_docs",  # type: ignore[assignment]
            status=status,
            answer=answer,
            trace=trace,
            error=error,
            latency_ms=latency_ms,
        )

    def _plan_api_call(self, question: str, context: Dict[str, Any]) -> ApiCallPlan:
        # Get base context from MCP client (endpoints.json) - this is the source of truth
        # MCP context provides endpoint definitions, orchestrator provides execution context
        # Merge them: MCP context takes precedence for endpoint definitions
        mcp_context = self._mcp_context
        
        # Check if selected_tools are provided in context (from capability registry)
        selected_tools = context.get("selected_tools", [])
        candidates: List[EndpointDefinition] = []
        preferred_endpoints: List[EndpointDefinition] = []
        
        if selected_tools:
            LOGGER.info(f"Matching {len(selected_tools)} selected tools to endpoints: {selected_tools}")
            # Use MCP client to find endpoints by tool ID
            for tool_id in selected_tools:
                mcp_endpoints = self._mcp_client.find_endpoints_by_tool_id(tool_id)
                if mcp_endpoints:
                    LOGGER.info(f"Found {len(mcp_endpoints)} MCP endpoints for tool '{tool_id}': {[e.id for e in mcp_endpoints]}")
                    for ep in mcp_endpoints:
                        if ep not in preferred_endpoints:
                            preferred_endpoints.append(ep)
                            candidates.append(ep)
                else:
                    # Fallback to legacy matching if MCP doesn't have it
                    LOGGER.warning(f"No MCP endpoint found for tool '{tool_id}', falling back to legacy matching")
                    # Legacy matching logic (keep for backward compatibility)
                    tool_name = tool_id.replace("api_", "")
                    for legacy_ep in self._endpoints:
                        if tool_name in legacy_ep.path.lower() or tool_name in legacy_ep.description.lower():
                            # Convert legacy endpoint to MCP format (create temporary)
                            # For now, just log and continue
                            LOGGER.debug(f"Legacy endpoint match: {legacy_ep.path}")
        
        # If no candidates from selected_tools, use MCP client to find by question
        if not candidates:
            # Try to find endpoints by keywords in question
            all_endpoints = self._mcp_client.list_endpoints()
            # Simple keyword matching
            question_lower = question.lower()
            for ep in all_endpoints:
                if any(keyword in question_lower for keyword in [ep.id.replace("api_", ""), ep.title.lower()]):
                    candidates.append(ep)
                    if len(candidates) >= self._top_k:
                        break
        
        # If still no candidates, use top endpoints
        if not candidates:
            candidates = self._mcp_client.list_endpoints()[: self._top_k]

        documentation = self._select_documentation(question)
        documentation_payload = [
            {"title": snippet.title, "source": snippet.source, "content": snippet.content}
            for snippet in documentation
        ]
        # Build planning context: MCP context (endpoints.json) + orchestrator context
        planning_context = {
            "request_context": context or {},
            "documentation": documentation_payload,
            "mcp_endpoints": mcp_context.get("endpoints", []),  # Endpoint definitions from JSON
            "base_url": mcp_context.get("base_url"),  # Base URL from JSON
            "authentication": mcp_context.get("authentication", {}),  # Auth config from JSON
        }
        context_dump = json.dumps(planning_context, default=str, ensure_ascii=False)
        endpoints_text = self._format_mcp_candidates(candidates)
        
        # Enhance question when we have preferred endpoints from selected_tools
        enhanced_question = question
        if preferred_endpoints and selected_tools:
            # Add explicit guidance about which endpoints to use
            preferred_paths = [ep.path for ep in preferred_endpoints[:3]]  # Limit to top 3
            # Extract parameter info from MCP endpoint definitions
            param_info = []
            date_range_hint = ""
            
            for ep in preferred_endpoints[:3]:
                # Build parameter description from MCP definition
                param_descriptions = []
                for param_name in ep.query_params:
                    param_def = ep.parameters.get(param_name)
                    if param_def:
                        param_desc = f"{param_name} ({param_def.type}"
                        if param_def.description:
                            param_desc += f": {param_def.description}"
                        if param_def.default is not None:
                            param_desc += f", default: {param_def.default}"
                        param_desc += ")"
                        param_descriptions.append(param_desc)
                
                if param_descriptions:
                    param_info.append(f"{ep.path}: {', '.join(param_descriptions)}")
                
                # Check if endpoint uses 'n' parameter and calculate it
                if "n" in ep.query_params:
                    n_days = None
                    start_date_str = None
                    end_date_str = None
                    
                    # Try to get from sub_queries context
                    if "sub_queries" in context:
                        sub_queries = context.get("sub_queries", [])
                        for sq in sub_queries:
                            sq_context = sq.get("context", {})
                            timeframe = sq_context.get("timeframe", {})
                            if isinstance(timeframe, dict):
                                start_date_str = timeframe.get("start")
                                end_date_str = timeframe.get("end")
                                if start_date_str and end_date_str:
                                    n_days = self._mcp_client.calculate_n_from_date_range(start_date_str, end_date_str)
                                    if n_days:
                                        break
                    
                    # Fallback: extract from question
                    if n_days is None:
                        question_lower = question.lower()
                        if "last" in question_lower:
                            import re
                            days_match = re.search(r'last\s+(\d+)\s+days?', question_lower)
                            if days_match:
                                n_days = int(days_match.group(1))
                            else:
                                months_match = re.search(r'last\s+(\d+)\s+months?', question_lower)
                                if months_match:
                                    months = int(months_match.group(1))
                                    n_days = months * 30  # Approximate
                    
                    if n_days is not None:
                        date_range_hint = (
                            f"\n\nCALCULATION: For endpoints that accept 'n' parameter, "
                            f"calculate n = {n_days} days from the requested date range. "
                            f"Use query parameter: n={n_days}"
                        )
            
            if param_info:
                # Escape curly braces in param_info to avoid template variable issues
                # Need to escape all curly braces that appear in the string
                param_info_escaped = []
                for info in param_info:
                    # Escape curly braces but preserve the f-string formatting for the list
                    escaped_info = info.replace("{", "{{").replace("}", "}}")
                    param_info_escaped.append(escaped_info)
                
                # Also escape the date_range_hint
                date_range_hint_escaped = date_range_hint.replace("{", "{{").replace("}", "}}") if date_range_hint else ""
                
                # Build the enhanced question with escaped content
                param_list = "\n".join(f"  - {info}" for info in param_info_escaped)
                enhanced_question = (
                    f"{question}\n\n"
                    f"IMPORTANT: Use one of these specific API endpoints: {', '.join(preferred_paths)}. "
                    f"Do NOT construct custom paths. Use the exact endpoint paths provided above.\n"
                    f"CRITICAL: Use ONLY the parameters documented for each endpoint:\n{param_list}"
                    + (f"\n{date_range_hint_escaped}" if date_range_hint_escaped else "")
                )
            else:
                enhanced_question = (
                    f"{question}\n\n"
                    f"IMPORTANT: Use one of these specific API endpoints: {', '.join(preferred_paths)}. "
                    f"Do NOT construct custom paths. Use the exact endpoint paths provided above. "
                    f"Check the endpoint definitions for the correct parameters."
                )
        
        planner_chain = (
            self._planner_prompt.partial(
                format_instructions=self._planner_parser.get_format_instructions()
            )
            | self.llm
            | self._planner_parser
        )

        try:
            # Escape any curly braces in endpoints_text that might be interpreted as template variables
            # This prevents LangChain from trying to interpret {n}, {startDate}, etc. as template variables
            endpoints_text_escaped = endpoints_text.replace("{", "{{").replace("}", "}}")
            
            plan = planner_chain.invoke(
                {
                    "question": enhanced_question,
                    "endpoints": endpoints_text_escaped,
                    "context": context_dump,
                }
            )
            
            # Validate and auto-correct plan parameters based on endpoint documentation
            plan = self._validate_and_correct_plan(plan, preferred_endpoints, documentation, context)
            
        except ValidationError as exc:
            LOGGER.exception("Failed to parse planner output")
            raise AgentError(
                message="Planner produced invalid output.",
                type="PlanningValidationError",
                details={"errors": exc.errors()},
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Planner invocation failed")
            raise AgentError(
                message="Planner invocation failed.",
                type=type(exc).__name__,
            ) from exc

        return plan

    def _validate_and_correct_plan(
        self, 
        plan: ApiCallPlan, 
        preferred_endpoints: List[EndpointDefinition],
        documentation: List[DocumentationSnippet],
        context: Dict[str, Any]
    ) -> ApiCallPlan:
        """
        Validate the plan's parameters against MCP endpoint definitions and auto-correct common mistakes.
        """
        if not plan.make_request or not preferred_endpoints:
            return plan
        
        # Find the endpoint being used
        endpoint = None
        for ep in preferred_endpoints:
            if ep.path == plan.path or ep.path.split('?')[0] == plan.path.split('?')[0]:
                endpoint = ep
                break
        
        if not endpoint:
            return plan
        
        # Use MCP client to validate and auto-correct parameters
        query_params = plan.query_params or {}
        
        # Auto-correct parameters using MCP client
        corrected_params = self._mcp_client.auto_correct_parameters(endpoint, query_params, context)
        
        # Validate corrected parameters
        is_valid, error_msg, normalized_params = self._mcp_client.validate_parameters(endpoint, corrected_params)
        
        if not is_valid:
            plan.make_request = False
            plan.failure_reason = error_msg or "Invalid parameters for endpoint"
            LOGGER.warning(f"Plan validation failed: {plan.failure_reason}")
            return plan
        
        # Update plan with corrected and normalized parameters
        if corrected_params != query_params or normalized_params != query_params:
            LOGGER.info(
                f"Auto-corrected plan parameters for {endpoint.id}: "
                f"{query_params} -> {normalized_params}"
            )
            plan.query_params = normalized_params
        
        return plan

    def _execute_plan(
        self, plan: ApiCallPlan, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], ApiCallPlan, List[TraceEvent]]:
        base_url = self._resolve_base_url(context)
        timeout = self._resolve_timeout(context)

        client = self._http_client
        close_client = False
        if client is None:
            client = httpx.Client(timeout=timeout)
            close_client = True

        plan_to_execute = plan
        response: Optional[httpx.Response] = None
        extra_events: List[TraceEvent] = []
        attempted_token_refresh = False

        try:
            for attempt in range(1, MAX_API_CALL_ATTEMPTS + 1):
                method = plan_to_execute.method.upper()
                if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}:
                    raise AgentError(
                        message=f"Unsupported HTTP method requested: {method}",
                        type="UnsupportedMethod",
                    )

                path = self._render_path(plan_to_execute.path, plan_to_execute.path_params)
                if not path.startswith("/"):
                    path = f"/{path}"
                url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

                headers, token_used = self._build_headers(context, plan_to_execute.headers)
                if not close_client:
                    merged_headers = dict(client.headers)
                    merged_headers.update(headers)
                    request_headers = merged_headers
                else:
                    request_headers = headers

                params = {
                    str(key): str(value) for key, value in (plan_to_execute.query_params or {}).items()
                }
                json_body = plan_to_execute.json_body if method in {"POST", "PUT", "PATCH"} else None

                try:
                    response = client.request(
                        method,
                        url,
                        params=params or None,
                        json=json_body,
                        headers=request_headers,
                    )
                except httpx.RequestError as exc:
                    raise AgentError(
                        message=f"HTTP request failed: {exc}",
                        type="HttpRequestError",
                        details={"url": str(exc.request.url) if exc.request else url},
                    ) from exc

                if response.status_code == 401:
                    should_retry, event = self._handle_unauthorised(
                        token_used, context, attempted_token_refresh
                    )
                    if event:
                        extra_events.append(event)
                    if should_retry:
                        attempted_token_refresh = True
                        continue

                if response.status_code == 400:
                    adjusted_plan, event = self._auto_adjust_plan_for_missing_params(
                        plan_to_execute, response, context
                    )
                    if adjusted_plan is not None:
                        plan_to_execute = adjusted_plan
                        if event:
                            extra_events.append(event)
                        continue

                break
            else:
                LOGGER.warning(
                    "Max API call attempts reached without successful response (last status: %s)",
                    response.status_code if response else "n/a",
                )

            if response is None:
                raise AgentError(
                    message="Failed to execute API request.",
                    type="UnknownExecutionError",
                )

            body_snippet = response.text[:2000]
            json_snippet = None
            try:
                json_snippet = response.json()
            except ValueError:
                json_snippet = None

            elapsed_ms = None
            if response.elapsed is not None:
                elapsed_ms = response.elapsed.total_seconds() * 1000

            request_headers_sanitised = {
                key: value
                for key, value in response.request.headers.items()
                if key.lower() != "authorization"
            }

            execution = {
                "url": str(response.request.url),
                "method": response.request.method,
                "status_code": response.status_code,
                "reason_phrase": response.reason_phrase,
                "elapsed_ms": elapsed_ms,
                "body_snippet": body_snippet,
                "json_snippet": json_snippet,
                "request_headers": request_headers_sanitised,
            }
            return execution, plan_to_execute, extra_events
        finally:
            if close_client:
                client.close()

    def _summarise_execution(
        self, question: str, plan: ApiCallPlan, execution: Dict[str, Any]
    ) -> str:
        json_body = execution.get("json_snippet")
        json_text = json.dumps(json_body, ensure_ascii=False)[:2000] if json_body is not None else ""
        body_snippet = execution.get("body_snippet") or ""

        summary_chain = self._summary_prompt | self.llm | RunnableLambda(lambda msg: msg.content)
        try:
            summary = summary_chain.invoke(
                {
                    "question": question,
                    "method": execution["method"],
                    "url": execution["url"],
                    "status_code": execution["status_code"],
                    "reason": plan.reason,
                    "body_snippet": body_snippet,
                    "json_snippet": json_text or "<no json>",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to summarise API response")
            return (
                f"HTTP {execution['status_code']} {execution.get('reason_phrase', '')}.\n"
                f"Response snippet: {body_snippet[:500]}"
            )

        return str(summary)

    def _select_endpoints(self, question: str) -> List[ApiEndpointSpec]:
        question_tokens = _tokenize(question)
        if not question_tokens:
            return self._endpoints[: self._top_k]

        token_counts: Dict[str, int] = {}
        for token in question_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        scored: List[tuple[int, ApiEndpointSpec]] = []
        for endpoint in self._endpoints:
            score = sum(token_counts.get(token, 0) for token in endpoint.tokens)
            if score > 0:
                scored.append((score, endpoint))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [endpoint for _, endpoint in scored[: self._top_k]]

    def _format_mcp_candidates(self, endpoints: List[EndpointDefinition]) -> str:
        """Format MCP endpoint definitions for the planner."""
        lines: List[str] = []
        for index, endpoint in enumerate(endpoints, start=1):
            # Build parameter description from MCP definition
            param_descriptions = []
            for param_name in endpoint.query_params:
                param_def = endpoint.parameters.get(param_name)
                if param_def:
                    param_desc = f"{param_name} ({param_def.type}"
                    if param_def.description:
                        param_desc += f": {param_def.description}"
                    if param_def.default is not None:
                        param_desc += f", default: {param_def.default}"
                    param_desc += ")"
                    param_descriptions.append(param_desc)
            
            # Format the endpoint line with prominent parameter info
            if param_descriptions:
                params_text = ", ".join(param_descriptions)
                lines.append(
                    f"{index}. {endpoint.method} {endpoint.path}\n"
                    f"   Title: {endpoint.title}\n"
                    f"   Description: {endpoint.description}\n"
                    f"   PARAMETERS: {params_text}\n"
                    f"   ID: {endpoint.id}"
                )
            else:
                lines.append(
                    f"{index}. {endpoint.method} {endpoint.path}\n"
                    f"   Title: {endpoint.title}\n"
                    f"   Description: {endpoint.description}\n"
                    f"   PARAMETERS: None\n"
                    f"   ID: {endpoint.id}"
                )
        return "\n".join(lines) if lines else "No candidate endpoints available."
    
    @staticmethod
    def _format_candidates(endpoints: Iterable[ApiEndpointSpec], documentation: List[DocumentationSnippet] = None) -> str:
        lines: List[str] = []
        for index, endpoint in enumerate(endpoints, start=1):
            # Try to find parameter information from documentation
            param_info = ""
            if documentation:
                # Find documentation snippet that matches this endpoint
                endpoint_path_base = endpoint.path.split('?')[0]  # Remove query params for matching
                for doc in documentation:
                    if endpoint_path_base.lower() in doc.content.lower() or endpoint.method.lower() + " " + endpoint_path_base.lower() in doc.content.lower():
                        # Extract parameter information
                        doc_content = doc.content
                        if "**Parameters:**" in doc_content:
                            param_section = doc_content.split("**Parameters:**")[-1].split("**")[0].strip()
                            if param_section:
                                param_info = f" | Parameters: {param_section[:200]}"  # Limit length
                                break
                        elif "Parameters:" in doc_content and "**Parameters:**" not in doc_content:
                            # Try to extract from non-bold Parameters line
                            for line in doc_content.split('\n'):
                                if line.strip().startswith("Parameters:") or "parameter" in line.lower() and "accepts" in line.lower():
                                    param_info = f" | {line.strip()[:200]}"
                                    break
                            if param_info:
                                break
            
            # Format the endpoint line with prominent parameter info
            if param_info:
                lines.append(
                    f"{index}. {endpoint.method} {endpoint.path}\n"
                    f"   Description: {endpoint.description}\n"
                    f"   ⚠️  PARAMETERS: {param_info}\n"
                    f"   Source: {endpoint.source}"
                )
            else:
                lines.append(
                    f"{index}. {endpoint.method} {endpoint.path} — {endpoint.description} (source: {endpoint.source})"
                )
        return "\n".join(lines) if lines else "No candidate endpoints available."

    @staticmethod
    def _render_path(path: str, path_params: Dict[str, Any]) -> str:
        rendered = path
        for key, value in (path_params or {}).items():
            rendered = rendered.replace(f":{key}", str(value))
            rendered = rendered.replace(f"{{{key}}}", str(value))
        if ":" in rendered or "{" in rendered or "}" in rendered:
            raise AgentError(
                message=f"Unable to resolve all path parameters for path '{path}'.",
                type="PathResolutionError",
                details={"provided_params": path_params},
            )
        return rendered

    def _resolve_base_url(self, context: Dict[str, Any]) -> str:
        """Resolve base URL, prioritizing MCP client config from endpoints.json."""
        # First try MCP client (from endpoints.json) - this is the source of truth
        mcp_base_url = self._mcp_client.get_base_url()
        if mcp_base_url:
            return mcp_base_url.rstrip("/")
        
        # Fallback to context or environment
        candidates = [
            context.get("api_base_url"),
            context.get("base_url"),
            os.getenv("API_AGENT_BASE_URL"),
            DEFAULT_BASE_URL,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.rstrip("/")
        return DEFAULT_BASE_URL

    @staticmethod
    def _resolve_timeout(context: Dict[str, Any]) -> float:
        timeout_candidates = [
            context.get("api_timeout"),
            os.getenv("API_AGENT_TIMEOUT"),
            DEFAULT_TIMEOUT_SECONDS,
        ]
        for candidate in timeout_candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                LOGGER.debug("Ignoring invalid timeout value: %s", candidate)
        return DEFAULT_TIMEOUT_SECONDS

    def _build_headers(
        self, context: Dict[str, Any], plan_headers: Dict[str, str]
    ) -> Tuple[Dict[str, str], Optional[str]]:
        headers: Dict[str, str] = {"Accept": "application/json"}

        context_headers = context.get("api_headers")
        if isinstance(context_headers, dict):
            for key, value in context_headers.items():
                if value is not None:
                    headers[str(key)] = str(value)

        for key, value in plan_headers.items():
            headers[key] = value

        bearer = self._get_firebase_token(context, context_headers=headers)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"

        return headers, bearer

    def _handle_unauthorised(
        self,
        token_used: Optional[str],
        context: Dict[str, Any],
        attempted_refresh: bool,
    ) -> Tuple[bool, Optional[TraceEvent]]:
        if not token_used or attempted_refresh:
            return False, None

        LOGGER.info("Received 401 response; attempting Firebase token refresh.")
        self._clear_token_cache()

        refreshed_token = self._get_firebase_token(context, force_refresh=True)
        if refreshed_token and refreshed_token != token_used:
            event = TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent="api_docs",  # type: ignore[assignment]
                message="Refreshed Firebase token after 401 response",
            )
            return True, event

        LOGGER.warning("Token refresh failed or returned the same token; not retrying request.")
        return False, None

    def _auto_adjust_plan_for_missing_params(
        self, plan: ApiCallPlan, response: httpx.Response, context: Dict[str, Any]
    ) -> Tuple[Optional[ApiCallPlan], Optional[TraceEvent]]:
        try:
            payload = response.json()
            body_text = json.dumps(payload)
        except Exception:
            body_text = response.text or ""

        if "startDateTime" not in body_text or "endDateTime" not in body_text:
            return None, None

        existing_params = dict(plan.query_params or {})
        if "startDateTime" in existing_params and "endDateTime" in existing_params:
            return None, None

        start_hour, end_hour = self._derive_default_hour_range(context)
        existing_params.setdefault("startDateTime", start_hour)
        existing_params.setdefault("endDateTime", end_hour)

        updated_plan = plan.model_copy(update={"query_params": existing_params})
        event = TraceEvent(
            event_type=TraceEventType.MESSAGE,
            agent="api_docs",  # type: ignore[assignment]
            message="Auto-filled startDateTime and endDateTime query parameters based on API error response",
            data={"startDateTime": start_hour, "endDateTime": end_hour},
        )
        return updated_plan, event

    def _select_documentation(self, question: str) -> List[DocumentationSnippet]:
        question_tokens = _tokenize(question)
        if not question_tokens:
            return self._doc_contexts[: self._top_k]

        token_counts: Dict[str, int] = {}
        for token in question_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        scored: List[tuple[int, DocumentationSnippet]] = []
        for snippet in self._doc_contexts:
            score = sum(token_counts.get(token, 0) for token in snippet.tokens)
            if score > 0:
                scored.append((score, snippet))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [snippet for _, snippet in scored[: self._top_k]]

    @staticmethod
    def _should_force_attempt(plan: ApiCallPlan) -> bool:
        if plan.make_request:
            return False
        if plan.method.upper() not in {"GET", "POST"}:
            return False
        reason_fragments = " ".join(
            filter(
                None,
                [
                    plan.reason.lower() if plan.reason else "",
                    plan.failure_reason.lower() if plan.failure_reason else "",
                ],
            )
        )
        keywords = [
            "documentation",
            "not specify",
            "cannot invent",
            "unspecified",
            "unknown parameter",
        ]
        return any(keyword in reason_fragments for keyword in keywords)

    @staticmethod
    def _derive_default_hour_range(context: Dict[str, Any]) -> Tuple[str, str]:
        tz_name = context.get("timezone") or os.getenv("API_AGENT_TIMEZONE") or "Asia/Kolkata"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # pragma: no cover - fallback for invalid timezone
            LOGGER.debug("Invalid timezone '%s'; defaulting to UTC for hour range.", tz_name)
            tz = timezone.utc

        now = datetime.now(tz)
        start = datetime(now.year, now.month, now.day, 0, tzinfo=tz)
        end = datetime(now.year, now.month, now.day, 23, tzinfo=tz)
        return start.strftime("%Y-%m-%d %H"), end.strftime("%Y-%m-%d %H")

    @staticmethod
    def _load_context_sources(location: Path) -> tuple[List[ApiEndpointSpec], List[DocumentationSnippet]]:
        if not location.exists():
            LOGGER.warning("API docs context file missing: %s", location)
            return [], []

        with location.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        sections = payload.get("sections", [])
        endpoints: List[ApiEndpointSpec] = []
        for section in sections:
            section_id = section.get("id", "")
            title = section.get("title", "")
            source = section.get("source", "")
            content = section.get("content", "") or ""
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                endpoint = ApiEndpointSpec.from_line(section_id, title, source, line)
                if endpoint is not None:
                    # Validate endpoint path before adding
                    if endpoint.path and endpoint.path.startswith("/") and len(endpoint.path) >= 2:
                        endpoints.append(endpoint)
                    else:
                        LOGGER.warning(f"Skipping invalid endpoint path: '{endpoint.path}' from line: {line[:100]}")

        contexts_section = payload.get("contexts", [])
        doc_contexts = [DocumentationSnippet.from_mapping(entry) for entry in contexts_section]

        LOGGER.info(
            "Loaded %d API endpoints and %d documentation snippets from context",
            len(endpoints),
            len(doc_contexts),
        )
        return endpoints, doc_contexts

    @staticmethod
    def _resolve_firebase_config(
        context: Dict[str, Any],
        *,
        context_headers: Optional[Dict[str, str]] = None,
    ) -> FirebaseAuthConfig:
        direct_token = (
            context.get("api_bearer_token")
            or context.get("firebase_id_token")
            or os.getenv("API_AGENT_BEARER_TOKEN")
            or os.getenv("FIREBASE_ID_TOKEN")
        )

        token_file_value = context.get("api_token_file") or os.getenv("FIREBASE_TOKEN_FILE")
        token_file = Path(token_file_value) if token_file_value else None

        config = FirebaseAuthConfig(
            direct_token=direct_token,
            token_file=token_file,

            web_api_key=context.get("firebase_web_api_key") or os.getenv("FIREBASE_WEB_API_KEY"),
            email=context.get("firebase_email") or os.getenv("FIREBASE_EMAIL"),
            password=context.get("firebase_password") or os.getenv("FIREBASE_PASSWORD"),
            context_headers=dict(context_headers or {}),
        )
        return config

    @staticmethod
    def _load_token_from_file(path: Path) -> Optional[str]:
        try:
            if not path.exists():
                return None
            token = path.read_text(encoding="utf-8").strip()
            return token or None
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read Firebase token file %s: %s", path, exc)
            return None

    @staticmethod
    def _cached_token_valid() -> bool:
        token = _TOKEN_CACHE.get("token")
        expires_at = _TOKEN_CACHE.get("expires_at")
        if not token:
            return False
        if expires_at is None:
            return True
        return datetime.now(timezone.utc) < expires_at

    @staticmethod
    def _set_token_cache(token: str, expires_in_seconds: Optional[int]) -> None:
        expires_at = None
        if expires_in_seconds:
            expiry_seconds = max(0, expires_in_seconds - TOKEN_REFRESH_GRACE_SECONDS)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
        _TOKEN_CACHE["token"] = token
        _TOKEN_CACHE["expires_at"] = expires_at

    @staticmethod
    def _clear_token_cache() -> None:
        _TOKEN_CACHE["token"] = None
        _TOKEN_CACHE["expires_at"] = None

    def _generate_token_via_password(self, config: FirebaseAuthConfig) -> Optional[Tuple[str, Optional[int]]]:
        if not config.web_api_key or not config.email or not config.password:
            return None

        url = f"{FIREBASE_ID_ENDPOINT}?key={config.web_api_key}"
        payload = {
            "email": config.email,
            "password": config.password,
            "returnSecureToken": True,
        }

        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            token = data.get("idToken")
            expires_in = data.get("expiresIn")
            if not token:
                LOGGER.error("Firebase authentication succeeded but no idToken returned.")
                return None
            expires_in_seconds = int(expires_in) if expires_in is not None else None
            LOGGER.info("Obtained Firebase ID token via email/password authentication.")
            return token, expires_in_seconds
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to generate Firebase token via email/password: %s", exc)
            return None

    def _get_firebase_token(
        self,
        context: Dict[str, Any],
        *,
        context_headers: Optional[Dict[str, str]] = None,
        force_refresh: bool = False,
    ) -> Optional[str]:
        config = self._resolve_firebase_config(context, context_headers=context_headers)

        if config.direct_token:
            return config.direct_token

        if not force_refresh and self._cached_token_valid():
            cached_token = _TOKEN_CACHE.get("token")
            if cached_token:
                return cached_token

        if config.token_file and config.token_file.exists():
            token = self._load_token_from_file(config.token_file)
            if token:
                self._set_token_cache(token, None)
                return token

        generated = self._generate_token_via_password(config)
        if generated:
            token, expires_in = generated
            self._set_token_cache(token, expires_in)
            return token

        return None


def compile_api_docs_agent() -> ApiDocsAgent:
    """Factory to align with orchestrator patterns."""
    return ApiDocsAgent()
