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

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTEXT_PATH = Path(__file__).resolve().parents[2] / "Docs" / "api_docs_context.yaml"
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
        method_match = re.search(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", line, re.IGNORECASE)
        path_match = re.search(r"`([^`]+)`", line)
        if not path_match:
            return None

        method = method_match.group(1).upper() if method_match else "GET"
        path = path_match.group(1).strip()
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
        top_k: int = 5,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        resources = get_resources()
        self.llm: AzureChatOpenAI = llm or resources.llm
        self._context_path = Path(context_path) if context_path else DEFAULT_CONTEXT_PATH
        self._top_k = top_k
        self._http_client = http_client
        self._endpoints, self._doc_contexts = self._load_context_sources(self._context_path)

        self._planner_parser = PydanticOutputParser(pydantic_object=ApiCallPlan)
        self._planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an integration specialist tasked with planning exactly one HTTP API call. "
                        "Choose an endpoint only from the provided candidates. "
                        "Do not invent paths or payload fields. "
                        "If you lack mandatory parameters, set make_request to false and explain why. "
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
        candidates = self._select_endpoints(question)
        if not candidates:
            candidates = self._endpoints[: self._top_k]

        documentation = self._select_documentation(question)
        documentation_payload = [
            {"title": snippet.title, "source": snippet.source, "content": snippet.content}
            for snippet in documentation
        ]
        planning_context = {
            "request_context": context or {},
            "documentation": documentation_payload,
        }
        context_dump = json.dumps(planning_context, default=str, ensure_ascii=False)
        endpoints_text = self._format_candidates(candidates)
        planner_chain = (
            self._planner_prompt.partial(
                format_instructions=self._planner_parser.get_format_instructions()
            )
            | self.llm
            | self._planner_parser
        )

        try:
            plan = planner_chain.invoke(
                {
                    "question": question,
                    "endpoints": endpoints_text,
                    "context": context_dump,
                }
            )
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

    @staticmethod
    def _format_candidates(endpoints: Iterable[ApiEndpointSpec]) -> str:
        lines: List[str] = []
        for index, endpoint in enumerate(endpoints, start=1):
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

    @staticmethod
    def _resolve_base_url(context: Dict[str, Any]) -> str:
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
                    endpoints.append(endpoint)

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
