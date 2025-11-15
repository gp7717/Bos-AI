"""MCP client for API endpoint definitions and execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINTS_PATH = Path(__file__).parent / "endpoints.json"


class EndpointParameter(BaseModel):
    """Parameter definition for an API endpoint."""

    type: str = Field(description="Parameter type (integer, string, date, etc.)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")
    example: Optional[Any] = Field(default=None, description="Example value")
    format: Optional[str] = Field(default=None, description="Format hint (e.g., 'date' for YYYY-MM-DD)")


class EndpointDefinition(BaseModel):
    """Definition of an API endpoint."""

    id: str = Field(description="Unique endpoint identifier")
    method: str = Field(description="HTTP method (GET, POST, etc.)")
    path: str = Field(description="API path (e.g., /api/net_profit_daily)")
    title: str = Field(description="Human-readable title")
    description: str = Field(description="Endpoint description")
    parameters: Dict[str, EndpointParameter] = Field(
        default_factory=dict, description="Available parameters"
    )
    query_params: List[str] = Field(default_factory=list, description="Query parameter names")
    path_params: List[str] = Field(default_factory=list, description="Path parameter names")
    request_body: Optional[Dict[str, Any]] = Field(default=None, description="Request body schema")
    response: Dict[str, Any] = Field(description="Response schema")
    authentication: str = Field(default="bearer", description="Authentication type")
    timeout_seconds: float = Field(default=30.0, description="Request timeout in seconds")


class EndpointsConfig(BaseModel):
    """Configuration for API endpoints."""

    version: str = Field(description="Configuration version")
    base_url: str = Field(description="Base URL for all endpoints")
    endpoints: List[EndpointDefinition] = Field(description="List of endpoint definitions")
    authentication: Dict[str, Any] = Field(default_factory=dict, description="Authentication config")


class ApiMCPClient:
    """MCP client for API endpoint management and execution."""

    def __init__(self, endpoints_path: Optional[Path] = None):
        """Initialize the API MCP client."""
        self.endpoints_path = endpoints_path or DEFAULT_ENDPOINTS_PATH
        self.config: Optional[EndpointsConfig] = None
        self._load_endpoints()

    def _load_endpoints(self) -> None:
        """Load endpoint definitions from JSON file."""
        if not self.endpoints_path.exists():
            logger.warning(f"Endpoints file not found: {self.endpoints_path}")
            self.config = EndpointsConfig(
                version="1.0",
                base_url="https://dashbackend-a3cbagbzg0hydhen.centralindia-01.azurewebsites.net",
                endpoints=[],
            )
            return

        try:
            with self.endpoints_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.config = EndpointsConfig(**data)
            logger.info(f"Loaded {len(self.config.endpoints)} API endpoints from {self.endpoints_path}")
        except Exception as e:
            logger.error(f"Failed to load endpoints from {self.endpoints_path}: {e}", exc_info=True)
            raise

    def get_endpoint(self, endpoint_id: str) -> Optional[EndpointDefinition]:
        """Get endpoint definition by ID."""
        if not self.config:
            return None
        return next((ep for ep in self.config.endpoints if ep.id == endpoint_id), None)

    def find_endpoints_by_path(self, path: str) -> List[EndpointDefinition]:
        """Find endpoints matching a path pattern."""
        if not self.config:
            return []
        path_base = path.split("?")[0]
        return [ep for ep in self.config.endpoints if ep.path == path_base or ep.path == path]

    def find_endpoints_by_tool_id(self, tool_id: str) -> List[EndpointDefinition]:
        """Find endpoints matching a tool ID (e.g., 'api_net_profit_daily')."""
        if not self.config:
            return []
        # Tool ID format: "api_net_profit_daily" -> endpoint ID should match
        return [ep for ep in self.config.endpoints if ep.id == tool_id]

    def list_endpoints(self) -> List[EndpointDefinition]:
        """List all available endpoints."""
        if not self.config:
            return []
        return self.config.endpoints

    def get_base_url(self) -> str:
        """Get the base URL for API calls."""
        if not self.config:
            return "https://dashbackend-a3cbagbzg0hydhen.centralindia-01.azurewebsites.net"
        return self.config.base_url

    def get_authentication_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        if not self.config:
            return {}
        return self.config.authentication

    def get_endpoint_context(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get full context for an endpoint including all metadata."""
        endpoint = self.get_endpoint(endpoint_id)
        if not endpoint:
            return None
        
        return {
            "id": endpoint.id,
            "method": endpoint.method,
            "path": endpoint.path,
            "title": endpoint.title,
            "description": endpoint.description,
            "parameters": {
                name: {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default,
                    "example": param.example,
                    "format": param.format,
                }
                for name, param in endpoint.parameters.items()
            },
            "query_params": endpoint.query_params,
            "path_params": endpoint.path_params,
            "response": endpoint.response,
            "authentication": endpoint.authentication,
            "timeout_seconds": endpoint.timeout_seconds,
        }

    def get_all_endpoints_context(self) -> Dict[str, Any]:
        """Get context for all endpoints from endpoints.json."""
        if not self.config:
            return {}
        
        endpoints_context = []
        for ep in self.config.endpoints:
            ep_ctx = self.get_endpoint_context(ep.id)
            if ep_ctx:
                endpoints_context.append(ep_ctx)
        
        return {
            "version": self.config.version,
            "base_url": self.config.base_url,
            "authentication": self.config.authentication,
            "endpoints": endpoints_context,
        }

    def validate_parameters(
        self, endpoint: EndpointDefinition, query_params: Dict[str, Any]
    ) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate and normalize parameters for an endpoint.
        
        Returns:
            (is_valid, error_message, normalized_params)
        """
        normalized = {}
        
        # Check for invalid parameters
        invalid_params = []
        for param_name in query_params:
            if param_name not in endpoint.query_params and param_name not in endpoint.path_params:
                invalid_params.append(param_name)
        
        if invalid_params:
            return (
                False,
                f"Invalid parameters for endpoint {endpoint.id}: {invalid_params}. "
                f"Valid parameters: {endpoint.query_params}",
                {},
            )
        
        # Validate and normalize each parameter
        for param_name in endpoint.query_params:
            param_def = endpoint.parameters.get(param_name)
            if not param_def:
                continue
            
            value = query_params.get(param_name)
            
            # Use default if not provided
            if value is None and param_def.default is not None:
                value = param_def.default
                normalized[param_name] = value
            elif value is not None:
                # Type conversion
                if param_def.type == "integer":
                    try:
                        normalized[param_name] = int(value)
                    except (ValueError, TypeError):
                        return (
                            False,
                            f"Parameter '{param_name}' must be an integer, got: {value}",
                            {},
                        )
                elif param_def.type == "string":
                    normalized[param_name] = str(value)
                else:
                    normalized[param_name] = value
        
        return True, None, normalized

    def calculate_n_from_date_range(
        self, start_date: str, end_date: str
    ) -> Optional[int]:
        """Calculate number of days from date range."""
        try:
            from datetime import datetime
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            return (end - start).days + 1  # +1 to include both dates
        except Exception as e:
            logger.warning(f"Failed to calculate days from date range: {e}")
            return None

    def auto_correct_parameters(
        self,
        endpoint: EndpointDefinition,
        query_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Auto-correct parameters based on endpoint definition.
        
        For endpoints that only accept 'n', convert date ranges to 'n'.
        """
        corrected = {}
        
        # Check if endpoint only accepts 'n' parameter
        if "n" in endpoint.query_params and len(endpoint.query_params) == 1:
            # This endpoint only accepts 'n'
            # Check if wrong parameters are provided
            invalid_params = ["start_date", "end_date", "startDate", "endDate", "granularity"]
            has_invalid = any(p in query_params for p in invalid_params)
            has_n = "n" in query_params
            
            if has_invalid and not has_n:
                # Try to calculate 'n' from date range
                start_date = None
                end_date = None
                
                # Get dates from invalid params
                for param in ["start_date", "startDate", "start"]:
                    if param in query_params:
                        start_date = str(query_params[param])
                        break
                for param in ["end_date", "endDate", "end"]:
                    if param in query_params:
                        end_date = str(query_params[param])
                        break
                
                # Try to get from context if not in params
                if not start_date or not end_date:
                    if context and "sub_queries" in context:
                        sub_queries = context.get("sub_queries", [])
                        for sq in sub_queries:
                            sq_context = sq.get("context", {})
                            timeframe = sq_context.get("timeframe", {})
                            if isinstance(timeframe, dict):
                                start_date = timeframe.get("start")
                                end_date = timeframe.get("end")
                                if start_date and end_date:
                                    break
                
                # Calculate n
                if start_date and end_date:
                    n_days = self.calculate_n_from_date_range(start_date, end_date)
                    if n_days:
                        corrected["n"] = n_days
                        logger.info(f"Auto-corrected: converted date range ({start_date} to {end_date}) to n={n_days}")
                else:
                    # Try to extract from question
                    question = context.get("focused_query", "") if context else ""
                    if "last" in question.lower():
                        import re
                        days_match = re.search(r"last\s+(\d+)\s+days?", question.lower())
                        if days_match:
                            corrected["n"] = int(days_match.group(1))
                        else:
                            months_match = re.search(r"last\s+(\d+)\s+months?", question.lower())
                            if months_match:
                                months = int(months_match.group(1))
                                corrected["n"] = months * 30  # Approximate
            elif has_n:
                corrected["n"] = query_params["n"]
        else:
            # Endpoint accepts other parameters, use as-is (after validation)
            corrected = query_params.copy()
        
        return corrected

    async def execute_endpoint(
        self,
        endpoint: EndpointDefinition,
        query_params: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute an API endpoint call.
        
        Returns:
            {
                "success": bool,
                "status_code": int,
                "data": Any,
                "error": Optional[str],
                "url": str
            }
        """
        query_params = query_params or {}
        path_params = path_params or {}
        headers = headers or {}
        timeout = timeout or endpoint.timeout_seconds
        
        # Build URL
        base_url = self.get_base_url()
        path = endpoint.path
        
        # Replace path parameters
        for param_name in endpoint.path_params:
            if param_name in path_params:
                path = path.replace(f":{param_name}", str(path_params[param_name]))
                path = path.replace(f"{{{param_name}}}", str(path_params[param_name]))
        
        url = f"{base_url.rstrip('/')}{path}"
        
        # Validate and normalize parameters
        is_valid, error_msg, normalized_params = self.validate_parameters(endpoint, query_params)
        if not is_valid:
            return {
                "success": False,
                "status_code": 0,
                "data": None,
                "error": error_msg,
                "url": url,
            }
        
        # Add normalized query params to URL
        if normalized_params:
            from urllib.parse import urlencode
            query_string = urlencode(normalized_params)
            url = f"{url}?{query_string}"
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=endpoint.method,
                    url=url,
                    headers=headers,
                )
                
                data = None
                try:
                    data = response.json()
                except Exception:
                    data = response.text
                
                return {
                    "success": response.is_success,
                    "status_code": response.status_code,
                    "data": data,
                    "error": None if response.is_success else f"HTTP {response.status_code}: {response.reason_phrase}",
                    "url": url,
                    "headers": dict(response.headers),
                }
        except Exception as e:
            logger.error(f"Failed to execute endpoint {endpoint.id}: {e}", exc_info=True)
            return {
                "success": False,
                "status_code": 0,
                "data": None,
                "error": str(e),
                "url": url,
            }

