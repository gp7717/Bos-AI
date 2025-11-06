"""Router Agent V2 - LangChain-based intent classification and slot extraction."""
import json
from typing import Dict, Any, Optional, Union
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.schemas import (
    TaskSpec, Intent, TimeRange,
    RouterAgentInput, RouterAgentOutput
)
from app.config.settings import settings
from app.services.metric_dictionary import metric_dictionary
from app.config.logging_config import get_logger
from app.core.base import BaseAgent, AgentMetadata, AgentType
from datetime import datetime, timedelta
import pytz
import re

logger = get_logger(__name__)


class RouterAgentV2(BaseAgent[RouterAgentInput, RouterAgentOutput]):
    """LangChain-based router agent for intent classification and slot extraction."""
    
    def __init__(self):
        """Initialize router agent with LangChain."""
        metadata = AgentMetadata(
            agent_id="router_v2",
            agent_type=AgentType.ROUTER,
            name="Router Agent V2",
            description="LangChain-based intent classification and slot extraction",
            version="2.0.0",
            capabilities=["intent_classification", "slot_extraction", "time_parsing"]
        )
        super().__init__(metadata, RouterAgentInput, RouterAgentOutput)
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment_name,
            temperature=settings.openai_temperature,
        )
        
        # Note: Using single braces since we're using SystemMessage directly (not ChatPromptTemplate)
        self.system_prompt = """You are an intent classification and slot extraction agent for an analytics Q&A system.

Your task is to parse user queries and extract:
1. Intent: One of: analytics.query, data.export, diagnostics, meta.help
2. Metrics: List of metrics requested (extract any metrics mentioned in the query)
3. Entities: Extract any entities mentioned in the query (dimensions, filters, categories, etc.)
4. Time: Time range - MUST extract the EXACT time range mentioned in the query
5. Filters: Additional filters (top_n, status, type, etc.)

CRITICAL: You MUST return ONLY valid JSON. Do not include any explanatory text, markdown, or comments before or after the JSON.

Return JSON with this structure:
{
  "intent": "analytics.query",
  "metrics": ["roas"],
  "entities": {},
  "time": {"range": "last_week", "tz": "Asia/Kolkata"},
  "filters": []
}

TIME RANGE EXTRACTION RULES:
- For "last 7 days" or "last week" -> use {"range": "last_7_days"} or {"range": "last_week"}
- For "last 30 days" or "last month" -> use {"range": "last_30_days"} or {"range": "last_month"}
- For specific dates -> use {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "tz": "Asia/Kolkata"}
- For relative ranges like "last N days" -> use {"range": "last_N_days"} where N is the number
- For "today" -> use {"range": "today", "tz": "Asia/Kolkata"}
- If no time range is mentioned, use {"range": null} - DO NOT assume a default
- ALWAYS extract the EXACT time range from the query - do not use hardcoded defaults

METRIC EXTRACTION RULES:
- Extract ONLY the metrics explicitly mentioned in the query
- For "count of customers" -> extract metric: ["customer_count"] or ["customers"] 
- For "sales" or "revenue" -> extract metric: ["sales"] or ["revenue"] ONLY if mentioned
- DO NOT infer metrics that aren't mentioned (e.g., don't add "sales" if query only mentions "customers")
- If query asks for a count, extract appropriate count metric (e.g., "customer_count", "order_count")
- If no metrics are mentioned, use empty array: []

ENTITY EXTRACTION RULES:
- Extract entities based on what the query actually mentions
- For "count of customers" -> entities: {"entity_type": "customer"} or similar
- DO NOT assume entities based on metrics (e.g., don't add "product" entity just because "sales" metric exists)
- Extract dimensions, filters, categories, or any entities explicitly mentioned

IMPORTANT RULES:
- filters MUST be an array/list, even if empty
- For top_n queries, use filters: [{"top_n": 5}]
- Return ONLY the JSON object, nothing else
- Extract entities mentioned in the query (any dimensions, categories, or filters mentioned)
- For queries asking for breakdown by dimension (e.g., "top N products", "by SKU", "by campaign"), extract the dimension as an entity
- Entity keys should be descriptive (e.g., "product", "channel", "geo", "campaign", "category", "customer", etc.)
- Entity values should be the actual value or dimension name mentioned in the query

Examples:
- "What was ROAS last week?" -> {"intent": "analytics.query", "metrics": ["roas"], "entities": {}, "time": {"range": "last_week", "tz": "Asia/Kolkata"}, "filters": []}
- "How many orders were cancelled in the last 7 days?" -> {"intent": "analytics.query", "metrics": ["orders"], "entities": {"status": "cancelled"}, "time": {"range": "last_7_days", "tz": "Asia/Kolkata"}, "filters": []}
- "Count of customers that came today?" -> {"intent": "analytics.query", "metrics": ["customer_count"], "entities": {"entity_type": "customer"}, "time": {"range": "today", "tz": "Asia/Kolkata"}, "filters": []}
- "Top 5 products by net sales" -> {"intent": "analytics.query", "metrics": ["net_sales"], "entities": {"product": "product"}, "time": {"range": null}, "filters": [{"top_n": 5}]}
"""
        
        logger.info(f"✅ [ROUTER_V2] Router Agent V2 initialized")
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to fix common formatting issues."""
        import re
        
        # Fix double braces (common when LLM sees escaped braces in prompt)
        # Replace {{ with { and }} with }
        json_str = json_str.replace('{{', '{').replace('}}', '}')
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix single quotes to double quotes (common LLM mistake)
        # But be careful not to break strings that legitimately contain quotes
        # Only replace at the start/end of property names and string values
        json_str = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', json_str)  # Property names
        json_str = re.sub(r':\s*\'([^\']*)\'(\s*[,}])', r': "\1"\2', json_str)  # String values
        
        return json_str.strip()
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks and extra text."""
        import re
        
        # Clean content - remove leading/trailing whitespace
        content = content.strip()
        
        # First, try to extract from markdown code blocks (most reliable)
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json {...} ```
            r'```\s*(\{.*?\})\s*```',      # ``` {...} ```
            r'`(\{.*?\})`',                # `{...}`
        ]
        
        for pattern in code_block_patterns:
            code_block_match = re.search(pattern, content, re.DOTALL)
            if code_block_match:
                try:
                    json_str = code_block_match.group(1).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Try to find the first complete JSON object using balanced brace matching
        brace_count = 0
        start_idx = -1
        found_objects = []
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found a complete JSON object
                    json_str = content[start_idx:i+1].strip()
                    # Try to clean up common issues
                    json_str = self._clean_json_string(json_str)
                    try:
                        parsed = json.loads(json_str)
                        # Validate it's a dict with expected keys
                        if isinstance(parsed, dict):
                            found_objects.append((start_idx, i+1, parsed))
                    except json.JSONDecodeError as e:
                        logger.debug(f"🔍 [ROUTER_V2] Failed to parse JSON object at {start_idx}-{i+1}: {str(e)}")
                    start_idx = -1
        
        # If we found valid JSON objects, return the first one
        if found_objects:
            logger.debug(f"🔍 [ROUTER_V2] Found {len(found_objects)} JSON object(s), using first")
            return found_objects[0][2]
        
        # Try to find JSON object using regex as fallback
        # Look for patterns that start with { and might contain JSON
        # This is less reliable but can catch some edge cases
        json_candidate_patterns = [
            r'\{[^{}]*"intent"[^{}]*\}',  # Simple pattern looking for "intent" key
        ]
        
        for pattern in json_candidate_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                # Try to expand the match to include full JSON object
                start = match.start()
                end = match.end()
                # Try expanding to find complete object
                brace_count = 0
                for i in range(start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content[start:i+1].strip()
                            json_str = self._clean_json_string(json_str)
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict) and 'intent' in parsed:
                                    logger.debug(f"🔍 [ROUTER_V2] Found valid JSON using regex fallback")
                                    return parsed
                            except json.JSONDecodeError:
                                break
                            break
        
        # Last resort: try to parse the entire content
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # If all else fails, log the content and raise error
        logger.error(
            f"❌ [ROUTER_V2] Failed to extract JSON from response | "
            f"content_length={len(content)} | content_preview={content[:500]}"
        )
        raise ValueError(
            f"Could not parse JSON from LLM response. "
            f"Response preview: {content[:200]}..."
        )
    
    async def execute(
        self, 
        inputs: Union[Dict[str, Any], RouterAgentInput], 
        context: Optional[Dict[str, Any]] = None
    ) -> RouterAgentOutput:
        """Execute router agent to parse query."""
        import time
        start_time = time.time()
        
        # Log inputs
        self._log_inputs(inputs, context)
        
        # Validate inputs
        validated_inputs = self.validate_inputs(inputs)
        
        query = validated_inputs.query if isinstance(validated_inputs, RouterAgentInput) else validated_inputs.get("query", "")
        user_id = validated_inputs.user_id if isinstance(validated_inputs, RouterAgentInput) else validated_inputs.get("user_id")
        session_id = validated_inputs.session_id if isinstance(validated_inputs, RouterAgentInput) else validated_inputs.get("session_id")
        
        logger.info(
            f"🔍 [ROUTER_V2] Parsing query | "
            f"query='{query[:100]}...' | "
            f"user_id={user_id} | session_id={session_id}"
        )
        
        try:
            # Create messages directly to avoid ChatPromptTemplate format string issues
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Parse this query: {query}")
            ]
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            content = response.content
            if isinstance(content, str):
                result = self._extract_json_from_response(content)
            else:
                result = content
            
            # Extract intent - trust LLM output, no hardcoded defaults
            intent_str = result.get('intent')
            if not intent_str:
                raise ValueError("Intent is required but was not extracted by LLM")
            
            try:
                intent = Intent(intent_str)
            except ValueError:
                raise ValueError(f"Invalid intent extracted by LLM: {intent_str}. Valid intents: {[i.value for i in Intent]}")
            
            # Extract time range
            time_range = self._parse_time_range(result.get('time', {}), query)
            
            # Extract metrics
            metrics = self._extract_metrics(result.get('metrics', []), query)
            
            # Extract entities
            entities = self._extract_entities(result.get('entities', {}), query)
            
            # Extract filters - ensure it's a list
            filters_raw = result.get('filters', [])
            # Convert dict to list of dicts if needed (e.g., {'top_n': 5} -> [{'top_n': 5}])
            if isinstance(filters_raw, dict):
                filters = [filters_raw]
            elif isinstance(filters_raw, list):
                filters = filters_raw
            else:
                filters = []
            
            task_spec = TaskSpec(
                intent=intent,
                metrics=metrics,
                entities=entities,
                time=time_range,
                filters=filters,
                user_id=user_id,
                session_id=session_id
            )
            
            logger.info(
                f"✅ [ROUTER_V2] Query parsed | "
                f"intent={intent.value} | metrics={metrics} | entities={entities}"
            )
            
            result = RouterAgentOutput(
                task_spec=task_spec,
                success=True,
                metadata={}
            )
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
        except Exception as e:
            logger.error(
                f"❌ [ROUTER_V2] Parsing failed | error={str(e)}",
                exc_info=True
            )
            
            result = RouterAgentOutput(
                task_spec=None,
                success=False,
                error=str(e),
                metadata={}
            )
            
            # Log error outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
    
    def _parse_time_range(self, time_data: Dict[str, Any], query: str) -> TimeRange:
        """Parse time range from LLM-extracted data - no hardcoded fallbacks."""
        # Ensure tz is never None (only fallback for timezone)
        tz = time_data.get('tz') or settings.default_timezone
        if not tz:
            tz = "Asia/Kolkata"
        
        # If LLM provided explicit start/end dates, use them
        if 'start' in time_data and 'end' in time_data:
            return TimeRange(start=time_data['start'], end=time_data['end'], tz=tz)
        
        # If LLM provided a range string, resolve it
        range_str = time_data.get('range')
        if range_str:
            start, end = self._resolve_relative_range(range_str)
            return TimeRange(start=start, end=end, range=range_str, tz=tz)
        
        # If LLM didn't provide time range, return None values (let planner handle it)
        # DO NOT use hardcoded defaults
        logger.warning(
            f"⚠️ [ROUTER_V2] No time range extracted by LLM | "
            f"query='{query[:100]}...' | time_data={time_data}"
        )
        return TimeRange(start=None, end=None, range=None, tz=tz)
    
    def _resolve_relative_range(self, range_str: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Resolve relative time range to actual dates - only called if LLM provides range."""
        tz = pytz.timezone(settings.default_timezone)
        now = datetime.now(tz)
        
        # Handle various range formats from LLM
        range_lower = range_str.lower().strip()
        
        # Parse "last_N_days" format
        if range_lower.startswith('last_') and range_lower.endswith('_days'):
            try:
                days = int(range_lower.replace('last_', '').replace('_days', ''))
                end = now
                start = end - timedelta(days=days)
                return start.date(), end.date()
            except ValueError:
                pass
        
        # Standard range strings
        if range_lower == 'last_week' or range_lower == 'last_7_days':
            end = now
            start = end - timedelta(days=7)
        elif range_lower == 'last_month' or range_lower == 'last_30_days':
            end = now
            start = end - timedelta(days=30)
        elif range_lower == 'last_quarter' or range_lower == 'last_90_days':
            end = now
            start = end - timedelta(days=90)
        elif range_lower == 'last_year' or range_lower == 'last_365_days':
            end = now
            start = end - timedelta(days=365)
        elif range_lower == 'yesterday':
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
        elif range_lower == 'today':
            start = now.replace(hour=0, minute=0, second=0)
            end = now
        elif range_lower == 'this_week':
            # Start of current week (Monday)
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
            end = now
        elif range_lower == 'this_month':
            # Start of current month
            start = now.replace(day=1, hour=0, minute=0, second=0)
            end = now
        else:
            # Unknown range format - log warning and return None
            logger.warning(
                f"⚠️ [ROUTER_V2] Unknown range format | range_str={range_str}"
            )
            return None, None
        
        return start.date(), end.date()
    
    def _extract_metrics(self, metrics_list: list, query: str) -> list[str]:
        """Extract and normalize metric names - trust LLM extraction, no hardcoded query scanning."""
        all_metrics = {m.metric_id.lower(): m.name.lower() for m in metric_dictionary.get_all_metrics()}
        extracted = []
        
        # Only use metrics explicitly extracted by LLM - no hardcoded query scanning
        for metric in metrics_list:
            metric_lower = metric.lower()
            if metric_lower in all_metrics:
                extracted.append(metric_lower)
            elif any(metric_lower in m for m in all_metrics.keys()):
                # Try to find closest match
                for key in all_metrics.keys():
                    if metric_lower in key:
                        extracted.append(key)
                        break
        
        # DO NOT scan query text for metrics - trust LLM extraction only
        return extracted
    
    def _extract_entities(self, entities_dict: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract and normalize entities - return entities as extracted by LLM without hardcoded logic."""
        # Return entities as-is from LLM extraction - no hardcoded normalization
        # The LLM should extract entities based on the query, and the schema context
        # will handle mapping entities to relevant tables
        return entities_dict if entities_dict else {}

