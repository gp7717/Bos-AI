"""Router Agent - Intent classification and slot extraction."""
import json
from typing import Dict, Any, Optional
from openai import AzureOpenAI
from app.models.schemas import TaskSpec, Intent, TimeRange, Channel
from app.config.settings import settings
from app.services.metric_dictionary import metric_dictionary
from app.config.logging_config import get_logger
import re
from datetime import datetime, timedelta
import pytz

logger = get_logger(__name__)


class RouterAgent:
    """Classifies user intent and extracts structured slots."""
    
    def __init__(self):
        """Initialize router agent with LLM."""
        logger.info(
            f"🔧 [ROUTER] Initializing Router Agent | "
            f"endpoint={settings.azure_openai_endpoint} | "
            f"model={settings.azure_openai_deployment_name} | "
            f"api_version={settings.azure_openai_api_version}"
        )
        try:
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.model = settings.azure_openai_deployment_name
            logger.info(f"✅ [ROUTER] Router Agent initialized successfully | model={self.model}")
        except Exception as e:
            logger.error(f"❌ [ROUTER] Failed to initialize AzureOpenAI client | error={str(e)}", exc_info=True)
            raise
    
    def parse(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> TaskSpec:
        """Parse user query into structured task spec."""
        logger.info(
            f"🔍 [ROUTER] Parsing query | "
            f"query='{query[:100]}...' | "
            f"user_id={user_id} | "
            f"session_id={session_id}"
        )
        
        # Use LLM for intent classification and slot extraction
        system_prompt = self._get_system_prompt()
        user_prompt = f"Parse this query: {query}"
        
        logger.debug(
            f"🤖 [ROUTER] Calling Azure OpenAI | "
            f"model={self.model} | "
            f"temperature={settings.openai_temperature} | "
            f"user_prompt_length={len(user_prompt)}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.openai_temperature,
                response_format={"type": "json_object"}
            )
            logger.debug(
                f"✅ [ROUTER] Azure OpenAI response received | "
                f"model={self.model} | "
                f"usage={response.usage.model_dump() if hasattr(response, 'usage') else 'N/A'}"
            )
        except Exception as e:
            logger.error(
                f"❌ [ROUTER] Azure OpenAI API call failed | "
                f"model={self.model} | "
                f"endpoint={settings.azure_openai_endpoint} | "
                f"error={str(e)}",
                exc_info=True
            )
            raise
        
        try:
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"📋 [ROUTER] Parsed LLM response | result={result}")
        except json.JSONDecodeError as e:
            logger.error(
                f"❌ [ROUTER] Failed to parse LLM response as JSON | "
                f"content={response.choices[0].message.content[:200]} | "
                f"error={str(e)}",
                exc_info=True
            )
            raise
        
        # Extract intent
        intent_str = result.get('intent', 'analytics.query')
        logger.debug(f"🎯 [ROUTER] Extracting intent | intent_str={intent_str}")
        try:
            intent = Intent(intent_str)
            logger.debug(f"✅ [ROUTER] Intent extracted | intent={intent.value}")
        except ValueError:
            logger.warning(f"⚠️ [ROUTER] Invalid intent, defaulting to ANALYTICS_QUERY | intent_str={intent_str}")
            intent = Intent.ANALYTICS_QUERY
        
        # Extract time range
        logger.debug(f"📅 [ROUTER] Extracting time range | time_data={result.get('time', {})}")
        time_range = self._parse_time_range(result.get('time', {}), query)
        logger.debug(f"✅ [ROUTER] Time range extracted | time_range={time_range.model_dump()}")
        
        # Extract metrics
        logger.debug(f"📊 [ROUTER] Extracting metrics | metrics_data={result.get('metrics', [])}")
        metrics = self._extract_metrics(result.get('metrics', []), query)
        logger.debug(f"✅ [ROUTER] Metrics extracted | metrics={metrics}")
        
        # Extract entities
        logger.debug(f"🏷️ [ROUTER] Extracting entities | entities_data={result.get('entities', {})}")
        entities = self._extract_entities(result.get('entities', {}), query)
        logger.debug(f"✅ [ROUTER] Entities extracted | entities={entities}")
        
        # Extract filters
        filters = result.get('filters', [])
        logger.debug(f"🔍 [ROUTER] Filters extracted | filters={filters}")
        
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
            f"✅ [ROUTER] Query parsing completed | "
            f"intent={intent.value} | "
            f"metrics_count={len(metrics)} | "
            f"entities_count={len(entities)}"
        )
        
        return task_spec
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for intent classification."""
        return """You are an intent classification and slot extraction agent for an analytics Q&A system.

Your task is to parse user queries and extract:
1. Intent: One of: analytics.query, data.export, diagnostics, meta.help
2. Metrics: List of metrics requested (e.g., ROAS, CPC, CTR, spend, revenue)
3. Entities: Channel (SP, SB, SD, META, GOOGLE), geo (city/region), campaign, product
4. Time: Time range (start, end, or relative like "last_week", "last_month")
5. Filters: Additional filters (status, type, etc.)

Return JSON with this structure:
{
  "intent": "analytics.query",
  "metrics": ["roas"],
  "entities": {"channel": "SB", "geo": "Delhi"},
  "time": {"range": "last_week", "tz": "Asia/Kolkata"},
  "filters": []
}

Examples:
- "What was ROAS last week for SB in Delhi?" -> {"intent": "analytics.query", "metrics": ["roas"], "entities": {"channel": "SB", "geo": "Delhi"}, "time": {"range": "last_week"}}
- "Top 5 products by net sales in Delhi last month" -> {"intent": "analytics.query", "metrics": ["net_sales"], "entities": {"geo": "Delhi"}, "time": {"range": "last_month"}}
"""
    
    def _parse_time_range(self, time_data: Dict[str, Any], query: str) -> TimeRange:
        """Parse time range from extracted data and query text."""
        tz = time_data.get('tz', settings.default_timezone)
        
        # Check for explicit dates
        if 'start' in time_data and 'end' in time_data:
            return TimeRange(start=time_data['start'], end=time_data['end'], tz=tz)
        
        # Parse relative time ranges
        range_str = time_data.get('range', '')
        if not range_str:
            # Try to extract from query
            range_str = self._extract_time_range_from_query(query)
        
        start, end = self._resolve_relative_range(range_str)
        
        return TimeRange(start=start, end=end, range=range_str, tz=tz)
    
    def _extract_time_range_from_query(self, query: str) -> str:
        """Extract time range keywords from query."""
        query_lower = query.lower()
        
        if 'last 7 days' in query_lower or 'last week' in query_lower:
            return 'last_week'
        elif 'last 30 days' in query_lower or 'last month' in query_lower:
            return 'last_month'
        elif 'last 90 days' in query_lower or 'last quarter' in query_lower:
            return 'last_quarter'
        elif 'last year' in query_lower:
            return 'last_year'
        elif 'yesterday' in query_lower:
            return 'yesterday'
        elif 'today' in query_lower:
            return 'today'
        
        return 'last_month'  # Default
    
    def _resolve_relative_range(self, range_str: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Resolve relative time range to actual dates."""
        tz = pytz.timezone(settings.default_timezone)
        now = datetime.now(tz)
        
        if range_str == 'last_week':
            end = now
            start = end - timedelta(days=7)
        elif range_str == 'last_month':
            end = now
            start = end - timedelta(days=30)
        elif range_str == 'last_quarter':
            end = now
            start = end - timedelta(days=90)
        elif range_str == 'yesterday':
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
        elif range_str == 'today':
            start = now.replace(hour=0, minute=0, second=0)
            end = now
        else:
            # Default to last month
            end = now
            start = end - timedelta(days=30)
        
        return start.date(), end.date()
    
    def _extract_metrics(self, metrics_list: list, query: str) -> list[str]:
        """Extract and normalize metric names."""
        # Get all available metrics
        all_metrics = {m.metric_id.lower(): m.name.lower() for m in metric_dictionary.get_all_metrics()}
        
        extracted = []
        query_lower = query.lower()
        
        # Check explicit metrics from LLM
        for metric in metrics_list:
            metric_lower = metric.lower()
            if metric_lower in all_metrics:
                extracted.append(metric_lower)
            elif any(metric_lower in m for m in all_metrics.keys()):
                # Partial match
                for key in all_metrics.keys():
                    if metric_lower in key:
                        extracted.append(key)
                        break
        
        # Also scan query for metric keywords
        for metric_id, metric_name in all_metrics.items():
            if metric_id in query_lower or metric_name in query_lower:
                if metric_id not in extracted:
                    extracted.append(metric_id)
        
        return extracted
    
    def _extract_entities(self, entities_dict: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract and normalize entities."""
        result = {}
        query_lower = query.lower()
        
        # Extract channel
        if 'channel' in entities_dict:
            result['channel'] = entities_dict['channel']
        else:
            # Try to extract from query
            if 'sp ' in query_lower or 'sponsored products' in query_lower:
                result['channel'] = 'SP'
            elif 'sb ' in query_lower or 'sponsored brands' in query_lower:
                result['channel'] = 'SB'
            elif 'sd ' in query_lower or 'sponsored display' in query_lower:
                result['channel'] = 'SD'
            elif 'meta' in query_lower or 'facebook' in query_lower:
                result['channel'] = 'META'
            elif 'google' in query_lower:
                result['channel'] = 'GOOGLE'
        
        # Extract geo
        if 'geo' in entities_dict:
            result['geo'] = entities_dict['geo']
        else:
            # Common Indian cities
            cities = ['delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune']
            for city in cities:
                if city in query_lower:
                    result['geo'] = city.capitalize()
                    break
        
        # Extract campaign
        if 'campaign' in entities_dict:
            result['campaign'] = entities_dict['campaign']
        
        # Extract product
        if 'product' in entities_dict:
            result['product'] = entities_dict['product']
        
        return result

