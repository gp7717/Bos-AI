"""Answer Composer Agent - Generates natural language answers."""
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from app.config.settings import settings
from app.models.schemas import QueryResponse, TaskSpec
from app.config.logging_config import get_logger
import json

logger = get_logger(__name__)


class AnswerComposerAgent:
    """Composes natural language answers from results."""
    
    def __init__(self):
        """Initialize answer composer."""
        logger.info(f"🔧 [COMPOSER] Initializing Answer Composer Agent | model={settings.azure_openai_deployment_name}")
        try:
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.model = settings.azure_openai_deployment_name
            logger.info(f"✅ [COMPOSER] Answer Composer Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ [COMPOSER] Failed to initialize | error={str(e)}", exc_info=True)
            raise
    
    def compose(self, task_spec: TaskSpec, results: Dict[str, Any], 
                execution_trace: Optional[str] = None) -> QueryResponse:
        """Compose answer from query results."""
        logger.info(
            f"📝 [COMPOSER] Composing answer | "
            f"intent={task_spec.intent.value} | "
            f"row_count={results.get('row_count', 0)}"
        )
        
        # Extract data
        raw_data = results.get('data', [])
        table = results.get('table', [])
        
        # Handle data type: QueryResponse.data expects Dict, table expects List[Dict]
        # If data is a list (tabular data), set data to None and use table instead
        if isinstance(raw_data, list):
            data = None  # Use table for list data, data field is for dict
            logger.debug(f"📊 [COMPOSER] Data is list, using table field | table_rows={len(table)}")
        elif isinstance(raw_data, dict):
            data = raw_data  # Keep as dict for structured data
            logger.debug(f"📊 [COMPOSER] Data is dict | data_keys={list(data.keys()) if data else []}")
        else:
            data = None  # Default to None for other types
            logger.debug(f"📊 [COMPOSER] Data type not recognized, setting to None")
        
        logger.debug(f"📊 [COMPOSER] Data extracted | data_type={type(data).__name__} | table_rows={len(table)}")
        
        # Generate reasoning trace
        logger.debug(f"🔗 [COMPOSER] Generating reasoning trace")
        reasoning_trace = self._generate_reasoning_trace(execution_trace, task_spec)
        
        # Generate natural language answer (use table for list data, data for dict)
        logger.debug(f"💬 [COMPOSER] Generating natural language answer")
        answer_data = table if data is None else data
        answer = self._generate_answer(task_spec, answer_data, table, reasoning_trace)
        logger.debug(f"✅ [COMPOSER] Answer generated | answer_length={len(answer)}")
        
        # Generate chart spec if applicable
        logger.debug(f"📊 [COMPOSER] Generating chart specification")
        chart_spec = self._generate_chart_spec(task_spec, answer_data, table)
        
        response = QueryResponse(
            answer=answer,
            data=data,  # Dict or None
            table=table,  # List[Dict] for tabular data
            chart_spec=chart_spec,
            reasoning_trace=reasoning_trace,
            request_id=results.get('request_id', ''),
            timestamp=results.get('timestamp')
        )
        
        logger.info(f"✅ [COMPOSER] Answer composition completed | answer_length={len(answer)}")
        
        return response
    
    def _generate_answer(self, task_spec: TaskSpec, data: Any, 
                        table: List[Dict[str, Any]], reasoning_trace: str) -> str:
        """Generate natural language answer."""
        
        # Build data summary with actual values
        data_summary = ""
        if table and len(table) > 0:
            # Show actual data values
            if len(table) <= 10:
                # Show all rows if small dataset
                data_summary = f"Data:\n{json.dumps(table, indent=2, default=str)}"
            else:
                # Show first few and summary if large dataset
                data_summary = f"Data (first 5 rows):\n{json.dumps(table[:5], indent=2, default=str)}\n... and {len(table) - 5} more rows"
        else:
            data_summary = "No data available."
        
        # Build prompt
        prompt = f"""You are a data analyst assistant. Generate a clear, concise answer to the user's query.

User Query Intent: {task_spec.intent.value}
Requested Metrics: {', '.join(task_spec.metrics)}
Time Range: {task_spec.time.start} to {task_spec.time.end or 'now'}

{data_summary}

Execution Trace:
{reasoning_trace}

Generate a natural language answer that:
1. Directly answers the user's question using the ACTUAL VALUES from the data above
2. Includes key numbers and insights from the data
3. Mentions any limitations or caveats
4. Is concise (2-3 sentences for simple queries, up to a paragraph for complex ones)

IMPORTANT: Use the actual numeric values from the data above, not placeholders like "$X".

Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_reasoning_trace(self, execution_trace: Optional[str], 
                                  task_spec: TaskSpec) -> str:
        """Generate reasoning trace."""
        if execution_trace:
            return execution_trace
        
        # Build trace from task spec
        trace_parts = []
        
        trace_parts.append(f"Intent: {task_spec.intent.value}")
        trace_parts.append(f"Metrics requested: {', '.join(task_spec.metrics)}")
        
        if task_spec.entities:
            trace_parts.append(f"Filters: {json.dumps(task_spec.entities)}")
        
        trace_parts.append(f"Time range: {task_spec.time.start} to {task_spec.time.end or 'now'}")
        trace_parts.append(f"Timezone: {task_spec.time.tz}")
        
        return " | ".join(trace_parts)
    
    def _generate_chart_spec(self, task_spec: TaskSpec, data: Any, 
                            table: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate chart specification if applicable."""
        if not table or len(table) == 0:
            return None
        
        # Determine chart type based on data
        if len(table) > 10:
            # Time series or large dataset
            chart_type = "line"
        elif len(task_spec.metrics) > 1:
            # Multiple metrics
            chart_type = "bar"
        else:
            # Single metric
            chart_type = "bar"
        
        # Build chart spec
        chart_spec = {
            "type": chart_type,
            "data": table[:100],  # Limit to 100 rows for chart
            "x_axis": "date" if "date" in str(table[0].keys()) else list(table[0].keys())[0],
            "y_axis": task_spec.metrics[0] if task_spec.metrics else list(table[0].keys())[-1],
            "title": f"{' vs '.join(task_spec.metrics)} over time"
        }
        
        return chart_spec

