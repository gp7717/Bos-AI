"""Composer Agent V2 - LangChain-based answer composition."""
from typing import Dict, Any, Optional, List, Union
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.schemas import (
    QueryResponse, TaskSpec,
    ComposerAgentInput, ComposerAgentOutput
)
from app.config.settings import settings
from app.config.logging_config import get_logger
from app.core.base import BaseAgent, AgentMetadata, AgentType
import json
from datetime import datetime

logger = get_logger(__name__)


class ComposerAgentV2(BaseAgent[ComposerAgentInput, ComposerAgentOutput]):
    """LangChain-based composer agent for answer generation."""
    
    def __init__(self):
        """Initialize composer agent with LangChain."""
        metadata = AgentMetadata(
            agent_id="composer_v2",
            agent_type=AgentType.COMPOSER,
            name="Composer Agent V2",
            description="LangChain-based natural language answer composition",
            version="2.0.0",
            capabilities=["answer_generation", "chart_spec_generation", "reasoning_trace"]
        )
        super().__init__(metadata, ComposerAgentInput, ComposerAgentOutput)
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment_name,
            temperature=0.3,
        )
        
        logger.info(f"✅ [COMPOSER_V2] Composer Agent V2 initialized")
    
    async def execute(
        self, 
        inputs: Union[Dict[str, Any], ComposerAgentInput], 
        context: Optional[Dict[str, Any]] = None
    ) -> ComposerAgentOutput:
        """Compose answer from query results."""
        import time
        start_time = time.time()
        
        # Log inputs
        self._log_inputs(inputs, context)
        
        # Validate inputs
        validated_inputs = self.validate_inputs(inputs)
        
        task_spec = validated_inputs.task_spec if isinstance(validated_inputs, ComposerAgentInput) else validated_inputs.get("task_spec")
        results = validated_inputs.results if isinstance(validated_inputs, ComposerAgentInput) else validated_inputs.get("results", {})
        execution_trace = validated_inputs.execution_trace if isinstance(validated_inputs, ComposerAgentInput) else validated_inputs.get("execution_trace")
        
        logger.info(
            f"📝 [COMPOSER_V2] Composing answer | "
            f"intent={task_spec.intent.value if task_spec else 'unknown'} | "
            f"row_count={results.get('row_count', 0)}"
        )
        
        try:
            # Extract data
            raw_data = results.get('data', [])
            table = results.get('table', [])
            
            if isinstance(raw_data, list):
                data = None
            elif isinstance(raw_data, dict):
                data = raw_data
            else:
                data = None
            
            # Generate reasoning trace
            reasoning_trace = self._generate_reasoning_trace(execution_trace, task_spec)
            
            # Generate answer
            answer_data = table if data is None else data
            answer = await self._generate_answer(task_spec, answer_data, table, reasoning_trace)
            
            # Generate chart spec
            chart_spec = self._generate_chart_spec(task_spec, answer_data, table)
            
            response = QueryResponse(
                answer=answer,
                data=data,
                table=table,
                chart_spec=chart_spec,
                reasoning_trace=reasoning_trace,
                request_id=results.get('request_id', ''),
                timestamp=results.get('timestamp', datetime.utcnow())
            )
            
            logger.info(f"✅ [COMPOSER_V2] Answer composition completed")
            
            result = ComposerAgentOutput(
                response=response,
                success=True,
                metadata={}
            )
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
        except Exception as e:
            logger.error(
                f"❌ [COMPOSER_V2] Answer composition failed | error={str(e)}",
                exc_info=True
            )
            
            result = ComposerAgentOutput(
                response=None,
                success=False,
                error=str(e),
                metadata={}
            )
            
            # Log error outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
    
    async def _generate_answer(
        self,
        task_spec: TaskSpec,
        data: Any,
        table: List[Dict[str, Any]],
        reasoning_trace: str
    ) -> str:
        """Generate natural language answer."""
        data_summary = ""
        if table and len(table) > 0:
            if len(table) <= 10:
                data_summary = f"Data:\n{json.dumps(table, indent=2, default=str)}"
            else:
                data_summary = f"Data (first 5 rows):\n{json.dumps(table[:5], indent=2, default=str)}\n... and {len(table) - 5} more rows"
        else:
            data_summary = "No data available."
        
        prompt = f"""You are a data analyst assistant. Generate a clear, concise answer to the user's query.

User Query Intent: {task_spec.intent.value if task_spec else 'unknown'}
Requested Metrics: {', '.join(task_spec.metrics) if task_spec else 'N/A'}
Time Range: {task_spec.time.start if task_spec else 'N/A'} to {task_spec.time.end or 'now' if task_spec else 'N/A'}

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
        
        # Create messages directly to avoid ChatPromptTemplate format string issues
        # JSON in data_summary may contain braces that would be interpreted as placeholders
        messages = [
            SystemMessage(content="You are a helpful data analyst assistant."),
            HumanMessage(content=prompt)
        ]
        response = await self.llm.ainvoke(messages)
        
        return response.content.strip()
    
    def _generate_reasoning_trace(self, execution_trace: Optional[str], task_spec: Optional[TaskSpec]) -> str:
        """Generate reasoning trace."""
        if execution_trace:
            return execution_trace
        
        if not task_spec:
            return "No execution trace available"
        
        trace_parts = []
        trace_parts.append(f"Intent: {task_spec.intent.value}")
        trace_parts.append(f"Metrics requested: {', '.join(task_spec.metrics)}")
        
        if task_spec.entities:
            trace_parts.append(f"Filters: {json.dumps(task_spec.entities)}")
        
        trace_parts.append(f"Time range: {task_spec.time.start} to {task_spec.time.end or 'now'}")
        trace_parts.append(f"Timezone: {task_spec.time.tz}")
        
        return " | ".join(trace_parts)
    
    def _generate_chart_spec(
        self,
        task_spec: Optional[TaskSpec],
        data: Any,
        table: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate chart specification."""
        if not table or len(table) == 0:
            return None
        
        if len(table) > 10:
            chart_type = "line"
        elif task_spec and len(task_spec.metrics) > 1:
            chart_type = "bar"
        else:
            chart_type = "bar"
        
        chart_spec = {
            "type": chart_type,
            "data": table[:100],
            "x_axis": "date" if "date" in str(table[0].keys()) else list(table[0].keys())[0],
            "y_axis": task_spec.metrics[0] if task_spec and task_spec.metrics else list(table[0].keys())[-1],
            "title": f"{' vs '.join(task_spec.metrics) if task_spec and task_spec.metrics else 'Data'} over time"
        }
        
        return chart_spec

