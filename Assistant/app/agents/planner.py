"""Planner Agent - Converts task specs into execution DAGs."""
import json
from typing import List, Dict, Any
from openai import AzureOpenAI
from app.models.schemas import TaskSpec, ExecutionPlan, PlanStep
from app.config.settings import settings
from app.services.tool_registry import tool_registry
from app.services.metric_dictionary import metric_dictionary
from app.services.schema_registry import schema_registry
from app.services.schema_context import SchemaContextService
from app.services.system_config import system_config
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    """Plans execution by converting task specs into DAGs."""
    
    def __init__(self):
        """Initialize planner agent."""
        logger.info(f"🔧 [PLANNER] Initializing Planner Agent | model={settings.azure_openai_deployment_name}")
        try:
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.model = settings.azure_openai_deployment_name
            self.schema_context = SchemaContextService()
            logger.info(f"✅ [PLANNER] Planner Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ [PLANNER] Failed to initialize | error={str(e)}", exc_info=True)
            raise
    
    def plan(self, task_spec: TaskSpec) -> ExecutionPlan:
        """Convert task spec into execution plan DAG."""
        logger.info(
            f"📊 [PLANNER] Generating execution plan | "
            f"intent={task_spec.intent.value} | "
            f"metrics={task_spec.metrics} | "
            f"entities={task_spec.entities}"
        )
        
        # Build tool catalog context
        logger.debug(f"🔧 [PLANNER] Building tool catalog")
        tool_catalog = self._build_tool_catalog()
        logger.debug(f"🔧 [PLANNER] Building metric catalog")
        metric_catalog = self._build_metric_catalog()
        logger.debug(f"🔧 [PLANNER] Building schema context")
        schema_catalog = self._build_schema_catalog(task_spec)
        logger.debug(f"✅ [PLANNER] Catalogs built | tools={len(tool_catalog)} | metrics={len(metric_catalog)} | schema_tables={len(schema_catalog.split('Table:')) - 1}")
        
        # Use LLM to generate plan
        system_prompt = self._get_system_prompt(tool_catalog, metric_catalog, schema_catalog)
        user_prompt = self._build_planning_prompt(task_spec)
        
        logger.debug(
            f"🤖 [PLANNER] Calling Azure OpenAI for plan generation | "
            f"model={self.model} | "
            f"prompt_length={len(user_prompt)}"
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
            logger.debug(f"✅ [PLANNER] Azure OpenAI response received")
        except Exception as e:
            logger.error(f"❌ [PLANNER] Azure OpenAI API call failed | error={str(e)}", exc_info=True)
            raise
        
        try:
            plan_data = json.loads(response.choices[0].message.content)
            logger.debug(f"📋 [PLANNER] Plan data parsed | steps={len(plan_data.get('steps', []))}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ [PLANNER] Failed to parse plan response | error={str(e)}", exc_info=True)
            raise
        
        # Validate and enhance plan
        logger.debug(f"🔍 [PLANNER] Parsing and validating plan")
        plan = self._parse_plan(plan_data, task_spec)
        logger.debug(f"✨ [PLANNER] Enhancing plan")
        plan = self._enhance_plan(plan, task_spec)
        
        logger.info(
            f"✅ [PLANNER] Execution plan generated | "
            f"steps={len(plan.steps)} | "
            f"outputs={list(plan.outputs.keys())}"
        )
        
        return plan
    
    def _build_tool_catalog(self) -> str:
        """Build tool catalog description for LLM."""
        tools = []
        for tool in tool_registry.get_all_tools():
            # Skip excluded tools based on config
            if system_config.is_tool_excluded(tool.tool_id):
                continue
            tool_info = {
                'id': tool.tool_id,
                'kind': tool.kind,
                'capabilities': [
                    {
                        'name': cap.name,
                        'metrics': cap.metrics,
                        'channels': cap.channels
                    }
                    for cap in tool.capabilities
                ]
            }
            tools.append(json.dumps(tool_info, indent=2))
        
        return "\n\n".join(tools)
    
    def _build_metric_catalog(self) -> str:
        """Build metric catalog description for LLM."""
        metrics = []
        for metric in metric_dictionary.get_all_metrics():
            metric_info = {
                'id': metric.metric_id,
                'name': metric.name,
                'formula': metric.formula,
                'dependencies': metric.dependencies
            }
            metrics.append(json.dumps(metric_info, indent=2))
        
        return "\n\n".join(metrics)
    
    def _build_schema_catalog(self, task_spec: TaskSpec) -> str:
        """Build schema catalog with relevant tables for LLM."""
        # Get relevant tables based on query context
        relevant_tables = self.schema_context.get_relevant_tables_for_query(
            task_spec.metrics,
            task_spec.entities
        )
        
        if not relevant_tables:
            # Fallback: get common tables from config
            logger.debug(f"⚠️ [PLANNER] No relevant tables found, using common tables")
            common_tables = system_config.get_common_fallback_tables()
            relevant_tables = []
            for table in common_tables:
                schema, table_name = table.split('.', 1)
                context = self.schema_context.get_table_context(schema, table_name)
                if context:
                    relevant_tables.append(context)
        
        # Format for LLM
        schema_catalog = self.schema_context.format_for_llm(relevant_tables)
        
        logger.debug(
            f"📊 [PLANNER] Schema catalog built | "
            f"tables={len(relevant_tables)} | "
            f"catalog_length={len(schema_catalog)}"
        )
        
        return schema_catalog
    
    def _get_system_prompt(self, tool_catalog: str, metric_catalog: str, schema_catalog: str) -> str:
        """Get system prompt for planning."""
        return f"""You are a planning agent that converts analytics queries into execution plans.

Available Tools:
{tool_catalog}

Available Metrics:
{metric_catalog}

Database Schema (for SQL generation):
{schema_catalog}

CRITICAL SQL GENERATION RULES:
1. **ALWAYS use the EXACT table names shown in the schema above**
2. **NEVER use generic table names like 'orders', 'sales', 'products'**
3. **Use the FULL qualified name format: schema.table_name (e.g., 'public.shopify_orders')**
4. **Match column names exactly as shown in the schema**
5. **Use parameterized queries with :param_name syntax**

Your task is to create an execution plan (DAG) that:
1. Identifies which tools/tables to use
2. Determines the order of operations
3. Specifies join keys and aggregations
4. Handles dependencies between steps
5. For SQL queries, you can now generate SQL directly using the schema information above

Return JSON with this structure:
{{
  "steps": [
    {{
      "id": "step1",
      "tool": "tool_id.capability_name",
      "inputs": {{
        "sql": "SELECT ... FROM schema.table_name WHERE ...",
        "params": {{"param_name": "value"}},
        "inputs": {{"key": "value"}}
      }},
      "depends_on": [],
      "output_key": "result1"
    }}
  ],
  "outputs": {{"table": "final_step_id", "narrative": true}}
}}

Rules:
- Each step must reference a valid tool and capability in the format: "tool_id.capability_name"
- **CRITICAL**: Tool format MUST be "tool_id.capability_name" (e.g., "sales_db.sql", "compute.aggregate")
- For SQL tools: use "tool_id.sql" format
- For compute tools: use "compute.aggregate", "compute.join", or "compute.calculate"
- Use depends_on to specify step dependencies
- For metrics requiring multiple data sources, create separate steps and join them
- Always include date range filters in inputs
- For SQL-based tools, you can include:
  * "sql" field with the generated SQL query (use parameterized queries with :param_name)
  * "params" field with parameter values
  * **MUST use EXACT table names from schema (e.g., 'public.shopify_orders', NOT 'orders')**
  * **MUST use exact column names as shown in the schema**
- If generating SQL, reference the schema tables above for accurate column names and relationships
"""
    
    def _build_planning_prompt(self, task_spec: TaskSpec) -> str:
        """Build user prompt for planning."""
        prompt = f"""Create an execution plan for this query:

Intent: {task_spec.intent.value}
Metrics: {', '.join(task_spec.metrics)}
Entities: {json.dumps(task_spec.entities)}
Time Range: {task_spec.time.start} to {task_spec.time.end or 'now'}
Timezone: {task_spec.time.tz}
Filters: {json.dumps(task_spec.filters)}

Generate a plan that:
1. Fetches the required data from appropriate tools
2. Joins data if needed (e.g., ads spend + sales revenue for ROAS)
3. Computes requested metrics
4. Applies filters and aggregations
"""
        return prompt
    
    def _parse_plan(self, plan_data: Dict[str, Any], task_spec: TaskSpec) -> ExecutionPlan:
        """Parse LLM output into ExecutionPlan."""
        steps = []
        for step_data in plan_data.get('steps', []):
            tool = step_data.get('tool', '')
            # Normalize tool format: ensure it has tool_id.capability_name format
            tool = self._normalize_tool_format(tool)
            
            step = PlanStep(
                id=step_data.get('id', f"step_{len(steps)}"),
                tool=tool,
                inputs=step_data.get('inputs', {}),
                depends_on=step_data.get('depends_on', []),
                output_key=step_data.get('output_key')
            )
            steps.append(step)
        
        return ExecutionPlan(
            steps=steps,
            outputs=plan_data.get('outputs', {}),
            metadata={'task_spec': task_spec.dict()}
        )
    
    def _normalize_tool_format(self, tool: str) -> str:
        """Normalize tool format to ensure it follows tool_id.capability_name pattern."""
        if not tool:
            return tool
        
        # If already in correct format (contains dot), return as is
        if '.' in tool:
            return tool
        
        # Normalize based on tool_id
        tool_id = tool
        
        # SQL tools: add default .sql capability
        if system_config.is_sql_tool(tool_id):
            normalized = f"{tool_id}.sql"
            logger.debug(f"🔧 [PLANNER] Normalized SQL tool | original={tool_id} | normalized={normalized}")
            return normalized
        
        # Compute tool: default to .aggregate capability
        if system_config.is_compute_tool(tool_id):
            normalized = f"{tool_id}.aggregate"
            logger.debug(f"🔧 [PLANNER] Normalized compute tool | original={tool_id} | normalized={normalized}")
            return normalized
        
        # API tools: check if they have capabilities defined
        tool_def = tool_registry.get_tool(tool_id)
        if tool_def and tool_def.capabilities:
            # Use first capability as default
            default_capability = tool_def.capabilities[0].name
            normalized = f"{tool_id}.{default_capability}"
            logger.debug(f"🔧 [PLANNER] Normalized API tool | original={tool_id} | normalized={normalized}")
            return normalized
        
        # If we can't normalize, return as is (will be caught by validation)
        logger.warning(f"⚠️ [PLANNER] Could not normalize tool format | tool={tool_id}")
        return tool
    
    def _enhance_plan(self, plan: ExecutionPlan, task_spec: TaskSpec) -> ExecutionPlan:
        """Enhance plan with additional context and validation."""
        # Add date range to all data access steps
        for step in plan.steps:
            # Extract tool_id (handle both normalized and non-normalized formats)
            tool_parts = step.tool.split('.')
            tool_id = tool_parts[0]
            
            if system_config.is_sql_tool(tool_id) or tool_id in system_config.get_date_range_tools():
                if 'inputs' not in step.inputs:
                    step.inputs['inputs'] = {}
                step.inputs['inputs']['date_start'] = str(task_spec.time.start)
                if task_spec.time.end:
                    step.inputs['inputs']['date_end'] = str(task_spec.time.end)
                step.inputs['inputs']['timezone'] = task_spec.time.tz
        
        # Add entity filters
        for step in plan.steps:
            if task_spec.entities:
                if 'channel' in task_spec.entities:
                    step.inputs.setdefault('inputs', {})['channel'] = task_spec.entities['channel']
                if 'geo' in task_spec.entities:
                    step.inputs.setdefault('inputs', {})['geo'] = task_spec.entities['geo']
        
        return plan

