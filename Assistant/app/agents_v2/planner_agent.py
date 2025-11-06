"""Planner Agent V2 - LangChain-based execution plan generation."""
import json
from typing import Dict, Any, Optional, List, Union
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.schemas import (
    TaskSpec, ExecutionPlan, PlanStep,
    PlannerAgentInput, PlannerAgentOutput
)
from app.config.settings import settings
from app.services.tool_registry import tool_registry
from app.services.metric_dictionary import metric_dictionary
from app.services.schema_context import SchemaContextService
from app.services.mcp_database_client import get_mcp_client
from app.services.system_config import system_config
from app.config.logging_config import get_logger
from app.core.base import BaseAgent, AgentMetadata, AgentType

logger = get_logger(__name__)


class PlannerAgentV2(BaseAgent[PlannerAgentInput, PlannerAgentOutput]):
    """LangChain-based planner agent for execution plan generation."""
    
    def __init__(self):
        """Initialize planner agent with LangChain."""
        metadata = AgentMetadata(
            agent_id="planner_v2",
            agent_type=AgentType.PLANNER,
            name="Planner Agent V2",
            description="LangChain-based execution plan generation",
            version="2.0.0",
            capabilities=["plan_generation", "dag_construction", "sql_generation"]
        )
        super().__init__(metadata, PlannerAgentInput, PlannerAgentOutput)
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment_name,
            temperature=settings.openai_temperature,
        )
        
        self.schema_context = SchemaContextService()
        
        logger.info(f"✅ [PLANNER_V2] Planner Agent V2 initialized")
    
    async def execute(
        self, 
        inputs: Union[Dict[str, Any], PlannerAgentInput], 
        context: Optional[Dict[str, Any]] = None
    ) -> PlannerAgentOutput:
        """Generate execution plan from task spec."""
        import time
        start_time = time.time()
        
        # Log inputs
        self._log_inputs(inputs, context)
        
        # Validate inputs
        validated_inputs = self.validate_inputs(inputs)
        
        task_spec = validated_inputs.task_spec if isinstance(validated_inputs, PlannerAgentInput) else validated_inputs.get("task_spec")
        
        if not task_spec:
            result = PlannerAgentOutput(
                plan=None,
                success=False,
                error="Task spec is required",
                metadata={}
            )
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            return result
        
        logger.info(
            f"📊 [PLANNER_V2] Generating execution plan | "
            f"intent={task_spec.intent.value} | metrics={task_spec.metrics}"
        )
        
        try:
            # Build context
            tool_catalog = self._build_tool_catalog()
            metric_catalog = self._build_metric_catalog()
            schema_catalog = await self._build_schema_catalog(task_spec)
            
            # Build prompt - format strings directly to avoid ChatPromptTemplate formatting issues
            system_prompt = self._get_system_prompt(tool_catalog, metric_catalog, schema_catalog)
            user_prompt = self._build_planning_prompt(task_spec)
            
            # Create messages directly without using ChatPromptTemplate to avoid format issues
            # ChatPromptTemplate tries to format strings even when they're already formatted
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            content = response.content
            if isinstance(content, str):
                plan_data = self._extract_json_from_response(content)
            else:
                plan_data = content
            
            # Parse and enhance plan
            plan = self._parse_plan(plan_data, task_spec)
            plan = self._enhance_plan(plan, task_spec)
            
            # Validate SQL queries for missing JOINs
            self._validate_plan_sql(plan)
            
            logger.info(
                f"✅ [PLANNER_V2] Execution plan generated | "
                f"steps={len(plan.steps)}"
            )
            
            result = PlannerAgentOutput(
                plan=plan,
                success=True,
                metadata={}
            )
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
        except Exception as e:
            logger.error(
                f"❌ [PLANNER_V2] Plan generation failed | error={str(e)}",
                exc_info=True
            )
            
            result = PlannerAgentOutput(
                plan=None,
                success=False,
                error=str(e),
                metadata={}
            )
            
            # Log error outputs
            execution_time_ms = (time.time() - start_time) * 1000
            self._log_outputs(result, execution_time_ms)
            
            return result
    
    def _build_tool_catalog(self) -> str:
        """Build tool catalog description."""
        tools = []
        for tool in tool_registry.get_all_tools():
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
        """Build metric catalog description."""
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
    
    async def _build_schema_catalog(self, task_spec: TaskSpec) -> str:
        """Build schema catalog with relevant tables using MCP client."""
        mcp_client = get_mcp_client()
        
        # Use MCP client to discover tables for metrics
        table_names = await mcp_client.discover_tables_for_metrics(task_spec.metrics)
        
        # Get schemas for discovered tables (on-demand)
        mcp_schemas = await mcp_client.get_schema_for_tables(table_names)
        
        # Convert to our format
        relevant_tables = []
        for schema_info in mcp_schemas:
            if schema_info:
                context = {
                    'schema': schema_info.get('schema', 'public'),
                    'table': schema_info.get('table', ''),
                    'full_name': schema_info.get('full_name', ''),
                    'columns': [
                        {
                            'name': col.get('name', ''),
                            'type': str(col.get('type', '')),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False),
                            'description': col.get('description')
                        }
                        for col in schema_info.get('columns', [])
                    ],
                    'primary_keys': schema_info.get('primary_keys', []),
                    'foreign_keys': schema_info.get('foreign_keys', []),
                    'date_column': schema_info.get('date_column')
                }
                relevant_tables.append(context)
        
        # Also get tables based on entities using schema context
        if task_spec.entities:
            entity_tables = await self.schema_context.get_relevant_tables_for_query(
                task_spec.metrics,
                task_spec.entities
            )
            # Merge with existing tables
            existing_names = {t.get('full_name') for t in relevant_tables}
            for table in entity_tables:
                if table.get('full_name') not in existing_names:
                    relevant_tables.append(table)
        
        if not relevant_tables:
            common_tables = system_config.get_common_fallback_tables()
            fallback_schemas = await mcp_client.get_schema_for_tables(common_tables)
            for schema_info in fallback_schemas:
                if schema_info:
                    context = {
                        'schema': schema_info.get('schema', 'public'),
                        'table': schema_info.get('table', ''),
                        'full_name': schema_info.get('full_name', ''),
                        'columns': [
                            {
                                'name': col.get('name', ''),
                                'type': str(col.get('type', '')),
                                'nullable': col.get('nullable', True),
                                'primary_key': col.get('primary_key', False),
                                'description': col.get('description')
                            }
                            for col in schema_info.get('columns', [])
                        ],
                        'primary_keys': schema_info.get('primary_keys', []),
                        'foreign_keys': schema_info.get('foreign_keys', []),
                        'date_column': schema_info.get('date_column')
                    }
                    relevant_tables.append(context)
        
        # Store available tables for validation
        self._last_schema_catalog_tables = {
            table.get('full_name', '') for table in relevant_tables
        }
        
        return self.schema_context.format_for_llm(relevant_tables)
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks and extra text."""
        import re
        
        # First, try to extract from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find the first complete JSON object (non-greedy, balanced braces)
        # This is more robust than greedy matching
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found a complete JSON object
                    json_str = content[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try next JSON object
                        start_idx = -1
                        continue
        
        # Fallback: try greedy match (original behavior)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"⚠️ [PLANNER_V2] Failed to parse JSON with greedy match | "
                    f"error={str(e)} | content_preview={content[:200]}"
                )
                raise
        
        # If all else fails, try to parse the entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(
                f"❌ [PLANNER_V2] Failed to extract JSON from response | "
                f"error={str(e)} | content_preview={content[:500]}"
            )
            raise ValueError(f"Could not parse JSON from LLM response: {str(e)}")
    
    def _get_system_prompt(self, tool_catalog: str, metric_catalog: str, schema_catalog: str) -> str:
        """Get system prompt for planning."""
        # Use f-string since we're creating messages directly (no ChatPromptTemplate formatting)
        return f"""You are a planning agent that converts analytics queries into execution plans.

Available Tools:
{tool_catalog}

Available Metrics:
{metric_catalog}

Database Schema (for SQL generation):
{schema_catalog}

CRITICAL SQL GENERATION RULES:
1. **⚠️ YOU CAN ONLY USE TABLES LISTED IN THE SCHEMA CATALOG ABOVE**
2. **⚠️ DO NOT reference tables that are not in the "AVAILABLE TABLES" list**
3. **⚠️ DO NOT invent table names (e.g., shopify_refunds, refunds, etc.) - only use tables from the catalog**
4. **ALWAYS use the EXACT table names shown in the schema catalog above**
5. **NEVER use generic table names like 'orders', 'sales', 'products'**
6. **Use the FULL qualified name format: schema.table_name (e.g., 'public.shopify_orders')**
7. **Match column names exactly as shown in the schema catalog**
8. **Use parameterized queries with :param_name syntax**
9. **Reference the schema catalog for table relationships, column locations, and common query patterns**
10. **If a table you need is not in the catalog, you cannot use it - work with available tables only**

CRITICAL JOIN REQUIREMENTS:
- **IF you reference a column from a table, you MUST include the JOIN for that table**
- **NEVER reference columns from tables that aren't joined in the FROM/JOIN clauses**
- **Before selecting a column, verify the table is in the FROM clause or properly joined**
- **Use the relationships and join examples provided in the schema catalog above**

CHECKLIST before generating SQL with table references:
1. List ALL table aliases used in SELECT clause
2. For EACH alias, verify it is either:
   - The base table in FROM clause, OR
   - Explicitly joined with a JOIN statement
3. Reference the schema catalog above for:
   - Exact table names and column locations
   - Required JOINs based on relationships
   - Common query patterns and examples
   - Critical rules for SQL generation
4. DOUBLE-CHECK: Every table alias in SELECT must appear in FROM or JOIN clause

Your task is to create an execution plan (DAG) that:
1. Identifies which tools/tables to use based on the query requirements
2. Determines the order of operations
3. Specifies join keys and aggregations using the schema information provided
4. Handles dependencies between steps
5. For SQL queries, generate SQL directly using the schema catalog above

QUERY ANALYSIS GUIDELINES:
- **Analyze the query to understand what data breakdown is needed:**
  * Does it need aggregation by specific dimensions (e.g., by product, by SKU, by date)?
  * Does it need totals/aggregates without breakdown?
  * Use the schema catalog to identify which tables and columns are needed

- **CRITICAL: Understand the difference between filtering and grouping:**
  * **Filtering by entity**: Use WHERE clause to filter rows (e.g., WHERE product = 'SKU')
  * **Grouping by entity**: Use GROUP BY to aggregate by dimension (e.g., GROUP BY pv.sku)
  * **For "top N" queries with entities**: You typically need GROUP BY, not WHERE filtering
  * **Example**: "top 5 SKUs" means GROUP BY sku, ORDER BY revenue DESC, LIMIT 5
  * **NOT**: WHERE sku = 'something' (that would filter, not group)

- **Use the schema catalog to:**
  * Find the correct tables for the query
  * Identify required JOINs based on relationships
  * Locate columns in the correct tables (check "COLUMNS NOT IN THIS TABLE" warnings)
  * Follow common query patterns when applicable
  * Adhere to critical rules for SQL generation

- **For "top N" queries:**
  * Determine what dimension to break down by (e.g., SKU, product, campaign)
  * Use GROUP BY on the appropriate dimension column
  * Aggregate metrics (e.g., SUM revenue) per dimension
  * **ALWAYS include ORDER BY ... DESC LIMIT N directly in your SQL query**
  * **DO NOT create separate compute steps for top N - handle it in SQL**
  * **DO NOT** use WHERE clause to filter by the dimension - use GROUP BY instead

Return JSON with this structure:
{{
  "steps": [
    {{
      "id": "step1",
      "tool": "tool_id.capability_name",
      "inputs": {{
        "sql": "SELECT ... FROM schema.table_name WHERE ... GROUP BY ... ORDER BY ... DESC LIMIT N",
        "params": {{"param_name": "value"}},
        "inputs": {{"key": "value"}}
      }},
      "depends_on": [],
      "output_key": "result1"
    }}
  ],
  "outputs": {{"table": "final_step_id", "narrative": true}}
}}

CRITICAL PLANNING RULES:
- **For queries that need breakdown by dimension (e.g., "top N SKUs", "top N products"):**
  * Generate a SINGLE SQL step that includes: SELECT, FROM, JOINs, WHERE, GROUP BY, ORDER BY, LIMIT
  * Do NOT create separate steps for aggregation and top N - do it all in one SQL query
  * Example: "top 5 SKUs" should generate SQL like: SELECT pv.sku, SUM(...) AS revenue FROM ... JOIN ... WHERE ... GROUP BY pv.sku ORDER BY revenue DESC LIMIT 5
- **Only use compute tools when you need to:**
  * Join data from multiple different data sources (e.g., ads data + sales data)
  * Perform complex calculations that cannot be done in SQL
  * Transform data structure (e.g., pivot, reshape)
- **For simple queries with breakdown and top N, use SQL directly - no compute steps needed**

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
        # Use f-string since we're creating messages directly (no ChatPromptTemplate formatting)
        entities_json = json.dumps(task_spec.entities)
        filters_json = json.dumps(task_spec.filters)
        
        return f"""Create an execution plan for this query:

Intent: {task_spec.intent.value}
Metrics: {', '.join(task_spec.metrics)}
Entities: {entities_json}
Time Range: {task_spec.time.start} to {task_spec.time.end or 'now'}
Timezone: {task_spec.time.tz}
Filters: {filters_json}

Generate a plan that:
1. Fetches the required data from appropriate tools
2. Joins data if needed (e.g., ads spend + sales revenue for ROAS)
3. Computes requested metrics
4. Applies filters and aggregations
"""
    
    def _parse_plan(self, plan_data: Dict[str, Any], task_spec: TaskSpec) -> ExecutionPlan:
        """Parse LLM output into ExecutionPlan."""
        steps = []
        for step_data in plan_data.get('steps', []):
            tool = step_data.get('tool', '')
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
            metadata={'task_spec': task_spec.model_dump()}
        )
    
    def _normalize_tool_format(self, tool: str) -> str:
        """Normalize tool format."""
        if not tool or '.' in tool:
            return tool
        
        tool_id = tool
        if system_config.is_sql_tool(tool_id):
            return f"{tool_id}.sql"
        if system_config.is_compute_tool(tool_id):
            return f"{tool_id}.aggregate"
        
        tool_def = tool_registry.get_tool(tool_id)
        if tool_def and tool_def.capabilities:
            return f"{tool_id}.{tool_def.capabilities[0].name}"
        
        return tool
    
    def _enhance_plan(self, plan: ExecutionPlan, task_spec: TaskSpec) -> ExecutionPlan:
        """Enhance plan with additional context."""
        for step in plan.steps:
            tool_parts = step.tool.split('.')
            tool_id = tool_parts[0]
            
            if system_config.is_sql_tool(tool_id) or tool_id in system_config.get_date_range_tools():
                if 'inputs' not in step.inputs:
                    step.inputs['inputs'] = {}
                # Only set date range if LLM extracted it - no hardcoded defaults
                if task_spec.time.start:
                    step.inputs['inputs']['date_start'] = str(task_spec.time.start)
                if task_spec.time.end:
                    step.inputs['inputs']['date_end'] = str(task_spec.time.end)
                if task_spec.time.tz:
                    step.inputs['inputs']['timezone'] = task_spec.time.tz
        
        for step in plan.steps:
            if task_spec.entities:
                if 'channel' in task_spec.entities:
                    step.inputs.setdefault('inputs', {})['channel'] = task_spec.entities['channel']
                if 'geo' in task_spec.entities:
                    step.inputs.setdefault('inputs', {})['geo'] = task_spec.entities['geo']
        
        return plan
    
    def _validate_plan_sql(self, plan: ExecutionPlan):
        """Validate SQL queries in plan for missing JOINs and non-existent tables."""
        import re
        
        # Get available tables from schema context
        available_tables = set()
        if hasattr(self, '_last_schema_catalog_tables'):
            available_tables = self._last_schema_catalog_tables
        else:
            # Fallback: get from schema registry
            from app.services.schema_registry import schema_registry
            available_tables = set(schema_registry.tables.keys())
        
        for step in plan.steps:
            sql = step.inputs.get('sql')
            if not sql:
                continue
            
            # Extract all table references from SQL (FROM and JOIN clauses)
            # Pattern: FROM schema.table alias or JOIN schema.table alias
            # Handle both: "FROM public.shopify_orders o" and "FROM shopify_orders o"
            table_pattern = r'(?:FROM|JOIN)\s+([a-z_]+\.[a-z_]+|[a-z_]+)\s+(\w+)'
            table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
            
            for table_ref, alias in table_matches:
                # Parse table reference
                if '.' in table_ref:
                    schema, table_name = table_ref.split('.', 1)
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = f"public.{table_ref}"
                
                # Check if table exists in available tables
                if full_table_name not in available_tables:
                    # Try alternative formats
                    alt_names = [
                        f"public.{table_ref}",
                        table_ref,
                        full_table_name
                    ]
                    found = False
                    for alt_name in alt_names:
                        if alt_name in available_tables:
                            found = True
                            break
                    
                    if not found:
                        # Get a sample of available tables for error message (limit to 10)
                        sample_tables = sorted(list(available_tables))[:10]
                        raise ValueError(
                            f"Step {step.id} references non-existent table: {full_table_name}. "
                            f"Available tables (sample): {', '.join(sample_tables)}"
                        )
            
            # Extract table aliases used in SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if not select_match:
                continue
            
            select_clause = select_match.group(1)
            # Find table references (e.g., pv.product_id, o.order_id)
            table_refs = set(re.findall(r'(\w+)\.\w+', select_clause))
            
            # Extract FROM clause base table alias
            from_match = re.search(r'FROM\s+\S+\s+(\w+)', sql, re.IGNORECASE)
            if not from_match:
                continue
            
            joined_tables = {from_match.group(1)}  # Base table alias
            
            # Extract JOINed table aliases
            join_matches = re.findall(r'JOIN\s+\S+\s+(\w+)', sql, re.IGNORECASE)
            joined_tables.update(join_matches)
            
            # Check if all referenced tables are joined
            missing_joins = table_refs - joined_tables
            if missing_joins:
                logger.warning(
                    f"⚠️ [PLANNER_V2] Missing JOINs detected in SQL | "
                    f"step_id={step.id} | "
                    f"missing_tables={missing_joins} | "
                    f"referenced_tables={table_refs} | "
                    f"joined_tables={joined_tables}"
                )
                # Log the SQL for debugging
                logger.debug(f"🔍 [PLANNER_V2] SQL with missing JOINs: {sql[:500]}...")
                
                # Try to auto-fix missing JOINs
                fixed_sql = self._fix_missing_joins(sql, step.id)
                if fixed_sql != sql:
                    logger.info(
                        f"🔧 [PLANNER_V2] Auto-fixed SQL JOINs | "
                        f"step_id={step.id} | "
                        f"missing_tables={missing_joins}"
                    )
                    step.inputs['sql'] = fixed_sql
                    logger.debug(f"✅ [PLANNER_V2] Fixed SQL: {fixed_sql[:500]}...")
    
    def _fix_missing_joins(self, sql: str, step_id: str) -> str:
        """Automatically fix missing JOINs in SQL based on referenced columns."""
        import re
        
        # Detect table aliases used in SELECT but not joined
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql
        
        select_clause = select_match.group(1)
        table_refs = set(re.findall(r'(\w+)\.\w+', select_clause))
        
        # Get FROM and existing JOINs
        from_match = re.search(r'FROM\s+(\S+)\s+(\w+)', sql, re.IGNORECASE)
        if not from_match:
            return sql
        
        base_table = from_match.group(1)
        base_alias = from_match.group(2)
        joined_tables = {base_alias}
        
        # Extract existing JOINs
        join_matches = re.findall(r'JOIN\s+(\S+)\s+(\w+)', sql, re.IGNORECASE)
        for join_table, join_alias in join_matches:
            joined_tables.add(join_alias)
        
        missing_joins = table_refs - joined_tables
        
        if not missing_joins:
            return sql
        
        # Common JOIN patterns for sales_db
        join_patterns = {
            'pv': {
                'table': 'public.shopify_product_variants',
                'alias': 'pv',
                'condition': 'oli.variant_id = pv.variant_id',
                'requires': ['oli']  # Must have oli joined first
            },
            'oli': {
                'table': 'public.shopify_order_line_items',
                'alias': 'oli',
                'condition': f'{base_alias}.order_id = oli.order_id',
                'requires': []
            },
            'p': {
                'table': 'public.shopify_products',
                'alias': 'p',
                'condition': 'pv.product_id = p.product_id',
                'requires': ['pv']
            }
        }
        
        # Build fixed SQL with missing JOINs
        fixed_sql = sql
        joins_to_add = []
        
        # Sort missing joins by dependencies (oli before pv, pv before p)
        join_order = ['oli', 'pv', 'p']
        for alias in join_order:
            if alias in missing_joins and alias in join_patterns:
                pattern = join_patterns[alias]
                # Check if prerequisites are met
                if all(req in joined_tables for req in pattern['requires']):
                    join_statement = f"LEFT JOIN {pattern['table']} {pattern['alias']} ON {pattern['condition']}"
                    joins_to_add.append(join_statement)
                    joined_tables.add(alias)
                    logger.info(f"🔧 [PLANNER_V2] Will add JOIN: {join_statement}")
        
        # Insert JOINs before WHERE, GROUP BY, ORDER BY, or LIMIT
        if joins_to_add:
            insert_match = re.search(r'(\s+(WHERE|GROUP BY|ORDER BY|LIMIT)\s+)', fixed_sql, re.IGNORECASE)
            if insert_match:
                join_text = '\n' + '\n'.join(joins_to_add)
                fixed_sql = fixed_sql[:insert_match.start()] + join_text + fixed_sql[insert_match.start():]
            else:
                # Append at end
                fixed_sql += '\n' + '\n'.join(joins_to_add)
        
        return fixed_sql

