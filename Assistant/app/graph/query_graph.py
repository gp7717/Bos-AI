"""LangGraph state graph for query processing."""
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from app.models.schemas import (
    TaskSpec, ExecutionPlan, PlanStep, QueryResponse,
    RouterAgentInput, RouterAgentOutput,
    PlannerAgentInput, PlannerAgentOutput,
    ComposerAgentInput, ComposerAgentOutput
)
from app.agents_v2.router_agent import RouterAgentV2
from app.agents_v2.planner_agent import PlannerAgentV2
from app.agents_v2.composer_agent import ComposerAgentV2
from app.agents.guardrail import GuardrailAgent
from app.services.mcp_database_client import get_mcp_client
from app.agents.computation import ComputationAgent
from app.services.system_config import system_config
from app.config.logging_config import get_logger
import pandas as pd
import uuid
from datetime import datetime

logger = get_logger(__name__)


class QueryState(TypedDict):
    """State structure for query processing graph."""
    query: str
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: str
    task_spec: Optional[TaskSpec]
    plan: Optional[ExecutionPlan]
    step_results: Dict[str, Any]
    execution_results: Dict[str, Any]
    response: Optional[QueryResponse]
    error: Optional[str]
    execution_trace: str


def create_query_graph() -> StateGraph:
    """Create LangGraph state graph for query processing."""
    
    # Initialize agents
    router = RouterAgentV2()
    planner = PlannerAgentV2()
    guardrail = GuardrailAgent()
    composer = ComposerAgentV2()
    computation = ComputationAgent()
    
    # Create graph
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("validate_task", validate_task_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("validate_plan", validate_plan_node)
    workflow.add_node("execute_plan", execute_plan_node)
    workflow.add_node("composer", composer_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add edges
    workflow.add_edge("router", "validate_task")
    workflow.add_conditional_edges(
        "validate_task",
        should_continue_after_validation,
        {
            "continue": "planner",
            "error": END
        }
    )
    workflow.add_edge("planner", "validate_plan")
    workflow.add_conditional_edges(
        "validate_plan",
        should_continue_after_plan_validation,
        {
            "continue": "execute_plan",
            "error": END
        }
    )
    workflow.add_edge("execute_plan", "composer")
    workflow.add_edge("composer", END)
    
    return workflow.compile()


async def router_node(state: QueryState) -> QueryState:
    """Router node - parse query into task spec."""
    logger.info(f"🔍 [GRAPH] Router node | query='{state['query'][:100]}...'")
    
    router = RouterAgentV2()
    
    # Create typed input
    router_input = RouterAgentInput(
        query=state["query"],
        user_id=state.get("user_id"),
        session_id=state.get("session_id")
    )
    
    # Execute with typed output
    result: RouterAgentOutput = await router.execute(router_input)
    
    if result.success:
        state["task_spec"] = result.task_spec
        logger.info(f"✅ [GRAPH] Router completed | intent={result.task_spec.intent.value if result.task_spec else 'unknown'}")
    else:
        state["error"] = result.error or "Router failed"
        logger.error(f"❌ [GRAPH] Router failed | error={state['error']}")
    
    return state


async def validate_task_node(state: QueryState) -> QueryState:
    """Validate task spec node."""
    logger.info(f"🛡️ [GRAPH] Validating task spec")
    
    guardrail = GuardrailAgent()
    task_spec = state.get("task_spec")
    
    if not task_spec:
        state["error"] = "Task spec is missing"
        return state
    
    validation = guardrail.validate_task_spec(task_spec)
    
    if not validation.is_valid:
        state["error"] = f"Task validation failed: {', '.join(validation.errors)}"
        logger.warning(f"⚠️ [GRAPH] Task validation failed | errors={validation.errors}")
    else:
        logger.info(f"✅ [GRAPH] Task validation passed")
    
    return state


def should_continue_after_validation(state: QueryState) -> str:
    """Determine next step after task validation."""
    if state.get("error"):
        return "error"
    return "continue"


async def planner_node(state: QueryState) -> QueryState:
    """Planner node - generate execution plan."""
    logger.info(f"📊 [GRAPH] Planner node")
    
    planner = PlannerAgentV2()
    
    # Create typed input
    planner_input = PlannerAgentInput(
        task_spec=state["task_spec"]
    )
    
    # Execute with typed output
    result: PlannerAgentOutput = await planner.execute(planner_input)
    
    if result.success:
        state["plan"] = result.plan
        logger.info(f"✅ [GRAPH] Planner completed | steps={len(result.plan.steps) if result.plan else 0}")
    else:
        state["error"] = result.error or "Planner failed"
        logger.error(f"❌ [GRAPH] Planner failed | error={state['error']}")
    
    return state


async def validate_plan_node(state: QueryState) -> QueryState:
    """Validate execution plan node."""
    logger.info(f"🛡️ [GRAPH] Validating plan")
    
    guardrail = GuardrailAgent()
    plan = state.get("plan")
    task_spec = state.get("task_spec")
    
    if not plan:
        state["error"] = "Plan is missing"
        return state
    
    validation = guardrail.validate_plan(plan, task_spec)
    
    if not validation.is_valid:
        state["error"] = f"Plan validation failed: {', '.join(validation.errors)}"
        logger.warning(f"⚠️ [GRAPH] Plan validation failed | errors={validation.errors}")
    else:
        logger.info(f"✅ [GRAPH] Plan validation passed")
    
    return state


def should_continue_after_plan_validation(state: QueryState) -> str:
    """Determine next step after plan validation."""
    if state.get("error"):
        return "error"
    return "continue"


async def execute_plan_node(state: QueryState) -> QueryState:
    """Execute plan node - run all plan steps."""
    logger.info(f"⚙️ [GRAPH] Executing plan")
    
    plan = state.get("plan")
    if not plan:
        state["error"] = "Plan is missing"
        return state
    
    step_results = {}
    executed_steps = set()
    
    # Execute steps in topological order
    while len(executed_steps) < len(plan.steps):
        ready_steps = [
            step for step in plan.steps
            if step.id not in executed_steps
            and all(dep in executed_steps for dep in step.depends_on)
        ]
        
        if not ready_steps:
            state["error"] = "Cannot resolve step dependencies"
            logger.error(f"❌ [GRAPH] Circular dependency detected")
            return state
        
        for step in ready_steps:
            logger.info(f"⚙️ [GRAPH] Executing step | step_id={step.id} | tool={step.tool}")
            
            try:
                # Pass task_spec to execute_step for top_n detection
                result = await execute_step(step, step_results, task_spec=state.get("task_spec"))
                step_results[step.id] = result
                executed_steps.add(step.id)
                logger.info(f"✅ [GRAPH] Step completed | step_id={step.id}")
            except Exception as e:
                logger.error(f"❌ [GRAPH] Step execution failed | step_id={step.id} | error={str(e)}")
                state["error"] = f"Step {step.id} failed: {str(e)}"
                return state
    
    # Aggregate results
    final_result = aggregate_results(plan, step_results)
    
    state["step_results"] = step_results
    state["execution_results"] = final_result
    state["execution_trace"] = build_execution_trace(plan, step_results)
    
    logger.info(f"✅ [GRAPH] Plan execution completed | row_count={final_result.get('row_count', 0)}")
    
    return state


async def execute_step(step: PlanStep, step_results: Dict[str, Any], task_spec: Optional[TaskSpec] = None) -> Dict[str, Any]:
    """Execute a single plan step."""
    import time
    import json
    start_time = time.time()
    
    # Log step inputs
    step_inputs_log = step.inputs.copy()
    if 'sql' in step_inputs_log and isinstance(step_inputs_log['sql'], str) and len(step_inputs_log['sql']) > 500:
        step_inputs_log['sql'] = step_inputs_log['sql'][:500] + "... (truncated)"
    
    logger.info(
        f"📥 [GRAPH_STEP] Input | "
        f"step_id={step.id} | "
        f"tool={step.tool} | "
        f"inputs={json.dumps(step_inputs_log, default=str)[:1000]}"
    )
    
    tool_parts = step.tool.split('.')
    tool_id = tool_parts[0]
    capability = tool_parts[1] if len(tool_parts) > 1 else None
    
    if system_config.is_tool_excluded(tool_id):
        return {
            'success': False,
            'error': f'Tool {tool_id} is excluded',
            'data': pd.DataFrame(),
            'row_count': 0,
            'columns': []
        }
    
    # Route to data access tools via MCP client
    if system_config.is_sql_tool(tool_id):
        mcp_client = get_mcp_client()
        
        # Extract SQL and params from step inputs
        sql = step.inputs.get('sql')
        params = step.inputs.get('params', {})
        
        if not sql:
            return {
                'success': False,
                'error': 'SQL query is required',
                'data': pd.DataFrame(),
                'row_count': 0,
                'columns': []
            }
        
        # Execute via MCP client
        result = await mcp_client.execute_query(sql, params)
        
        if result.get('success'):
            df_result = pd.DataFrame(result.get('data', []))
            step_result = {
                'success': True,
                'data': df_result,
                'row_count': result.get('row_count', 0),
                'columns': result.get('columns', [])
            }
        else:
            step_result = {
                'success': False,
                'error': result.get('error', 'Query execution failed'),
                'data': pd.DataFrame(),
                'row_count': 0,
                'columns': []
            }
        
        # Log outputs
        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"📤 [GRAPH_STEP] Output | "
            f"step_id={step.id} | "
            f"row_count={step_result['row_count']} | "
            f"columns={step_result['columns']} | "
            f"execution_time_ms={execution_time_ms:.2f}"
        )
        
        return step_result
    
    # Route to computation tools
    elif system_config.is_compute_tool(tool_id):
        computation = ComputationAgent()
        
        if capability == 'aggregate' or capability == 'join':
            # Get input data from previous steps
            left_step_id = step.inputs.get('inputs', {}).get('left')
            if not left_step_id and step.depends_on:
                left_step_id = step.depends_on[0]
            
            right_step_id = step.inputs.get('inputs', {}).get('right')
            
            left_data = step_results.get(left_step_id, {}).get('data')
            right_data = step_results.get(right_step_id, {}).get('data') if right_step_id else None
            
            if left_data is None:
                raise ValueError(f"No input data for step {step.id}")
            
            if right_data is not None and capability == 'join':
                join_keys = step.inputs.get('inputs', {}).get('join_keys', [])
                result_data = computation.join(left_data, right_data, join_keys)
            else:
                result_data = left_data
            
            # Apply formulas if any
            formulas = step.inputs.get('inputs', {}).get('formulas', [])
            for formula in formulas:
                if '=' in formula:
                    metric_id = formula.split('=', 1)[0].strip()
                    result_data = computation.compute(metric_id, result_data)
            
            step_result = {
                'success': True,
                'data': result_data,
                'row_count': len(result_data) if result_data is not None else 0,
                'columns': list(result_data.columns) if result_data is not None else []
            }
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"📤 [GRAPH_STEP] Output | "
                f"step_id={step.id} | "
                f"row_count={step_result['row_count']} | "
                f"columns={step_result['columns']} | "
                f"execution_time_ms={execution_time_ms:.2f}"
            )
            
            return step_result
        
        elif capability == 'filter':
            # Handle compute.filter for top N operations
            left_step_id = step.inputs.get('inputs', {}).get('left')
            if not left_step_id and step.depends_on:
                left_step_id = step.depends_on[0]
            
            left_data = step_results.get(left_step_id, {}).get('data')
            if left_data is None:
                raise ValueError(f"No input data for filter step {step.id}")
            
            # Extract top_n from step inputs or task_spec
            step_inputs = step.inputs.get('inputs', {})
            filters = step_inputs.get('filters', [])
            
            top_n_filter = None
            for f in filters:
                if isinstance(f, dict) and 'top_n' in f:
                    top_n_filter = f['top_n']
                    break
            
            if not top_n_filter and task_spec:
                for f in task_spec.filters:
                    if isinstance(f, dict) and 'top_n' in f:
                        top_n_filter = f['top_n']
                        break
            
            # Try to infer from step name
            if not top_n_filter:
                import re
                match = re.search(r'top[_\s]*(\d+)', step.id.lower())
                if match:
                    top_n_filter = int(match.group(1))
                else:
                    top_n_filter = 5  # Default
            
            # Determine sort column
            sort_by = step_inputs.get('sort_by', [])
            if not sort_by and isinstance(left_data, pd.DataFrame) and len(left_data.columns) > 0:
                numeric_cols = left_data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    sort_by = [numeric_cols[0]]
                else:
                    sort_by = [left_data.columns[0]]
            
            ascending = step_inputs.get('ascending', False)
            result_data = computation.top_n(left_data, sort_by, top_n_filter, ascending)
            
            logger.info(
                f"🔝 [GRAPH] Executed filter (top_n) operation | "
                f"step_id={step.id} | limit={top_n_filter} | sort_by={sort_by}"
            )
            
            step_result = {
                'success': True,
                'data': result_data,
                'row_count': len(result_data) if result_data is not None else 0,
                'columns': list(result_data.columns) if result_data is not None else []
            }
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"📤 [GRAPH_STEP] Output | "
                f"step_id={step.id} | "
                f"row_count={step_result['row_count']} | "
                f"columns={step_result['columns']} | "
                f"execution_time_ms={execution_time_ms:.2f}"
            )
            
            return step_result
        
        elif capability == 'calculate':
            left_step_id = step.inputs.get('inputs', {}).get('left')
            if not left_step_id and step.depends_on:
                left_step_id = step.depends_on[0]
            
            left_data = step_results.get(left_step_id, {}).get('data')
            if left_data is None:
                raise ValueError(f"No input data for calculation step {step.id}")
            
            # Check if this is a top N operation
            # 1. Check step name for "top_n" or "top" keywords
            # 2. Check step inputs filters
            # 3. Check task_spec filters
            step_inputs = step.inputs.get('inputs', {})
            filters = step_inputs.get('filters', [])
            
            # Check step name for top_n hint
            is_top_n_step = 'top_n' in step.id.lower() or 'top' in step.id.lower()
            
            top_n_filter = None
            
            # First, check step inputs filters
            for f in filters:
                if isinstance(f, dict) and 'top_n' in f:
                    top_n_filter = f['top_n']
                    break
            
            # If not found in step inputs, check task_spec filters
            if not top_n_filter and task_spec:
                for f in task_spec.filters:
                    if isinstance(f, dict) and 'top_n' in f:
                        top_n_filter = f['top_n']
                        break
            
            # If step name suggests top_n but no filter found, try to infer from step name
            if is_top_n_step and not top_n_filter:
                # Try to extract number from step name (e.g., "top_5" -> 5)
                import re
                match = re.search(r'top[_\s]*(\d+)', step.id.lower())
                if match:
                    top_n_filter = int(match.group(1))
                    logger.info(f"🔍 [GRAPH] Inferred top_n={top_n_filter} from step name {step.id}")
            
            # If top_n filter exists or step name suggests top_n, use top_n operation instead of compute
            if top_n_filter or is_top_n_step:
                if not top_n_filter:
                    # Default to 5 if step name suggests top_n but no number found
                    top_n_filter = 5
                    logger.info(f"🔍 [GRAPH] Using default top_n=5 for step {step.id}")
                # Determine sort column from step inputs or use first numeric column
                sort_by = step_inputs.get('sort_by', [])
                if not sort_by and isinstance(left_data, pd.DataFrame) and len(left_data.columns) > 0:
                    # Try to find a numeric column that looks like a metric
                    numeric_cols = left_data.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        sort_by = [numeric_cols[0]]  # Use first numeric column
                    else:
                        sort_by = [left_data.columns[0]]  # Fallback to first column
                
                ascending = step_inputs.get('ascending', False)
                result_data = computation.top_n(left_data, sort_by, top_n_filter, ascending)
                
                logger.info(
                    f"🔝 [GRAPH] Executed top_n operation | "
                    f"step_id={step.id} | limit={top_n_filter} | sort_by={sort_by}"
                )
            else:
                # Regular metric computation - requires valid metric_id
                metric_id = step.inputs.get('output_key') or step_inputs.get('metric_id')
                if not metric_id:
                    # Check if step.id looks like a metric (should be in metric dictionary)
                    from app.services.metric_dictionary import metric_dictionary
                    if metric_dictionary.get_metric(step.id):
                        metric_id = step.id
                    else:
                        raise ValueError(
                            f"Step {step.id} uses compute.calculate but no valid metric_id provided. "
                            f"For data transformations (like top N), use compute.aggregate with filters."
                        )
                
                result_data = computation.compute(metric_id, left_data)
            
            step_result = {
                'success': True,
                'data': result_data,
                'row_count': len(result_data) if result_data is not None else 0,
                'columns': list(result_data.columns) if result_data is not None else []
            }
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"📤 [GRAPH_STEP] Output | "
                f"step_id={step.id} | "
                f"row_count={step_result['row_count']} | "
                f"columns={step_result['columns']} | "
                f"execution_time_ms={execution_time_ms:.2f}"
            )
            
            return step_result
    
    # Log error output
    execution_time_ms = (time.time() - start_time) * 1000
    error_msg = f"Unknown tool or capability: {step.tool}"
    logger.info(
        f"📤 [GRAPH_STEP] Output (error) | "
        f"step_id={step.id} | "
        f"error={error_msg} | "
        f"execution_time_ms={execution_time_ms:.2f}"
    )
    
    raise ValueError(error_msg)


def aggregate_results(plan: ExecutionPlan, step_results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results from all steps."""
    output_key = plan.outputs.get('table', None)
    
    # Find the step that matches the output_key
    # First try to find by output_key, then fall back to last step
    target_step_id = None
    
    if output_key:
        # Look for a step with matching output_key
        for step in plan.steps:
            if step.output_key == output_key:
                target_step_id = step.id
                logger.debug(f"🔍 [GRAPH] Found output step by output_key | output_key={output_key} | step_id={target_step_id}")
                break
        
        # If not found by output_key, try using output_key as step_id directly
        if not target_step_id and output_key in step_results:
            target_step_id = output_key
            logger.debug(f"🔍 [GRAPH] Using output_key as step_id | step_id={target_step_id}")
    
    # Fallback to last step if no match found
    if not target_step_id and plan.steps:
        target_step_id = plan.steps[-1].id
        logger.debug(f"🔍 [GRAPH] Using last step as fallback | step_id={target_step_id}")
    
    if target_step_id and target_step_id in step_results:
        final_result = step_results[target_step_id]
        data = final_result.get('data', pd.DataFrame())
        
        logger.debug(
            f"📊 [GRAPH] Aggregating results | "
            f"target_step_id={target_step_id} | "
            f"data_type={type(data).__name__} | "
            f"is_empty={data.empty if isinstance(data, pd.DataFrame) else len(data) == 0 if isinstance(data, list) else 'N/A'}"
        )
        
        if isinstance(data, pd.DataFrame):
            if not data.empty:
                table = data.to_dict('records')
                logger.debug(f"✅ [GRAPH] Converted DataFrame to records | row_count={len(table)}")
            else:
                table = []
                logger.warning(f"⚠️ [GRAPH] DataFrame is empty | step_id={target_step_id}")
        elif isinstance(data, list):
            table = data
            logger.debug(f"✅ [GRAPH] Using list data directly | row_count={len(table)}")
        else:
            table = []
            logger.warning(f"⚠️ [GRAPH] Unknown data type | type={type(data).__name__} | step_id={target_step_id}")
        
        return {
            'data': table,
            'table': table,
            'row_count': len(table),
            'request_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow()
        }
    
    logger.warning(
        f"⚠️ [GRAPH] Could not find output step | "
        f"output_key={output_key} | "
        f"target_step_id={target_step_id} | "
        f"available_steps={list(step_results.keys())}"
    )
    
    return {
        'data': [],
        'table': [],
        'row_count': 0,
        'request_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow()
    }


def build_execution_trace(plan: ExecutionPlan, step_results: Dict[str, Any]) -> str:
    """Build execution trace string."""
    trace_parts = []
    for step in plan.steps:
        step_result = step_results.get(step.id, {})
        trace_parts.append(
            f"Step {step.id}: {step.tool} -> {step_result.get('row_count', 0)} rows"
        )
    return " | ".join(trace_parts)


async def composer_node(state: QueryState) -> QueryState:
    """Composer node - generate final answer."""
    logger.info(f"📝 [GRAPH] Composer node")
    
    composer = ComposerAgentV2()
    
    # Create typed input
    composer_input = ComposerAgentInput(
        task_spec=state["task_spec"],
        results=state.get("execution_results", {}),
        execution_trace=state.get("execution_trace")
    )
    
    # Execute with typed output
    result: ComposerAgentOutput = await composer.execute(composer_input)
    
    if result.success:
        response = result.response
        if response:
            response.request_id = state.get("request_id", str(uuid.uuid4()))
        state["response"] = response
        logger.info(f"✅ [GRAPH] Composer completed")
    else:
        state["error"] = result.error or "Composer failed"
        logger.error(f"❌ [GRAPH] Composer failed | error={state['error']}")
    
    return state

