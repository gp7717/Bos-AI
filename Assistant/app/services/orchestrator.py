"""Orchestrator - Main conductor implementing ReAct/Plan-&-Execute pattern."""
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent
from app.agents.guardrail import GuardrailAgent
from app.agents.data_access import get_data_access_agent
from app.agents.computation import ComputationAgent
from app.agents.composer import AnswerComposerAgent
from app.models.schemas import TaskSpec, ExecutionPlan, PlanStep, QueryResponse, AgentResponse
from app.config.logging_config import get_logger
import pandas as pd
import traceback
import json

logger = get_logger(__name__)


class Orchestrator:
    """Main orchestrator coordinating all agents."""
    
    def __init__(self):
        """Initialize orchestrator with all agents."""
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.guardrail = GuardrailAgent()
        self.computation = ComputationAgent()
        self.composer = AnswerComposerAgent()
        
        # Execution state
        self.step_results: Dict[str, Any] = {}
    
    async def process_query(self, query: str, user_id: Optional[str] = None, 
                           session_id: Optional[str] = None) -> QueryResponse:
        """Process user query end-to-end."""
        request_id = str(uuid.uuid4())
        
        logger.info(
            f"🎯 [ORCHESTRATOR] Starting query processing | "
            f"request_id={request_id} | "
            f"query='{query[:100]}...' | "
            f"user_id={user_id} | "
            f"session_id={session_id}"
        )
        
        try:
            # Step 1: Parse query (Router)
            logger.info(f"📋 [ORCHESTRATOR] Step 1: Parsing query with Router Agent | request_id={request_id}")
            task_spec = self.router.parse(query, user_id, session_id)
            logger.info(
                f"✅ [ORCHESTRATOR] Query parsed | "
                f"request_id={request_id} | "
                f"intent={task_spec.intent.value} | "
                f"metrics={task_spec.metrics} | "
                f"entities={task_spec.entities}"
            )
            logger.debug(f"📝 [ORCHESTRATOR] TaskSpec details | request_id={request_id} | task_spec={task_spec.model_dump_json()}")
            
            # Step 2: Validate task spec
            logger.info(f"🛡️ [ORCHESTRATOR] Step 2: Validating task spec with Guardrail | request_id={request_id}")
            task_validation = self.guardrail.validate_task_spec(task_spec)
            logger.debug(
                f"🔍 [ORCHESTRATOR] Task validation result | "
                f"request_id={request_id} | "
                f"is_valid={task_validation.is_valid} | "
                f"errors={task_validation.errors} | "
                f"warnings={task_validation.warnings}"
            )
            if not task_validation.is_valid:
                logger.warning(
                    f"❌ [ORCHESTRATOR] Task validation failed | "
                    f"request_id={request_id} | "
                    f"errors={task_validation.errors}"
                )
                return self._create_error_response(
                    request_id,
                    f"Invalid query: {', '.join(task_validation.errors)}"
                )
            
            # Step 3: Plan execution (Planner)
            logger.info(f"📊 [ORCHESTRATOR] Step 3: Generating execution plan with Planner | request_id={request_id}")
            plan = self.planner.plan(task_spec)
            logger.info(
                f"✅ [ORCHESTRATOR] Execution plan generated | "
                f"request_id={request_id} | "
                f"steps_count={len(plan.steps)} | "
                f"outputs={list(plan.outputs.keys())}"
            )
            logger.debug(f"📋 [ORCHESTRATOR] Plan details | request_id={request_id} | plan={json.dumps(plan.model_dump(), default=str)}")
            
            # Step 4: Validate plan (Guardrail)
            logger.info(f"🛡️ [ORCHESTRATOR] Step 4: Validating execution plan | request_id={request_id}")
            plan_validation = self.guardrail.validate_plan(plan, task_spec)
            logger.debug(
                f"🔍 [ORCHESTRATOR] Plan validation result | "
                f"request_id={request_id} | "
                f"is_valid={plan_validation.is_valid} | "
                f"errors={plan_validation.errors}"
            )
            if not plan_validation.is_valid:
                logger.warning(
                    f"❌ [ORCHESTRATOR] Plan validation failed | "
                    f"request_id={request_id} | "
                    f"errors={plan_validation.errors}"
                )
                return self._create_error_response(
                    request_id,
                    f"Invalid plan: {', '.join(plan_validation.errors)}"
                )
            
            # Step 5: Execute plan
            logger.info(f"⚙️ [ORCHESTRATOR] Step 5: Executing plan | request_id={request_id} | steps={len(plan.steps)}")
            execution_results = self._execute_plan(plan, task_spec)
            logger.info(
                f"✅ [ORCHESTRATOR] Plan execution completed | "
                f"request_id={request_id} | "
                f"row_count={execution_results.get('row_count', 0)}"
            )
            
            # Step 6: Compose answer
            logger.info(f"📝 [ORCHESTRATOR] Step 6: Composing answer with Composer | request_id={request_id}")
            execution_trace = self._build_execution_trace(plan, execution_results)
            logger.debug(f"🔗 [ORCHESTRATOR] Execution trace | request_id={request_id} | trace={execution_trace}")
            
            response = self.composer.compose(
                task_spec,
                execution_results,
                execution_trace=execution_trace
            )
            response.request_id = request_id
            
            logger.info(
                f"🎉 [ORCHESTRATOR] Query processing completed successfully | "
                f"request_id={request_id} | "
                f"answer_length={len(response.answer)}"
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"💥 [ORCHESTRATOR] Query processing failed | "
                f"request_id={request_id} | "
                f"error={str(e)}",
                exc_info=True
            )
            return self._create_error_response(
                request_id,
                f"Error processing query: {str(e)}",
                traceback=traceback.format_exc()
            )
    
    def _execute_plan(self, plan: ExecutionPlan, task_spec: TaskSpec) -> Dict[str, Any]:
        """Execute plan steps in order."""
        logger.debug(f"🔄 [ORCHESTRATOR] Starting plan execution | steps={len(plan.steps)}")
        self.step_results = {}
        
        # Execute steps in topological order (respecting dependencies)
        executed_steps = set()
        
        while len(executed_steps) < len(plan.steps):
            # Find steps ready to execute (no pending dependencies)
            ready_steps = [
                step for step in plan.steps
                if step.id not in executed_steps
                and all(dep in executed_steps for dep in step.depends_on)
            ]
            
            logger.debug(
                f"🔍 [ORCHESTRATOR] Finding ready steps | "
                f"executed={len(executed_steps)}/{len(plan.steps)} | "
                f"ready={len(ready_steps)}"
            )
            
            if not ready_steps:
                logger.error(
                    f"❌ [ORCHESTRATOR] Circular dependency or missing dependencies | "
                    f"executed_steps={executed_steps} | "
                    f"total_steps={[s.id for s in plan.steps]}"
                )
                raise ValueError("Cannot resolve step dependencies")
            
            # Execute ready steps
            for step in ready_steps:
                logger.info(
                    f"⚙️ [ORCHESTRATOR] ========== EXECUTING STEP ========== | "
                    f"step_id={step.id} | "
                    f"tool={step.tool} | "
                    f"depends_on={step.depends_on}"
                )
                logger.info(
                    f"📥 [ORCHESTRATOR] STEP INPUT | "
                    f"step_id={step.id} | "
                    f"inputs={step.inputs} | "
                    f"output_key={step.output_key}"
                )
                
                try:
                    result = self._execute_step(step, task_spec)
                    
                    logger.info(
                        f"📤 [ORCHESTRATOR] STEP OUTPUT | "
                        f"step_id={step.id} | "
                        f"success={result.get('success', False)} | "
                        f"row_count={result.get('row_count', 0)} | "
                        f"columns={result.get('columns', [])} | "
                        f"error={result.get('error', 'None')}"
                    )
                    
                    # Log data preview if available
                    if result.get('data') is not None:
                        data = result.get('data')
                        if isinstance(data, pd.DataFrame):
                            logger.info(
                                f"📊 [ORCHESTRATOR] STEP DATA PREVIEW | "
                                f"step_id={step.id} | "
                                f"shape={data.shape} | "
                                f"columns={list(data.columns)} | "
                                f"first_row={data.head(1).to_dict('records') if not data.empty else 'Empty DataFrame'}"
                            )
                        elif isinstance(data, list):
                            logger.info(
                                f"📊 [ORCHESTRATOR] STEP DATA PREVIEW | "
                                f"step_id={step.id} | "
                                f"list_length={len(data)} | "
                                f"first_item={data[0] if data else 'Empty list'}"
                            )
                    
                    self.step_results[step.id] = result
                    executed_steps.add(step.id)
                    
                    logger.info(
                        f"✅ [ORCHESTRATOR] Step completed | "
                        f"step_id={step.id} | "
                        f"success={result.get('success', False)} | "
                        f"row_count={result.get('row_count', 0)}"
                    )
                    logger.info(f"═══════════════════════════════════════")
                except Exception as e:
                    logger.error(
                        f"❌ [ORCHESTRATOR] Step execution failed | "
                        f"step_id={step.id} | "
                        f"tool={step.tool} | "
                        f"error={str(e)}",
                        exc_info=True
                    )
                    raise
        
        # Final aggregation/computation
        logger.info(f"📊 [ORCHESTRATOR] ========== AGGREGATING RESULTS ==========")
        logger.info(
            f"📥 [ORCHESTRATOR] Aggregation INPUT | "
            f"plan_outputs={plan.outputs} | "
            f"total_steps={len(plan.steps)} | "
            f"step_results_keys={list(self.step_results.keys())}"
        )
        
        final_result = self._aggregate_results(plan, task_spec)
        
        logger.info(
            f"📤 [ORCHESTRATOR] Aggregation OUTPUT | "
            f"row_count={final_result.get('row_count', 0)} | "
            f"table_length={len(final_result.get('table', []))} | "
            f"data_length={len(final_result.get('data', []))}"
        )
        
        if final_result.get('table'):
            logger.info(
                f"📊 [ORCHESTRATOR] Final result preview | "
                f"first_row={final_result.get('table', [])[0] if final_result.get('table') else 'Empty'}"
            )
        
        logger.info(f"═══════════════════════════════════════")
        
        return final_result
    
    def _execute_step(self, step: PlanStep, task_spec: TaskSpec) -> Dict[str, Any]:
        """Execute a single plan step."""
        tool_parts = step.tool.split('.')
        tool_id = tool_parts[0]
        
        logger.debug(
            f"🔧 [ORCHESTRATOR] Executing step | "
            f"step_id={step.id} | "
            f"tool_id={tool_id} | "
            f"tool_parts={tool_parts}"
        )
        
        # Route to appropriate agent
        if tool_id in ['sales_db', 'amazon_ads_db', 'meta_ads_db', 'google_ads_db', 'amazon_ads_api']:
            # Data access agent
            logger.info(f"📊 [ORCHESTRATOR] Routing to DataAccess agent | tool_id={tool_id} | step_id={step.id}")
            logger.info(f"📥 [ORCHESTRATOR] DataAccess INPUT | step_id={step.id} | tool_id={tool_id} | inputs={step.inputs}")
            
            agent = get_data_access_agent(tool_id)
            if agent:
                result = agent.execute(step.inputs)
                
                logger.info(
                    f"📤 [ORCHESTRATOR] DataAccess OUTPUT | "
                    f"step_id={step.id} | "
                    f"tool_id={tool_id} | "
                    f"row_count={result.get('row_count', 0)} | "
                    f"columns={result.get('columns', [])} | "
                    f"data_preview={result.get('data', [])[:2] if isinstance(result.get('data'), list) else 'N/A'}"
                )
                
                df_result = pd.DataFrame(result.get('data', []))
                
                logger.info(
                    f"📊 [ORCHESTRATOR] DataAccess DataFrame | "
                    f"step_id={step.id} | "
                    f"shape={df_result.shape} | "
                    f"columns={list(df_result.columns)} | "
                    f"is_empty={df_result.empty}"
                )
                
                return {
                    'success': True,
                    'data': df_result,
                    'row_count': result.get('row_count', 0),
                    'columns': result.get('columns', [])
                }
            else:
                logger.warning(f"⚠️ [ORCHESTRATOR] DataAccess agent not found | tool_id={tool_id}")
        
        elif tool_id == 'compute':
            # Computation agent
            capability = tool_parts[1] if len(tool_parts) > 1 else 'aggregate'
            logger.debug(f"🧮 [ORCHESTRATOR] Routing to Computation agent | capability={capability}")
            
            if capability == 'aggregate':
                # Get input data from previous steps
                # The planner may set 'left' to refer to the previous step, or use 'input' or step depends_on
                left_step_id = step.inputs.get('inputs', {}).get('left')
                if not left_step_id and step.depends_on:
                    # If no explicit left step, use the first dependency
                    left_step_id = step.depends_on[0]
                    logger.debug(f"🔗 [ORCHESTRATOR] Using first dependency as left_step_id | left_step_id={left_step_id}")
                
                right_step_id = step.inputs.get('inputs', {}).get('right')
                
                logger.info(
                    f"🔗 [ORCHESTRATOR] Preparing computation | "
                    f"step_id={step.id} | "
                    f"left_step_id={left_step_id} | "
                    f"right_step_id={right_step_id} | "
                    f"depends_on={step.depends_on} | "
                    f"available_steps={list(self.step_results.keys())}"
                )
                
                # Log all available step results
                for step_id, step_result in self.step_results.items():
                    logger.debug(
                        f"📋 [ORCHESTRATOR] Available step result | "
                        f"step_id={step_id} | "
                        f"row_count={step_result.get('row_count', 0)} | "
                        f"columns={step_result.get('columns', [])}"
                    )
                
                left_result = self.step_results.get(left_step_id, {})
                left_data = left_result.get('data')
                
                logger.info(
                    f"📥 [ORCHESTRATOR] Compute INPUT | "
                    f"step_id={step.id} | "
                    f"left_step_id={left_step_id} | "
                    f"left_result_keys={list(left_result.keys())} | "
                    f"left_data_type={type(left_data).__name__} | "
                    f"left_data_exists={left_data is not None}"
                )
                
                if left_data is not None:
                    if isinstance(left_data, pd.DataFrame):
                        logger.info(
                            f"📊 [ORCHESTRATOR] Left DataFrame | "
                            f"step_id={step.id} | "
                            f"shape={left_data.shape} | "
                            f"columns={list(left_data.columns)} | "
                            f"is_empty={left_data.empty} | "
                            f"first_row={left_data.head(1).to_dict('records') if not left_data.empty else 'Empty'}"
                        )
                    else:
                        logger.info(f"📊 [ORCHESTRATOR] Left data (non-DataFrame) | step_id={step.id} | type={type(left_data).__name__} | value={str(left_data)[:200]}")
                
                right_data = self.step_results.get(right_step_id, {}).get('data') if right_step_id else None
                
                logger.info(
                    f"📊 [ORCHESTRATOR] Retrieved data | "
                    f"step_id={step.id} | "
                    f"left_data_type={type(left_data).__name__} | "
                    f"left_data_rows={len(left_data) if hasattr(left_data, '__len__') and not isinstance(left_data, str) else 'N/A'} | "
                    f"right_data={right_data is not None}"
                )
                
                if right_data is not None:
                    # Join
                    join_keys = step.inputs.get('inputs', {}).get('join_keys', [])
                    logger.info(
                        f"🔗 [ORCHESTRATOR] Joining dataframes | "
                        f"left_rows={len(left_data) if left_data is not None else 0} | "
                        f"right_rows={len(right_data) if right_data is not None else 0} | "
                        f"join_keys={join_keys}"
                    )
                    
                    # Log column names for debugging
                    if left_data is not None and len(left_data) > 0:
                        logger.debug(f"📊 [ORCHESTRATOR] Left dataframe columns | columns={list(left_data.columns)}")
                    if right_data is not None and len(right_data) > 0:
                        logger.debug(f"📊 [ORCHESTRATOR] Right dataframe columns | columns={list(right_data.columns)}")
                    
                    result_data = self.computation.join(left_data, right_data, join_keys)
                    logger.info(
                        f"{'✅' if len(result_data) > 0 else '⚠️'} [ORCHESTRATOR] Join completed | "
                        f"result_rows={len(result_data)} | "
                        f"left_rows={len(left_data) if left_data is not None else 0} | "
                        f"right_rows={len(right_data) if right_data is not None else 0}"
                    )
                else:
                    result_data = left_data
                    if result_data is None:
                        logger.warning(
                            f"⚠️ [ORCHESTRATOR] left_data is None for compute step | "
                            f"step_id={step.id} | "
                            f"left_step_id={left_step_id} | "
                            f"available_steps={list(self.step_results.keys())}"
                        )
                    else:
                        logger.debug(
                            f"📊 [ORCHESTRATOR] Using single dataframe | "
                            f"rows={len(result_data) if result_data is not None else 0} | "
                            f"columns={list(result_data.columns) if hasattr(result_data, 'columns') else 'N/A'}"
                        )
                
                # Apply formulas (if any)
                formulas = step.inputs.get('inputs', {}).get('formulas', [])
                logger.debug(f"🧮 [ORCHESTRATOR] Applying formulas | formulas={formulas}")
                
                if formulas:
                    for formula in formulas:
                        # Parse formula (e.g., "roas=revenue/spend")
                        if '=' in formula:
                            metric_id, formula_expr = formula.split('=', 1)
                            logger.debug(f"🔢 [ORCHESTRATOR] Computing metric | metric_id={metric_id.strip()} | formula={formula_expr}")
                            if result_data is not None:
                                result_data = self.computation.compute(metric_id.strip(), result_data)
                            else:
                                logger.warning(f"⚠️ [ORCHESTRATOR] Cannot apply formula, result_data is None")
                
                # Ensure result_data is a DataFrame
                if result_data is None:
                    logger.error(f"❌ [ORCHESTRATOR] result_data is None after computation | step_id={step.id}")
                    return {
                        'success': False,
                        'error': 'No data available for computation',
                        'data': pd.DataFrame(),
                        'row_count': 0,
                        'columns': []
                    }
                
                # Log final result before returning
                logger.info(
                    f"📤 [ORCHESTRATOR] Compute OUTPUT | "
                    f"step_id={step.id} | "
                    f"result_rows={len(result_data)} | "
                    f"result_columns={list(result_data.columns) if hasattr(result_data, 'columns') else 'N/A'} | "
                    f"is_empty={result_data.empty if isinstance(result_data, pd.DataFrame) else 'N/A'}"
                )
                
                if isinstance(result_data, pd.DataFrame) and not result_data.empty:
                    logger.info(
                        f"📊 [ORCHESTRATOR] Compute result preview | "
                        f"step_id={step.id} | "
                        f"first_row={result_data.head(1).to_dict('records')}"
                    )
                elif isinstance(result_data, pd.DataFrame) and result_data.empty:
                    logger.warning(f"⚠️ [ORCHESTRATOR] Compute result is empty DataFrame | step_id={step.id}")
                
                return {
                    'success': True,
                    'data': result_data,
                    'row_count': len(result_data) if result_data is not None else 0,
                    'columns': list(result_data.columns) if result_data is not None else []
                }
        
        logger.error(f"❌ [ORCHESTRATOR] Unknown tool | tool_id={tool_id}")
        return {
            'success': False,
            'error': f'Unknown tool: {tool_id}'
        }
    
    def _aggregate_results(self, plan: ExecutionPlan, task_spec: TaskSpec) -> Dict[str, Any]:
        """Aggregate results from all steps."""
        # Get final step output
        output_step_id = plan.outputs.get('table', plan.steps[-1].id if plan.steps else None)
        
        logger.debug(
            f"📊 [ORCHESTRATOR] Aggregating results | "
            f"output_step_id={output_step_id} | "
            f"available_steps={list(self.step_results.keys())} | "
            f"plan_outputs={plan.outputs}"
        )
        
        if output_step_id and output_step_id in self.step_results:
            final_result = self.step_results[output_step_id]
            data = final_result.get('data', pd.DataFrame())
            
            logger.debug(
                f"📋 [ORCHESTRATOR] Final result found | "
                f"step_id={output_step_id} | "
                f"data_type={type(data).__name__} | "
                f"data_is_empty={data.empty if isinstance(data, pd.DataFrame) else 'N/A'}"
            )
            
            # Convert to list of dicts for JSON serialization
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    table = data.to_dict('records')
                    logger.info(
                        f"✅ [ORCHESTRATOR] Converted DataFrame to table | "
                        f"rows={len(table)} | "
                        f"columns={list(data.columns)}"
                    )
                else:
                    table = []
                    logger.warning(f"⚠️ [ORCHESTRATOR] DataFrame is empty | step_id={output_step_id}")
            elif isinstance(data, list):
                table = data
                logger.debug(f"✅ [ORCHESTRATOR] Data is already a list | rows={len(table)}")
            else:
                table = []
                logger.warning(
                    f"⚠️ [ORCHESTRATOR] Unexpected data type | "
                    f"type={type(data).__name__} | "
                    f"data={str(data)[:200]}"
                )
            
            return {
                'data': table,
                'table': table,
                'row_count': len(table),
                'request_id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow()
            }
        
        # Fallback: try to get data from the last step
        if plan.steps:
            last_step_id = plan.steps[-1].id
            logger.warning(
                f"⚠️ [ORCHESTRATOR] Output step not found, using last step | "
                f"output_step_id={output_step_id} | "
                f"last_step_id={last_step_id} | "
                f"available_steps={list(self.step_results.keys())}"
            )
            
            if last_step_id in self.step_results:
                final_result = self.step_results[last_step_id]
                data = final_result.get('data', pd.DataFrame())
                
                if isinstance(data, pd.DataFrame) and not data.empty:
                    table = data.to_dict('records')
                    logger.info(f"✅ [ORCHESTRATOR] Using last step data | rows={len(table)}")
                    return {
                        'data': table,
                        'table': table,
                        'row_count': len(table),
                        'request_id': str(uuid.uuid4()),
                        'timestamp': datetime.utcnow()
                    }
        
        logger.error(
            f"❌ [ORCHESTRATOR] No results found | "
            f"output_step_id={output_step_id} | "
            f"available_steps={list(self.step_results.keys())}"
        )
        
        return {
            'data': [],
            'table': [],
            'row_count': 0,
            'request_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow()
        }
    
    def _build_execution_trace(self, plan: ExecutionPlan, results: Dict[str, Any]) -> str:
        """Build execution trace for answer composition."""
        trace_parts = []
        
        for step in plan.steps:
            step_result = self.step_results.get(step.id, {})
            trace_parts.append(
                f"Step {step.id}: {step.tool} -> {step_result.get('row_count', 0)} rows"
            )
        
        return " | ".join(trace_parts)
    
    def _create_error_response(self, request_id: str, error_message: str, 
                               traceback: Optional[str] = None) -> QueryResponse:
        """Create error response."""
        return QueryResponse(
            answer=f"I encountered an error: {error_message}",
            data=None,
            table=None,
            chart_spec=None,
            reasoning_trace=traceback or error_message,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )

