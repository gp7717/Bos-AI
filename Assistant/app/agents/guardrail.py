"""Guardrail Agent - Validates plans and queries for safety."""
from typing import List, Dict, Any
from app.models.schemas import ExecutionPlan, PlanStep, ValidationResult, TaskSpec
from app.services.tool_registry import tool_registry
from app.services.schema_registry import schema_registry
from app.services.metric_dictionary import metric_dictionary
from app.config.logging_config import get_logger
import re

logger = get_logger(__name__)


class GuardrailAgent:
    """Validates plans and queries against schemas and policies."""
    
    def __init__(self):
        """Initialize guardrail agent."""
        self.max_result_rows = 200000
        self.max_date_range_days = 365
    
    def validate_plan(self, plan: ExecutionPlan, task_spec: TaskSpec) -> ValidationResult:
        """Validate execution plan."""
        logger.info(f"🛡️ [GUARDRAIL] Validating execution plan | steps={len(plan.steps)}")
        errors = []
        warnings = []
        suggestions = []
        
        # Validate each step
        logger.debug(f"🔍 [GUARDRAIL] Validating individual steps")
        for step in plan.steps:
            step_errors, step_warnings, step_suggestions = self._validate_step(step, task_spec)
            errors.extend(step_errors)
            warnings.extend(step_warnings)
            suggestions.extend(step_suggestions)
            if step_errors:
                logger.debug(f"❌ [GUARDRAIL] Step validation errors | step_id={step.id} | errors={step_errors}")
        
        # Validate plan structure
        logger.debug(f"🔍 [GUARDRAIL] Validating plan structure")
        plan_errors, plan_warnings = self._validate_plan_structure(plan)
        errors.extend(plan_errors)
        warnings.extend(plan_warnings)
        
        # Validate metrics
        logger.debug(f"🔍 [GUARDRAIL] Validating metrics | metrics={task_spec.metrics}")
        metric_errors = self._validate_metrics(task_spec.metrics)
        errors.extend(metric_errors)
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
        logger.info(
            f"{'✅' if result.is_valid else '❌'} [GUARDRAIL] Plan validation completed | "
            f"is_valid={result.is_valid} | "
            f"errors={len(errors)} | "
            f"warnings={len(warnings)}"
        )
        
        return result
    
    def _validate_step(self, step: PlanStep, task_spec: TaskSpec) -> tuple:
        """Validate a single plan step."""
        errors = []
        warnings = []
        suggestions = []
        
        # Validate tool exists
        tool_parts = step.tool.split('.')
        if len(tool_parts) < 2:
            errors.append(f"Invalid tool format in step {step.id}: {step.tool}")
            return errors, warnings, suggestions
        
        tool_id = tool_parts[0]
        capability_name = tool_parts[1]
        
        tool = tool_registry.get_tool(tool_id)
        if not tool:
            logger.debug(f"❌ [GUARDRAIL] Unknown tool | step_id={step.id} | tool_id={tool_id}")
            errors.append(f"Unknown tool in step {step.id}: {tool_id}")
            return errors, warnings, suggestions
        
        # For SQL tools, capabilities are implicit (any capability name is allowed)
        # For other tools (api, compute), validate that capability exists
        if tool.kind == 'sql':
            logger.debug(f"✅ [GUARDRAIL] SQL tool validated | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
            # SQL tools don't have explicit capabilities - validate SQL safety instead
            sql_errors, sql_warnings = self._validate_sql_step(step)
            errors.extend(sql_errors)
            warnings.extend(sql_warnings)
        else:
            # Validate capability exists for non-SQL tools
            capability = tool_registry.get_capability(tool_id, capability_name)
            if not capability:
                logger.debug(f"❌ [GUARDRAIL] Unknown capability | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
                errors.append(f"Unknown capability in step {step.id}: {capability_name}")
                return errors, warnings, suggestions
            logger.debug(f"✅ [GUARDRAIL] Capability validated | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
        
        # Check quota limits
        if 'max_result_rows' in tool.quotas:
            max_rows = tool.quotas['max_result_rows']
            if max_rows < self.max_result_rows:
                warnings.append(f"Step {step.id} has row limit: {max_rows}")
        
        # Validate date ranges
        if 'inputs' in step.inputs:
            inputs = step.inputs.get('inputs', {})
            if 'date_start' in inputs and 'date_end' in inputs:
                date_range_valid, date_error = self._validate_date_range(
                    inputs['date_start'],
                    inputs['date_end']
                )
                if not date_range_valid:
                    errors.append(date_error)
        
        return errors, warnings, suggestions
    
    def _validate_sql_step(self, step: PlanStep) -> tuple:
        """Validate SQL-related step (returns errors, warnings)."""
        errors = []
        warnings = []
        
        # Check if SQL template is used (preferred over raw SQL)
        if 'sql_template' not in step.inputs.get('inputs', {}):
            # Allow it, but could add warning
            pass
        
        # If raw SQL is provided (LLM-generated), validate it
        if 'sql' in step.inputs:
            sql = step.inputs['sql']
            sql_errors = self._validate_sql_safety(sql)
            errors.extend(sql_errors)
            
            # Validate SQL against schema
            schema_errors, schema_warnings = self._validate_sql_against_schema(sql, step)
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)
        
        return errors, warnings
    
    def _validate_sql_against_schema(self, sql: str, step: PlanStep) -> tuple:
        """Validate LLM-generated SQL against schema registry."""
        errors = []
        warnings = []
        
        # Extract table names from SQL (handle FROM, JOIN, UPDATE, etc.)
        table_pattern = r'(?:FROM|JOIN|UPDATE|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)'
        tables = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Common incorrect table name mappings
        table_name_corrections = {
            'orders': 'public.shopify_orders',
            'order': 'public.shopify_orders',
            'sales': 'public.shopify_orders',
            'products': 'public.shopify_product_variants',
            'product': 'public.shopify_product_variants',
            'amazon_ads': 'public.amazon_product_metrics_daily',
            'meta_ads': 'public.dw_meta_ads_attribution',
            'google_ads': 'public.dw_google_ads_attribution',
        }
        
        if tables:
            for table in tables:
                original_table = table
                # Normalize table name
                if '.' not in table:
                    # Check if it's a known incorrect name
                    if table.lower() in table_name_corrections:
                        correct_table = table_name_corrections[table.lower()]
                        errors.append(
                            f"Invalid table name '{table}'. "
                            f"Use the correct table name: '{correct_table}'. "
                            f"Generic table names like 'orders', 'sales', etc. are not allowed."
                        )
                        logger.error(
                            f"❌ [GUARDRAIL] Invalid table name detected | "
                            f"invalid={table} | "
                            f"should_be={correct_table}"
                        )
                    else:
                        table = f'public.{table}'
                
                schema, table_name = table.split('.', 1)
                table_def = schema_registry.get_table(schema, table_name)
                
                if not table_def:
                    # Check if it's a known incorrect name
                    if original_table.lower() in table_name_corrections:
                        correct_table = table_name_corrections[original_table.lower()]
                        errors.append(
                            f"Invalid table name '{original_table}'. "
                            f"Use the correct table name: '{correct_table}'"
                        )
                    else:
                        warnings.append(
                            f"Table {table} not found in schema registry. "
                            f"Available tables: {', '.join(list(schema_registry.tables.keys())[:5])}..."
                        )
                else:
                    logger.debug(f"✅ [GUARDRAIL] Table validated | table={table}")
        
        # Check for parameterized queries (recommended)
        if ':' not in sql and '%' not in sql:
            warnings.append("SQL query does not appear to use parameterized queries")
        
        return errors, warnings
    
    def _validate_sql_safety(self, sql: str) -> List[str]:
        """Validate SQL for safety (no dangerous operations)."""
        errors = []
        sql_upper = sql.upper()
        
        # Block dangerous operations
        dangerous_patterns = [
            (r'\bDROP\s+TABLE\b', 'DROP TABLE not allowed'),
            (r'\bDELETE\s+FROM\b', 'DELETE not allowed'),
            (r'\bTRUNCATE\b', 'TRUNCATE not allowed'),
            (r'\bUPDATE\s+.*\s+SET\b', 'UPDATE not allowed'),
            (r'\bINSERT\s+INTO\b', 'INSERT not allowed'),
            (r'\bCREATE\s+TABLE\b', 'CREATE TABLE not allowed'),
            (r'\bALTER\s+TABLE\b', 'ALTER TABLE not allowed'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, sql_upper):
                errors.append(f"SQL safety violation: {message}")
        
        # Warn about SELECT * (but don't block)
        if re.search(r'SELECT\s+\*', sql_upper):
            errors.append("SQL warning: SELECT * should be avoided in production")
        
        return errors
    
    def _validate_date_range(self, start: str, end: str) -> tuple:
        """Validate date range is reasonable."""
        from datetime import datetime, timedelta
        
        try:
            start_date = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end.replace('Z', '+00:00'))
            
            if end_date < start_date:
                return False, "End date must be after start date"
            
            days_diff = (end_date - start_date).days
            if days_diff > self.max_date_range_days:
                return False, f"Date range exceeds maximum of {self.max_date_range_days} days"
            
            return True, None
        except Exception as e:
            return False, f"Invalid date format: {str(e)}"
    
    def _validate_plan_structure(self, plan: ExecutionPlan) -> tuple:
        """Validate plan DAG structure."""
        errors = []
        warnings = []
        
        # Check for cycles (basic check)
        step_ids = {step.id for step in plan.steps}
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} depends on unknown step: {dep}")
        
        # Check for orphaned steps
        referenced_steps = set()
        for step in plan.steps:
            referenced_steps.update(step.depends_on)
        
        root_steps = step_ids - referenced_steps
        if len(root_steps) == 0 and len(plan.steps) > 0:
            warnings.append("No root steps found (possible circular dependency)")
        
        return errors, warnings
    
    def _validate_metrics(self, metrics: List[str]) -> List[str]:
        """Validate requested metrics exist."""
        errors = []
        
        for metric in metrics:
            metric_def = metric_dictionary.get_metric(metric)
            if not metric_def:
                errors.append(f"Unknown metric: {metric}")
        
        return errors
    
    def validate_task_spec(self, task_spec: TaskSpec) -> ValidationResult:
        """Validate task specification."""
        logger.info(
            f"🛡️ [GUARDRAIL] Validating task spec | "
            f"intent={task_spec.intent.value} | "
            f"metrics={task_spec.metrics}"
        )
        errors = []
        warnings = []
        
        # Validate intent
        if task_spec.intent.value == 'unknown':
            warnings.append("Intent could not be determined")
            logger.warning(f"⚠️ [GUARDRAIL] Intent could not be determined")
        
        # Validate time range
        if task_spec.time.start and task_spec.time.end:
            logger.debug(f"📅 [GUARDRAIL] Validating date range | start={task_spec.time.start} | end={task_spec.time.end}")
            valid, error = self._validate_date_range(
                str(task_spec.time.start),
                str(task_spec.time.end)
            )
            if not valid:
                errors.append(error)
                logger.debug(f"❌ [GUARDRAIL] Date range validation failed | error={error}")
        
        # Validate metrics
        logger.debug(f"📊 [GUARDRAIL] Validating metrics | metrics={task_spec.metrics}")
        metric_errors = self._validate_metrics(task_spec.metrics)
        errors.extend(metric_errors)
        if metric_errors:
            logger.debug(f"❌ [GUARDRAIL] Metric validation errors | errors={metric_errors}")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=[]
        )
        
        logger.info(
            f"{'✅' if result.is_valid else '❌'} [GUARDRAIL] Task spec validation completed | "
            f"is_valid={result.is_valid} | "
            f"errors={len(errors)} | "
            f"warnings={len(warnings)}"
        )
        
        return result

