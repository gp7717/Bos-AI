"""Guardrail Agent - Validates plans and queries for safety."""
from typing import List, Dict, Any
from app.models.schemas import ExecutionPlan, PlanStep, ValidationResult, TaskSpec
from app.services.tool_registry import tool_registry
from app.services.schema_registry import schema_registry
from app.services.metric_dictionary import metric_dictionary
from app.services.system_config import system_config
from app.models.schemas import SchemaTable
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
            # Provide helpful error message with format guidance
            tool_id = step.tool
            # Skip excluded tools
            if system_config.is_tool_excluded(tool_id):
                errors.append(system_config.get_message('amazon_disabled', tool_id=tool_id))
                return errors, warnings, suggestions
            if system_config.is_sql_tool(tool_id):
                suggested_format = f"{tool_id}.sql"
                errors.append(
                    f"Invalid tool format in step {step.id}: {step.tool}. "
                    f"{system_config.get_message('sql_tool_format', suggested_format=suggested_format)}"
                )
            elif system_config.is_compute_tool(tool_id):
                errors.append(
                    f"Invalid tool format in step {step.id}: {step.tool}. "
                    f"{system_config.get_message('compute_tool_format')}"
                )
            else:
                errors.append(
                    system_config.get_message('invalid_tool_format', step_id=step.id, tool=step.tool)
                )
            return errors, warnings, suggestions
        
        tool_id = tool_parts[0]
        capability_name = tool_parts[1]
        
        # Skip excluded tools
        if system_config.is_tool_excluded(tool_id):
            errors.append(system_config.get_message('amazon_disabled', tool_id=tool_id))
            return errors, warnings, suggestions
        
        tool = tool_registry.get_tool(tool_id)
        if not tool:
            logger.debug(f"❌ [GUARDRAIL] Unknown tool | step_id={step.id} | tool_id={tool_id}")
            errors.append(f"Unknown tool in step {step.id}: {tool_id}")
            return errors, warnings, suggestions
        
        # For SQL tools, capabilities are implicit (any capability name is allowed)
        # For compute tools, allow common capabilities even if not in registry
        # For other tools (api), validate that capability exists
        if tool.kind == 'sql':
            logger.debug(f"✅ [GUARDRAIL] SQL tool validated | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
            # SQL tools don't have explicit capabilities - validate SQL safety instead
            sql_errors, sql_warnings = self._validate_sql_step(step)
            errors.extend(sql_errors)
            warnings.extend(sql_warnings)
        elif tool.kind == 'compute':
            # For compute tools, allow standard capabilities: aggregate, join, calculate, filter
            valid_compute_capabilities = ['aggregate', 'join', 'calculate', 'filter']
            if capability_name in valid_compute_capabilities:
                logger.debug(f"✅ [GUARDRAIL] Compute capability validated | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
            else:
                # Check if it exists in registry
                capability = tool_registry.get_capability(tool_id, capability_name)
                if not capability:
                    logger.debug(f"❌ [GUARDRAIL] Unknown compute capability | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
                    errors.append(f"Unknown capability in step {step.id}: {capability_name}. Valid compute capabilities: {', '.join(valid_compute_capabilities)}")
                    return errors, warnings, suggestions
                logger.debug(f"✅ [GUARDRAIL] Compute capability validated (from registry) | step_id={step.id} | tool_id={tool_id} | capability={capability_name}")
        else:
            # Validate capability exists for non-SQL, non-compute tools (e.g., API tools)
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
            
            # Auto-correct common column mistakes first
            corrected_sql = self._auto_correct_column_mistakes(sql)
            if corrected_sql != sql:
                logger.info(f"🔧 [GUARDRAIL] Auto-corrected SQL in step validation | step_id={step.id}")
                # Update the step inputs with corrected SQL
                step.inputs['sql'] = corrected_sql
                sql = corrected_sql
            
            sql_errors, sql_warnings = self._validate_sql_safety(sql)
            errors.extend(sql_errors)
            warnings.extend(sql_warnings)
            
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
        
        # Get table name corrections from config
        table_name_corrections = system_config.get_table_name_corrections()
        
        # Block excluded table usage
        for table in tables:
            if system_config.is_table_excluded(table):
                errors.append(system_config.get_message('amazon_table_disabled', table=table))
                logger.error(f"❌ [GUARDRAIL] Excluded table detected and blocked | table={table}")
        
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
        
        # Auto-correct common column mistakes before validation
        corrected_sql = self._auto_correct_column_mistakes(sql)
        if corrected_sql != sql:
            logger.info(f"🔧 [GUARDRAIL] Auto-corrected SQL before validation | original_length={len(sql)} | corrected_length={len(corrected_sql)}")
            sql = corrected_sql
        
        # Validate columns exist in tables
        column_errors = self._validate_sql_columns(sql, tables)
        errors.extend(column_errors)
        
        # Check for parameterized queries (recommended)
        if ':' not in sql and '%' not in sql:
            warnings.append("SQL query does not appear to use parameterized queries")
        
        return errors, warnings
    
    def _validate_sql_columns(self, sql: str, tables: List[str]) -> List[str]:
        """Validate that columns referenced in SQL actually exist in the tables."""
        errors = []
        
        # Extract column references from SQL (SELECT, WHERE, GROUP BY, ORDER BY, etc.)
        # Pattern: table_alias.column_name or just column_name
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
        column_matches = re.findall(column_pattern, sql, re.IGNORECASE)
        
        # Also find standalone column names (that might be ambiguous)
        standalone_column_pattern = r'(?:SELECT|WHERE|GROUP BY|ORDER BY|HAVING)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        standalone_columns = re.findall(standalone_column_pattern, sql, re.IGNORECASE)
        
        # Build table alias map
        alias_map = {}
        for table in tables:
            # Extract alias if present (e.g., "FROM table t" -> alias is "t")
            table_alias_pattern = rf'\bFROM\s+{re.escape(table)}\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
            alias_match = re.search(table_alias_pattern, sql, re.IGNORECASE)
            if alias_match:
                alias_map[alias_match.group(1)] = table
            else:
                # No alias, use table name as alias
                table_name = table.split('.')[-1] if '.' in table else table
                alias_map[table_name] = table
        
        # Validate columns with table aliases
        for alias, column_name in column_matches:
            if alias not in alias_map:
                continue  # Skip if alias not found
            
            table_name = alias_map[alias]
            schema, table = table_name.split('.', 1) if '.' in table_name else ('public', table_name)
            table_def = schema_registry.get_table(schema, table)
            
            if table_def:
                column_names = [col.name.lower() for col in table_def.columns]
                if column_name.lower() not in column_names:
                    # Suggest similar columns or related tables
                    suggestions = self._suggest_column_alternatives(
                        column_name, table_def, schema, table
                    )
                    error_msg = (
                        f"Column '{column_name}' does not exist in table '{table_name}'. "
                        f"{suggestions}"
                    )
                    errors.append(error_msg)
                    logger.error(
                        f"❌ [GUARDRAIL] Invalid column detected | "
                        f"table={table_name} | "
                        f"column={column_name} | "
                        f"available_columns={column_names[:5]}"
                    )
        
        return errors
    
    def _suggest_column_alternatives(self, invalid_column: str, table_def: SchemaTable, 
                                     schema: str, table: str) -> str:
        """Suggest alternative columns or related tables when column doesn't exist."""
        suggestions = []
        
        # Check if column exists in related tables
        column_lower = invalid_column.lower()
        
        # Common column name patterns
        if 'order_date' in column_lower:
            # order_date does not exist - use created_at
            if 'shopify_orders' in table.lower() or 'orders' in table.lower():
                suggestions.append(
                    "The 'order_date' column does not exist in shopify_orders. "
                    "Use 'created_at' instead: "
                    "WHERE o.created_at BETWEEN :date_start AND :date_end"
                )
        
        if 'product_id' in column_lower:
            # product_id is often in related tables, not in order_line_items
            if 'order_line_items' in table.lower():
                suggestions.append(
                    "The 'product_id' column does not exist in shopify_order_line_items. "
                    "Use 'variant_id' instead, or JOIN with shopify_product_variants table "
                    "to get product_id: "
                    "JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id"
                )
            elif 'product_variants' in table.lower():
                suggestions.append("The 'product_id' column exists in shopify_product_variants. "
                                 "Make sure you're joining correctly.")
        
        # Check for similar column names
        similar_columns = []
        for col in table_def.columns:
            col_lower = col.name.lower()
            # Check for partial matches
            if any(word in col_lower for word in column_lower.split('_') if len(word) > 3):
                similar_columns.append(col.name)
        
        if similar_columns:
            suggestions.append(f"Did you mean: {', '.join(similar_columns[:3])}?")
        
        # Check foreign keys for related tables
        if table_def.foreign_keys:
            suggestions.append(
                f"Available relationships: {', '.join([fk.get('referred_table', '') for fk in table_def.foreign_keys[:2]])}"
            )
        
        return " ".join(suggestions) if suggestions else "Please check the table schema for available columns."
    
    def _auto_correct_column_mistakes(self, sql: str) -> str:
        """Auto-correct common column name mistakes in SQL."""
        column_corrections = system_config.get_column_corrections()
        corrected_sql = sql
        
        for correction in column_corrections:
            pattern = correction.get('pattern')
            replacement = correction.get('replacement')
            if pattern and replacement:
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    original = corrected_sql
                    corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                    if corrected_sql != original:
                        logger.info(
                            f"🔧 [GUARDRAIL] Auto-corrected column | "
                            f"pattern={pattern} | "
                            f"description={correction.get('description', '')}"
                        )
        
        return corrected_sql
    
    def _validate_sql_safety(self, sql: str) -> tuple[List[str], List[str]]:
        """Validate SQL for safety (no dangerous operations). Returns (errors, warnings)."""
        errors = []
        warnings = []
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
        
        # Warn about SELECT * (but don't block - use warning for testing)
        if re.search(r'SELECT\s+\*', sql_upper):
            warnings.append("SQL warning: SELECT * should be avoided in production")
        
        return errors, warnings
    
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

