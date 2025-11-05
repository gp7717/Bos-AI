"""SQL Generator Agent - Uses LLM with schema context to generate SQL queries."""
import json
from typing import Dict, Any, Optional, List
from openai import AzureOpenAI
from app.config.settings import settings
from app.services.schema_context import SchemaContextService
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class SQLGeneratorAgent:
    """Generates SQL queries using LLM with schema context."""
    
    def __init__(self):
        """Initialize SQL generator agent."""
        logger.info(f"🔧 [SQL_GENERATOR] Initializing SQL Generator Agent | model={settings.azure_openai_deployment_name}")
        self.schema_context = SchemaContextService()
        try:
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.model = settings.azure_openai_deployment_name
            logger.info(f"✅ [SQL_GENERATOR] SQL Generator Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ [SQL_GENERATOR] Failed to initialize | error={str(e)}", exc_info=True)
            raise
    
    def generate_sql(
        self,
        query_description: str,
        tables: List[str],
        filters: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        entities: Optional[Dict[str, Any]] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL query using LLM with schema context."""
        logger.info(
            f"🔍 [SQL_GENERATOR] Generating SQL | "
            f"tables={tables} | "
            f"metrics={metrics} | "
            f"description={query_description[:100]}..."
        )
        
        # Get schema context for relevant tables
        table_contexts = []
        for table in tables:
            schema, table_name = table.split('.', 1) if '.' in table else ('public', table)
            context = self.schema_context.get_table_context(schema, table_name)
            if context:
                table_contexts.append(context)
        
        if not table_contexts:
            logger.warning(f"⚠️ [SQL_GENERATOR] No schema context found for tables | tables={tables}")
            return {
                'sql': None,
                'params': {},
                'error': 'No schema context available for specified tables'
            }
        
        # Format schema for LLM
        schema_prompt = self.schema_context.format_for_llm(table_contexts)
        
        # Build prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_sql_prompt(
            query_description,
            schema_prompt,
            filters,
            metrics,
            entities,
            date_start,
            date_end
        )
        
        logger.debug(
            f"🤖 [SQL_GENERATOR] Calling Azure OpenAI for SQL generation | "
            f"model={self.model} | "
            f"tables={len(table_contexts)}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more deterministic SQL
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            logger.info(
                f"✅ [SQL_GENERATOR] SQL generated successfully | "
                f"sql_length={len(result.get('sql', ''))} | "
                f"has_params={bool(result.get('params'))}"
            )
            
            return {
                'sql': result.get('sql'),
                'params': result.get('params', {}),
                'explanation': result.get('explanation', ''),
                'tables_used': tables
            }
            
        except Exception as e:
            logger.error(
                f"❌ [SQL_GENERATOR] SQL generation failed | "
                f"error={str(e)}",
                exc_info=True
            )
            return {
                'sql': None,
                'params': {},
                'error': str(e)
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for SQL generation."""
        return """You are an expert SQL query generator. Your task is to generate safe, efficient SQL queries based on schema information and user requirements.

Rules:
1. ALWAYS use parameterized queries (use :param_name for parameters, not string interpolation)
2. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, ALTER, etc.)
3. Use proper JOIN syntax (prefer explicit JOINs over WHERE clauses for joins)
4. Include appropriate WHERE clauses for date filtering
5. Use proper column names exactly as they appear in the schema
6. Handle NULL values appropriately
7. Consider performance (use indexes, limit result sets when appropriate)

Output format (JSON):
{
  "sql": "SELECT ... FROM ... WHERE ...",
  "params": {
    "param_name": "value"
  },
  "explanation": "Brief explanation of the query"
}

Important:
- Use :param_name syntax for parameters in SQL
- Never use string interpolation or f-strings
- Always validate table and column names against provided schema
- Include date range filters when dates are specified"""
    
    def _build_sql_prompt(
        self,
        query_description: str,
        schema_prompt: str,
        filters: Optional[Dict[str, Any]],
        metrics: Optional[List[str]],
        entities: Optional[Dict[str, Any]],
        date_start: Optional[str],
        date_end: Optional[str]
    ) -> str:
        """Build user prompt for SQL generation."""
        prompt_parts = [
            "Generate a SQL query based on the following requirements:",
            "",
            f"**Query Description:** {query_description}",
            ""
        ]
        
        if metrics:
            prompt_parts.append(f"**Metrics to Calculate:** {', '.join(metrics)}")
            prompt_parts.append("")
        
        if entities:
            prompt_parts.append(f"**Entities/Filters:** {json.dumps(entities, indent=2)}")
            prompt_parts.append("")
        
        if date_start or date_end:
            prompt_parts.append(f"**Date Range:** {date_start or 'start'} to {date_end or 'end'}")
            prompt_parts.append("")
        
        if filters:
            prompt_parts.append(f"**Additional Filters:** {json.dumps(filters, indent=2)}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "**Available Schema:**",
            schema_prompt,
            "",
            "Generate a parameterized SQL query that:",
            "1. Selects the required columns/metrics",
            "2. Applies date filters using :date_start and :date_end parameters",
            "3. Applies entity filters (channel, geo, etc.)",
            "4. Uses proper JOINs if multiple tables are needed",
            "5. Includes appropriate aggregations if needed",
            "",
            "Return the query as JSON with 'sql', 'params', and 'explanation' fields."
        ])
        
        return "\n".join(prompt_parts)

