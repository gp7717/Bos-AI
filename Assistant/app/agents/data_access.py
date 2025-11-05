"""Data Access Agents - Domain-specific data retrieval."""
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.config.settings import settings
from app.services.schema_registry import schema_registry
from app.services.system_config import system_config
from app.config.logging_config import get_logger
import pandas as pd
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(__name__)


class DataAccessAgent:
    """Base class for data access agents."""
    
    def __init__(self, tool_id: str):
        """Initialize data access agent."""
        self.tool_id = tool_id
        self.engine: Optional[Engine] = None
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data access query."""
        raise NotImplementedError


class SalesDBAgent(DataAccessAgent):
    """Sales database agent."""
    
    def __init__(self):
        """Initialize sales DB agent."""
        super().__init__("sales_db")
        logger.info(f"🔧 [SALES_DB] Initializing SalesDBAgent")
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
        
        # Get schema information from registry
        primary_table = system_config.get_tool_primary_table(self.tool_id)
        if primary_table:
            schema, table_name = primary_table.split('.', 1)
            self.table_schema = schema_registry.get_table(schema, table_name)
            if self.table_schema:
                logger.info(f"✅ [SALES_DB] Schema loaded | table={table_name} | columns={len(self.table_schema.columns)}")
            else:
                logger.warning(f"⚠️ [SALES_DB] Schema not found for {primary_table}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sales database query."""
        query_inputs = inputs.get('inputs', {})
        
        logger.info(
            f"📊 [SALES_DB] Executing query | "
            f"date_start={query_inputs.get('date_start')} | "
            f"date_end={query_inputs.get('date_end')} | "
            f"geo={query_inputs.get('geo')}"
        )
        
        # Check if SQL is provided (LLM-generated) or use template-based
        if 'sql' in inputs:
            sql = inputs['sql']
            params = inputs.get('params', {})
            
            # Auto-correct common table name and column mistakes
            sql = self._correct_table_names(sql)
            
            logger.info(
                f"🔍 [SALES_DB] Using LLM-generated SQL | "
                f"sql={sql[:200]}... | params={params}"
            )
        else:
            # Build SQL query using template-based approach
            sql, params = self._build_query(query_inputs)
            logger.info(f"🔍 [SALES_DB] Generated SQL (template-based) | sql={sql} | params={params}")
        
        try:
            # Execute query with parameterized values for safety
            with self.engine.connect() as conn:
                result = pd.read_sql(text(sql), conn, params=params)
            
            logger.info(
                f"✅ [SALES_DB] Query executed successfully | "
                f"row_count={len(result)} | "
                f"columns={list(result.columns)}"
            )
            
            if len(result) == 0:
                logger.warning(
                    f"⚠️ [SALES_DB] Query returned 0 rows | "
                    f"date_start={query_inputs.get('date_start')} | "
                    f"date_end={query_inputs.get('date_end')}"
                )
            
            return {
                'data': result.to_dict('records'),
                'row_count': len(result),
                'columns': list(result.columns)
            }
        except Exception as e:
            error_str = str(e)
            logger.error(
                f"❌ [SALES_DB] Query execution failed | "
                f"error={error_str} | "
                f"sql={sql[:200]}...",
                exc_info=True
            )
            
            # Provide helpful error messages for common issues
            if 'does not exist' in error_str or 'UndefinedColumn' in error_str:
                # Extract column name from error
                import re
                column_match = re.search(r"column\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+does not exist", error_str, re.IGNORECASE)
                if column_match:
                    column_name = column_match.group(1)
                    suggestion = self._suggest_column_fix(column_name, sql)
                    if suggestion:
                        error_msg = f"{error_str}\n\n💡 Suggestion: {suggestion}"
                        logger.info(f"💡 [SALES_DB] Column fix suggestion | column={column_name} | suggestion={suggestion}")
                        raise ValueError(error_msg) from e
            
            raise
    
    def _build_query(self, inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build SQL query from inputs using schema registry."""
        params = {}
        
        # Get date column from schema registry
        date_column = schema_registry.get_date_column('public', 'shopify_orders')
        if not date_column:
            date_column = 'created_at'  # Fallback
            logger.debug(f"⚠️ [SALES_DB] Date column not found in schema, using default: {date_column}")
        else:
            logger.debug(f"✅ [SALES_DB] Using date column from schema: {date_column}")
        
        # Build base query - check if columns exist in schema
        base_query = """
        SELECT 
            o.order_id,
            o.order_name,
            o.created_at,
            o.total_price_amount as revenue,
            o.ship_city,
            o.ship_country
        """
        
        # Add join columns if tables exist in schema
        line_items_table = schema_registry.get_table('public', 'shopify_order_line_items')
        if line_items_table:
            base_query += """,
            li.quantity,
            li.discounted_unit_price_amount as unit_price"""
        
        variants_table = schema_registry.get_table('public', 'shopify_product_variants')
        if variants_table:
            base_query += """,
            pv.sku,
            pv.product_title"""
        
        base_query += """
        FROM public.shopify_orders o
        """
        
        # Add joins if tables exist
        if line_items_table:
            base_query += """
        LEFT JOIN public.shopify_order_line_items li ON o.order_id = li.order_id
        """
        if variants_table:
            base_query += """
        LEFT JOIN public.shopify_product_variants pv ON li.variant_id = pv.variant_id
        """
        
        base_query += " WHERE 1=1"
        
        # Add date filter with proper parameterization
        if 'date_start' in inputs and inputs['date_start']:
            base_query += f" AND o.{date_column} >= :date_start"
            # Convert date string to proper format for timestamp comparison
            date_start = inputs['date_start']
            if isinstance(date_start, str):
                # If it's just a date, add time component
                if 'T' not in date_start and ' ' not in date_start:
                    date_start = f"{date_start} 00:00:00"
            params['date_start'] = date_start
            logger.debug(f"📅 [SALES_DB] Added date_start filter | date_start={date_start}")
        
        if 'date_end' in inputs and inputs['date_end']:
            base_query += f" AND o.{date_column} <= :date_end"
            date_end = inputs['date_end']
            if isinstance(date_end, str):
                # If it's just a date, add time component for end of day
                if 'T' not in date_end and ' ' not in date_end:
                    date_end = f"{date_end} 23:59:59"
            params['date_end'] = date_end
            logger.debug(f"📅 [SALES_DB] Added date_end filter | date_end={date_end}")
        
        # Add geo filter with parameterization
        if 'geo' in inputs and inputs['geo']:
            base_query += " AND LOWER(o.ship_city) LIKE :geo"
            params['geo'] = f"%{inputs['geo'].lower()}%"
            logger.debug(f"🌍 [SALES_DB] Added geo filter | geo={inputs['geo']}")
        
        # Add limit
        limit = inputs.get('limit', 10000)
        base_query += f" LIMIT :limit"
        params['limit'] = limit
        
        return base_query, params
    
    def _correct_table_names(self, sql: str) -> str:
        """Auto-correct common table name and column mistakes in SQL."""
        import re
        
        # Get SQL corrections from config
        sql_corrections = system_config.get_sql_corrections()
        column_corrections = system_config.get_column_corrections()
        
        corrected_sql = sql
        
        # Apply table name corrections
        for correction in sql_corrections:
            pattern = correction.get('pattern')
            replacement = correction.get('replacement')
            if pattern and replacement:
                # Only replace if the table is not already qualified (doesn't have a dot)
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    # Check if already qualified
                    match = re.search(pattern, corrected_sql, re.IGNORECASE)
                    if match and '.' not in match.group(0):
                        corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
        
        # Apply column name corrections
        for correction in column_corrections:
            pattern = correction.get('pattern')
            replacement = correction.get('replacement')
            description = correction.get('description', '')
            if pattern and replacement:
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    original_sql = corrected_sql
                    # Use regex replacement with backreferences
                    corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                    if corrected_sql != original_sql:
                        logger.warning(
                            f"⚠️ [SALES_DB] Auto-corrected column name | "
                            f"pattern={pattern} | "
                            f"description={description}"
                        )
        
        if corrected_sql != sql:
            logger.warning(
                f"⚠️ [SALES_DB] Auto-corrected SQL | "
                f"original={sql[:150]}... | corrected={corrected_sql[:150]}..."
            )
        
        return corrected_sql
    
    def _suggest_column_fix(self, invalid_column: str, sql: str) -> Optional[str]:
        """Suggest fixes for invalid column names."""
        suggestions = []
        column_lower = invalid_column.lower()
        
        # Check which table is being queried
        if 'order_line_items' in sql.lower() or 'oli' in sql.lower():
            if 'product_id' in column_lower:
                suggestions.append(
                    "The 'product_id' column does not exist in shopify_order_line_items. "
                    "Use 'variant_id' instead, or JOIN with shopify_product_variants to get product_id:\n"
                    "JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id\n"
                    "Then use: pv.product_id"
                )
        
        # Check for date column issues
        if 'order_date' in column_lower:
            suggestions.append(
                "The 'order_date' column does not exist in shopify_orders. "
                "Use 'created_at' instead: "
                "WHERE o.created_at BETWEEN :date_start AND :date_end"
            )
        
        # Get actual columns from schema
        if 'order_line_items' in sql.lower():
            table_def = schema_registry.get_table('public', 'shopify_order_line_items')
            if table_def:
                available_columns = [col.name for col in table_def.columns]
                # Find similar columns
                similar = [col for col in available_columns if any(word in col.lower() for word in column_lower.split('_') if len(word) > 3)]
                if similar:
                    suggestions.append(f"Similar columns available: {', '.join(similar[:3])}")
        
        return "\n".join(suggestions) if suggestions else None


class AmazonAdsDBAgent(DataAccessAgent):
    """Amazon Ads database agent."""
    
    def __init__(self):
        """Initialize Amazon Ads DB agent."""
        super().__init__("amazon_ads_db")
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
    
    def _correct_table_names(self, sql: str) -> str:
        """Auto-correct common table name mistakes in SQL."""
        import re
        corrections = {
            r'\bFROM\s+amazon_ads\b': 'FROM public.amazon_product_metrics_daily',
            r'\bJOIN\s+amazon_ads\b': 'JOIN public.amazon_product_metrics_daily',
        }
        corrected_sql = sql
        for pattern, replacement in corrections.items():
            if re.search(pattern, corrected_sql, re.IGNORECASE):
                match = re.search(pattern, corrected_sql, re.IGNORECASE)
                if match and '.' not in match.group(0):
                    corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
        if corrected_sql != sql:
            logger.warning(f"⚠️ [AMAZON_ADS_DB] Auto-corrected table names | original={sql[:100]}... | corrected={corrected_sql[:100]}...")
        return corrected_sql
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Amazon Ads database query."""
        query_inputs = inputs.get('inputs', {})
        
        logger.info(
            f"📊 [AMAZON_ADS_DB] Executing query | "
            f"date_start={query_inputs.get('date_start')} | "
            f"date_end={query_inputs.get('date_end')} | "
            f"channel={query_inputs.get('channel')}"
        )
        
        # Check if SQL is provided (LLM-generated) or use template-based
        if 'sql' in inputs:
            sql = inputs['sql']
            params = inputs.get('params', {})
            # Auto-correct common table name mistakes
            sql = self._correct_table_names(sql)
            logger.info(f"🔍 [AMAZON_ADS_DB] Using LLM-generated SQL | sql={sql[:200]}...")
        else:
            # Build SQL query
            sql = self._build_query(query_inputs)
            params = {}
            logger.info(f"🔍 [AMAZON_ADS_DB] Generated SQL (template-based) | sql={sql}")
        
        try:
            # Execute query
            with self.engine.connect() as conn:
                if params:
                    result = pd.read_sql(text(sql), conn, params=params)
                else:
                    result = pd.read_sql(sql, conn)
            
            logger.info(
                f"✅ [AMAZON_ADS_DB] Query executed successfully | "
                f"row_count={len(result)} | "
                f"columns={list(result.columns)}"
            )
            
            if len(result) == 0:
                logger.warning(
                    f"⚠️ [AMAZON_ADS_DB] Query returned 0 rows | "
                    f"date_start={query_inputs.get('date_start')} | "
                    f"date_end={query_inputs.get('date_end')}"
                )
            
            return {
                'data': result.to_dict('records'),
                'row_count': len(result),
                'columns': list(result.columns)
            }
        except Exception as e:
            logger.error(
                f"❌ [AMAZON_ADS_DB] Query execution failed | "
                f"error={str(e)} | "
                f"sql={sql[:200]}...",
                exc_info=True
            )
            raise
    
    def _build_query(self, inputs: Dict[str, Any]) -> str:
        """Build SQL query from inputs."""
        base_query = """
        SELECT 
            campaign_id,
            campaign_name,
            date,
            spend,
            clicks,
            impressions,
            sales,
            orders,
            roas,
            acos,
            ctr,
            cpc
        FROM public.amazon_product_metrics_daily
        WHERE 1=1
        """
        
        # Add date filter
        if 'date_start' in inputs:
            base_query += f" AND date >= '{inputs['date_start']}'"
        if 'date_end' in inputs:
            base_query += f" AND date <= '{inputs['date_end']}'"
        
        # Add channel filter (if applicable)
        if 'channel' in inputs:
            # Channel mapping would be in campaign_name or separate field
            # For now, we'll filter by campaign name pattern
            channel = inputs['channel']
            if channel == 'SP':
                base_query += " AND campaign_name LIKE '%SP%'"
            elif channel == 'SB':
                base_query += " AND campaign_name LIKE '%SB%'"
            elif channel == 'SD':
                base_query += " AND campaign_name LIKE '%SD%'"
        
        # Add geo filter (if applicable)
        if 'geo' in inputs:
            # Geo filtering would depend on campaign setup
            pass
        
        # Add limit
        limit = inputs.get('limit', 100000)
        base_query += f" LIMIT {limit}"
        
        return base_query


class MetaAdsDBAgent(DataAccessAgent):
    """Meta Ads database agent."""
    
    def __init__(self):
        """Initialize Meta Ads DB agent."""
        super().__init__("meta_ads_db")
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
    
    def _correct_table_names(self, sql: str) -> str:
        """Auto-correct common table name mistakes in SQL."""
        import re
        # Get SQL corrections from config
        sql_corrections = system_config.get_sql_corrections()
        
        corrected_sql = sql
        for correction in sql_corrections:
            pattern = correction.get('pattern')
            replacement = correction.get('replacement')
            if pattern and replacement:
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    match = re.search(pattern, corrected_sql, re.IGNORECASE)
                    if match and '.' not in match.group(0):
                        corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
        if corrected_sql != sql:
            logger.warning(f"⚠️ [META_ADS_DB] Auto-corrected table names | original={sql[:100]}... | corrected={corrected_sql[:100]}...")
        return corrected_sql
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Meta Ads database query."""
        query_inputs = inputs.get('inputs', {})
        
        logger.info(
            f"📊 [META_ADS_DB] Executing query | "
            f"date_start={query_inputs.get('date_start')} | "
            f"date_end={query_inputs.get('date_end')}"
        )
        
        # Check if SQL is provided (LLM-generated) or use template-based
        if 'sql' in inputs:
            sql = inputs['sql']
            params = inputs.get('params', {})
            # Auto-correct common table name mistakes
            sql = self._correct_table_names(sql)
            logger.info(f"🔍 [META_ADS_DB] Using LLM-generated SQL | sql={sql[:200]}...")
        else:
            # Build SQL query
            sql = self._build_query(query_inputs)
            params = {}
            logger.info(f"🔍 [META_ADS_DB] Generated SQL (template-based) | sql={sql}")
        
        try:
            # Execute query
            with self.engine.connect() as conn:
                if params:
                    result = pd.read_sql(text(sql), conn, params=params)
                else:
                    result = pd.read_sql(sql, conn)
            
            logger.info(
                f"✅ [META_ADS_DB] Query executed successfully | "
                f"row_count={len(result)} | "
                f"columns={list(result.columns)}"
            )
            
            if len(result) == 0:
                logger.warning(
                    f"⚠️ [META_ADS_DB] Query returned 0 rows | "
                    f"date_start={query_inputs.get('date_start')} | "
                    f"date_end={query_inputs.get('date_end')}"
                )
            
            return {
                'data': result.to_dict('records'),
                'row_count': len(result),
                'columns': list(result.columns)
            }
        except Exception as e:
            logger.error(
                f"❌ [META_ADS_DB] Query execution failed | "
                f"error={str(e)} | "
                f"sql={sql[:200]}...",
                exc_info=True
            )
            raise
    
    def _build_query(self, inputs: Dict[str, Any]) -> str:
        """Build SQL query from inputs."""
        base_query = """
        SELECT 
            campaign_id,
            campaign_name,
            date_start,
            hour,
            spend,
            clicks,
            impressions,
            ctr,
            cpc,
            cpm,
            attributed_orders_revenue as revenue,
            attributed_orders_count as orders
        FROM public.dw_meta_ads_attribution
        WHERE 1=1
        """
        
        # Add date filter
        if 'date_start' in inputs:
            base_query += f" AND date_start >= '{inputs['date_start']}'"
        if 'date_end' in inputs:
            base_query += f" AND date_start <= '{inputs['date_end']}'"
        
        # Add limit
        limit = inputs.get('limit', 100000)
        base_query += f" LIMIT {limit}"
        
        return base_query


class AmazonAdsAPIAgent(DataAccessAgent):
    """Amazon Ads API agent (for live API calls)."""
    
    def __init__(self):
        """Initialize Amazon Ads API agent."""
        super().__init__("amazon_ads_api")
        self.base_url = "https://advertising-api.amazon.com"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Amazon Ads API call."""
        query_inputs = inputs.get('inputs', {})
        
        logger.info(
            f"🌐 [AMAZON_ADS_API] Making API call | "
            f"channels={query_inputs.get('channels')} | "
            f"date_range={query_inputs.get('date_range')} | "
            f"metrics={query_inputs.get('metrics')}"
        )
        
        # This would integrate with actual Amazon Ads API
        # For now, return mock structure
        
        # In production, this would:
        # 1. Get OAuth token
        # 2. Make API call to Amazon Ads
        # 3. Parse response
        # 4. Return structured data
        
        logger.warning(f"⚠️ [AMAZON_ADS_API] API integration not yet implemented")
        
        return {
            'data': [],
            'row_count': 0,
            'columns': [],
            'error': 'Amazon Ads API integration not yet implemented'
        }


# Factory function
def get_data_access_agent(tool_id: str) -> Optional[DataAccessAgent]:
    """Get appropriate data access agent for tool."""
    # Skip excluded agents
    if system_config.is_tool_excluded(tool_id):
        logger.warning(f"⚠️ [DATA_ACCESS] Excluded tool | tool_id={tool_id}")
        return None
    
    agents = {
        'sales_db': SalesDBAgent,
        'meta_ads_db': MetaAdsDBAgent,
    }
    
    agent_class = agents.get(tool_id)
    if agent_class:
        return agent_class()
    return None

