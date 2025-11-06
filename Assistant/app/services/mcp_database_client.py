"""MCP Database Client - Client wrapper for MCP database server."""
from typing import Dict, Any, List, Optional
from app.services.mcp_database_server import get_mcp_server
from app.services.system_config import system_config
from app.config.logging_config import get_logger

# Global MCP client instance
_mcp_client: Optional['MCPDatabaseClient'] = None

logger = get_logger(__name__)


class MCPDatabaseClient:
    """Client wrapper for MCP database server."""
    
    def __init__(self):
        """Initialize MCP database client."""
        self.server = get_mcp_server()
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ [MCP_DB_CLIENT] MCP Database Client initialized")
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Dict with 'data', 'row_count', 'columns', 'success', 'error'
        """
        return await self.server.execute_sql(sql, params)
    
    async def get_schema_for_tables(
        self,
        table_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get schema information for multiple tables.
        
        Args:
            table_names: List of table names (format: 'schema.table' or 'table')
            
        Returns:
            List of table schema dictionaries
        """
        schemas = []
        
        for table_name in table_names:
            # Check cache first
            if table_name in self._schema_cache:
                schemas.append(self._schema_cache[table_name])
                continue
            
            # Parse table name
            if '.' in table_name:
                schema, table = table_name.split('.', 1)
            else:
                schema = 'public'
                table = table_name
            
            # Get schema from server
            result = await self.server.get_table_schema(schema, table)
            if result.get('success') and result.get('schema'):
                schema_info = result['schema']
                self._schema_cache[table_name] = schema_info
                schemas.append(schema_info)
            else:
                logger.warning(
                    f"⚠️ [MCP_DB_CLIENT] Failed to get schema | table={table_name}"
                )
        
        return schemas
    
    async def discover_tables_for_metrics(
        self,
        metrics: List[str]
    ) -> List[str]:
        """
        Discover relevant tables for given metrics.
        
        Args:
            metrics: List of metric IDs
            
        Returns:
            List of table full names (schema.table)
        """
        table_names = set()
        
        # Use system config to get tables for metrics
        for metric in metrics:
            tables = system_config.get_tables_for_metric(metric)
            for table in tables:
                if '.' not in table:
                    table = f"public.{table}"
                table_names.add(table)
        
        # If no tables found, use common fallback
        if not table_names:
            common_tables = system_config.get_common_fallback_tables()
            for table in common_tables:
                if '.' not in table:
                    table = f"public.{table}"
                table_names.add(table)
        
        logger.info(
            f"🔍 [MCP_DB_CLIENT] Discovered tables | "
            f"metrics={metrics} | tables={list(table_names)}"
        )
        
        return list(table_names)
    
    async def list_tables(
        self,
        schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all tables in the database.
        
        Args:
            schema: Optional schema name
            
        Returns:
            List of table information dictionaries
        """
        result = await self.server.list_tables(schema)
        if result.get('success'):
            return result.get('tables', [])
        return []
    
    async def get_table_relationships(
        self,
        schema: str,
        table: str
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a table.
        
        Args:
            schema: Schema name
            table: Table name
            
        Returns:
            List of relationship dictionaries
        """
        result = await self.server.get_table_relationships(schema, table)
        if result.get('success'):
            return result.get('relationships', [])
        return []
    
    def clear_schema_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()
        logger.debug("🧹 [MCP_DB_CLIENT] Schema cache cleared")


def get_mcp_client() -> 'MCPDatabaseClient':
    """Get or create global MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPDatabaseClient()
    return _mcp_client


# Convenience alias
mcp_database_client = get_mcp_client()

