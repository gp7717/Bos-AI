"""Data Access - Simplified database query executor using MCP client."""
from typing import Dict, Any, Optional
from app.services.mcp_database_client import get_mcp_client
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseQueryExecutor:
    """Simple database query executor using MCP client."""
    
    def __init__(self):
        """Initialize database query executor."""
        self.client = get_mcp_client()
        logger.info("✅ [DATA_ACCESS] DatabaseQueryExecutor initialized")
    
    async def execute(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute database query.
        
        Args:
            inputs: Dict containing 'sql' and optional 'params'
            
        Returns:
            Dict with 'data', 'row_count', 'columns', 'success', 'error'
        """
        sql = inputs.get('sql')
        params = inputs.get('params', {})
        
        if not sql:
            return {
                'success': False,
                'error': 'SQL query is required',
                'data': [],
                'row_count': 0,
                'columns': []
            }
        
        # Execute via MCP client
        result = await self.client.execute_query(sql, params)
        return result


# Factory function for backward compatibility
def get_data_access_agent(tool_id: str) -> Optional[DatabaseQueryExecutor]:
    """
    Get database query executor (backward compatibility).
    
    Args:
        tool_id: Tool identifier (for compatibility, not used in simplified version)
        
    Returns:
        DatabaseQueryExecutor instance
    """
    # In simplified version, all tools use the same executor
    # Tool-specific logic is handled by the planner/query graph
    return DatabaseQueryExecutor()
