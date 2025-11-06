"""MCP Database Server - Provides database tools via Model Context Protocol."""
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.engine import Engine
from app.config.settings import settings
from app.config.logging_config import get_logger
from app.services.schema_registry import schema_registry
import pandas as pd
import re
import json

logger = get_logger(__name__)


class MCPDatabaseServer:
    """MCP server for database operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize MCP database server."""
        self.database_url = database_url or settings.database_url
        self.engine: Optional[Engine] = None
        self._initialize_engine()
        logger.info("✅ [MCP_DB_SERVER] MCP Database Server initialized")
    
    def _initialize_engine(self):
        """Initialize database engine with connection pooling."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow
            )
            logger.info("✅ [MCP_DB_SERVER] Database engine initialized")
        except Exception as e:
            logger.error(f"❌ [MCP_DB_SERVER] Failed to initialize engine | error={str(e)}")
            raise
    
    def _validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for safety (read-only, no DDL/DML)."""
        sql_upper = sql.strip().upper()
        
        # Block dangerous operations
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
        ]
        
        for keyword in dangerous_keywords:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False, f"SQL contains forbidden keyword: {keyword}"
        
        # Only allow SELECT statements
        if not sql_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, None
    
    async def execute_sql(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query with safety checks.
        
        Args:
            sql: SQL query string (SELECT only)
            params: Query parameters for parameterized queries
            
        Returns:
            Dict with 'data', 'row_count', 'columns', 'success', 'error'
        """
        import time
        start_time = time.time()
        
        logger.info(
            f"🔍 [MCP_DB_SERVER] Executing SQL | "
            f"sql_preview={sql[:200]}... | params={params}"
        )
        
        # Validate SQL
        is_valid, error_msg = self._validate_sql(sql)
        if not is_valid:
            logger.error(f"❌ [MCP_DB_SERVER] SQL validation failed | error={error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'data': [],
                'row_count': 0,
                'columns': []
            }
        
        try:
            # Execute query
            with self.engine.connect() as conn:
                result = pd.read_sql(text(sql), conn, params=params or {})
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"✅ [MCP_DB_SERVER] Query executed | "
                    f"row_count={len(result)} | "
                    f"columns={list(result.columns)} | "
                    f"execution_time_ms={execution_time_ms:.2f}"
                )
                
                return {
                    'success': True,
                    'data': result.to_dict('records'),
                    'row_count': len(result),
                    'columns': list(result.columns),
                    'execution_time_ms': execution_time_ms
                }
        except Exception as e:
            error_str = str(e)
            execution_time_ms = (time.time() - start_time) * 1000
            
            logger.error(
                f"❌ [MCP_DB_SERVER] Query execution failed | "
                f"error={error_str} | "
                f"execution_time_ms={execution_time_ms:.2f}",
                exc_info=True
            )
            
            return {
                'success': False,
                'error': error_str,
                'data': [],
                'row_count': 0,
                'columns': [],
                'execution_time_ms': execution_time_ms
            }
    
    async def list_tables(
        self,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all tables in the database.
        
        Args:
            schema: Optional schema name (defaults to 'public')
            
        Returns:
            Dict with 'tables' list containing table information
        """
        try:
            inspector = inspect(self.engine)
            schemas_to_check = [schema] if schema else ['public']
            
            tables = []
            for schema_name in schemas_to_check:
                table_names = inspector.get_table_names(schema=schema_name)
                for table_name in table_names:
                    tables.append({
                        'schema': schema_name,
                        'table': table_name,
                        'full_name': f"{schema_name}.{table_name}"
                    })
            
            logger.info(f"✅ [MCP_DB_SERVER] Listed tables | count={len(tables)}")
            
            return {
                'success': True,
                'tables': tables,
                'count': len(tables)
            }
        except Exception as e:
            logger.error(
                f"❌ [MCP_DB_SERVER] Failed to list tables | error={str(e)}",
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'tables': [],
                'count': 0
            }
    
    async def get_table_schema(
        self,
        schema: str,
        table: str
    ) -> Dict[str, Any]:
        """
        Get detailed schema for a specific table.
        
        Args:
            schema: Schema name
            table: Table name
            
        Returns:
            Dict with table schema information
        """
        try:
            # Get from schema registry first (cached)
            table_def = schema_registry.get_table(schema, table)
            if not table_def:
                # Fallback to direct inspection
                inspector = inspect(self.engine)
                columns = inspector.get_columns(table, schema=schema)
                pk_constraint = inspector.get_pk_constraint(table, schema=schema)
                fk_constraints = inspector.get_foreign_keys(table, schema=schema)
                
                table_info = {
                    'schema': schema,
                    'table': table,
                    'full_name': f"{schema}.{table}",
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True),
                            'primary_key': False
                        }
                        for col in columns
                    ],
                    'primary_keys': pk_constraint.get('constrained_columns', []) if pk_constraint else [],
                    'foreign_keys': [
                        {
                            'columns': fk.get('constrained_columns', []),
                            'referred_table': fk.get('referred_table', ''),
                            'referred_columns': fk.get('referred_columns', [])
                        }
                        for fk in fk_constraints
                    ],
                    'date_column': schema_registry.get_date_column(schema, table)
                }
            else:
                # Use schema registry data
                table_info = {
                    'schema': schema,
                    'table': table,
                    'full_name': f"{schema}.{table}",
                    'columns': [
                        {
                            'name': col.name,
                            'type': str(col.type),
                            'nullable': col.nullable,
                            'primary_key': col.primary_key,
                            'description': col.description
                        }
                        for col in table_def.columns
                    ],
                    'primary_keys': table_def.primary_keys,
                    'foreign_keys': table_def.foreign_keys,
                    'date_column': schema_registry.get_date_column(schema, table)
                }
            
            logger.info(
                f"✅ [MCP_DB_SERVER] Retrieved schema | "
                f"table={schema}.{table} | "
                f"columns={len(table_info['columns'])}"
            )
            
            return {
                'success': True,
                'schema': table_info
            }
        except Exception as e:
            logger.error(
                f"❌ [MCP_DB_SERVER] Failed to get table schema | "
                f"table={schema}.{table} | error={str(e)}",
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'schema': None
            }
    
    async def get_table_relationships(
        self,
        schema: str,
        table: str
    ) -> Dict[str, Any]:
        """
        Get join relationships for a table.
        
        Args:
            schema: Schema name
            table: Table name
            
        Returns:
            Dict with relationships information
        """
        try:
            inspector = inspect(self.engine)
            
            # Get foreign keys (outgoing relationships)
            fk_constraints = inspector.get_foreign_keys(table, schema=schema)
            relationships = []
            
            for fk in fk_constraints:
                relationships.append({
                    'type': 'foreign_key',
                    'local_column': fk.get('constrained_columns', [])[0] if fk.get('constrained_columns') else '',
                    'foreign_table': fk.get('referred_table', ''),
                    'foreign_column': fk.get('referred_columns', [])[0] if fk.get('referred_columns') else '',
                    'purpose': f"Join {schema}.{table} to {fk.get('referred_table', '')}"
                })
            
            # Get reverse relationships (tables that reference this table)
            # This requires checking all tables, which can be expensive
            # For now, we'll use schema registry if available
            table_def = schema_registry.get_table(schema, table)
            if table_def and hasattr(table_def, 'foreign_keys'):
                # Schema registry might have reverse relationships
                pass
            
            logger.info(
                f"✅ [MCP_DB_SERVER] Retrieved relationships | "
                f"table={schema}.{table} | "
                f"count={len(relationships)}"
            )
            
            return {
                'success': True,
                'relationships': relationships
            }
        except Exception as e:
            logger.error(
                f"❌ [MCP_DB_SERVER] Failed to get relationships | "
                f"table={schema}.{table} | error={str(e)}",
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'relationships': []
            }


# Global MCP server instance
_mcp_server: Optional[MCPDatabaseServer] = None


def get_mcp_server() -> MCPDatabaseServer:
    """Get or create global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPDatabaseServer()
    return _mcp_server

