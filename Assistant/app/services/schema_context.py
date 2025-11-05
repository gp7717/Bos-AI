"""Schema Context Service - Provides schema metadata in MCP-style format for LLM."""
from typing import Dict, List, Optional, Any
from app.services.schema_registry import schema_registry
from app.config.logging_config import get_logger
import json

logger = get_logger(__name__)


class SchemaContextService:
    """Provides schema metadata in formats optimized for LLM consumption."""
    
    def __init__(self):
        """Initialize schema context service."""
        self.registry = schema_registry
    
    def get_table_context(self, schema: str, table: str) -> Optional[Dict[str, Any]]:
        """Get detailed table context for a specific table."""
        table_def = self.registry.get_table(schema, table)
        if not table_def:
            return None
        
        return {
            "schema": schema,
            "table": table,
            "full_name": f"{schema}.{table}",
            "columns": [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "description": col.description
                }
                for col in table_def.columns
            ],
            "primary_keys": table_def.primary_keys,
            "foreign_keys": table_def.foreign_keys,
            "date_column": self.registry.get_date_column(schema, table),
            "sample_columns": [col.name for col in table_def.columns[:10]]  # First 10 columns
        }
    
    def get_schema_summary(self, schemas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get summary of all tables in specified schemas."""
        if schemas is None:
            schemas = ['public']
        
        summary = {}
        for table_key, table_def in self.registry.tables.items():
            schema_name, table_name = table_key.split('.', 1)
            if schema_name in schemas:
                if schema_name not in summary:
                    summary[schema_name] = {}
                
                summary[schema_name][table_name] = {
                    "columns_count": len(table_def.columns),
                    "primary_keys": table_def.primary_keys,
                    "date_column": self.registry.get_date_column(schema_name, table_name),
                    "key_columns": [
                        col.name for col in table_def.columns 
                        if any(keyword in col.name.lower() for keyword in ['id', 'key', 'campaign', 'order', 'date'])
                    ]
                }
        
        return summary
    
    def get_relevant_tables_for_query(self, metrics: List[str], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant tables based on query context (metrics and entities)."""
        relevant_tables = []
        
        # Map metrics/entities to likely tables
        table_keywords = {
            'revenue': ['shopify_orders', 'dw_meta_ads_attribution', 'dw_google_ads_attribution'],
            'spend': ['amazon_product_metrics_daily', 'dw_meta_ads_attribution', 'dw_google_ads_attribution'],
            'orders': ['shopify_orders'],
            'campaign': ['amazon_product_metrics_daily', 'dw_meta_ads_attribution', 'dw_google_ads_attribution'],
            'product': ['shopify_product_variants', 'shopify_order_line_items'],
            'ads': ['amazon_product_metrics_daily', 'dw_meta_ads_attribution', 'dw_google_ads_attribution'],
            'meta': ['dw_meta_ads_attribution'],
            'google': ['dw_google_ads_attribution'],
            'amazon': ['amazon_product_metrics_daily']
        }
        
        # Find relevant tables
        table_names = set()
        for metric in metrics:
            metric_lower = metric.lower()
            for keyword, tables in table_keywords.items():
                if keyword in metric_lower:
                    table_names.update(tables)
        
        for entity_key, entity_value in entities.items():
            entity_str = str(entity_value).lower()
            for keyword, tables in table_keywords.items():
                if keyword in entity_str:
                    table_names.update(tables)
        
        # Get full context for each table
        for table_name in table_names:
            context = self.get_table_context('public', table_name)
            if context:
                relevant_tables.append(context)
        
        return relevant_tables
    
    def format_for_llm(self, tables: List[Dict[str, Any]]) -> str:
        """Format table schemas in a format optimized for LLM SQL generation."""
        formatted = []
        
        # Add critical header with exact table names
        formatted.append("=" * 80)
        formatted.append("CRITICAL: Use EXACT table names as shown below. DO NOT use generic names!")
        formatted.append("=" * 80)
        formatted.append("")
        
        for table in tables:
            full_name = table['full_name']
            schema = table['schema']
            table_name = table['table']
            
            # Emphasize exact table names
            table_info = f"""## Table: {full_name}

⚠️ **IMPORTANT: Use the EXACT table name '{full_name}' in SQL queries.**
⚠️ **DO NOT use generic names like 'orders', 'sales', etc.**
⚠️ **Full qualified name: {full_name}**
⚠️ **Schema: {schema}, Table: {table_name}**

**Columns:**
"""
            formatted.append(table_info)
            
            for col in table['columns']:
                pk_marker = " [PRIMARY KEY]" if col['primary_key'] else ""
                nullable_marker = "" if col['nullable'] else " [NOT NULL]"
                formatted.append(f"- `{col['name']}` ({col['type']}){pk_marker}{nullable_marker}")
            
            formatted.append(f"\n**Primary Keys:** {', '.join(table['primary_keys']) if table['primary_keys'] else 'None'}")
            
            if table['foreign_keys']:
                formatted.append(f"\n**Foreign Keys:**")
                for fk in table['foreign_keys']:
                    formatted.append(f"  - {', '.join(fk['columns'])} → {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
            
            if table['date_column']:
                formatted.append(f"\n**Date Column:** `{table['date_column']}` (use for date filtering with :date_start and :date_end parameters)")
            
            formatted.append("\n")
        
        return "\n".join(formatted)
    
    def get_join_suggestions(self, table1: str, table2: str) -> List[Dict[str, str]]:
        """Get suggested join keys between two tables."""
        schema1, name1 = table1.split('.', 1) if '.' in table1 else ('public', table1)
        schema2, name2 = table2.split('.', 1) if '.' in table2 else ('public', table2)
        
        return self.registry.get_join_keys(schema1, name1, schema2, name2)
    
    def build_mcp_tool_definition(self, schema: str, table: str) -> Dict[str, Any]:
        """Build MCP-style tool definition for a database table."""
        table_def = self.registry.get_table(schema, table)
        if not table_def:
            return None
        
        return {
            "name": f"query_{table}",
            "description": f"Query the {schema}.{table} table",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute (must be SELECT only, no DDL/DML)"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters for parameterized queries"
                    }
                },
                "required": ["sql"]
            },
            "schema": {
                "schema": schema,
                "table": table,
                "columns": [
                    {
                        "name": col.name,
                        "type": str(col.type),
                        "nullable": col.nullable
                    }
                    for col in table_def.columns
                ],
                "primary_keys": table_def.primary_keys,
                "foreign_keys": table_def.foreign_keys,
                "date_column": self.registry.get_date_column(schema, table)
            }
        }

