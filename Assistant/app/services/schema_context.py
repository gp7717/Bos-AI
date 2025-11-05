"""Schema Context Service - Provides schema metadata in MCP-style format for LLM."""
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from app.services.schema_registry import schema_registry
from app.services.system_config import system_config
from app.config.logging_config import get_logger
import json

logger = get_logger(__name__)


class SchemaContextService:
    """Provides schema metadata in formats optimized for LLM consumption."""
    
    def __init__(self):
        """Initialize schema context service."""
        self.registry = schema_registry
        self.table_metadata = self._load_table_metadata()
    
    def _load_table_metadata(self) -> Dict[str, Any]:
        """Load table metadata from YAML config."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "table_metadata.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"✅ [SCHEMA_CONTEXT] Loaded table metadata from {config_path}")
            return config.get('tables', {})
        except Exception as e:
            logger.warning(f"⚠️ [SCHEMA_CONTEXT] Failed to load table metadata | error={str(e)}")
            return {}
    
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
        
        # Get metric-to-table mappings from config
        metric_mappings = system_config.get_metric_table_mappings()
        
        # Find relevant tables
        table_names = set()
        for metric in metrics:
            tables = system_config.get_tables_for_metric(metric)
            table_names.update(tables)
        
        for entity_key, entity_value in entities.items():
            entity_str = str(entity_value).lower()
            # Check if entity matches any metric keyword
            for keyword in metric_mappings.keys():
                if keyword in entity_str:
                    tables = system_config.get_tables_for_metric(keyword)
                    table_names.update(tables)
        
        # Get full context for each table
        for table_name in table_names:
            context = self.get_table_context('public', table_name)
            if context:
                relevant_tables.append(context)
        
        return relevant_tables
    
    def format_for_llm(self, tables: List[Dict[str, Any]]) -> str:
        """Format table schemas in structured, minimal format optimized for LLM SQL generation."""
        formatted = []
        
        formatted.append("=" * 80)
        formatted.append("DATABASE SCHEMA CONTEXT (Query-Relevant Tables Only)")
        formatted.append("=" * 80)
        formatted.append("")
        formatted.append("⚠️ CRITICAL: Use EXACT table names as shown. DO NOT use generic names like 'orders', 'sales'.")
        formatted.append("")
        
        for table in tables:
            full_name = table['full_name']
            schema = table['schema']
            table_name = table['table']
            
            # Get metadata if available
            metadata = self.table_metadata.get(full_name, {})
            purpose = metadata.get('purpose', f'Table: {full_name}')
            
            formatted.append(f"## {full_name}")
            formatted.append(f"Purpose: {purpose}")
            formatted.append("")
            
            # Key columns with purposes
            if metadata.get('key_columns'):
                formatted.append("Key Columns:")
                for col_meta in metadata['key_columns']:
                    col_name = col_meta['name']
                    col_purpose = col_meta.get('purpose', '')
                    col_type = col_meta.get('type', '')
                    
                    markers = []
                    if col_meta.get('primary_key'):
                        markers.append("PRIMARY KEY")
                    if col_meta.get('foreign_key'):
                        markers.append("FOREIGN KEY")
                    if col_meta.get('date_column'):
                        markers.append("DATE COLUMN")
                    
                    marker_str = f" [{', '.join(markers)}]" if markers else ""
                    formatted.append(f"  • {col_name} ({col_type}){marker_str}")
                    formatted.append(f"    → {col_purpose}")
                    
                    # Show foreign key reference explicitly
                    if col_meta.get('references'):
                        formatted.append(f"    → References: {col_meta['references']}")
                
                formatted.append("")
            else:
                # Fallback to basic column list
                formatted.append("Columns:")
                for col in table['columns'][:15]:  # Limit to first 15 columns
                    pk_marker = " [PK]" if col['primary_key'] else ""
                    formatted.append(f"  • {col['name']} ({col['type']}){pk_marker}")
                formatted.append("")
            
            # Relationships with purpose
            if metadata.get('relationships'):
                formatted.append("Relationships:")
                for rel in metadata['relationships']:
                    rel_type = rel.get('type', '')
                    local_col = rel.get('local_column', '')
                    foreign_tbl = rel.get('foreign_table', '')
                    foreign_col = rel.get('foreign_column', '')
                    rel_purpose = rel.get('purpose', '')
                    
                    formatted.append(f"  • {rel_type}: {full_name}.{local_col} → {foreign_tbl}.{foreign_col}")
                    formatted.append(f"    → {rel_purpose}")
                    formatted.append(f"    Example: JOIN {foreign_tbl} ON {full_name}.{local_col} = {foreign_tbl}.{foreign_col}")
                formatted.append("")
            elif table.get('foreign_keys'):
                formatted.append("Foreign Keys:")
                for fk in table['foreign_keys']:
                    fk_cols = fk['columns'].split(', ') if isinstance(fk['columns'], str) else fk['columns']
                    ref_table = fk.get('referred_table', '')
                    ref_cols = fk['referred_columns'].split(', ') if isinstance(fk['referred_columns'], str) else fk['referred_columns']
                    formatted.append(f"  • {full_name}.{fk_cols[0]} → {ref_table}.{ref_cols[0]}")
                formatted.append("")
            
            # JSON columns with extraction examples
            if metadata.get('json_columns'):
                formatted.append("JSON Columns (Special Handling Required):")
                for json_col in metadata['json_columns']:
                    col_name = json_col['name']
                    col_purpose = json_col.get('purpose', '')
                    extraction = json_col.get('extraction_example', '')
                    formatted.append(f"  • {col_name}: {col_purpose}")
                    if extraction:
                        formatted.append(f"    Extraction: {extraction}")
                formatted.append("")
            
            # Table-specific quirks
            if metadata.get('quirks'):
                formatted.append("⚠️ Important Notes:")
                for quirk in metadata['quirks']:
                    formatted.append(f"  • {quirk}")
                formatted.append("")
            
            # Date column info
            date_col = table.get('date_column') or metadata.get('date_column') or self.registry.get_date_column(schema, table_name)
            if date_col:
                formatted.append(f"Date Filtering: Use `{date_col}` with BETWEEN :date_start AND :date_end")
                formatted.append(f"⚠️ DO NOT use 'order_date' or other names - use '{date_col}'")
                formatted.append("")
            
            formatted.append("---")
            formatted.append("")
        
        # Relationship summary
        formatted.append("=" * 80)
        formatted.append("COMMON JOIN PATTERNS")
        formatted.append("=" * 80)
        formatted.append("")
        formatted.append("Order → Line Items → Product Variants:")
        formatted.append("  FROM public.shopify_orders o")
        formatted.append("  JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id")
        formatted.append("  JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id")
        formatted.append("")
        formatted.append("Note: product_id is in shopify_product_variants, NOT in shopify_order_line_items")
        formatted.append("")
        
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

