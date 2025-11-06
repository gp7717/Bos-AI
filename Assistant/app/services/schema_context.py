"""Schema Context Service - Provides schema metadata in MCP-style format for LLM."""
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from app.services.schema_registry import schema_registry
from app.services.system_config import system_config
from app.services.mcp_database_client import get_mcp_client
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
            # Return full config (includes tables, common_patterns, critical_rules)
            return config
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
    
    async def get_relevant_tables_for_query(self, metrics: List[str], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant tables based on query context (metrics and entities) using MCP client."""
        mcp_client = get_mcp_client()
        
        # Use MCP client to discover tables for metrics
        table_names = await mcp_client.discover_tables_for_metrics(metrics)
        
        # DO NOT add hardcoded tables based on entity keywords
        # Let MCP client and system config handle table discovery based on actual metrics
        # Trust the LLM's entity extraction and metric extraction
        
        # Process entities to find additional relevant tables
        tables_metadata = self.table_metadata.get('tables', {})
        for entity_key, entity_value in entities.items():
            entity_str = str(entity_value).lower()
            
            # Check table metadata for entity-related tables
            for full_table_name, table_meta in tables_metadata.items():
                purpose = str(table_meta.get('purpose', '')).lower()
                all_columns = table_meta.get('all_columns', [])
                key_columns = table_meta.get('key_columns', [])
                
                # Check if entity matches column names or table purpose
                if entity_str in purpose:
                    if full_table_name not in table_names:
                        table_names.append(full_table_name)
                
                # Check columns for entity match
                for col in all_columns + key_columns:
                    col_name = str(col.get('name', '')).lower()
                    col_purpose = str(col.get('purpose', '')).lower()
                    if entity_str in col_name or entity_str in col_purpose:
                        if full_table_name not in table_names:
                            table_names.append(full_table_name)
                        # Also add related tables from relationships
                        relationships = table_meta.get('relationships', [])
                        for rel in relationships:
                            foreign_table = rel.get('foreign_table', '')
                            if foreign_table:
                                if '.' not in foreign_table:
                                    foreign_table = f"public.{foreign_table}"
                                if foreign_table not in table_names:
                                    table_names.append(foreign_table)
        
        # If no tables found, use common fallback tables
        if not table_names:
            common_tables = system_config.get_common_fallback_tables()
            for table in common_tables:
                if '.' not in table:
                    table = f"public.{table}"
                if table not in table_names:
                    table_names.append(table)
        
        # Get schemas for discovered tables using MCP client (on-demand)
        mcp_schemas = await mcp_client.get_schema_for_tables(table_names)
        
        # Convert MCP schemas to our format
        relevant_tables = []
        for schema_info in mcp_schemas:
            if schema_info:
                context = {
                    'schema': schema_info.get('schema', 'public'),
                    'table': schema_info.get('table', ''),
                    'full_name': schema_info.get('full_name', ''),
                    'columns': [
                        {
                            'name': col.get('name', ''),
                            'type': str(col.get('type', '')),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False),
                            'description': col.get('description')
                        }
                        for col in schema_info.get('columns', [])
                    ],
                    'primary_keys': schema_info.get('primary_keys', []),
                    'foreign_keys': schema_info.get('foreign_keys', []),
                    'date_column': schema_info.get('date_column')
                }
                relevant_tables.append(context)
        
        # Also include related tables based on relationships
        additional_tables = set()
        for table in relevant_tables:
            full_name = table.get('full_name', '')
            table_meta = tables_metadata.get(full_name, {})
            relationships = table_meta.get('relationships', [])
            for rel in relationships:
                foreign_table = rel.get('foreign_table', '')
                if foreign_table:
                    if '.' not in foreign_table:
                        foreign_table = f"public.{foreign_table}"
                    if not any(t.get('full_name') == foreign_table for t in relevant_tables):
                        additional_tables.add(foreign_table)
        
        # Get schemas for additional related tables
        if additional_tables:
            additional_schemas = await mcp_client.get_schema_for_tables(list(additional_tables))
            for schema_info in additional_schemas:
                if schema_info:
                    context = {
                        'schema': schema_info.get('schema', 'public'),
                        'table': schema_info.get('table', ''),
                        'full_name': schema_info.get('full_name', ''),
                        'columns': [
                            {
                                'name': col.get('name', ''),
                                'type': str(col.get('type', '')),
                                'nullable': col.get('nullable', True),
                                'primary_key': col.get('primary_key', False),
                                'description': col.get('description')
                            }
                            for col in schema_info.get('columns', [])
                        ],
                        'primary_keys': schema_info.get('primary_keys', []),
                        'foreign_keys': schema_info.get('foreign_keys', []),
                        'date_column': schema_info.get('date_column')
                    }
                    relevant_tables.append(context)
        
        logger.debug(f"📊 [SCHEMA_CONTEXT] Selected {len(relevant_tables)} relevant tables: {[t.get('full_name') for t in relevant_tables]}")
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
        formatted.append("⚠️ CRITICAL: You can ONLY use tables listed below. DO NOT reference tables that are not in this list.")
        formatted.append("")
        formatted.append("AVAILABLE TABLES FOR THIS QUERY:")
        for table in tables:
            full_name = table.get('full_name', '')
            formatted.append(f"  • {full_name}")
        formatted.append("")
        formatted.append("=" * 80)
        formatted.append("")
        
        for table in tables:
            full_name = table['full_name']
            schema = table['schema']
            table_name = table['table']
            
            # Get metadata if available (from tables section)
            tables_metadata = self.table_metadata.get('tables', {})
            metadata = tables_metadata.get(full_name, {})
            purpose = metadata.get('purpose', f'Table: {full_name}')
            
            formatted.append(f"## {full_name}")
            formatted.append(f"Purpose: {purpose}")
            formatted.append("")
            
            # All columns (comprehensive list)
            if metadata.get('all_columns'):
                formatted.append("All Columns in This Table:")
                for col_meta in metadata['all_columns']:
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
                    if col_meta.get('nullable') is False:
                        markers.append("NOT NULL")
                    
                    marker_str = f" [{', '.join(markers)}]" if markers else ""
                    formatted.append(f"  • {col_name} ({col_type}){marker_str}")
                    if col_purpose:
                        formatted.append(f"    → {col_purpose}")
                    
                    # Show foreign key reference explicitly
                    if col_meta.get('references'):
                        formatted.append(f"    → References: {col_meta['references']}")
                
                formatted.append("")
            
            # Key columns with purposes (for quick reference)
            if metadata.get('key_columns'):
                formatted.append("Key Columns (Quick Reference):")
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
            
            # Columns NOT in this table (common mistakes)
            if metadata.get('columns_not_in_table'):
                formatted.append("⚠️ COLUMNS NOT IN THIS TABLE (Common Mistakes):")
                for col_warning in metadata['columns_not_in_table']:
                    formatted.append(f"  • {col_warning}")
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
                    join_example = rel.get('join_example', '')
                    
                    formatted.append(f"  • {rel_type}: {full_name}.{local_col} → {foreign_tbl}.{foreign_col}")
                    formatted.append(f"    → {rel_purpose}")
                    if join_example:
                        formatted.append(f"    Example: {join_example}")
                    else:
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
        
        # Add common patterns from metadata if available
        if self.table_metadata and isinstance(self.table_metadata, dict):
            common_patterns = self.table_metadata.get('common_patterns', {}) or {}
            if common_patterns:
                formatted.append("=" * 80)
                formatted.append("COMMON QUERY PATTERNS")
                formatted.append("=" * 80)
                formatted.append("")
                for pattern_name, pattern_data in common_patterns.items():
                    if isinstance(pattern_data, dict):
                        description = pattern_data.get('description', '')
                        sql = pattern_data.get('sql', '')
                        notes = pattern_data.get('notes', [])
                        
                        formatted.append(f"### {pattern_name.replace('_', ' ').title()}")
                        formatted.append(f"{description}")
                        formatted.append("")
                        formatted.append("SQL Example:")
                        formatted.append("```sql")
                        formatted.append(sql)
                        formatted.append("```")
                        formatted.append("")
                        if notes:
                            formatted.append("Notes:")
                            for note in notes:
                                formatted.append(f"  • {note}")
                            formatted.append("")
                formatted.append("---")
                formatted.append("")
        
        # Add critical rules from metadata if available
        if self.table_metadata and isinstance(self.table_metadata, dict):
            critical_rules = self.table_metadata.get('critical_rules', [])
            if critical_rules:
                formatted.append("=" * 80)
                formatted.append("CRITICAL RULES FOR SQL GENERATION")
                formatted.append("=" * 80)
                formatted.append("")
                for rule in critical_rules:
                    if isinstance(rule, dict):
                        rule_name = rule.get('rule', '')
                        rule_desc = rule.get('description', '')
                        formatted.append(f"### {rule_name}")
                        formatted.append(rule_desc)
                        formatted.append("")
                formatted.append("---")
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

