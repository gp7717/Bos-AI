"""Database Knowledge Base - Centralized knowledge base for database schema and query building."""
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import yaml
from app.services.schema_registry import schema_registry
from app.services.system_config import system_config
from app.services.schema_context import SchemaContextService
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseKnowledgeBase:
    """Centralized knowledge base for database schema, relationships, and query building."""
    
    def __init__(self):
        """Initialize database knowledge base."""
        self.schema_registry = schema_registry
        self.system_config = system_config
        self.schema_context = SchemaContextService()
        self.table_metadata = self.schema_context.table_metadata
        logger.info("✅ [DB_KNOWLEDGE_BASE] Initialized")
    
    # ==================== Table Information ====================
    
    def get_table_info(self, schema: str, table: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive table information combining all sources."""
        full_name = f"{schema}.{table}"
        
        # Get from schema registry
        table_def = self.schema_registry.get_table(schema, table)
        if not table_def:
            return None
        
        # Get from metadata YAML
        metadata = self.table_metadata.get('tables', {}).get(full_name, {})
        
        # Combine information
        info = {
            'full_name': full_name,
            'schema': schema,
            'table': table,
            'purpose': metadata.get('purpose', f'Table: {full_name}'),
            'columns': [
                {
                    'name': col.name,
                    'type': str(col.type),
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'description': None
                }
                for col in table_def.columns
            ],
            'primary_keys': table_def.primary_keys,
            'foreign_keys': table_def.foreign_keys,
            'date_column': self.schema_registry.get_date_column(schema, table) or metadata.get('date_column'),
            'relationships': metadata.get('relationships', []),
            'key_columns': metadata.get('key_columns', []),
            'all_columns_metadata': metadata.get('all_columns', []),
            'quirks': metadata.get('quirks', []),
            'json_columns': metadata.get('json_columns', []),
            'columns_not_in_table': metadata.get('columns_not_in_table', []),
            'alias_suggestions': metadata.get('alias_suggestions', [])
        }
        
        # Enhance columns with metadata
        if metadata.get('all_columns'):
            col_metadata_map = {col['name']: col for col in metadata['all_columns']}
            for col in info['columns']:
                if col['name'] in col_metadata_map:
                    meta = col_metadata_map[col['name']]
                    col['description'] = meta.get('purpose', '')
                    col['date_column'] = meta.get('date_column', False)
                    col['json_column'] = meta.get('json_column', False)
        
        return info
    
    def get_tables_for_tool(self, tool_id: str) -> List[Dict[str, Any]]:
        """Get all tables relevant to a specific tool."""
        tables = []
        
        # Get primary table from system config
        primary_table = self.system_config.get_tool_primary_table(tool_id)
        if primary_table:
            schema, table_name = primary_table.split('.', 1)
            table_info = self.get_table_info(schema, table_name)
            if table_info:
                tables.append(table_info)
        
        # Get related tables from relationships
        if tables:
            related = self._get_related_tables(tables[0]['full_name'])
            for rel_table in related:
                schema, table_name = rel_table.split('.', 1)
                table_info = self.get_table_info(schema, table_name)
                if table_info:
                    tables.append(table_info)
        
        return tables
    
    async def get_tables_for_metrics(self, metrics: List[str], entities: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get relevant tables based on metrics and entities."""
        relevant_tables = await self.schema_context.get_relevant_tables_for_query(
            metrics, entities or {}
        )
        
        # Convert to our format
        result = []
        for table in relevant_tables:
            full_name = table.get('full_name', '')
            if full_name:
                schema, table_name = full_name.split('.', 1)
                table_info = self.get_table_info(schema, table_name)
                if table_info:
                    result.append(table_info)
        
        return result
    
    # ==================== Join Information ====================
    
    def get_join_path(self, from_table: str, to_table: str) -> Optional[List[Dict[str, str]]]:
        """Get join path between two tables (handles multi-hop joins)."""
        from_schema, from_name = from_table.split('.', 1) if '.' in from_table else ('public', from_table)
        to_schema, to_name = to_table.split('.', 1) if '.' in to_table else ('public', to_table)
        
        # Direct relationship check
        from_info = self.get_table_info(from_schema, from_name)
        to_info = self.get_table_info(to_schema, to_name)
        
        if not from_info or not to_info:
            return None
        
        # Check direct relationships
        for rel in from_info.get('relationships', []):
            foreign_table = rel.get('foreign_table', '')
            if foreign_table == to_table or foreign_table == to_name:
                return [{
                    'from_table': from_table,
                    'to_table': to_table,
                    'local_column': rel.get('local_column', ''),
                    'foreign_column': rel.get('foreign_column', ''),
                    'join_example': rel.get('join_example', ''),
                    'purpose': rel.get('purpose', '')
                }]
        
        # Try reverse relationship
        for rel in to_info.get('relationships', []):
            foreign_table = rel.get('foreign_table', '')
            if foreign_table == from_table or foreign_table == from_name:
                return [{
                    'from_table': to_table,
                    'to_table': from_table,
                    'local_column': rel.get('foreign_column', ''),
                    'foreign_column': rel.get('local_column', ''),
                    'join_example': rel.get('join_example', ''),
                    'purpose': rel.get('purpose', '')
                }]
        
        # Multi-hop join (BFS search)
        return self._find_join_path(from_table, to_table, max_hops=3)
    
    def _find_join_path(self, from_table: str, to_table: str, max_hops: int = 3) -> Optional[List[Dict[str, str]]]:
        """Find join path using BFS."""
        from_schema, from_name = from_table.split('.', 1) if '.' in from_table else ('public', from_table)
        to_schema, to_name = to_table.split('.', 1) if '.' in to_table else ('public', to_table)
        
        visited = set()
        queue = [(from_table, [])]
        
        while queue and len(queue[0][1]) < max_hops:
            current_table, path = queue.pop(0)
            
            if current_table in visited:
                continue
            visited.add(current_table)
            
            current_schema, current_name = current_table.split('.', 1) if '.' in current_table else ('public', current_table)
            current_info = self.get_table_info(current_schema, current_name)
            
            if not current_info:
                continue
            
            # Check all relationships
            for rel in current_info.get('relationships', []):
                foreign_table = rel.get('foreign_table', '')
                if '.' not in foreign_table:
                    foreign_table = f"public.{foreign_table}"
                
                if foreign_table == to_table:
                    # Found path!
                    full_path = path + [{
                        'from_table': current_table,
                        'to_table': to_table,
                        'local_column': rel.get('local_column', ''),
                        'foreign_column': rel.get('foreign_column', ''),
                        'join_example': rel.get('join_example', ''),
                        'purpose': rel.get('purpose', '')
                    }]
                    return full_path
                
                if foreign_table not in visited:
                    new_path = path + [{
                        'from_table': current_table,
                        'to_table': foreign_table,
                        'local_column': rel.get('local_column', ''),
                        'foreign_column': rel.get('foreign_column', ''),
                        'join_example': rel.get('join_example', ''),
                        'purpose': rel.get('purpose', '')
                    }]
                    queue.append((foreign_table, new_path))
        
        return None
    
    def suggest_joins(self, tables: List[str]) -> List[Dict[str, Any]]:
        """Suggest join relationships for a list of tables."""
        joins = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                join_path = self.get_join_path(table1, table2)
                if join_path:
                    joins.extend(join_path)
        
        return joins
    
    def get_required_joins(self, tables: List[str], columns: List[str]) -> List[Dict[str, str]]:
        """Get required joins based on referenced columns (e.g., pv.sku requires shopify_product_variants)."""
        required_joins = []
        table_aliases = {}
        
        # Extract table aliases from columns (e.g., "pv.sku" -> alias "pv")
        for col in columns:
            if '.' in col:
                alias = col.split('.')[0]
                if alias not in table_aliases:
                    # Try to find which table this alias refers to
                    table_aliases[alias] = self.find_table_for_alias(alias, tables)
        
        # Check if all referenced tables are in the list
        for alias, table in table_aliases.items():
            if table and table not in tables:
                # Need to join this table
                # Find best join path from existing tables
                for existing_table in tables:
                    join_path = self.get_join_path(existing_table, table)
                    if join_path:
                        required_joins.extend(join_path)
                        break
        
        return required_joins
    
    def find_table_for_alias(self, alias: str, tables: List[str]) -> Optional[str]:
        """Find which table an alias likely refers to."""
        # Check metadata for alias suggestions
        for table in tables:
            schema, table_name = table.split('.', 1) if '.' in table else ('public', table)
            info = self.get_table_info(schema, table_name)
            if info and alias.lower() in [a.lower() for a in info.get('alias_suggestions', [])]:
                return table
        
        # Common alias patterns
        alias_lower = alias.lower()
        for table in tables:
            table_lower = table.lower()
            if alias_lower in table_lower or any(word.startswith(alias_lower) for word in table_lower.split('_')):
                return table
        
        return None
    
    # ==================== Column Information ====================
    
    def get_columns_for_metric(self, metric: str, table: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get columns that might contain data for a metric."""
        columns = []
        
        # Get tables for metric
        if table:
            tables = [table]
        else:
            tables = self.system_config.get_tables_for_metric(metric)
            if not tables:
                return columns
        
        # Search columns in those tables
        for table_name in tables:
            schema, table_name_only = table_name.split('.', 1) if '.' in table_name else ('public', table_name)
            info = self.get_table_info(schema, table_name_only)
            if not info:
                continue
            
            metric_lower = metric.lower()
            for col in info.get('all_columns_metadata', info.get('columns', [])):
                col_name = col.get('name', '') if isinstance(col, dict) else col.name
                col_purpose = col.get('purpose', '') if isinstance(col, dict) else ''
                
                if metric_lower in col_name.lower() or metric_lower in col_purpose.lower():
                    columns.append({
                        'table': table_name,
                        'column': col_name,
                        'type': col.get('type', '') if isinstance(col, dict) else str(col.type),
                        'purpose': col_purpose
                    })
        
        return columns
    
    def validate_column(self, schema: str, table: str, column: str) -> Tuple[bool, Optional[str]]:
        """Validate if a column exists, returns (is_valid, error_message)."""
        info = self.get_table_info(schema, table)
        if not info:
            return False, f"Table {schema}.{table} does not exist"
        
        # Check in columns
        for col in info['columns']:
            if col['name'].lower() == column.lower():
                return True, None
        
        # Check columns_not_in_table for common mistakes
        for warning in info.get('columns_not_in_table', []):
            if column.lower() in warning.lower():
                return False, warning
        
        # Suggest similar columns
        similar = [col['name'] for col in info['columns'] 
                  if any(word in col['name'].lower() for word in column.lower().split('_') if len(word) > 3)]
        
        error_msg = f"Column '{column}' does not exist in {schema}.{table}"
        if similar:
            error_msg += f". Similar columns: {', '.join(similar[:3])}"
        
        return False, error_msg
    
    # ==================== Query Building Helpers ====================
    
    async def build_query_context(self, tool_id: str, metrics: List[str] = None, entities: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build comprehensive query context for an agent."""
        context = {
            'tool_id': tool_id,
            'tables': [],
            'joins': [],
            'date_column': None,
            'common_patterns': self.table_metadata.get('common_patterns', {}),
            'critical_rules': self.table_metadata.get('critical_rules', [])
        }
        
        # Get tables
        tool_tables = self.get_tables_for_tool(tool_id)
        if metrics or entities:
            metric_tables = await self.get_tables_for_metrics(metrics or [], entities or {})
            # Merge and deduplicate
            all_tables = {t['full_name']: t for t in tool_tables + metric_tables}
            context['tables'] = list(all_tables.values())
        else:
            context['tables'] = tool_tables
        
        # Get joins
        if len(context['tables']) > 1:
            table_names = [t['full_name'] for t in context['tables']]
            context['joins'] = self.suggest_joins(table_names)
        
        # Get date column
        if context['tables']:
            primary_table = context['tables'][0]
            context['date_column'] = primary_table.get('date_column')
        
        return context
    
    def get_query_template(self, tool_id: str, query_type: str = 'select') -> Optional[str]:
        """Get query template for a tool and query type."""
        # Get common patterns
        patterns = self.table_metadata.get('common_patterns', {})
        
        # Look for tool-specific pattern
        pattern_key = f"{tool_id}_{query_type}"
        if pattern_key in patterns:
            return patterns[pattern_key].get('sql', '')
        
        # Look for generic pattern
        if query_type in patterns:
            return patterns[query_type].get('sql', '')
        
        return None
    
    def format_schema_for_agent(self, tool_id: str, metrics: List[str] = None, entities: Dict[str, Any] = None) -> str:
        """Format schema information for agent consumption (similar to schema_context but tool-specific)."""
        context = self.build_query_context(tool_id, metrics, entities)
        
        formatted = []
        formatted.append("=" * 80)
        formatted.append(f"DATABASE KNOWLEDGE BASE - {tool_id.upper()}")
        formatted.append("=" * 80)
        formatted.append("")
        
        # Tables
        formatted.append("AVAILABLE TABLES:")
        for table in context['tables']:
            formatted.append(f"  • {table['full_name']} - {table['purpose']}")
        formatted.append("")
        
        # Detailed table info
        for table in context['tables']:
            formatted.append(f"## {table['full_name']}")
            formatted.append(f"Purpose: {table['purpose']}")
            formatted.append("")
            
            # Key columns
            if table.get('key_columns'):
                formatted.append("Key Columns:")
                for col in table['key_columns']:
                    col_name = col.get('name', '') if isinstance(col, dict) else col
                    col_purpose = col.get('purpose', '') if isinstance(col, dict) else ''
                    formatted.append(f"  • {col_name}: {col_purpose}")
                formatted.append("")
            
            # Date column
            if table.get('date_column'):
                formatted.append(f"Date Filtering: Use `{table['date_column']}` for date ranges")
                formatted.append("")
            
            # Relationships
            if table.get('relationships'):
                formatted.append("Relationships:")
                for rel in table['relationships']:
                    formatted.append(f"  • {rel.get('purpose', '')}")
                    if rel.get('join_example'):
                        formatted.append(f"    {rel['join_example']}")
                formatted.append("")
            
            # Quirks
            if table.get('quirks'):
                formatted.append("⚠️ Important Notes:")
                for quirk in table['quirks']:
                    formatted.append(f"  • {quirk}")
                formatted.append("")
            
            formatted.append("---")
            formatted.append("")
        
        # Joins
        if context['joins']:
            formatted.append("SUGGESTED JOINS:")
            for join in context['joins']:
                formatted.append(f"  • {join.get('purpose', '')}")
                if join.get('join_example'):
                    formatted.append(f"    {join['join_example']}")
            formatted.append("")
        
        # Critical rules
        if context['critical_rules']:
            formatted.append("CRITICAL RULES:")
            for rule in context['critical_rules']:
                if isinstance(rule, dict):
                    formatted.append(f"  • {rule.get('rule', '')}: {rule.get('description', '')}")
            formatted.append("")
        
        return "\n".join(formatted)


# Global instance
db_knowledge_base = DatabaseKnowledgeBase()

