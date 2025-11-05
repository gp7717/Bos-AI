"""System Configuration Service - Loads and provides access to system configuration."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class SystemConfig:
    """Manages system-wide configuration from YAML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize system config from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        self.config: Dict[str, Any] = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: Path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"✅ [SYSTEM_CONFIG] Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"❌ [SYSTEM_CONFIG] Failed to load config | error={str(e)}", exc_info=True)
            self.config = {}
    
    # Tool-related methods
    def get_sql_tools(self) -> List[str]:
        """Get list of SQL tool IDs."""
        return self.config.get('tools', {}).get('sql_tools', [])
    
    def get_api_tools(self) -> List[str]:
        """Get list of API tool IDs."""
        return self.config.get('tools', {}).get('api_tools', [])
    
    def get_compute_tools(self) -> List[str]:
        """Get list of compute tool IDs."""
        return self.config.get('tools', {}).get('compute_tools', [])
    
    def get_excluded_tool_patterns(self) -> List[str]:
        """Get list of tool patterns to exclude."""
        return self.config.get('tools', {}).get('excluded_patterns', [])
    
    def get_date_range_tools(self) -> List[str]:
        """Get list of tools that need date range enhancement."""
        return self.config.get('tools', {}).get('date_range_tools', [])
    
    def is_tool_excluded(self, tool_id: str) -> bool:
        """Check if a tool should be excluded based on patterns."""
        tool_lower = tool_id.lower()
        excluded_patterns = self.get_excluded_tool_patterns()
        return any(pattern.lower() in tool_lower for pattern in excluded_patterns)
    
    def is_sql_tool(self, tool_id: str) -> bool:
        """Check if a tool is a SQL tool."""
        return tool_id in self.get_sql_tools()
    
    def is_api_tool(self, tool_id: str) -> bool:
        """Check if a tool is an API tool."""
        return tool_id in self.get_api_tools()
    
    def is_compute_tool(self, tool_id: str) -> bool:
        """Check if a tool is a compute tool."""
        return tool_id in self.get_compute_tools()
    
    # Table-related methods
    def get_common_fallback_tables(self) -> List[str]:
        """Get list of common fallback tables."""
        return self.config.get('tables', {}).get('common_fallback', [])
    
    def get_table_name_corrections(self) -> Dict[str, str]:
        """Get mapping of incorrect table names to correct ones."""
        return self.config.get('tables', {}).get('name_corrections', {})
    
    def get_excluded_table_patterns(self) -> List[str]:
        """Get list of table patterns to exclude."""
        return self.config.get('tables', {}).get('excluded_patterns', [])
    
    def get_sql_corrections(self) -> List[Dict[str, str]]:
        """Get list of SQL correction patterns."""
        return self.config.get('tables', {}).get('sql_corrections', [])
    
    def get_column_corrections(self) -> List[Dict[str, str]]:
        """Get list of column correction patterns."""
        return self.config.get('tables', {}).get('column_corrections', [])
    
    def is_table_excluded(self, table_name: str) -> bool:
        """Check if a table should be excluded based on patterns."""
        table_lower = table_name.lower()
        excluded_patterns = self.get_excluded_table_patterns()
        return any(pattern.lower() in table_lower for pattern in excluded_patterns)
    
    def get_correct_table_name(self, table_name: str) -> Optional[str]:
        """Get corrected table name if available."""
        corrections = self.get_table_name_corrections()
        return corrections.get(table_name.lower())
    
    # Metric-to-table mappings
    def get_metric_table_mappings(self) -> Dict[str, List[str]]:
        """Get mappings of metrics to tables."""
        return self.config.get('metric_table_mappings', {})
    
    def get_tables_for_metric(self, metric: str) -> List[str]:
        """Get list of tables that might contain data for a metric."""
        mappings = self.get_metric_table_mappings()
        metric_lower = metric.lower()
        
        # Direct match
        if metric_lower in mappings:
            return mappings[metric_lower]
        
        # Partial match
        matching_tables = []
        for keyword, tables in mappings.items():
            if keyword in metric_lower:
                matching_tables.extend(tables)
        
        return list(set(matching_tables))  # Remove duplicates
    
    # Tool-specific table mappings
    def get_tool_table_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get tool-specific table mappings."""
        return self.config.get('tool_table_mappings', {})
    
    def get_tool_primary_table(self, tool_id: str) -> Optional[str]:
        """Get primary table for a tool."""
        mappings = self.get_tool_table_mappings()
        tool_mapping = mappings.get(tool_id, {})
        return tool_mapping.get('primary_table')
    
    def get_tool_date_column(self, tool_id: str) -> Optional[str]:
        """Get date column for a tool."""
        mappings = self.get_tool_table_mappings()
        tool_mapping = mappings.get(tool_id, {})
        return tool_mapping.get('date_column')
    
    # Message templates
    def get_messages(self) -> Dict[str, str]:
        """Get message templates."""
        return self.config.get('messages', {})
    
    def get_message(self, key: str, **kwargs) -> str:
        """Get formatted message template."""
        messages = self.get_messages()
        template = messages.get(key, key)
        try:
            return template.format(**kwargs)
        except KeyError:
            logger.warning(f"⚠️ [SYSTEM_CONFIG] Missing format key in message template | key={key}")
            return template


# Global config instance
system_config = SystemConfig()

