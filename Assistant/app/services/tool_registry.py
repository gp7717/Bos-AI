"""Tool Registry - Declarative catalog of tools and capabilities."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from app.models.schemas import ToolDefinition, ToolCapability
from app.config.settings import settings


class ToolRegistry:
    """Manages tool definitions and capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize tool registry from YAML config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "tools.yaml"
        
        self.tools: Dict[str, ToolDefinition] = {}
        self._load_tools(config_path)
    
    def _load_tools(self, config_path: Path):
        """Load tools from YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for tool_config in config.get('tools', []):
            tool_def = ToolDefinition(**tool_config)
            self.tools[tool_def.tool_id] = tool_def
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool definition by ID."""
        return self.tools.get(tool_id)
    
    def get_tools_by_tag(self, tag: str) -> List[ToolDefinition]:
        """Get all tools with a specific observability tag."""
        return [
            tool for tool in self.tools.values()
            if tag in tool.observability_tags
        ]
    
    def get_tools_by_kind(self, kind: str) -> List[ToolDefinition]:
        """Get all tools of a specific kind."""
        return [
            tool for tool in self.tools.values()
            if tool.kind == kind
        ]
    
    def get_capability(self, tool_id: str, capability_name: str) -> Optional[ToolCapability]:
        """Get specific capability from a tool."""
        tool = self.get_tool(tool_id)
        if not tool:
            return None
        
        for cap in tool.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def can_fulfill_metric(self, metric: str) -> List[ToolDefinition]:
        """Find tools that can provide a specific metric."""
        matching_tools = []
        for tool in self.tools.values():
            for cap in tool.capabilities:
                if metric.lower() in [m.lower() for m in cap.metrics]:
                    matching_tools.append(tool)
                    break
        return matching_tools
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_join_keys(self, tool_id: str) -> List[Dict[str, str]]:
        """Get join keys for a tool."""
        tool = self.get_tool(tool_id)
        if tool:
            return tool.join_keys
        return []


# Global registry instance
tool_registry = ToolRegistry()

