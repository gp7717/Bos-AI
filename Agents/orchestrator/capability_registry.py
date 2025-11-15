"""Capability registry system for intelligent agent and tool selection."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from Agents.core.models import AgentCapability, AgentName, Skill, Tool

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """
    Central registry of all agents, their skills, tools, and metadata.
    Enables intelligent selection of agents and tools based on query requirements.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the capability registry.

        Args:
            config_path: Path to capabilities YAML config file. If None, uses default.
        """
        self.agents: Dict[str, AgentCapability] = {}
        self.tools_by_metric: Dict[str, List[Tool]] = {}
        self.tools_by_id: Dict[str, Tool] = {}
        self.skills_by_capability: Dict[str, List[Skill]] = {}

        if config_path:
            self.load_from_config(config_path)
        else:
            # Use default path
            default_path = Path(__file__).parent / "capabilities.yaml"
            if default_path.exists():
                self.load_from_config(str(default_path))
            else:
                logger.warning(
                    f"Capabilities config not found at {default_path}, using minimal defaults. "
                    "Some features may not work optimally."
                )
                self._load_defaults()

    def load_from_config(self, config_path: str) -> None:
        """Load capabilities from YAML configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            agents_config = config.get("agents", [])
            for agent_config in agents_config:
                self._load_agent(agent_config)

            # Build metric-to-tool mappings
            self._build_metric_mappings()

            logger.info(
                f"Loaded {len(self.agents)} agents with {len(self.tools_by_id)} tools from {config_path}"
            )
        except Exception as exc:
            logger.error(f"Failed to load capabilities from {config_path}: {exc}", exc_info=True)
            self._load_defaults()

    def _load_agent(self, agent_config: Dict[str, Any]) -> None:
        """Load a single agent configuration."""
        agent_name = agent_config.get("name")
        if not agent_name:
            logger.warning("Skipping agent config without name")
            return

        # Load tools
        tools = []
        for tool_config in agent_config.get("tools", []):
            tool = Tool(
                id=tool_config.get("id", ""),
                name=tool_config.get("name", ""),
                type=tool_config.get("type", "api_endpoint"),
                agent=agent_name,  # type: ignore[assignment]
                capabilities=tool_config.get("capabilities", []),
                metrics=tool_config.get("metrics", []),
                parameters=tool_config.get("parameters", {}),
                metadata=tool_config.get("metadata", {}),
            )
            tools.append(tool)
            self.tools_by_id[tool.id] = tool

        # Load skills
        skills = []
        for skill_config in agent_config.get("skills", []):
            skill = Skill(
                id=skill_config.get("id", ""),
                name=skill_config.get("name", ""),
                agent=agent_name,  # type: ignore[assignment]
                description=skill_config.get("description", ""),
                tools=skill_config.get("tools", []),
                use_cases=skill_config.get("use_cases", []),
                preferences=skill_config.get("preferences", []),
            )
            skills.append(skill)

            # Index by capability type
            capability_type = skill.name.lower().replace(" ", "_")
            if capability_type not in self.skills_by_capability:
                self.skills_by_capability[capability_type] = []
            self.skills_by_capability[capability_type].append(skill)

        # Create agent capability
        capability = AgentCapability(
            agent_name=agent_name,  # type: ignore[assignment]
            description=agent_config.get("description", ""),
            skills=skills,
            tools=tools,
            strengths=agent_config.get("strengths", []),
            limitations=agent_config.get("limitations", []),
            preferred_for=agent_config.get("preferred_for", []),
            requires=agent_config.get("requires", []),
        )

        self.agents[agent_name] = capability

    def _build_metric_mappings(self) -> None:
        """Build metric-to-tool mappings for fast lookup."""
        self.tools_by_metric.clear()
        for tool in self.tools_by_id.values():
            for metric in tool.metrics:
                metric_lower = metric.lower()
                if metric_lower not in self.tools_by_metric:
                    self.tools_by_metric[metric_lower] = []
                self.tools_by_metric[metric_lower].append(tool)

    def _load_defaults(self) -> None:
        """Load default capabilities if config file is not available."""
        # This will be populated from YAML file, but provides fallback
        logger.info("Using default capability registry (minimal)")
        # Could add hardcoded defaults here if needed

    def find_tools_for_metric(self, metric: str) -> List[Tool]:
        """
        Find tools that provide a specific metric.

        Args:
            metric: Metric name (e.g., "net_profit", "revenue")

        Returns:
            List of tools that provide this metric
        """
        metric_lower = metric.lower().strip()
        # Direct match
        tools = list(self.tools_by_metric.get(metric_lower, []))

        # Fuzzy matching for variations
        if not tools:
            for tool_metric, tool_list in self.tools_by_metric.items():
                if metric_lower in tool_metric or tool_metric in metric_lower:
                    if tool_list:  # Ensure tool_list is not None
                        tools.extend(tool_list)

        # Remove duplicates by tool ID (since Tool objects aren't hashable)
        # Use a dict to track unique tools by ID for deduplication
        unique_tools_dict: Dict[str, Tool] = {}
        for tool in tools:
            if tool and hasattr(tool, 'id') and tool.id:
                # Use tool.id as key to ensure uniqueness
                unique_tools_dict[tool.id] = tool
        
        return list(unique_tools_dict.values())

    def find_agents_for_skill(self, skill_name: str) -> List[AgentCapability]:
        """
        Find agents that have a specific skill.

        Args:
            skill_name: Skill name (e.g., "data_retrieval", "metric_calculation")

        Returns:
            List of agent capabilities with this skill
        """
        skill_lower = skill_name.lower().replace(" ", "_")
        matching_agents = []

        for agent_cap in self.agents.values():
            for skill in agent_cap.skills:
                if skill_lower in skill.name.lower() or skill_lower in skill.id.lower():
                    matching_agents.append(agent_cap)
                    break

        return matching_agents

    def find_best_agent(
        self, query: str, requirements: List[str], preferred_metrics: Optional[List[str]] = None
    ) -> Optional[AgentCapability]:
        """
        Intelligently select the best agent for a query based on requirements.

        Args:
            query: User query text
            requirements: List of requirements (e.g., ["net_profit", "last_4_months"])
            preferred_metrics: List of metrics the user wants

        Returns:
            Best matching agent capability, or None if no match
        """
        query_lower = query.lower()

        # If metrics are specified, check if API has them
        if preferred_metrics:
            for metric in preferred_metrics:
                tools = self.find_tools_for_metric(metric)
                # Prefer API tools (they're usually faster and pre-calculated)
                api_tools = [t for t in tools if t.type == "api_endpoint" and t.agent == "api_docs"]
                if api_tools:
                    api_agent = self.agents.get("api_docs")
                    if api_agent:
                        logger.info(f"Selected api_docs agent for metric: {metric}")
                        return api_agent

        # Check for specific keywords in query
        if any(keyword in query_lower for keyword in ["api", "endpoint", "http"]):
            api_agent = self.agents.get("api_docs")
            if api_agent:
                return api_agent

        # Check for computation keywords
        if any(
            keyword in query_lower
            for keyword in ["calculate", "compute", "forecast", "analyze", "statistics"]
        ):
            comp_agent = self.agents.get("computation")
            if comp_agent:
                return comp_agent

        # Default to SQL for data retrieval
        sql_agent = self.agents.get("sql")
        if sql_agent:
            return sql_agent

        return None

    def get_tool_metadata(self, tool_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Dictionary with tool metadata
        """
        tool = self.tools_by_id.get(tool_id)
        if tool:
            return {
                "id": tool.id,
                "name": tool.name,
                "type": tool.type,
                "agent": tool.agent,
                "capabilities": tool.capabilities,
                "metrics": tool.metrics,
                "parameters": tool.parameters,
                "metadata": tool.metadata,
            }
        return {}

    def register_agent(self, capability: AgentCapability) -> None:
        """
        Register a new agent capability at runtime.

        Args:
            capability: Agent capability to register
        """
        self.agents[capability.agent_name] = capability
        for tool in capability.tools:
            self.tools_by_id[tool.id] = tool
        self._build_metric_mappings()
        logger.info(f"Registered agent: {capability.agent_name}")

    def extract_metrics_from_query(self, query: str) -> List[str]:
        """
        Extract metric names from a query using keyword matching.

        Args:
            query: User query text

        Returns:
            List of detected metric names
        """
        query_lower = query.lower()
        detected_metrics = []

        # Common metric keywords
        metric_keywords = {
            "net profit": "net_profit",
            "profit": "net_profit",
            "revenue": "revenue",
            "sales": "sales",
            "cogs": "cogs",
            "roas": "roas",
            "ad spend": "ad_spend",
            "orders": "orders",
            "conversion": "conversion_rate",
            "cpa": "cpa",
            "aov": "aov",
        }

        for keyword, metric in metric_keywords.items():
            if keyword in query_lower:
                detected_metrics.append(metric)

        return detected_metrics

    def get_agent_capability(self, agent_name: AgentName) -> Optional[AgentCapability]:
        """Get capability profile for a specific agent."""
        return self.agents.get(agent_name)

    def list_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self.tools_by_id.values())

    def list_all_agents(self) -> List[AgentCapability]:
        """Get all registered agents."""
        return list(self.agents.values())


__all__ = ["CapabilityRegistry"]

