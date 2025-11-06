"""Agent registry for plug-and-play agent management."""
from typing import Dict, Optional, List, Type
from app.core.base import BaseAgent, AgentMetadata, AgentType
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """Registry for managing agents in the system."""
    
    _instance: Optional["AgentRegistry"] = None
    _agents: Dict[str, BaseAgent] = {}
    _agent_classes: Dict[str, Type[BaseAgent]] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._agent_classes = {}
        return cls._instance
    
    def register(self, agent: BaseAgent, override: bool = False) -> bool:
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance to register
            override: Whether to override existing agent
            
        Returns:
            True if registration successful, False otherwise
        """
        agent_id = agent.metadata.agent_id
        
        if agent_id in self._agents and not override:
            logger.warning(
                f"⚠️ [REGISTRY] Agent already registered | "
                f"agent_id={agent_id} | override={override}"
            )
            return False
        
        if not agent.is_enabled():
            logger.warning(
                f"⚠️ [REGISTRY] Agent is disabled, skipping registration | "
                f"agent_id={agent_id}"
            )
            return False
        
        self._agents[agent_id] = agent
        logger.info(
            f"✅ [REGISTRY] Agent registered | "
            f"agent_id={agent_id} | "
            f"type={agent.metadata.agent_type.value} | "
            f"version={agent.metadata.version}"
        )
        return True
    
    def register_class(self, agent_class: Type[BaseAgent], metadata: AgentMetadata, override: bool = False) -> bool:
        """
        Register an agent class for lazy instantiation.
        
        Args:
            agent_class: Agent class to register
            metadata: Agent metadata
            override: Whether to override existing class
            
        Returns:
            True if registration successful, False otherwise
        """
        agent_id = metadata.agent_id
        
        if agent_id in self._agent_classes and not override:
            logger.warning(
                f"⚠️ [REGISTRY] Agent class already registered | "
                f"agent_id={agent_id} | override={override}"
            )
            return False
        
        self._agent_classes[agent_id] = agent_class
        logger.info(
            f"✅ [REGISTRY] Agent class registered | "
            f"agent_id={agent_id} | "
            f"type={metadata.agent_type.value}"
        )
        return True
    
    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get registered agent instance.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        if agent_id in self._agents:
            return self._agents[agent_id]
        
        # Try lazy instantiation
        if agent_id in self._agent_classes:
            agent_class = self._agent_classes[agent_id]
            try:
                # This would need metadata to be stored separately
                # For now, return None if not instantiated
                logger.debug(
                    f"🔍 [REGISTRY] Agent class found but not instantiated | "
                    f"agent_id={agent_id}"
                )
            except Exception as e:
                logger.error(
                    f"❌ [REGISTRY] Failed to instantiate agent | "
                    f"agent_id={agent_id} | error={str(e)}"
                )
        
        logger.warning(f"⚠️ [REGISTRY] Agent not found | agent_id={agent_id}")
        return None
    
    def get_all(self, agent_type: Optional[AgentType] = None) -> List[BaseAgent]:
        """
        Get all registered agents, optionally filtered by type.
        
        Args:
            agent_type: Optional filter by agent type
            
        Returns:
            List of agent instances
        """
        agents = list(self._agents.values())
        
        if agent_type:
            agents = [a for a in agents if a.metadata.agent_type == agent_type]
        
        return agents
    
    def get_ids(self, agent_type: Optional[AgentType] = None) -> List[str]:
        """
        Get all registered agent IDs, optionally filtered by type.
        
        Args:
            agent_type: Optional filter by agent type
            
        Returns:
            List of agent IDs
        """
        if agent_type:
            return [
                agent_id for agent_id, agent in self._agents.items()
                if agent.metadata.agent_type == agent_type
            ]
        return list(self._agents.keys())
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"🗑️ [REGISTRY] Agent unregistered | agent_id={agent_id}")
            return True
        
        if agent_id in self._agent_classes:
            del self._agent_classes[agent_id]
            logger.info(f"🗑️ [REGISTRY] Agent class unregistered | agent_id={agent_id}")
            return True
        
        logger.warning(f"⚠️ [REGISTRY] Agent not found for unregistration | agent_id={agent_id}")
        return False
    
    def is_registered(self, agent_id: str) -> bool:
        """Check if agent is registered."""
        return agent_id in self._agents or agent_id in self._agent_classes
    
    def clear(self):
        """Clear all registered agents."""
        self._agents.clear()
        self._agent_classes.clear()
        logger.info("🗑️ [REGISTRY] All agents cleared")


# Global registry instance
agent_registry = AgentRegistry()

