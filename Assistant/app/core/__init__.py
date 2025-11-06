"""Core module for agent base classes and interfaces."""
from app.core.base import BaseAgent, AgentMetadata
from app.core.registry import AgentRegistry

__all__ = ["BaseAgent", "AgentMetadata", "AgentRegistry"]

