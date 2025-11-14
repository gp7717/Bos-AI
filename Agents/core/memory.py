"""Helper utilities for agents to interact with the shared memory/scratchpad."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Agents.core.models import AgentName, MemoryEntry, Scratchpad


def get_scratchpad(context: Optional[Dict[str, Any]] = None) -> Optional[Scratchpad]:
    """
    Extract scratchpad from context dictionary.
    
    Args:
        context: Context dictionary that may contain a scratchpad
        
    Returns:
        Scratchpad instance if found, None otherwise
    """
    if not context:
        return None
    
    scratchpad = context.get("scratchpad")
    if isinstance(scratchpad, Scratchpad):
        return scratchpad
    
    # Handle dict representation (for serialization scenarios)
    if isinstance(scratchpad, dict):
        try:
            return Scratchpad(**scratchpad)
        except Exception:
            return None
    
    return None


def add_to_memory(
    context: Optional[Dict[str, Any]],
    agent: AgentName,
    content: str,
    category: str = "finding",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Add an entry to the shared memory/scratchpad.
    
    This is a convenience function for agents to write to memory.
    The scratchpad must be present in the context for this to work.
    
    Args:
        context: Context dictionary containing scratchpad
        agent: Name of the agent adding the entry
        content: Content to add
        category: Category of entry (finding, data_summary, insight, context, error)
        metadata: Optional metadata dictionary
        
    Returns:
        True if entry was added successfully, False otherwise
    """
    scratchpad = get_scratchpad(context)
    if scratchpad is None:
        return False
    
    try:
        scratchpad.add(
            agent=agent,
            content=content,
            category=category,  # type: ignore[arg-type]
            metadata=metadata,
        )
        return True
    except Exception:
        return False


def read_memory(
    context: Optional[Dict[str, Any]],
    agent: Optional[AgentName] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[MemoryEntry]:
    """
    Read entries from the shared memory/scratchpad.
    
    Args:
        context: Context dictionary containing scratchpad
        agent: Optional agent name to filter by
        category: Optional category to filter by
        limit: Optional limit on number of entries to return
        
    Returns:
        List of memory entries matching the criteria
    """
    scratchpad = get_scratchpad(context)
    if scratchpad is None:
        return []
    
    entries = scratchpad.entries
    
    # Filter by agent if specified
    if agent:
        entries = [e for e in entries if e.agent == agent]
    
    # Filter by category if specified
    if category:
        entries = [e for e in entries if e.category == category]
    
    # Apply limit if specified
    if limit:
        entries = entries[-limit:]
    
    return entries


def get_memory_summary(context: Optional[Dict[str, Any]], limit: int = 20) -> str:
    """
    Get a formatted summary of memory entries.
    
    Args:
        context: Context dictionary containing scratchpad
        limit: Maximum number of recent entries to include
        
    Returns:
        Formatted string summary
    """
    scratchpad = get_scratchpad(context)
    if scratchpad is None:
        return "No shared memory available."
    
    return scratchpad.get_summary()


def get_data_summaries(context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Get all data summaries from memory, keyed by agent name.
    
    Args:
        context: Context dictionary containing scratchpad
        
    Returns:
        Dictionary mapping agent names to their data summaries
    """
    scratchpad = get_scratchpad(context)
    if scratchpad is None:
        return {}
    
    return scratchpad.get_data_summaries()


def get_findings(context: Optional[Dict[str, Any]]) -> List[str]:
    """
    Get all findings from memory across all agents.
    
    Args:
        context: Context dictionary containing scratchpad
        
    Returns:
        List of finding strings
    """
    scratchpad = get_scratchpad(context)
    if scratchpad is None:
        return []
    
    return scratchpad.get_findings()


__all__ = [
    "add_to_memory",
    "get_data_summaries",
    "get_findings",
    "get_memory_summary",
    "get_scratchpad",
    "read_memory",
]

