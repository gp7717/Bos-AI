"""State definitions for the standalone SQL LangGraph agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class SQLAgentState(MessagesState, total=False):
    """Typed state passed between nodes in the SQL agent graph.

    Extends the built-in :class:`langgraph.graph.MessagesState` with optional
    diagnostic fields that downstream nodes can use for logging or analytics.
    The core contract (``messages`` list) stays compatible with LangGraph's
    expectations, so nodes from the tutorial can operate without changes.
    """

    metadata: Dict[str, Any]
    error: Optional[str]
    last_query: Optional[str]
    last_query_result: Optional[Dict[str, Any]]
    table_context: Dict[str, Any]
    statistics: Dict[str, Any]
    final_answer: Optional[str]


def ensure_messages(state: SQLAgentState) -> List[BaseMessage]:
    """Return the message list from *state*, initialising if necessary.

    LangGraph guarantees that ``messages`` exists, but when we craft unit
    tests or manually instantiate the state we may forget to seed it.  This
    helper keeps node implementations tidy and defensive.
    """

    messages: Optional[List[BaseMessage]] = state.get("messages")  # type: ignore[assignment]
    if messages is None:
        messages = []
        state["messages"] = messages
    return messages


