"""Composer agent that fuses individual agent outputs into a final answer."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any, Iterable, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from Agents.core.models import AgentResult, OrchestratorResponse, TabularResult
from Agents.QueryAgent.config import get_resources

logger = logging.getLogger(__name__)


def _normalize_tabular_value(value: Any) -> Any:
    """
    Normalize a tabular cell value to ensure it's JSON-serializable and table-renderable.
    
    Converts complex objects (dicts, lists) to comma-separated strings for better readability,
    handles Decimal types, and preserves simple types (str, int, float, bool, None).
    
    Args:
        value: The value to normalize
        
    Returns:
        Normalized value that can be safely serialized to JSON and rendered in tables
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    
    # Handle lists/arrays - convert to comma-separated string
    if isinstance(value, list):
        if not value:
            return ""
        # Convert each element to a readable string
        normalized_items = []
        for item in value:
            if isinstance(item, dict):
                # Try to extract a meaningful field from the object
                readable = _extract_readable_value(item)
                normalized_items.append(readable)
            elif isinstance(item, (str, int, float, bool)):
                normalized_items.append(str(item))
            else:
                normalized_items.append(str(item))
        return ", ".join(normalized_items)
    
    # Handle single dict/object - convert to readable format
    if isinstance(value, dict):
        return _extract_readable_value(value)
    
    # For any other type, convert to string
    return str(value)


def _extract_readable_value(obj: dict) -> str:
    """
    Extract a readable string representation from a dictionary/object.
    
    Tries to find common fields like 'name', 'title', 'product_name', etc.
    Falls back to a comma-separated key-value format or JSON string.
    
    Args:
        obj: Dictionary to extract readable value from
        
    Returns:
        Readable string representation
    """
    if not isinstance(obj, dict) or not obj:
        return str(obj)
    
    # Priority order for extracting readable fields
    preferred_fields = [
        "name", "title", "product_name", "productName",
        "label", "description", "id", "value", "text"
    ]
    
    # Try to find a preferred field
    for field in preferred_fields:
        if field in obj and obj[field] is not None:
            return str(obj[field])
    
    # If no preferred field found, try to create a readable format
    # Use first non-empty string value, or create key-value pairs
    for key, val in obj.items():
        if val is not None and val != "":
            if isinstance(val, (str, int, float, bool)):
                return str(val)
    
    # Fallback: create comma-separated key-value pairs for readability
    try:
        # Limit to first few key-value pairs to avoid overly long strings
        items = list(obj.items())[:3]
        kv_pairs = [f"{k}: {v}" for k, v in items if v is not None]
        if kv_pairs:
            return ", ".join(kv_pairs)
    except Exception:
        pass
    
    # Final fallback: JSON string
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _normalize_tabular_data(tabular: Optional[TabularResult]) -> Optional[TabularResult]:
    """
    Normalize all values in tabular data to ensure proper serialization.
    
    Args:
        tabular: The TabularResult to normalize
        
    Returns:
        A new TabularResult with normalized values, or None if input is None
    """
    if not tabular or not tabular.rows:
        return tabular
    
    normalized_rows = []
    for row in tabular.rows:
        if isinstance(row, dict):
            normalized_row = {
                str(key): _normalize_tabular_value(value)
                for key, value in row.items()
            }
            normalized_rows.append(normalized_row)
        else:
            # If row is not a dict, try to normalize it as-is
            normalized_rows.append(_normalize_tabular_value(row))
    
    return TabularResult(
        columns=tabular.columns,
        rows=normalized_rows,
        row_count=tabular.row_count,
    )


class Composer:
    """LLM-backed composer that consolidates agent answers."""

    def __init__(self, llm: Optional[AzureChatOpenAI] = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Senior Business Intelligence Agent. "
                        "Provide a concise and direct answer to the user query. Do not provide any explanation or context. "
                        "If any metrics is 0 or not available , dont mention that in the answer. "
                        "Structured metrics appear in the agent summaries as entries beginning with 'structured'. "
                        "Treat structured metrics as authoritative, especially for financial figures such as revenue, spend, ROAS, CPA, conversion rate, and orders. "
                        "Only claim a metric is missing when it is absent from all structured summaries. "
                        "Prefer structured data from quantitative agents over narrative text, and never repeat statements that contradict available structured metrics. "
                        "Do not explain your process or how conclusions were reached. "
                        "If no data is available or no rows are returned, clearly state that no data was returned."
                        "Use Indian Rupees (Rs.) with commas and two decimals for all monetary amounts (e.g., Rs.1,75,206.00), never ₹ or INR. "
                        "Be professional. Do not use emojis, hashtags, or unnecessary formatting—only use '\\n' for new lines. "
                        "Do not include boilerplate, incomplete placeholders, or repeat explanations. Only summarize the most important findings and information relevant to the user question."
                    ),
                ),
                (
                    "user",
                    (
                        "User question: {question}\n"
                        "Planner rationale: {planner_rationale}\n"
                        "Agent summaries:\n{agent_summaries}"
                    ),
                ),
            ]
        )
        self._simple_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful Business Intelligence assistant. "
                        "Respond naturally and conversationally to the user's query. "
                        "Be friendly, professional, and concise. "
                        "For greetings, respond warmly and offer to help with data queries or analysis. "
                        "Do not use emojis, hashtags, or unnecessary formatting—only use '\\n' for new lines. "
                        "Keep responses brief and to the point."
                    ),
                ),
                (
                    "user",
                    "User query: {question}",
                ),
            ]
        )

    def compose(
        self,
        *,
        question: str,
        planner_rationale: str,
        agent_results: Iterable[AgentResult],
        context: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        results = list(agent_results)
        summaries = []
        tabular = self._select_tabular(results)
        
        # Normalize tabular data to ensure proper serialization for tables
        normalized_tabular = _normalize_tabular_data(tabular)
        
        for result in results:
            status = result.status
            base = f"Agent: {result.agent} | status: {status}."
            if result.answer:
                base += f" Answer: {result.answer}"
            if result.error:
                base += f" Error: {result.error.message}"
            summaries.append(base)

        llm_response = self._prompt | self.llm | RunnableLambda(lambda message: message.content)
        answer = llm_response.invoke(
            {
                "question": question,
                "planner_rationale": planner_rationale,
                "agent_summaries": "\n".join(summaries) or "No agents produced outputs.",
            }
        )

        return OrchestratorResponse(
            answer=str(answer),
            data=normalized_tabular,
            agent_results=results,
            metadata=metadata or {},
        )

    def compose_simple_response(
        self,
        *,
        question: str,
        router_rationale: str = "",
        context: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        """
        Compose a response for simple queries that don't require agents.
        Uses a conversational prompt instead of the BI-focused prompt.
        """
        llm_response = self._simple_prompt | self.llm | RunnableLambda(lambda message: message.content)
        answer = llm_response.invoke(
            {
                "question": question,
            }
        )

        response_metadata = metadata or {}
        if router_rationale:
            response_metadata["router_rationale"] = router_rationale

        return OrchestratorResponse(
            answer=str(answer),
            data=None,
            agent_results=[],
            metadata=response_metadata,
        )

    @staticmethod
    def _select_tabular(results: Iterable[AgentResult]) -> Optional[TabularResult]:
        for result in results:
            if result.tabular:
                return result.tabular
        return None


__all__ = ["Composer"]


