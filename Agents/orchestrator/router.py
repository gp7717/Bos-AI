"""Router for classifying queries and determining if agents are needed."""

from __future__ import annotations

from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import RouterDecision
from Agents.QueryAgent.config import get_resources
from .planner import Planner

RouteType = Literal["simple_response", "needs_agents"]


class _RouterResponse(BaseModel):
    route_type: RouteType
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class Router:
    """LLM-based router that classifies queries to determine if agents are needed."""

    def __init__(self, llm: AzureChatOpenAI | None = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._parser = PydanticOutputParser(pydantic_object=_RouterResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a query router for a Business Intelligence system. "
                        "Classify user queries into two categories:\n"
                        "- 'simple_response': Greetings, casual conversation, or queries that don't require data retrieval, computation, or API calls. Examples: 'Hi', 'Hello', 'How are you?', 'Thanks', 'Goodbye'.\n"
                        "- 'needs_agents': Queries that require data retrieval (SQL), calculations, API calls, or any specialized agent. Examples: 'Show me sales data', 'Calculate revenue', 'What tables are available?', 'Get API documentation'.\n"
                        "Be conservative: if a query might need data or tools, classify it as 'needs_agents'. "
                        "Return JSON that matches the format instructions."
                    ),
                ),
                (
                    "user",
                    "Question: {question}\n"
                    "Context keys: {context_keys}\n"
                    "Format instructions: {format_instructions}",
                ),
            ]
        )

    def route(
        self,
        *,
        question: str,
        context: dict | None = None,
    ) -> RouterDecision:
        """
        Classify a query to determine if agents are needed.
        
        Returns RouterDecision with route_type ('simple_response' or 'needs_agents').
        """
        try:
            response = (
                self._prompt.partial(
                    format_instructions=self._parser.get_format_instructions()
                )
                | self.llm
                | self._parser
            ).invoke(
                {
                    "question": question,
                    "context_keys": list((context or {}).keys()),
                }
            )
        except Exception:  # pragma: no cover - fallback resilience
            # Fallback to planner heuristics to determine if agents are needed
            candidates = Planner._heuristic_candidates(question)
            # If planner finds candidates, route to agents; otherwise simple response
            route_type: RouteType = "needs_agents" if candidates else "simple_response"
            response = _RouterResponse(
                route_type=route_type,
                rationale="Using heuristic fallback due to router error",
                confidence=0.5,
            )

        return RouterDecision(
            route_type=response.route_type,
            rationale=response.rationale,
            confidence=float(min(1.0, max(0.0, response.confidence))),
        )


__all__ = ["Router", "RouteType"]

