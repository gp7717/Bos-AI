"""Planner for selecting which agents to execute for a query."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Mapping, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import AgentName, PlannerDecision
from Agents.orchestrator.capability_classifier import CapabilityClassifier
from Agents.orchestrator.context_index import AgentContextRetriever, RetrievedContext
from Agents.QueryAgent.config import get_resources


logger = logging.getLogger(__name__)

_ALLOWED_AGENTS: Sequence[AgentName] = ("docs", "sql", "computation")
_MAX_CONTEXT_SNIPPETS = 2
_MAX_SNIPPET_CHARS = 400


class _PlannerResponse(BaseModel):
    agents: List[AgentName]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class Planner:
    """Combines retrieval, capability scoring, and LLM arbitration to pick agents."""

    def __init__(
        self,
        llm: AzureChatOpenAI | None = None,
        *,
        retriever: AgentContextRetriever | None = None,
        classifier: CapabilityClassifier | None = None,
    ) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._retriever = retriever or AgentContextRetriever.from_default_sources()
        self._classifier = classifier or CapabilityClassifier.default()

        self._parser = PydanticOutputParser(pydantic_object=_PlannerResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the orchestration planner for Bos-AI. Given the user's question and the "
                    "evidence for each agent, select the ordered list of agents that should run. "
                    "Choose only from: docs (documentation), sql (warehouse queries), computation (ad-hoc maths). "
                    "Return JSON that matches the format instructions exactly.",
                ),
                (
                    "user",
                    "Question:\n{question}\n\n"
                    "Disabled agents: {disabled}\n\n"
                    "Agent evidence:\n{agent_evidence}\n\n"
                    "Format instructions: {format_instructions}",
                ),
            ]
        )

    def plan(
        self,
        *,
        question: str,
        prefer: Sequence[AgentName] = (),
        disable: Sequence[AgentName] = (),
        context: dict | None = None,
    ) -> PlannerDecision:
        retrieval_hits = self._retriever.retrieve(question)
        capability_scores = self._classifier.score(question, retrieval_hits=retrieval_hits)

        agents_to_consider = self._apply_preferences(
            [agent for agent, _ in sorted(capability_scores.items(), key=lambda item: item[1], reverse=True)],
            prefer,
            disable,
        )
        if not agents_to_consider:
            agents_to_consider = [agent for agent in _ALLOWED_AGENTS if agent not in disable]

        agent_evidence = self._build_agent_evidence(
            capability_scores,
            retrieval_hits,
            prefer=prefer,
            disable=disable,
        )

        try:
            prompt_messages = self._prompt.format_prompt(
                question=question,
                disabled=list(disable),
                agent_evidence=json.dumps(agent_evidence, ensure_ascii=False, indent=2),
                format_instructions=self._parser.get_format_instructions(),
            ).to_messages()
            llm_response = self.llm.invoke(prompt_messages)
            raw_content = getattr(llm_response, "content", llm_response)
            response = self._parser.parse(raw_content)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Planner LLM arbitration failed; falling back to capability scores", exc_info=exc)
            fallback_agents = self._fallback_agents(capability_scores, disable=disable)
            return PlannerDecision(
                rationale="Using capability score ordering due to planner error",
                chosen_agents=tuple(fallback_agents),
                confidence=0.4,
                guardrails={
                    "disabled": list(disable),
                    "capability_scores": capability_scores,
                },
            )

        filtered_agents = [
            agent for agent in response.agents if agent in agents_to_consider and agent not in disable
        ]
        if not filtered_agents:
            filtered_agents = self._fallback_agents(capability_scores, disable=disable)

        return PlannerDecision(
            rationale=response.rationale,
            chosen_agents=tuple(filtered_agents),
            confidence=float(min(1.0, max(0.0, response.confidence))),
            guardrails={
                "disabled": list(disable),
                "capability_scores": capability_scores,
                "retrieval_hits": {agent: self._serialise_hits(hits) for agent, hits in retrieval_hits.items()},
            },
        )

    def _build_agent_evidence(
        self,
        capability_scores: Mapping[AgentName, float],
        retrieval_hits: Mapping[AgentName, Sequence[RetrievedContext]],
        *,
        prefer: Sequence[AgentName],
        disable: Sequence[AgentName],
    ) -> List[Dict[str, object]]:
        evidence: List[Dict[str, object]] = []
        for agent in _ALLOWED_AGENTS:
            profile = self._classifier.profile(agent)
            hits = retrieval_hits.get(agent, [])
            evidence.append(
                {
                    "agent": agent,
                    "score": round(capability_scores.get(agent, 0.0), 3),
                    "preferred": agent in prefer,
                    "disabled": agent in disable,
                    "description": profile.description,
                    "strengths": list(profile.strengths),
                    "top_context": self._serialise_hits(hits[:_MAX_CONTEXT_SNIPPETS]),
                }
            )
        return evidence

    @staticmethod
    def _serialise_hits(hits: Sequence[RetrievedContext]) -> List[Dict[str, object]]:
        serialised = []
        for hit in hits:
            serialised.append(
                {
                    "document_id": hit.document.document_id,
                    "title": hit.document.title,
                    "score": round(hit.score, 3),
                    "snippet": Planner._truncate(hit.document.content),
                    "metadata": dict(hit.document.metadata),
                }
            )
        return serialised

    @staticmethod
    def _truncate(text: str) -> str:
        trimmed = text.strip()
        if len(trimmed) <= _MAX_SNIPPET_CHARS:
            return trimmed
        return f"{trimmed[:_MAX_SNIPPET_CHARS].rstrip()}…"

    @staticmethod
    def _apply_preferences(
        candidates: Iterable[AgentName],
        prefer: Sequence[AgentName],
        disable: Sequence[AgentName],
    ) -> List[AgentName]:
        prefer_set = list(dict.fromkeys(agent for agent in prefer if agent not in disable))
        filtered = [agent for agent in candidates if agent in _ALLOWED_AGENTS and agent not in disable]

        ordered: List[AgentName] = []
        for agent in prefer_set:
            if agent not in ordered and agent in filtered:
                ordered.append(agent)
        for agent in filtered:
            if agent not in ordered:
                ordered.append(agent)
        return ordered

    @staticmethod
    def _fallback_agents(
        capability_scores: Mapping[AgentName, float],
        *,
        disable: Sequence[AgentName],
    ) -> List[AgentName]:
        candidates = [
            agent
            for agent, _ in sorted(capability_scores.items(), key=lambda item: item[1], reverse=True)
            if agent not in disable
        ]
        if not candidates:
            candidates = [agent for agent in _ALLOWED_AGENTS if agent not in disable]
        return candidates or ["sql"]


__all__ = ["Planner"]
