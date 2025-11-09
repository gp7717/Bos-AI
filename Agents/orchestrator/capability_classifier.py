from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from Agents.core.models import AgentName
from Agents.orchestrator.context_index import RetrievedContext


_TOKEN_REGEX = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CapabilityProfile:
    agent: AgentName
    description: str
    strengths: Sequence[str]
    keywords: Set[str]
    handles_numeric: bool = False
    prefers_structured_data: bool = False
    understands_api_contracts: bool = False
    score_bias: float = 0.0


def _tokenise(question: str) -> Set[str]:
    return set(_TOKEN_REGEX.findall(question.lower()))


def _extract_features(question: str) -> Dict[str, bool]:
    lowered = question.lower()
    return {
        "mentions_endpoint": any(keyword in lowered for keyword in ["/api", "endpoint", "http", "route"]),
        "asks_how": lowered.strip().startswith(("how", "when", "where")),
        "asks_what": lowered.strip().startswith(("what", "which")),
        "contains_numbers": bool(re.search(r"\d", question)),
        "mentions_sql": any(keyword in lowered for keyword in ["sql", "query", "database", "table"]),
        "mentions_timeframe": any(keyword in lowered for keyword in ["year", "month", "week", "day", "quarter"]),
        "asks_calculate": any(keyword in lowered for keyword in ["calculate", "compute", "difference", "ratio", "forecast", "estimate"]),
        "mentions_doc": any(keyword in lowered for keyword in ["doc", "documentation", "api docs"]),
    }


class CapabilityClassifier:
    """Assigns suitability scores per agent based on capability metadata."""

    def __init__(self, profiles: Sequence[CapabilityProfile]) -> None:
        self._profiles = list(profiles)
        self._profiles_by_agent = {profile.agent: profile for profile in profiles}

    @classmethod
    def default(cls) -> "CapabilityClassifier":
        profiles = [
            CapabilityProfile(
                agent="docs",
                description="Understands REST endpoints, request/response contracts, and troubleshooting notes.",
                strengths=[
                    "Explaining API behaviour",
                    "Listing request parameters",
                    "Diagnosing HTTP errors",
                ],
                keywords={
                    "api",
                    "endpoint",
                    "route",
                    "request",
                    "response",
                    "status",
                    "payload",
                    "net",
                    "profit",
                    "webhook",
                    "auth",
                },
                understands_api_contracts=True,
                score_bias=0,
            ),
            CapabilityProfile(
                agent="sql",
                description="Executes analytical SQL over the warehouse schema and returns tabular results.",
                strengths=[
                    "Aggregations over time",
                    "Joins across attribution and order tables",
                    "Filtering datasets by campaign or channel",
                ],
                keywords={
                    "table",
                    "tables",
                    "query",
                    "database",
                    "sql",
                    "select",
                    "orders",
                    "revenue",
                    "cogs",
                    "cohort",
                    "report",
                    "dataset",
                    "fetch",
                    "pull",
                    "list",
                },
                handles_numeric=True,
                prefers_structured_data=True,
            ),
            CapabilityProfile(
                agent="computation",
                description="Performs arithmetic or statistical reasoning with numbers provided in the prompt or retrieved context.",
                strengths=[
                    "Ad-hoc financial maths",
                    "Scenario modelling with provided metrics",
                    "Ratios, growth rates, and variance",
                ],
                keywords={
                    "calculate",
                    "compute",
                    "difference",
                    "ratio",
                    "growth",
                    "percent",
                    "percentage",
                    "forecast",
                    "projection",
                    "scenario",
                    "margin",
                },
                handles_numeric=True,
            ),
        ]
        return cls(profiles)

    def score(
        self,
        question: str,
        *,
        retrieval_hits: Optional[Mapping[AgentName, Sequence[RetrievedContext]]] = None,
    ) -> Dict[AgentName, float]:
        tokens = _tokenise(question)
        features = _extract_features(question)
        scores: Dict[AgentName, float] = {}

        for profile in self._profiles:
            score = 0.1 + profile.score_bias  # base

            keyword_overlap = len(tokens & profile.keywords)
            score += keyword_overlap * 0.6

            if profile.understands_api_contracts and (
                features["mentions_endpoint"] or features["asks_how"] or features["asks_what"] or features["mentions_doc"]
            ):
                score += 0.8

            if profile.prefers_structured_data and (
                features["mentions_sql"] or features["mentions_timeframe"]
            ):
                score += 0.7

            if profile.handles_numeric and (
                features["contains_numbers"] or features["asks_calculate"]
            ):
                score += 0.7

            if retrieval_hits and profile.agent in retrieval_hits:
                hits = retrieval_hits[profile.agent]
                if hits:
                    # Use the best score as a signal, dampened to avoid overpowering keywords
                    score += min(hits[0].score, 5.0) * 0.2

            scores[profile.agent] = score

        return scores

    def profile(self, agent: AgentName) -> CapabilityProfile:
        return self._profiles_by_agent[agent]


__all__ = ["CapabilityClassifier", "CapabilityProfile"]


