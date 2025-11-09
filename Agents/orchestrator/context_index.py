from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml

from Agents.core.models import AgentName


_TOKEN_REGEX = re.compile(r"[a-z0-9]+")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContextDocument:
    """Single retrievable document for an agent."""

    agent: AgentName
    document_id: str
    title: str
    content: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievedContext:
    """Result of retrieving contextual snippets for an agent."""

    document: ContextDocument
    score: float


def _tokenise(text: str) -> List[str]:
    return _TOKEN_REGEX.findall(text.lower())


class _BM25Index:
    """Lightweight BM25 implementation for small corpora."""

    def __init__(self, documents: Sequence[ContextDocument]) -> None:
        self._documents = list(documents)
        self._doc_tokens: List[Counter[str]] = []
        self._doc_lengths: List[int] = []
        self._doc_freq: MutableMapping[str, int] = defaultdict(int)

        for doc in self._documents:
            tokens = Counter(_tokenise(f"{doc.title} {doc.content}"))
            self._doc_tokens.append(tokens)
            length = sum(tokens.values())
            self._doc_lengths.append(length)
            for token in tokens:
                self._doc_freq[token] += 1

        self._avgdl = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._num_docs = len(self._documents)
        self._k1 = 1.5
        self._b = 0.75

    def query(self, text: str, *, top_k: int = 3) -> List[RetrievedContext]:
        if not self._documents or not text.strip():
            return []

        query_tokens = _tokenise(text)
        if not query_tokens:
            return []

        scores: List[tuple[float, int]] = []
        for idx, tokens in enumerate(self._doc_tokens):
            score = 0.0
            doc_len = self._doc_lengths[idx] or 1
            for token in query_tokens:
                tf = tokens.get(token)
                if not tf:
                    continue
                idf = self._idf(token)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / (self._avgdl or 1))
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda item: item[0], reverse=True)
        results: List[RetrievedContext] = []
        for score, idx in scores[:top_k]:
            results.append(RetrievedContext(document=self._documents[idx], score=score))
        return results

    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + ((self._num_docs - df + 0.5) / (df + 0.5)))


class AgentContextRetriever:
    """Retrieves relevant contextual snippets for each agent."""

    def __init__(self, indexes: Mapping[AgentName, _BM25Index]) -> None:
        self._indexes = dict(indexes)

    @classmethod
    def from_default_sources(cls) -> "AgentContextRetriever":
        documents = _load_default_documents()
        indexes = {
            agent: _BM25Index(agent_documents)
            for agent, agent_documents in documents.items()
            if agent_documents
        }
        return cls(indexes)

    def available_agents(self) -> Sequence[AgentName]:
        return list(self._indexes.keys())

    def retrieve(
        self, question: str, *, top_k: int = 3
    ) -> Dict[AgentName, List[RetrievedContext]]:
        return {
            agent: index.query(question, top_k=top_k)
            for agent, index in self._indexes.items()
        }

    def retrieve_for_agent(
        self, agent: AgentName, question: str, *, top_k: int = 3
    ) -> List[RetrievedContext]:
        index = self._indexes.get(agent)
        if index is None:
            return []
        return index.query(question, top_k=top_k)


def _load_default_documents() -> Dict[AgentName, List[ContextDocument]]:
    documents: Dict[AgentName, List[ContextDocument]] = {
        "docs": _load_api_docs_documents(),
        "sql": _load_sql_documents(),
        "computation": _load_computation_documents(),
    }
    return documents


def _load_api_docs_documents() -> List[ContextDocument]:
    context_path = _PROJECT_ROOT / "Docs" / "docs_context.yaml"
    if not context_path.exists():
        return []

    payload = yaml.safe_load(context_path.read_text(encoding="utf-8")) or {}
    documents: List[ContextDocument] = []

    for section in payload.get("sections", []):
        content = section.get("content", "")
        if not content:
            continue
        documents.append(
            ContextDocument(
                agent="docs",
                document_id=section.get("id", ""),
                title=section.get("title", ""),
                content=content,
                metadata={
                    "source": section.get("source"),
                    "kind": "section",
                },
            )
        )

    for entry in payload.get("contexts", []):
        content = entry.get("content", "")
        if not content:
            continue
        documents.append(
            ContextDocument(
                agent="docs",
                document_id=entry.get("id", ""),
                title=entry.get("title", ""),
                content=content,
                metadata={
                    "source": entry.get("source"),
                    "section": entry.get("section"),
                    "kind": "context",
                },
            )
        )

    return documents


def _load_sql_documents() -> List[ContextDocument]:
    context_path = _PROJECT_ROOT / "Agents" / "QueryAgent" / "table_context.yaml"
    if not context_path.exists():
        return []

    payload = yaml.safe_load(context_path.read_text(encoding="utf-8")) or {}
    tables: Mapping[str, Dict[str, Any]] = payload.get("tables", {})
    documents: List[ContextDocument] = []

    for table_name, config in tables.items():
        description = config.get("description", "")
        columns = config.get("columns", [])
        column_text = ", ".join(columns)
        combined_content = f"{description}\nColumns: {column_text}".strip()
        documents.append(
            ContextDocument(
                agent="sql",
                document_id=table_name,
                title=table_name,
                content=combined_content,
                metadata={"columns": columns},
            )
        )

    return documents


def _load_computation_documents() -> List[ContextDocument]:
    descriptions = [
        (
            "ad-hoc-computation",
            "Ad-hoc business calculations",
            (
                "Performs arithmetic with numbers provided directly in the question or context. "
                "Ideal for margin, growth, ratio, forecasting, and scenario modelling when data is supplied."
            ),
        ),
        (
            "data-synthesis",
            "Combining provided figures",
            (
                "Aggregates or compares metrics supplied in the prompt (e.g., revenue vs expenses) "
                "without querying external databases."
            ),
        ),
    ]
    documents: List[ContextDocument] = []
    for doc_id, title, content in descriptions:
        documents.append(
            ContextDocument(
                agent="computation",
                document_id=doc_id,
                title=title,
                content=content,
                metadata={"kind": "capability"},
            )
        )
    return documents


__all__ = [
    "AgentContextRetriever",
    "ContextDocument",
    "RetrievedContext",
]


