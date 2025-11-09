from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from Agents.QueryAgent.config import get_resources
from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentResult,
    TraceEvent,
    TraceEventType,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT_PATH = _PROJECT_ROOT / "Docs" / "docs_context.yaml"
_LEGACY_CONTEXT_PATH = Path("Docs") / "docs_context.yaml"

TOP_K_CONTEXTS = 4

_MUTATION_KEYWORDS = {
    "delete",
    "remove",
    "destroy",
    "drop",
    "truncate",
    "update",
    "modify",
    "put",
    "patch",
}

_MUTATION_HTTP_VERBS = {"DELETE", "PUT", "PATCH"}


@dataclass(frozen=True)
class ContextChunk:
    """One retrievable chunk of API documentation."""

    identifier: str
    title: str
    source: str
    section: str
    content: str


@dataclass(frozen=True)
class _ScoredChunk:
    chunk: ContextChunk
    score: float
    matches: Tuple[str, ...]


class ApiAgent:
    """Agent that answers API documentation questions via retrieval + LLM."""

    def __init__(
        self,
        *,
        context_path: Optional[Path] = None,
        llm: Optional[Any] = None,
        top_k: int = TOP_K_CONTEXTS,
    ) -> None:
        if llm is None:
            resources = get_resources()
            self.llm = resources.llm
        else:
            self.llm = llm
        self.context_path = context_path or DEFAULT_CONTEXT_PATH
        self.top_k = max(1, int(top_k))
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert assistant for the dashboard backend REST API. "
                        "Use ONLY the provided context snippets to answer the question. "
                        "If the context lacks sufficient detail, say so explicitly. "
                        "Provide structured, concise answers with endpoint names, HTTP methods, "
                        "required parameters, and key behaviours. Include source section names when helpful."
                    ),
                ),
                (
                    "user",
                    "Context:\n{context}\n\nQuestion: {question}",
                ),
            ]
        )
        self._context_chunks: List[ContextChunk] = self._load_context_chunks()
        logger.info(
            "ApiAgent initialised",
            extra={
                "context_path": str(self.context_path),
                "chunk_count": len(self._context_chunks),
            },
        )

    def _load_context_chunks(self) -> List[ContextChunk]:
        if not self.context_path.exists() and self.context_path == DEFAULT_CONTEXT_PATH:
            legacy_path = _LEGACY_CONTEXT_PATH.resolve()
            if legacy_path.exists():
                logger.info(
                    "ApiAgent falling back to legacy context path",
                    extra={"legacy_path": str(legacy_path)},
                )
                self.context_path = legacy_path

        if not self.context_path.exists():
            raise FileNotFoundError(
                f"API docs context file not found: {self.context_path}"
            )
        payload = yaml.safe_load(self.context_path.read_text(encoding="utf-8"))
        items = payload.get("contexts") or payload.get("sections") or []
        chunks: List[ContextChunk] = []
        for item in items:
            try:
                chunks.append(
                    ContextChunk(
                        identifier=item["id"],
                        title=item["title"],
                        source=item["source"],
                        section=item.get("section", ""),
                        content=item["content"],
                    )
                )
            except KeyError as exc:  # pragma: no cover - defensive
                logger.warning("Skipping malformed context entry", extra={"error": str(exc)})
        return chunks

    def invoke(
        self,
        question: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Answer a question using retrieved API documentation context."""

        if self._is_mutating_question(question):
            message = (
                "Safety policy: the API Docs agent cannot assist with update or delete "
                "operations or other mutating REST calls."
            )
            logger.warning("Blocked mutating API documentation question", extra={"question": question})
            trace = [
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="docs",  # type: ignore[assignment]
                    message="Blocked mutating REST operation question",
                    data={"question": question},
                )
            ]
            error = AgentError(message=message, type="GuardrailViolation")
            return AgentResult(
                agent="docs",
                status=AgentExecutionStatus.failed,
                error=error,
                trace=trace,
            )

        if not question or not question.strip():
            error = AgentError(message="Empty API question.", type="ValidationError")
            return AgentResult(
                agent="docs",
                status=AgentExecutionStatus.failed,
                error=error,
                trace=[],
            )

        query_tokens = _tokenise(question)
        scored_chunks: List[_ScoredChunk] = [
            _ScoredChunk(chunk=chunk, score=score, matches=matches)
            for chunk, score, matches in (
                self._score_chunk(chunk, query_tokens) for chunk in self._context_chunks
            )
        ]
        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        positive = [item for item in scored_chunks if item.score > 0]
        top_chunks = (positive or scored_chunks)[: self.top_k]

        if not top_chunks:
            logger.warning("No API documentation chunks available for retrieval")
            top_chunks = []

        retrieval_trace = self._build_retrieval_trace(top_chunks)
        context_payload = _format_context([item.chunk for item in top_chunks])

        prompt_value = self._prompt.format_prompt(context=context_payload, question=question)
        messages = prompt_value.to_messages()
        try:
            answer_message = self.llm.invoke(messages)
            answer = getattr(answer_message, "content", str(answer_message))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ApiAgent LLM invocation failed")
            error = AgentError(message=str(exc), type=type(exc).__name__)
            return AgentResult(
                agent="docs",
                status=AgentExecutionStatus.failed,
                error=error,
                trace=retrieval_trace,
            )

        return AgentResult(
            agent="docs",
            status=AgentExecutionStatus.succeeded,
            answer=str(answer),
            trace=retrieval_trace
            + [
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="docs",  # type: ignore[assignment]
                    message="Generated API documentation answer",
                    data={
                        "context_chunks_used": [item.chunk.identifier for item in top_chunks]
                    },
                )
            ],
        )

    def _score_chunk(
        self,
        chunk: ContextChunk,
        query_tokens: Sequence[str],
    ) -> Tuple[ContextChunk, float, Tuple[str, ...]]:
        if not query_tokens:
            return chunk, 0.0, ()
        text = " ".join([chunk.title, chunk.section, chunk.content])
        chunk_tokens = _tokenise(text)
        if not chunk_tokens:
            return chunk, 0.0, ()

        matches = tuple(sorted({token for token in query_tokens if token in chunk_tokens}))
        if not matches:
            return chunk, 0.0, ()

        overlap = sum(chunk_tokens.count(token) for token in matches)

        # simple TF overlap normalised by log length to dampen very long chunks
        score = overlap / math.log(len(chunk_tokens) + 10, 10)
        return chunk, score, matches

    def _build_retrieval_trace(self, chunks: Iterable[_ScoredChunk]) -> List[TraceEvent]:
        events = []
        for chunk in chunks:
            events.append(
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="docs",  # type: ignore[assignment]
                    message="Retrieved API documentation chunk",
                    data={
                        "chunk_id": chunk.chunk.identifier,
                        "title": chunk.chunk.title,
                        "source": chunk.chunk.source,
                        "section": chunk.chunk.section,
                        "score": chunk.score,
                        "matches": list(chunk.matches),
                    },
                )
            )
        return events

    def _is_mutating_question(self, question: str) -> bool:
        tokens = set(_tokenise(question))
        if tokens & _MUTATION_KEYWORDS:
            return True
        upper_question = question.upper()
        return any(verb in upper_question for verb in _MUTATION_HTTP_VERBS)


def _tokenise(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def _format_context(chunks: Iterable[ContextChunk]) -> str:
    formatted_sections = []
    for chunk in chunks:
        header = f"{chunk.title} ({chunk.section}) — {chunk.source}"
        formatted_sections.append(f"{header}\n{chunk.content}".strip())
    return "\n\n---\n\n".join(formatted_sections)


def compile_docs_agent(**kwargs: Any) -> ApiAgent:
    """Factory aligning with orchestrator expectations."""
    return ApiAgent(**kwargs)


