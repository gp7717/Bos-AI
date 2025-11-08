"""
Retrieval-backed agent for answering questions about REST API endpoints.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from Agents.QueryAgent.config import get_resources
from Agents.core.models import (
    AgentError,
    AgentExecutionStatus,
    AgentResult,
    TraceEvent,
    TraceEventType,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONTEXT_PATH = Path("Docs") / "api_docs_context.yaml"


@dataclass(frozen=True)
class ApiDocSection:
    """Lightweight representation of a context chunk."""

    id: str
    title: str
    source: str
    content: str
    tokens: List[str]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @classmethod
    def from_mapping(cls, mapping: dict) -> "ApiDocSection":
        content = mapping.get("content", "")
        title = mapping.get("title", "")
        tokens = cls._tokenize(f"{title}\n{content}")
        return cls(
            id=mapping.get("id", ""),
            title=title,
            source=mapping.get("source", ""),
            content=content,
            tokens=tokens,
        )


class ApiDocsAgent:
    """Agent that answers questions about documented APIs using retrieval + LLM."""

    def __init__(
        self,
        *,
        llm: Optional[AzureChatOpenAI] = None,
        context_path: Path | str | None = None,
        top_k: int = 3,
    ) -> None:
        resources = get_resources()
        self.llm: AzureChatOpenAI = llm or resources.llm
        self._context_path = Path(context_path) if context_path else DEFAULT_CONTEXT_PATH
        self._top_k = top_k
        self._sections: List[ApiDocSection] = self._load_sections(self._context_path)

        system_prompt = (
            "You are an assistant that answers questions about the Bos-AI REST API. "
            "Use ONLY the provided context snippets. If the answer cannot be found, "
            "reply that the documentation does not cover the request. "
            "Always cite endpoint paths and HTTP verbs when applicable."
        )
        user_prompt = (
            "User question:\n{question}\n\n"
            "Relevant documentation snippets:\n{context}\n\n"
            "Provide a concise, actionable answer. "
            "If the question references multiple endpoints, list them separately."
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

    @staticmethod
    def _load_sections(location: Path) -> List[ApiDocSection]:
        if not location.exists():
            LOGGER.warning("API docs context file missing: %s", location)
            return []
        with location.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        raw_sections = payload.get("sections", [])
        sections = [ApiDocSection.from_mapping(section) for section in raw_sections]
        LOGGER.info("Loaded %d API documentation sections", len(sections))
        return sections

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _rank_sections(self, question: str) -> List[ApiDocSection]:
        question_tokens = self._tokenize(question)
        if not question_tokens or not self._sections:
            return []

        token_counts = {}
        for token in question_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        scored_sections: List[tuple[float, ApiDocSection]] = []
        for section in self._sections:
            if not section.tokens:
                continue
            score = sum(token_counts.get(token, 0) for token in section.tokens)
            if score > 0:
                scored_sections.append((score, section))

        scored_sections.sort(key=lambda item: item[0], reverse=True)
        return [section for _, section in scored_sections[: self._top_k]]

    def _format_context(self, sections: Iterable[ApiDocSection]) -> str:
        lines: List[str] = []
        for section in sections:
            entry = {
                "title": section.title,
                "source": section.source,
                "content": section.content.strip(),
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
        return "\n".join(lines)

    def invoke(self, question: str, *, context: Optional[dict] = None) -> AgentResult:
        selected_sections = self._rank_sections(question)

        if not selected_sections and self._sections:
            # Fall back to first section to avoid empty context; still explain uncertainty.
            selected_sections = self._sections[:1]

        if not selected_sections:
            error_message = (
                "API documentation context is unavailable; unable to answer the question."
            )
            LOGGER.error(error_message)
            return AgentResult(
                agent="api_docs",  # type: ignore[assignment]
                status=AgentExecutionStatus.failed,
                error=AgentError(message=error_message, type="MissingContext"),
                trace=[
                    TraceEvent(
                        event_type=TraceEventType.ERROR,
                        agent="api_docs",  # type: ignore[assignment]
                        message=error_message,
                    )
                ],
            )

        context_payload = self._format_context(selected_sections)
        llm_chain = self._prompt | self.llm | RunnableLambda(lambda msg: msg.content)
        answer = llm_chain.invoke({"question": question, "context": context_payload})

        trace_data = [
            {
                "id": section.id,
                "title": section.title,
                "source": section.source,
            }
            for section in selected_sections
        ]

        trace = [
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent="api_docs",  # type: ignore[assignment]
                message="Selected API documentation sections",
                data={
                    "matches": trace_data,
                    "question": question,
                },
            )
        ]

        messages = [
            TraceEvent(
                event_type=TraceEventType.MESSAGE,
                agent="api_docs",  # type: ignore[assignment]
                message="Generated answer from documentation",
            )
        ]
        trace.extend(messages)

        return AgentResult(
            agent="api_docs",  # type: ignore[assignment]
            status=AgentExecutionStatus.succeeded,
            answer=str(answer),
            trace=trace,
        )


def compile_api_docs_agent() -> ApiDocsAgent:
    """Factory to align with orchestrator patterns."""
    return ApiDocsAgent()
"""Retrieval-backed agent that answers questions about dashboard REST APIs."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

DEFAULT_CONTEXT_PATH = (
    Path(__file__).resolve().parent.parent / "Docs" / "api_docs_context.yaml"
)

TOP_K_CONTEXTS = 4


@dataclass(frozen=True)
class ContextChunk:
    """One retrievable chunk of API documentation."""

    identifier: str
    title: str
    source: str
    section: str
    content: str


class ApiDocsAgent:
    """Agent that answers API documentation questions via retrieval + LLM."""

    def __init__(
        self,
        *,
        context_path: Optional[Path] = None,
        llm: Optional[Any] = None,
    ) -> None:
        if llm is None:
            resources = get_resources()
            self.llm = resources.llm
        else:
            self.llm = llm
        self.context_path = context_path or DEFAULT_CONTEXT_PATH
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
            "ApiDocsAgent initialised",
            extra={
                "context_path": str(self.context_path),
                "chunk_count": len(self._context_chunks),
            },
        )

    def _load_context_chunks(self) -> List[ContextChunk]:
        if not self.context_path.exists():
            raise FileNotFoundError(
                f"API docs context file not found: {self.context_path}"
            )
        payload = yaml.safe_load(self.context_path.read_text(encoding="utf-8"))
        items = payload.get("contexts", [])
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

        if not question or not question.strip():
            error = AgentError(message="Empty API question.", type="ValidationError")
            return AgentResult(
                agent="api_docs",
                status=AgentExecutionStatus.failed,
                error=error,
                trace=[],
            )

        query_tokens = _tokenise(question)
        scored_chunks = [
            (self._score_chunk(chunk, query_tokens), chunk) for chunk in self._context_chunks
        ]
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:TOP_K_CONTEXTS] if score > 0]

        if not top_chunks:
            # fall back to the highest scored chunk even if score is zero
            top_chunks = [scored_chunks[0][1]] if scored_chunks else []

        retrieval_trace = self._build_retrieval_trace(top_chunks)
        context_payload = _format_context(top_chunks)

        llm_chain = self._prompt | self.llm | RunnableLambda(lambda msg: msg.content)
        try:
            answer = llm_chain.invoke({"context": context_payload, "question": question})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ApiDocsAgent LLM invocation failed")
            error = AgentError(message=str(exc), type=type(exc).__name__)
            return AgentResult(
                agent="api_docs",
                status=AgentExecutionStatus.failed,
                error=error,
                trace=retrieval_trace,
            )

        return AgentResult(
            agent="api_docs",
            status=AgentExecutionStatus.succeeded,
            answer=str(answer),
            trace=retrieval_trace
            + [
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="api_docs",  # type: ignore[assignment]
                    message="Generated API documentation answer",
                    data={"context_chunks_used": [chunk.identifier for chunk in top_chunks]},
                )
            ],
        )

    def _score_chunk(
        self,
        chunk: ContextChunk,
        query_tokens: Sequence[str],
    ) -> float:
        if not query_tokens:
            return 0.0
        text = " ".join([chunk.title, chunk.section, chunk.content])
        chunk_tokens = _tokenise(text)
        if not chunk_tokens:
            return 0.0

        overlap = sum(chunk_tokens.count(token) for token in set(query_tokens))
        if overlap == 0:
            return 0.0

        # simple TF overlap normalised by log length to dampen very long chunks
        score = overlap / math.log(len(chunk_tokens) + 10, 10)
        return score

    def _build_retrieval_trace(self, chunks: Iterable[ContextChunk]) -> List[TraceEvent]:
        events = []
        for chunk in chunks:
            events.append(
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="api_docs",  # type: ignore[assignment]
                    message="Retrieved API documentation chunk",
                    data={
                        "chunk_id": chunk.identifier,
                        "title": chunk.title,
                        "source": chunk.source,
                        "section": chunk.section,
                    },
                )
            )
        return events


def _tokenise(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def _format_context(chunks: Iterable[ContextChunk]) -> str:
    formatted_sections = []
    for chunk in chunks:
        header = f"{chunk.title} ({chunk.section}) — {chunk.source}"
        formatted_sections.append(f"{header}\n{chunk.content}".strip())
    return "\n\n---\n\n".join(formatted_sections)


