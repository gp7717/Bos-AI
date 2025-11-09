from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from Agents.QueryAgent.config import get_resources
from Agents.core.models import (
    AgentExecutionStatus,
    AgentResult,
    TraceEvent,
    TraceEventType,
)


class _ValidationVerdict(BaseModel):
    verdict: str = Field(pattern="^(pass|fail)$")
    reason: str


_FAIL_TEXT = {
    "empty_answer": "Answer validation failed: no answer was produced.",
    "missing_numeric": "Answer validation failed: numeric result expected but not found in answer.",
}


def _serialise_agent_results(results: List[AgentResult]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for result in results:
        entry: Dict[str, Any] = {
            "agent": result.agent,
            "status": result.status.value,
        }
        if result.answer:
            entry["answer"] = result.answer[:500]
        if result.tabular:
            entry["row_count"] = result.tabular.row_count
        if result.error:
            entry["error"] = {
                "message": result.error.message,
                "type": result.error.type,
            }
        payload.append(entry)
    return payload


@dataclass
class AnswerValidationAgent:
    """Validates that the composed answer satisfies the user's question."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._parser = PydanticOutputParser(pydantic_object=_ValidationVerdict)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an exacting auditor. "
                        "Determine whether the final assistant answer satisfies the user's question. "
                        "Base your judgement only on the provided data and reasoning steps. "
                        "If the answer is incomplete, contradictory, evasive, or unsupported by the provided evidence, mark it as fail. "
                        "Respond using the specified JSON schema."
                    ),
                ),
                (
                    "user",
                    "Question:\n{question}\n\n"
                    "Final answer:\n{answer}\n\n"
                    "Supporting agent outputs:\n{agent_outputs}\n\n"
                    "Format instructions:\n{format_instructions}",
                ),
            ]
        )

    def invoke(
        self,
        *,
        question: str,
        final_answer: Optional[str],
        agent_results: List[AgentResult],
    ) -> AgentResult:
        heuristics_reason = self._heuristic_failure(question, final_answer, agent_results)
        trace: List[TraceEvent] = []
        if heuristics_reason:
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="validator",  # type: ignore[assignment]
                    message=heuristics_reason,
                )
            )
            return AgentResult(
                agent="validator",
                status=AgentExecutionStatus.failed,
                answer=heuristics_reason,
                trace=trace,
            )

        prompt_value = self._prompt.format_prompt(
            question=question.strip(),
            answer=(final_answer or "").strip(),
            agent_outputs=json.dumps(_serialise_agent_results(agent_results), ensure_ascii=False, indent=2),
            format_instructions=self._parser.get_format_instructions(),
        )
        messages = prompt_value.to_messages()
        llm_response = self.llm.invoke(messages)
        raw_content = getattr(llm_response, "content", llm_response)
        try:
            verdict = self._parser.parse(raw_content)  # type: ignore[arg-type]
            satisfied = verdict.verdict.lower() == "pass"
            status = AgentExecutionStatus.succeeded if satisfied else AgentExecutionStatus.failed
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.MESSAGE,
                    agent="validator",  # type: ignore[assignment]
                    message="Validation completed",
                    data={"verdict": verdict.verdict, "reason": verdict.reason},
                )
            )
            return AgentResult(
                agent="validator",
                status=status,
                answer=verdict.reason,
                trace=trace,
            )
        except Exception as exc:  # pragma: no cover - defensive
            trace.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    agent="validator",  # type: ignore[assignment]
                    message="Validation LLM parsing failed",
                    data={"error": str(exc), "raw_content": raw_content},
                )
            )
            return AgentResult(
                agent="validator",
                status=AgentExecutionStatus.failed,
                error=None,
                answer="Validation failed: unable to parse LLM judgement.",
                trace=trace,
            )

    @staticmethod
    def _heuristic_failure(
        question: str,
        final_answer: Optional[str],
        agent_results: List[AgentResult],
    ) -> Optional[str]:
        answer = (final_answer or "").strip()
        if not answer:
            return _FAIL_TEXT["empty_answer"]

        lower_question = question.lower()
        if any(phrase in lower_question for phrase in ("how many", "count", "number of")):
            if not re.search(r"\d", answer):
                return _FAIL_TEXT["missing_numeric"]

        # If every agent result is failed, flag automatically.
        if agent_results and all(result.status != AgentExecutionStatus.succeeded for result in agent_results):
            return "Validation failed: all upstream agents failed; final answer lacks supporting evidence."

        return None


def compile_validation_agent(**kwargs: Any) -> AnswerValidationAgent:
    return AnswerValidationAgent(**kwargs)


