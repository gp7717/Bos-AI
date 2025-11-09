"""Composer agent that fuses individual agent outputs into a final answer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from Agents.core.models import AgentResult, OrchestratorResponse, TabularResult
from Agents.QueryAgent.config import get_resources


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

    def compose(
        self,
        *,
        question: str,
        planner_rationale: str,
        agent_results: Iterable[AgentResult],
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        results = list(agent_results)
        summaries = [self._summarise_result(result) for result in results]
        tabular = self._select_tabular(results)

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
            data=tabular,
            agent_results=results,
            metadata=metadata or {},
        )

    @staticmethod
    def _select_tabular(results: Iterable[AgentResult]) -> Optional[TabularResult]:
        for result in results:
            if result.tabular:
                return result.tabular
        return None

    def _summarise_result(self, result: AgentResult) -> str:
        parts: List[str] = [f"agent={result.agent}", f"status={result.status}"]

        tabular_summary = self._format_tabular_summary(result.tabular)
        if tabular_summary:
            parts.append(tabular_summary)

        if result.answer:
            parts.append(f"text_answer={result.answer}")

        if result.error:
            parts.append(f"error={result.error.message}")

        return " | ".join(parts)

    def _format_tabular_summary(self, tabular: Optional[TabularResult]) -> Optional[str]:
        if not tabular or not tabular.rows:
            return None

        columns = tabular.columns or list(tabular.rows[0].keys())
        if not columns:
            return None

        sample_rows = tabular.rows[: min(3, len(tabular.rows))]
        samples = []
        for index, row in enumerate(sample_rows, start=1):
            if not isinstance(row, dict):
                continue
            metrics = self._extract_relevant_metrics(row, columns)
            if metrics:
                samples.append(f"row{index}: " + ", ".join(metrics))
        if not samples:
            return None

        column_preview = ", ".join(columns[:6])
        if len(columns) > 6:
            column_preview += ", ..."

        rows_reported = tabular.row_count or len(tabular.rows)
        return (
            f"structured rows={rows_reported} columns=({column_preview}) "
            f"key_metrics={'; '.join(samples)}"
        )

    def _extract_relevant_metrics(self, row: Dict[str, Any], columns: List[str]) -> List[str]:
        key_columns = {
            "campaign": {"campaign_name", "name", "entity", "label"},
            "campaign_id": {"campaign_id", "id"},
            "orders": {
                "orders",
                "total_orders",
                "orders_count",
                "order_count",
                "onsite_purchases",
            },
            "revenue": {
                "revenue",
                "total_revenue",
                "gross_revenue",
                "value_onsite_web_purchase",
                "onsite_purchase_value",
            },
            "spend": {"spend", "total_spend", "ad_spend"},
            "roas": {"roas", "return_on_ad_spend"},
            "cpa": {"cpa", "cost_per_order", "cost_per_acquisition"},
            "conversion_rate": {"conversion_rate", "conv_rate", "cr"},
        }

        metrics: List[str] = []
        lower_column_map = {column.lower(): column for column in columns}

        def render_metric(label: str, aliases: set[str]) -> None:
            for alias in aliases:
                column = lower_column_map.get(alias)
                if column is None:
                    continue
                value = row.get(column)
                if value in (None, "", []):
                    continue
                metrics.append(f"{label}={value}")
                return

        for label, aliases in key_columns.items():
            render_metric(label, aliases)

        return metrics


__all__ = ["Composer"]


