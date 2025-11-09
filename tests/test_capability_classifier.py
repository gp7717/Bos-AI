from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Agents.orchestrator.capability_classifier import CapabilityClassifier


@pytest.fixture(scope="module")
def classifier() -> CapabilityClassifier:
    return CapabilityClassifier.default()


def test_docs_question_scores_highest_for_docs(classifier: CapabilityClassifier) -> None:
    question = "How does the /api/net_profit endpoint calculate profit?"
    scores = classifier.score(question)
    best_agent = max(scores, key=scores.get)
    assert best_agent == "docs"


def test_sql_question_scores_highest_for_sql(classifier: CapabilityClassifier) -> None:
    question = "Write a query to list Shopify orders with revenue and COGS by month."
    scores = classifier.score(question)
    best_agent = max(scores, key=scores.get)
    assert best_agent == "sql"


def test_calculation_question_scores_highest_for_computation(classifier: CapabilityClassifier) -> None:
    question = "Calculate the growth ratio between 2024 revenue (Rs 12.4M) and 2023 revenue (Rs 9.8M)."
    scores = classifier.score(question)
    best_agent = max(scores, key=scores.get)
    assert best_agent == "computation"

