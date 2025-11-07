"""Command line entry point for the standalone SQL agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage

from .config import ConfigurationError
from .sql_agent import compile_sql_agent
from .state import SQLAgentState


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalise_message(message: BaseMessage) -> str:
    if hasattr(message, "content") and isinstance(message.content, str):
        return message.content
    if hasattr(message, "content"):
        try:
            return json.dumps(message.content, indent=2, default=str)
        except TypeError:
            return str(message.content)
    return str(message)


def _print_messages(messages: Iterable[BaseMessage]) -> None:
    for message in messages:
        role = getattr(message, "type", getattr(message, "role", "message"))
        print(f"[{role.upper()}] {_normalise_message(message)}")


def run_cli(question: str, stream: bool = False) -> int:
    """Execute the agent for *question* and print the result."""

    try:
        graph = compile_sql_agent()
    except ConfigurationError as exc:  # pragma: no cover - command line convenience
        logger.exception("Configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    state: SQLAgentState = {"messages": [HumanMessage(content=question)]}

    if stream:
        for step in graph.stream(state, stream_mode="values"):
            if "messages" in step and step["messages"]:
                _print_messages(step["messages"][-1:])
    else:
        final_state = graph.invoke(state)
        _print_messages(final_state.get("messages", []))
        final_answer = final_state.get("final_answer")
        if final_answer:
            print("\nFinal answer:\n" + final_answer)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the custom SQL LangGraph agent")
    parser.add_argument("question", help="Natural language question to answer with SQL")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate steps instead of only printing the final answer",
    )

    args = parser.parse_args(argv)
    return run_cli(question=args.question, stream=args.stream)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


