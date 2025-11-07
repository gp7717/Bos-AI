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


DEFAULT_LOG_LEVEL = "INFO"


def _configure_logging(log_level: str | None = None) -> logging.Logger:
    level_name = (log_level or DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured", extra={"level": level_name})
    return logger


logger = _configure_logging()


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

    logger.info("Starting CLI execution", extra={"stream": stream})
    try:
        graph = compile_sql_agent()
    except ConfigurationError as exc:  # pragma: no cover - command line convenience
        logger.exception("Configuration error during graph compilation", extra={"question": question})
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error compiling SQL agent graph", extra={"question": question})
        print(f"Unexpected error while preparing the agent: {exc}", file=sys.stderr)
        return 3

    state: SQLAgentState = {"messages": [HumanMessage(content=question)]}

    try:
        if stream:
            logger.debug("Streaming mode enabled; beginning graph execution")
            for idx, step in enumerate(graph.stream(state, stream_mode="values")):
                logger.debug("Stream step completed", extra={"step_index": idx, "keys": list(step.keys())})
                if "messages" in step and step["messages"]:
                    _print_messages(step["messages"][-1:])
        else:
            logger.debug("Invoke mode enabled; executing graph")
            final_state = graph.invoke(state)
            logger.debug(
                "Graph invocation complete",
                extra={
                    "state_keys": list(final_state.keys()),
                    "message_count": len(final_state.get("messages", [])),
                },
            )
            _print_messages(final_state.get("messages", []))
            final_answer = final_state.get("final_answer")
            if final_answer:
                print("\nFinal answer:\n" + final_answer)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Agent execution failed", extra={"question": question, "stream": stream})
        print(f"Agent execution failed: {exc}", file=sys.stderr)
        return 4

    logger.info("CLI execution completed successfully")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the custom SQL LangGraph agent")
    parser.add_argument("question", help="Natural language question to answer with SQL")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate steps instead of only printing the final answer",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the default log level (e.g. DEBUG, INFO, WARNING)",
    )

    args = parser.parse_args(argv)
    global logger
    if args.log_level:
        logger = _configure_logging(args.log_level)
    logger.debug("Parsed CLI arguments", extra={"args": vars(args)})
    try:
        return run_cli(question=args.question, stream=args.stream)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Fatal error running CLI")
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


