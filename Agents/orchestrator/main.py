"""Command-line entry point for launching the orchestrator server."""

from __future__ import annotations

import argparse
from typing import Sequence

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    """Construct the shared CLI parser for orchestrator launch commands."""
    parser = argparse.ArgumentParser(description="Run the Agentic orchestrator FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (for development only)",
    )
    return parser


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally from a provided sequence for reuse."""
    return build_parser().parse_args(args=args)


def main(args: Sequence[str] | None = None) -> None:
    """Entry point for running the orchestrator via `python -m`."""
    parsed = parse_args(args)
    uvicorn.run(
        "Agents.orchestrator.server:app",
        host=parsed.host,
        port=parsed.port,
        reload=parsed.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


