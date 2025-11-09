"""Top-level entry point for running the Agentic orchestrator FastAPI server."""

from __future__ import annotations

from typing import Sequence

import uvicorn

from Agents.orchestrator.main import parse_args
from Agents.orchestrator.server import create_app


app = create_app()


def main(args: Sequence[str] | None = None) -> None:
    """Launch the orchestrator server, reusing the shared CLI parser."""
    parsed = parse_args(args)
    target = "main:app" if parsed.reload else app
    uvicorn.run(
        target,
        host=parsed.host,
        port=parsed.port,
        reload=parsed.reload,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


