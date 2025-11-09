"""Public exports for the API documentation agent."""

from .agent import ApiAgent, compile_docs_agent

ApiDocsAgent = ApiAgent
compile_api_docs_agent = compile_docs_agent

__all__ = [
    "ApiAgent",
    "compile_docs_agent",
    "ApiDocsAgent",
    "compile_api_docs_agent",
]
