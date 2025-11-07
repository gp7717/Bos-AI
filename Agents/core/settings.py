"""Environment-driven runtime settings for the Agentic multi-agent system."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_PATHS = (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
)


for path in _ENV_PATHS:
    if path.exists():
        load_dotenv(path, override=False)


class AzureOpenAISettings(BaseModel):
    endpoint: AnyHttpUrl
    deployment: str
    api_key: str
    api_version: str = Field(default="2024-02-15-preview")


class AgentRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_", case_sensitive=False)

    azure_openai: Optional[AzureOpenAISettings] = None
    database_url: Optional[str] = None
    allowed_tables: tuple[str, ...] = Field(default=())
    default_schema: str = Field(default="public")
    sql_dialect: str = Field(default="PostgreSQL")
    planner_model: str = Field(default="gpt-4o-mini")
    composer_model: str = Field(default="gpt-4o-mini")
    computation_sandbox_root: Path = Field(default_factory=lambda: Path.cwd() / "tmp" / "agentic")
    enable_tracing: bool = False

    @property
    def has_sql_support(self) -> bool:
        return self.database_url is not None


@functools.lru_cache(maxsize=1)
def load_settings() -> AgentRuntimeSettings:
    """Load and cache runtime settings for orchestrator components."""

    return AgentRuntimeSettings()  # type: ignore[call-arg]


__all__ = ["AgentRuntimeSettings", "AzureOpenAISettings", "load_settings"]


