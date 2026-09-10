"""Runtime configuration.

Everything is environment-driven so the same image runs locally and on Azure
App Service, where these arrive as application settings. No secret is ever
written to the repository or returned by the API.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ImplicitLab"
    version: str = "1.0.0"
    environment: str = "development"

    # --- Azure OpenAI -----------------------------------------------------
    # When these are absent the insight layer falls back to a deterministic
    # template engine. The API surface is identical either way, and the
    # response says which engine produced the text.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-5-mini"
    azure_openai_api_version: str = "2025-01-01-preview"
    llm_timeout_seconds: float = 45.0
    llm_max_completion_tokens: int = 3000
    #: Reasoning depth for models that accept it. The statistics are already
    #: computed; this call is exposition, not analysis.
    llm_reasoning_effort: str = "low"

    # --- Storage ----------------------------------------------------------
    # SQLite is sufficient and honest for a demo: one file, no server, and the
    # whole dataset is portable. A real deployment would swap this for Postgres,
    # which is a one-file change because everything goes through storage.py.
    database_path: str = "data/implicitlab.db"

    # --- Behaviour --------------------------------------------------------
    bootstrap_resamples: int = 2000
    retain_sessions: bool = True

    @property
    def llm_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
