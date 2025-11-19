"""Configuration management for code search."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchConfig(BaseSettings):
    """Configuration for code search server."""

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/code_search_embeddings",
        description="PostgreSQL connection URL with pgvector extension"
    )
    table_name: str = Field(default="embeddings", description="Table name for embeddings")

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )

    # Search
    default_top_k: int = Field(default=5, ge=1, le=100, description="Default number of results")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum allowed results")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CODE_SEARCH_",
        extra="ignore"
    )


@lru_cache()
def get_config(env_file: Optional[Path] = None) -> SearchConfig:
    """Get cached configuration instance."""
    if env_file:
        return SearchConfig(_env_file=env_file)
    return SearchConfig()
