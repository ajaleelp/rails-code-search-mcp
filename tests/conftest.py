"""Pytest configuration and fixtures."""

import pytest
from sqlalchemy import create_engine

from rails_code_search.config import SearchConfig


@pytest.fixture
def test_config():
    """Test configuration."""
    return SearchConfig(
        database_url="sqlite:///:memory:",
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def test_engine(test_config):
    """Test database engine."""
    return create_engine(test_config.database_url)
