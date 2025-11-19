"""Test configuration management."""

from rails_code_search.config import SearchConfig


def test_config_defaults():
    """Test configuration with defaults."""
    config = SearchConfig(openai_api_key="test-key")
    assert config.openai_api_key == "test-key"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.default_top_k == 5
    assert config.max_results == 20


def test_config_custom():
    """Test configuration with custom values."""
    config = SearchConfig(
        openai_api_key="test-key",
        database_url="postgresql://test",
        default_top_k=10,
    )
    assert config.database_url == "postgresql://test"
    assert config.default_top_k == 10
