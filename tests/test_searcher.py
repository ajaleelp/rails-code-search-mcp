"""Test code searcher."""

import pytest
from rails_code_search.searcher import CodeSearcher


def test_searcher_init(test_engine):
    """Test searcher initialization."""
    searcher = CodeSearcher(test_engine)
    assert searcher.engine == test_engine
    assert searcher.table_name == "embeddings"


def test_searcher_custom_table(test_engine):
    """Test searcher with custom table name."""
    searcher = CodeSearcher(test_engine, table_name="custom_embeddings")
    assert searcher.table_name == "custom_embeddings"
