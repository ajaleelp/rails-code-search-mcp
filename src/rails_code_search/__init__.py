"""Rails Code Search MCP Server.

Semantic code search for Rails applications using OpenAI embeddings.
"""

__version__ = "0.1.0"

from .config import SearchConfig
from .searcher import CodeSearcher

__all__ = ["SearchConfig", "CodeSearcher", "__version__"]
