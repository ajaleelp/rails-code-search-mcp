"""FastAPI server for code search."""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from .config import get_config
from .searcher import CodeSearcher


class SearchRequest(BaseModel):
    """Code search request."""

    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    feature: Optional[str] = Field(None, description="Optional feature filter")
    class_name: Optional[str] = Field(None, description="Optional class name filter")
    model: str = Field(default="text-embedding-3-small", description="Embedding model")


class SearchResult(BaseModel):
    """Single search result."""

    feature: str
    path: str
    start_line: Optional[int]
    class_name: Optional[str]
    method_name: Optional[str]
    snippet: str
    distance: float


class SearchResponse(BaseModel):
    """Code search response."""

    results: List[SearchResult]


def create_app(config=None) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: Optional SearchConfig instance. If None, loads from environment.

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = get_config()

    app = FastAPI(
        title="Rails Code Search API",
        description="Semantic code search for Rails applications",
        version="0.1.0"
    )

    # Initialize components
    engine = create_engine(config.database_url, future=True)
    openai_client = OpenAI(api_key=config.openai_api_key)
    searcher = CodeSearcher(engine, config.table_name)

    @app.on_event("startup")
    def register_vector_extension():
        """Register pgvector extension on startup."""
        with engine.connect() as conn:
            raw_conn = conn.connection.driver_connection
            register_vector(raw_conn)

    @app.get("/healthz")
    def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/search", response_model=SearchResponse)
    def search_code(payload: SearchRequest):
        """Search code by natural language query.

        Embeds the query and searches for similar code snippets using vector similarity.
        """
        try:
            # Generate embedding for query
            embedding_resp = openai_client.embeddings.create(
                model=payload.model,
                input=[payload.query]
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        # Search for similar code
        query_embedding = embedding_resp.data[0].embedding
        results = searcher.search(
            query_embedding=query_embedding,
            top_k=payload.top_k,
            feature=payload.feature,
            class_name=payload.class_name,
        )

        return {"results": results}

    return app


# Default app instance for uvicorn
app = create_app()
