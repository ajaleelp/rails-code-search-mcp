"""Vector search functionality for code embeddings."""

from typing import Dict, List, Optional

from pgvector.psycopg2 import register_vector
from sqlalchemy import text
from sqlalchemy.engine import Engine


class CodeSearcher:
    """Search code embeddings using vector similarity."""

    def __init__(self, engine: Engine, table_name: str = "embeddings"):
        """Initialize searcher with database engine.

        Args:
            engine: SQLAlchemy engine connected to PostgreSQL with pgvector
            table_name: Name of the embeddings table
        """
        self.engine = engine
        self.table_name = table_name

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        feature: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> List[Dict]:
        """Search for similar code snippets.

        Args:
            query_embedding: Vector embedding of the search query
            top_k: Number of results to return
            feature: Optional feature filter
            class_name: Optional class name filter

        Returns:
            List of matching code snippets with metadata
        """
        filters = []
        params = {
            "query_embedding": str(query_embedding),
            "top_k": top_k,
        }

        if feature:
            filters.append("feature = :feature")
            params["feature"] = feature

        if class_name:
            filters.append("class_name = :class_name")
            params["class_name"] = class_name

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk,
                   embedding <-> :query_embedding AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <-> :query_embedding
            LIMIT :top_k
        """

        with self.engine.connect() as conn:
            raw_conn = conn.connection.driver_connection
            register_vector(raw_conn)
            rows = conn.execute(text(sql), params).fetchall()

        results = []
        for row in rows:
            results.append({
                "feature": row.feature,
                "path": row.path,
                "start_line": row.start_line,
                "class_name": row.class_name,
                "method_name": row.method_name,
                "snippet": row.chunk,
                "distance": float(row.distance),
            })

        return results
