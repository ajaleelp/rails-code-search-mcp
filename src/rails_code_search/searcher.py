"""Vector search functionality for code embeddings."""

from pathlib import Path
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

    def list_features(self) -> List[Dict]:
        """List all unique features in the database.

        Returns:
            List of features with their code chunk counts
        """
        sql = f"""
            SELECT feature, COUNT(*) as count
            FROM {self.table_name}
            GROUP BY feature
            ORDER BY feature
        """

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()

        return [{"feature": row.feature, "count": row.count} for row in rows]

    def search_by_class(
        self,
        class_name: str,
        feature: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Find all methods in a specific class.

        Args:
            class_name: Name of the class to search for
            feature: Optional feature filter
            limit: Maximum number of results

        Returns:
            List of code snippets from the specified class
        """
        filters = ["class_name = :class_name"]
        params = {"class_name": class_name, "limit": limit}

        if feature:
            filters.append("feature = :feature")
            params["feature"] = feature

        where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk
            FROM {self.table_name}
            {where_clause}
            ORDER BY path, start_line
            LIMIT :limit
        """

        with self.engine.connect() as conn:
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
            })

        return results

    def search_by_feature(
        self,
        feature: str,
        limit: int = 100
    ) -> List[Dict]:
        """List all code in a specific feature.

        Args:
            feature: Feature tag to search for
            limit: Maximum number of results

        Returns:
            List of code snippets in the feature
        """
        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk
            FROM {self.table_name}
            WHERE feature = :feature
            ORDER BY path, start_line
            LIMIT :limit
        """

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"feature": feature, "limit": limit}).fetchall()

        results = []
        for row in rows:
            results.append({
                "feature": row.feature,
                "path": row.path,
                "start_line": row.start_line,
                "class_name": row.class_name,
                "method_name": row.method_name,
                "snippet": row.chunk,
            })

        return results

    def get_file_chunks(
        self,
        file_path: str,
        feature: Optional[str] = None
    ) -> List[Dict]:
        """Get all code chunks from a specific file.

        Args:
            file_path: Path to the file
            feature: Optional feature filter

        Returns:
            List of code chunks from the file, ordered by line number
        """
        filters = ["path = :path"]
        params = {"path": file_path}

        if feature:
            filters.append("feature = :feature")
            params["feature"] = feature

        where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk
            FROM {self.table_name}
            {where_clause}
            ORDER BY start_line
        """

        with self.engine.connect() as conn:
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
            })

        return results

    def get_surrounding_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 5,
        feature: Optional[str] = None
    ) -> List[Dict]:
        """Get code chunks around a specific line number in a file.

        Args:
            file_path: Path to the file
            line_number: Target line number
            context_lines: Number of chunks to retrieve before/after
            feature: Optional feature filter

        Returns:
            List of code chunks around the target line
        """
        filters = ["path = :path"]
        params = {"path": file_path, "line_number": line_number, "limit": context_lines * 2 + 1}

        if feature:
            filters.append("feature = :feature")
            params["feature"] = feature

        where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk,
                   ABS(start_line - :line_number) as distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance, start_line
            LIMIT :limit
        """

        with self.engine.connect() as conn:
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
                "distance_from_target": row.distance,
            })

        return results

    def search_related_code(
        self,
        class_or_method: str,
        top_k: int = 10,
        feature: Optional[str] = None
    ) -> List[Dict]:
        """Find code that references a specific class or method name.

        Args:
            class_or_method: Class or method name to search for
            top_k: Number of results to return
            feature: Optional feature filter

        Returns:
            List of code chunks that mention the class/method
        """
        filters = ["chunk ILIKE :search_term"]
        params = {"search_term": f"%{class_or_method}%", "limit": top_k}

        if feature:
            filters.append("feature = :feature")
            params["feature"] = feature

        where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT feature, path, start_line, class_name, method_name, chunk
            FROM {self.table_name}
            {where_clause}
            ORDER BY start_line
            LIMIT :limit
        """

        with self.engine.connect() as conn:
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
            })

        return results
