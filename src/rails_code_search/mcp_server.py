"""MCP server for Rails code search."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import anyio
import click
from mcp.server import NotificationOptions, Server as MCPServer
from mcp.server.stdio import stdio_server
import mcp.types as mcp_types
from openai import OpenAI
from sqlalchemy import create_engine

from . import __version__
from .config import get_config, SearchConfig
from .searcher import CodeSearcher


SEARCH_CODE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural language search query describing the code you want to find"
        },
        "top_k": {
            "type": "integer",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
            "description": "Number of results to return"
        },
        "feature": {
            "type": "string",
            "description": "Optional feature tag to filter results (e.g., 'auth', 'payments')"
        },
        "class_name": {
            "type": "string",
            "description": "Optional class name to filter results"
        },
    },
    "required": ["query"],
}


class CodeSearchMCPServer:
    """MCP server for semantic code search."""

    def __init__(self, config: SearchConfig):
        """Initialize MCP server with configuration.

        Args:
            config: SearchConfig instance with database and OpenAI settings
        """
        self.config = config
        self.engine = create_engine(config.database_url)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.searcher = CodeSearcher(self.engine, config.table_name)

    def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        return [
            {
                "name": "search_code",
                "description": "Search codebase using natural language queries. Returns semantically similar code snippets with file paths, line numbers, and class/method context.",
                "input_schema": SEARCH_CODE_INPUT_SCHEMA,
            }
        ]

    def search_code(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code search.

        Args:
            payload: Search parameters including query, top_k, feature, class_name

        Returns:
            Dictionary with search results

        Raises:
            ValueError: If query is missing or invalid
        """
        query = payload.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' must be a non-empty string")

        top_k = payload.get("top_k", self.config.default_top_k)
        if not isinstance(top_k, int) or top_k < 1 or top_k > self.config.max_results:
            raise ValueError(f"'top_k' must be between 1 and {self.config.max_results}")

        feature = payload.get("feature")
        class_name = payload.get("class_name")

        # Generate embedding for query
        try:
            embedding_resp = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=[query]
            )
            query_embedding = embedding_resp.data[0].embedding
        except Exception as exc:
            raise ValueError(f"Failed to generate embedding: {exc}") from exc

        # Search for similar code
        results = self.searcher.search(
            query_embedding=query_embedding,
            top_k=top_k,
            feature=feature,
            class_name=class_name,
        )

        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }


@click.group(context_settings={"auto_envvar_prefix": "CODE_SEARCH"})
@click.option("--database-url", envvar="CODE_SEARCH_DATABASE_URL", required=True, help="PostgreSQL database URL")
@click.option("--openai-api-key", envvar="CODE_SEARCH_OPENAI_API_KEY", required=True, help="OpenAI API key")
@click.option("--embedding-model", envvar="CODE_SEARCH_EMBEDDING_MODEL", default="text-embedding-3-small", help="OpenAI embedding model")
@click.option("--table-name", envvar="CODE_SEARCH_TABLE_NAME", default="embeddings", help="Embeddings table name")
@click.pass_context
def cli(ctx: click.Context, database_url: str, openai_api_key: str, embedding_model: str, table_name: str) -> None:
    """Initialize MCP server for code search."""
    config = SearchConfig(
        database_url=database_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        table_name=table_name,
    )
    ctx.ensure_object(dict)
    ctx.obj["server"] = CodeSearchMCPServer(config)


@cli.command("list-tools")
@click.pass_context
def list_tools(ctx: click.Context) -> None:
    """List available MCP tools."""
    server: CodeSearchMCPServer = ctx.obj["server"]
    click.echo(_json_dump(server.list_tools()))


@cli.command("search-code")
@click.option("--query", required=True, help="Natural language search query")
@click.option("--top-k", type=int, default=5, help="Number of results to return")
@click.option("--feature", help="Optional feature filter")
@click.option("--class-name", help="Optional class name filter")
@click.pass_context
def search_code(ctx: click.Context, query: str, top_k: int, feature: Optional[str], class_name: Optional[str]) -> None:
    """Search code using natural language query."""
    server: CodeSearchMCPServer = ctx.obj["server"]
    payload = {"query": query, "top_k": top_k}
    if feature:
        payload["feature"] = feature
    if class_name:
        payload["class_name"] = class_name

    try:
        result = server.search_code(payload)
        click.echo(_json_dump(result))
    except ValueError as exc:
        click.echo(_json_dump({"error": str(exc)}), err=True)
        raise SystemExit(1) from exc


@cli.command("serve")
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start MCP stdio server for integrations like Claude Desktop."""
    server: CodeSearchMCPServer = ctx.obj["server"]
    anyio.run(_run_mcp_server, server)


async def _run_mcp_server(app_server: CodeSearchMCPServer) -> None:
    """Run the MCP server with stdio transport."""
    mcp_server = MCPServer(name="rails-code-search", version=__version__)

    @mcp_server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="search_code",
                description="Search codebase using natural language queries. Returns semantically similar code snippets with file paths, line numbers, and class/method context.",
                inputSchema=SEARCH_CODE_INPUT_SCHEMA,
            )
        ]

    @mcp_server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[mcp_types.TextContent]:
        if name != "search_code":
            raise ValueError(f"Unknown tool '{name}'")

        payload = arguments or {}

        def search_wrapper() -> Dict[str, Any]:
            try:
                return app_server.search_code(payload)
            except ValueError as exc:
                raise ValueError(str(exc))

        result = await anyio.to_thread.run_sync(search_wrapper)
        result_json = _json_dump(result)

        return [
            mcp_types.TextContent(
                type="text",
                text=result_json,
            )
        ]

    init_options = mcp_server.create_initialization_options(NotificationOptions())
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, init_options)


def _json_dump(payload: Any) -> str:
    """Serialize payload to JSON string."""
    return json.dumps(payload, indent=2, default=str)


main = cli


if __name__ == "__main__":
    main()
