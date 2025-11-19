"""Command-line interface for code search."""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine

from .config import get_config
from .searcher import CodeSearcher


@click.group()
@click.version_option()
def main():
    """Rails Code Search CLI."""
    pass


@main.command()
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--feature", "-f", help="Filter by feature")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--env-file", type=click.Path(exists=True), help="Path to .env file")
def search(query: str, feature: str, top_k: int, env_file: str):
    """Search code by natural language query."""
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    config = get_config()

    # Initialize components
    engine = create_engine(config.database_url)
    openai_client = OpenAI(api_key=config.openai_api_key)
    searcher = CodeSearcher(engine, config.table_name)

    # Generate embedding
    click.echo(f"Searching for: {query}")
    response = openai_client.embeddings.create(
        model=config.embedding_model,
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # Search
    results = searcher.search(
        query_embedding=query_embedding,
        top_k=top_k,
        feature=feature
    )

    if not results:
        click.echo("No results found.")
        return

    # Display results
    for idx, result in enumerate(results, 1):
        click.echo(f"\n{idx}. distance={result['distance']:.4f} feature={result['feature']}")
        click.echo(f"   {result['path']}:{result['start_line']}")
        if result['class_name']:
            click.echo(f"   {result['class_name']}#{result['method_name']}")
        snippet_lines = result['snippet'].split('\n')
        preview = '\n'.join(snippet_lines[:6])
        click.echo(f"\n{preview}")
        click.echo("-" * 60)


@main.command()
def serve():
    """Start the FastAPI server."""
    import uvicorn
    from .server import app

    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
