#!/usr/bin/env python3
"""Query stored embeddings to retrieve the most relevant code snippets."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import Vector, register_vector
from sqlalchemy import create_engine

from app.services.vector_store import search_embeddings


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Search embeddings by text query")
  parser.add_argument("--query", required=True, help="Natural language query")
  parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
  parser.add_argument("--feature", default=None, help="Filter by feature tag if provided")
  parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
  return parser.parse_args()


def main():
  load_dotenv()
  args = parse_args()

  db_url = os.environ.get("DATABASE_URL")
  if not db_url:
    raise SystemExit("DATABASE_URL not set")
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise SystemExit("OPENAI_API_KEY not set")

  engine = create_engine(db_url, future=True)
  client = OpenAI(api_key=api_key)

  # Embed the query text
  response = client.embeddings.create(model=args.model, input=[args.query])
  query_embedding = response.data[0].embedding

  # Prepare SQL for similarity search
  with engine.connect() as conn:
    raw_conn = conn.connection.driver_connection
    register_vector(raw_conn)

  rows = search_embeddings(
    engine=engine,
    query_embedding=query_embedding,
    top_k=args.top_k,
    feature=args.feature,
  )

  if not rows:
    print("No matches found.")
    return

  for idx, row in enumerate(rows, start=1):
    snippet_preview = row["snippet"].splitlines()
    preview = "\n".join(snippet_preview[:6])
    print(
      f"{idx}. distance={row['distance']:.4f} feature={row['feature']} "
      f"class={row['class_name']} method={row['method_name']} path={row['path']}:{row['start_line']}"
    )
    print(preview)
    print("-" * 60)


if __name__ == "__main__":
  main()
