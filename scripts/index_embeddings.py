#!/usr/bin/env python3
"""Load chunked code snippets, embed them with OpenAI, and store in Postgres."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List

from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, text


def parse_args() -> argparse.Namespace:
  """CLI options for batch embedding."""
  parser = argparse.ArgumentParser(description="Embed chunked files into Postgres")
  parser.add_argument("--input", required=True, help="Path to JSONL chunk file")
  parser.add_argument("--feature", required=True, help="Logical feature tag for these chunks")
  parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
  parser.add_argument("--batch-size", type=int, default=32, help="Chunks per embedding request")
  parser.add_argument("--limit", type=int, default=None, help="Max chunks to process (for testing)")
  return parser.parse_args()


def iter_chunks(path: Path, limit: int | None = None) -> Iterator[Dict]:
  """Yield chunk objects from a JSONL file, respecting the optional limit."""
  count = 0
  with path.open("r", encoding="utf-8") as fh:
    for line in fh:
      if not line.strip():
        continue
      yield json.loads(line)
      count += 1
      if limit is not None and count >= limit:
        break


def insert_batch(engine, rows: List[Dict]) -> None:
  """Persist prepared rows into the embeddings table."""
  if not rows:
    return
  insert_sql = text(
    """
    INSERT INTO embeddings (feature, path, start_line, chunk, class_name, method_name, embedding)
    VALUES (:feature, :path, :start_line, :chunk, :class_name, :method_name, :embedding)
    """
  )
  with engine.begin() as conn:
    raw_conn = conn.connection.driver_connection
    register_vector(raw_conn)
    conn.execute(insert_sql, rows)


def process_batch(batch: List[Dict], client: OpenAI, engine, model: str) -> None:
  """Embed the current batch and write it to Postgres."""
  inputs = [row["chunk"] for row in batch]
  response = client.embeddings.create(model=model, input=inputs)

  prepared_rows = []
  for row, embedding in zip(batch, response.data):
    prepared_rows.append(
      {
        "feature": row["feature"],
        "path": row["path"],
        "start_line": row["start_line"],
        "chunk": row["chunk"],
        "class_name": row.get("class_name"),
        "method_name": row.get("method_name"),
        "embedding": embedding.embedding,
      }
    )

  insert_batch(engine, prepared_rows)


def main() -> None:
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

  input_path = Path(args.input)
  batch: List[Dict] = []

  for chunk in iter_chunks(input_path, args.limit):
    batch.append(
      {
        "feature": args.feature,
        "path": chunk["path"],
        "start_line": chunk.get("start_line"),
        "chunk": chunk["text"],
        "class_name": chunk.get("class_name"),
        "method_name": chunk.get("method_name"),
      }
    )
    if len(batch) >= args.batch_size:
      process_batch(batch, client, engine, args.model)
      batch = []

  if batch:
    process_batch(batch, client, engine, args.model)


if __name__ == "__main__":
  main()
