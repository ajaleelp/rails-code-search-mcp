DROP TABLE IF EXISTS embeddings;

CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  feature TEXT NOT NULL,
  path TEXT NOT NULL,
  start_line INT,
  class_name TEXT,
  method_name TEXT,
  chunk TEXT NOT NULL,
  embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_feature ON embeddings (feature);
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
