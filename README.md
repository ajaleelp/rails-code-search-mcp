# Rails Code Search MCP Server

Semantic code search for Rails applications using OpenAI embeddings and PostgreSQL pgvector.

## Features

- 🔍 **Semantic search** - Find code using natural language queries
- 🎯 **Ruby-aware chunking** - Preserves method and class context
- 📦 **Vector storage** - PostgreSQL with pgvector for fast similarity search
- 🚀 **FastAPI server** - RESTful API for code search
- 🔧 **MCP integration** - Ready for Claude Desktop integration
- 📝 **CLI tools** - Command-line interface for all operations

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 15+ with pgvector extension
- OpenAI API key
- Ruby (for parsing Ruby files)

## Installation

### 1. Install from source

```bash
git clone https://github.com/jaleel/rails-code-search-mcp.git
cd rails-code-search-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e '.[dev]'
```

### 2. Set up PostgreSQL

```bash
# Create database
createdb code_search_embeddings

# Enable pgvector extension
psql code_search_embeddings -c "CREATE EXTENSION vector;"

# Create schema
psql code_search_embeddings < scripts/schema.sql
```

### 3. Configure environment

Create a `.env` file:

```env
CODE_SEARCH_DATABASE_URL=postgresql://localhost:5432/code_search_embeddings
CODE_SEARCH_OPENAI_API_KEY=sk-...
```

## Usage

### 1. Chunk your codebase

```bash
# Chunk a Rails app directory
python scripts/chunk_codebase.py \
  --root /path/to/rails/app \
  --output chunks.jsonl \
  --max-lines 80
```

### 2. Index embeddings

```bash
# Generate embeddings and store in database
python scripts/index_embeddings.py \
  --input chunks.jsonl \
  --feature your_feature_name \
  --batch-size 32
```

### 3. Search code

**Via CLI:**

```bash
# Search all code
rails-code-search search --query "payment processing logic"

# Search specific feature
rails-code-search search \
  --query "stripe webhook handler" \
  --feature payments \
  --top-k 10
```

**Via API:**

```bash
# Start the server
rails-code-search serve

# Or use uvicorn directly
uvicorn rails_code_search.server:app --reload
```

Then query:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user authentication flow",
    "feature": "auth",
    "top_k": 5
  }'
```

**Response:**

```json
{
  "results": [
    {
      "feature": "auth",
      "path": "app/services/auth/session_service.rb",
      "start_line": 42,
      "class_name": "Auth::SessionService",
      "method_name": "authenticate_user",
      "snippet": "def authenticate_user...",
      "distance": 0.234
    }
  ]
}
```

## API Endpoints

### `GET /healthz`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### `POST /search`

Search code by natural language query.

**Request:**
```json
{
  "query": "string (required)",
  "top_k": 5,
  "feature": "optional_feature_name",
  "class_name": "optional_class_name",
  "model": "text-embedding-3-small"
}
```

**Response:**
```json
{
  "results": [
    {
      "feature": "string",
      "path": "string",
      "start_line": 42,
      "class_name": "string",
      "method_name": "string",
      "snippet": "string",
      "distance": 0.234
    }
  ]
}
```

## Configuration

Configuration can be provided via:
1. Environment variables (prefixed with `CODE_SEARCH_`)
2. `.env` file
3. `search_config.yml` file

### Environment Variables

- `CODE_SEARCH_DATABASE_URL` - PostgreSQL connection string
- `CODE_SEARCH_OPENAI_API_KEY` - OpenAI API key
- `CODE_SEARCH_EMBEDDING_MODEL` - Embedding model (default: text-embedding-3-small)
- `CODE_SEARCH_DEFAULT_TOP_K` - Default number of results (default: 5)
- `CODE_SEARCH_MAX_RESULTS` - Maximum allowed results (default: 20)

See `config/search_config.example.yml` for all options.

## How It Works

1. **Chunking**: Uses Ruby's Ripper parser to extract methods with full class context
2. **Embedding**: Generates vector embeddings using OpenAI's API
3. **Storage**: Stores embeddings in PostgreSQL with pgvector extension
4. **Search**: Finds similar code using cosine similarity

### Chunking Strategy

- **Ruby files**: Method-level chunks with class context preserved
- **YAML files**: Line-based chunking (configurable max lines)
- **Other files**: Line-based chunking with overlap

Each chunk includes:
- File path and line numbers
- Class and method names (for Ruby)
- Code snippet with context
- Feature tag for filtering

## Development

### Run tests

```bash
pytest
```

### Format code

```bash
black src/ tests/
ruff check src/ tests/
```

### Type checking

```bash
mypy src/
```

## Examples

### Index a Rails app

```bash
# Chunk different parts of the app
python scripts/chunk_codebase.py --root app/services --output services.jsonl
python scripts/chunk_codebase.py --root app/models --output models.jsonl
python scripts/chunk_codebase.py --root app/controllers --output controllers.jsonl

# Index each part with feature tags
python scripts/index_embeddings.py --input services.jsonl --feature services
python scripts/index_embeddings.py --input models.jsonl --feature models
python scripts/index_embeddings.py --input controllers.jsonl --feature controllers
```

### Search examples

```bash
# Find authentication code
rails-code-search search -q "user login authentication"

# Find payment processing
rails-code-search search -q "stripe payment webhooks" -f payments

# Find background jobs
rails-code-search search -q "async job processing" -f jobs -k 10
```

## Cost Estimation

Using `text-embedding-3-small`:
- **Price**: $0.02 per 1M tokens
- **Average file**: ~500 tokens
- **10,000 files**: ~5M tokens = **$0.10**

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- Built on [pgvector](https://github.com/pgvector/pgvector)
- Uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- Inspired by semantic code search tools
