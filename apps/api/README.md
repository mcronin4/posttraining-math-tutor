# Math Tutor API

FastAPI backend for the LLM Math Tutor.

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn src.main:app --reload --port 8000
```

## Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /chat` - Tutoring chat endpoint

