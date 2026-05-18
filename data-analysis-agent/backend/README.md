# Backend

FastAPI service for the Data Analysis Agent.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Or use the backend helper:

```bash
make dev
```

## Current Endpoints

- `GET /health` returns service status for local development and frontend connectivity checks.

## Planned Areas

- SQLModel persistence with SQLite
- Dataset upload and storage services
- General-purpose coding agent abstraction over the OpenAI API
- Python execution runtime for pandas/numpy/cloudpickle workflows
- Server-Sent Events for streamed agent traces
