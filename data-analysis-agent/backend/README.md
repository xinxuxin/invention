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
- `POST /api/sessions` creates an analysis session and a default `main` branch.
- `GET /api/sessions/{session_id}` returns session metadata and branches.
- `POST /api/sessions/{session_id}/datasets` uploads one or more `.pkl` files, stores originals and initial snapshots, creates initial version nodes, and returns generic object profiles.
- `GET /api/sessions/{session_id}/datasets` lists datasets in a session.
- `GET /api/sessions/{session_id}/datasets/{dataset_id}` returns one dataset profile and current version metadata.

## Security Note

Loading arbitrary pickle files is unsafe in production because pickle payloads can execute code during deserialization. This take-home assumes trusted demo files. A production implementation should move pickle loading into an isolated worker or sandbox with strict resource limits.

## Planned Areas

- SQLModel persistence with SQLite
- General-purpose coding agent abstraction over the OpenAI API
- Python execution runtime for pandas/numpy/cloudpickle workflows
- Server-Sent Events for streamed agent traces
