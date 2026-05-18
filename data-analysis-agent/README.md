# Data Analysis Agent

A take-home full-stack project for uploading arbitrary pickle datasets and chatting with a general-purpose Python coding agent that can inspect, transform, visualize, branch, and export session state.

This repository currently contains the initial scaffold:

- FastAPI backend with a `/health` endpoint
- Session creation and pickle dataset upload endpoints
- SQLite metadata storage with SQLModel
- Filesystem storage for original pickle files and current snapshots
- Generic Python object profiling for DataFrames, Series, ndarrays, mappings, collections, nested JSON-like values, and custom objects
- React + Vite + TypeScript frontend shell
- Tailwind styling with shadcn-inspired primitives
- Frontend health check against the backend
- Placeholder dashboard layout for chat, trace events, datasets, and inspector panels

## Monorepo Layout

```text
data-analysis-agent/
  backend/
    app/
    tests/
    pyproject.toml
    README.md
  frontend/
    src/
    package.json
    vite.config.ts
    tailwind.config.js
  README.md
  .env.example
```

## Backend Setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The API should be available at `http://localhost:8000`.

Backend helper scripts are available through `make dev`, `make test`, and `make lint`.

## Backend API

- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/datasets`
- `GET /api/sessions/{session_id}/datasets`
- `GET /api/sessions/{session_id}/datasets/{dataset_id}`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app should be available at `http://localhost:5173`.

## Environment

Copy `.env.example` to `.env` and fill in values as implementation expands.

```bash
cp .env.example .env
```

## Implementation Notes

Future phases will add the OpenAI-backed coding agent abstraction, Python execution runtime, SSE trace streaming, branch/fork mutation operations beyond the initial version graph, visualization artifacts, confirmation flows, and CSV export.

## Security Note

Loading arbitrary pickle files is unsafe in production because pickle payloads can execute code during deserialization. This take-home currently assumes trusted demo files. A production implementation should load user-submitted pickles only inside an isolated worker or sandbox with strict resource limits and no access to sensitive host resources.
