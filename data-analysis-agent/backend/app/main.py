from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.artifacts import router as artifacts_router
from app.api.chat import router as chat_router
from app.api.confirmations import router as confirmations_router
from app.api.export import router as export_router
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.sessions import router as sessions_router
from app.core.config import get_settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).rstrip("/") for origin in settings.backend_cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(confirmations_router)
    app.include_router(artifacts_router)
    app.include_router(history_router)
    app.include_router(export_router)

    return app


app = create_app()
