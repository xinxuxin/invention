from collections.abc import Generator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)


def init_db() -> None:
    from app.models.entities import (  # noqa: F401
        AnalysisSession,
        Artifact,
        Branch,
        ChatMessage,
        Dataset,
        PendingConfirmation,
        VersionNode,
    )

    SQLModel.metadata.create_all(engine)
    _migrate_sqlite()


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def _migrate_sqlite() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    additions = {
        "analysis_sessions": {
            "active_branch_id": "TEXT",
            "active_dataset_id": "TEXT",
        },
        "datasets": {
            "dataset_key": "TEXT",
        },
        "branches": {
            "current_version_id": "TEXT",
            "root_version_id": "TEXT",
        },
        "version_nodes": {
            "parent_version_id": "TEXT",
            "mutation_summary": "TEXT",
            "created_by_message_id": "TEXT",
        },
        "chat_messages": {
            "pending_action": "JSON",
        },
    }
    with engine.begin() as connection:
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())
        for table, columns in additions.items():
            if table not in existing_tables:
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table)}
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}"))
