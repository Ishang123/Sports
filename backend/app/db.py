from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _db_url() -> str:
    db_path = resolve_db_path()
    return f"sqlite:///{db_path}"


def resolve_db_path() -> str:
    raw = os.getenv("INTEGRITY_DB_PATH", str(REPO_ROOT / "integrity.db"))
    return str(Path(raw).expanduser().resolve())


engine: Engine | None = None
SessionLocal: sessionmaker | None = None


def configure_engine(db_path: str | None = None) -> None:
    global engine, SessionLocal
    if db_path:
        os.environ["INTEGRITY_DB_PATH"] = db_path
    # SQLite + threaded FastAPI can exhaust QueuePool under frequent polling.
    # NullPool opens/closes per checkout and avoids pool timeout buildup.
    engine = create_engine(
        _db_url(),
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session() -> Session:
    if SessionLocal is None:
        configure_engine()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from app.models_db import (  # noqa: F401
        Entity,
        EntityScore,
        EntityWindowFeature,
        Market,
        MarketMapping,
        ModelRegistry,
        Trade,
    )

    if engine is None:
        configure_engine()
    Base.metadata.create_all(bind=engine)


configure_engine()
