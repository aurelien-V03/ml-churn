"""Connexion a la base PostgreSQL du projet."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_URL = "postgresql+psycopg://mlchurn:mlchurn@localhost:5432/mlchurn"


def database_url() -> str:
    """URL SQLAlchemy, lue depuis DATABASE_URL (.env a la racine du projet)."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(database_url(), future=True)


def get_session() -> Session:
    return sessionmaker(bind=get_engine(), future=True)()


def ensure_schema(schema: str) -> None:
    """Cree le schema s'il n'existe pas (le volume Docker peut etre neuf)."""
    with get_engine().begin() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
