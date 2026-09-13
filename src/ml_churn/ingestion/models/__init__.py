"""Modeles SQLAlchemy des tables du medaillon."""

from ml_churn.ingestion.models.base import Base
from ml_churn.ingestion.models.bronze import (
    BRONZE_SCHEMA,
    CatalogueBronze,
    ChurnSaasCompletBronze,
    ChurnSaasEchantillonBronze,
)
from ml_churn.ingestion.models.silver import SILVER_SCHEMA, ChurnSaasSilver

__all__ = [
    "BRONZE_SCHEMA",
    "SILVER_SCHEMA",
    "Base",
    "CatalogueBronze",
    "ChurnSaasCompletBronze",
    "ChurnSaasEchantillonBronze",
    "ChurnSaasSilver",
]
