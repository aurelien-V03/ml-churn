"""Modeles SQLAlchemy des tables du medaillon."""

from ml_churn.ingestion.models.base import Base
from ml_churn.ingestion.models.bronze import (
    BRONZE_SCHEMA,
    CatalogueBronze,
    ChurnSaasCompletBronze,
    ChurnSaasEchantillonBronze,
)
from ml_churn.ingestion.models.gold import (
    GOLD_SCHEMA,
    MODALITES_ONE_HOT,
    CatalogueGold,
    ChurnSaasGold,
    nom_colonne_one_hot,
)
from ml_churn.ingestion.models.silver import (
    SILVER_SCHEMA,
    CatalogueSilver,
    ChurnSaasSilver,
)

__all__ = [
    "BRONZE_SCHEMA",
    "GOLD_SCHEMA",
    "MODALITES_ONE_HOT",
    "SILVER_SCHEMA",
    "Base",
    "CatalogueBronze",
    "CatalogueGold",
    "CatalogueSilver",
    "ChurnSaasCompletBronze",
    "ChurnSaasEchantillonBronze",
    "ChurnSaasGold",
    "ChurnSaasSilver",
    "nom_colonne_one_hot",
]
