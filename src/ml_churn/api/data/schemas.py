"""Corps des requetes et des reponses, valides par Pydantic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Le service repond."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Les modeles sont entraines et prets a predire."""

    ready: bool
    models: dict[str, list[str]] = Field(
        description="Modeles disponibles, par famille (`churn`, `clv`)"
    )


class PredictRequest(BaseModel):
    """Un client a evaluer, par le modele demande.

    `data` reprend les noms de colonnes de la couche gold. Les colonnes
    inconnues du modele sont ignorees : une ligne gold complete peut etre
    envoyee telle quelle, `client_id` et `churn` compris.
    """

    model: str = Field(description="Nom du modele, par exemple `xgboost`")
    data: dict[str, Any] = Field(description="Colonnes gold du client a evaluer")


class PredictChurnResponse(BaseModel):
    """Probabilite de churn et decision au seuil du modele."""

    model: str
    probability: float = Field(description="Probabilite de churn, entre 0 et 1")
    threshold: float = Field(description="Seuil a partir duquel l'alerte est levee")
    churn: bool = Field(description="`probability >= threshold`")


class PredictClvResponse(BaseModel):
    """Valeur vie client estimee."""

    model: str
    lifetime_value_eur: float = Field(description="Valeur vie client estimee, en euros")
