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


class UiConfigResponse(BaseModel):
    """Ce dont la page de test a besoin pour appeler le service."""

    api_key: str = Field(description="Cle attendue dans l'en-tete X-API-Key")


class DriftRequest(BaseModel):
    """Population courante a comparer a celle de l'entrainement."""

    data: list[dict[str, Any]] = Field(
        description="Lignes observees, colonnes de la couche gold"
    )


class ColonneDeriveResponse(BaseModel):
    """Derive mesuree sur une colonne."""

    column: str
    ks: float = Field(description="Ecart maximal entre les deux repartitions")
    p_value: float = Field(description="Sous 0.05, l'ecart n'est pas un hasard")
    psi: float = Field(description="Population Stability Index")
    verdict: str = Field(description="stable, moderee ou importante")
    # Calcule sur la p-valeur complete : arrondie, une valeur de 0.049
    # deviendrait 0.05 et cesserait d'etre signalee.
    significant: bool = Field(description="p-valeur sous 0.05")


class DriftResponse(BaseModel):
    """Derive entre la population d'entrainement et une population courante."""

    reference: str = Field(description="Fichier servant de reference")
    current: str = Field(description="Fichier compare a la reference")
    columns: list[ColonneDeriveResponse]
