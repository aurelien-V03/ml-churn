"""API REST exposant les modeles de prediction.

Quatre endpoints :

- `GET /health`       : le service repond, sans rien supposer des modeles.
- `GET /ready`        : les deux familles de modeles sont entrainees.
- `POST /predict-churn` : probabilite de churn d'un client.
- `POST /predict-clv`   : valeur vie client estimee, en euros.

La distinction entre `health` et `ready` est celle des sondes Kubernetes : un
service peut etre vivant mais pas encore pret, le temps d'entrainer ses
modeles.

La page de test est servie a la racine. Ouverte directement depuis le disque
elle fonctionne aussi, en visant `127.0.0.1:8000` : c'est ce que CORS autorise
ci-dessous.

Usage :
    uv run uvicorn ml_churn.api.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ml_churn.api.data import (
    HealthResponse,
    PredictChurnResponse,
    PredictClvResponse,
    PredictRequest,
    ReadyResponse,
)
from ml_churn.api.registry import MODELES as FABRIQUES
from ml_churn.api.registry import Modele, entrainer_tout

# Page de test de l'API, servie a la racine. Elle vit hors du package `api` :
# c'est une interface, pas une brique du service.
UI_DIR = Path(__file__).parent.parent / "ui"

# Decimales de la probabilite renvoyee. La decision, elle, se prend sur la
# valeur complete : arrondir d'abord ferait basculer un client a 0.3951 du bon
# cote d'un seuil a 0.40.
DECIMALES = 2

# Rempli au demarrage, une entree par famille. Tant qu'une famille manque, le
# service n'est pas pret -- ce que `/ready` signale.
MODELES: dict[str, dict[str, Modele]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Entraine les modeles une fois pour toutes, au demarrage du service."""
    MODELES.update(entrainer_tout())
    yield
    MODELES.clear()


app = FastAPI(
    title="ml-churn",
    description="Prediction du churn et de la valeur vie client.",
    lifespan=lifespan,
)

# La page servie a la racine n'en a pas besoin -- meme origine -- mais elle
# reste utilisable ouverte directement depuis le disque, ou l'origine vaut
# `null`. Service local de developpement : toutes les origines sont admises.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _modele(famille: str, nom: str) -> Modele:
    """Modele demande, ou l'erreur HTTP correspondante."""
    disponibles = MODELES.get(famille, {})
    modele = disponibles.get(nom)
    if modele is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"modele inconnu : {nom!r}, disponibles {sorted(disponibles)}",
        )

    return modele


def _verifier(modele: Modele, donnees: dict) -> None:
    """Refuse une ligne incomplete avant d'atteindre le modele."""
    manquantes = modele.colonnes_manquantes(donnees)
    if manquantes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"colonnes_manquantes": manquantes},
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Le service tourne."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Les deux familles de modeles sont entrainees.

    Repond 503 tant qu'une famille manque : un orchestrateur doit pouvoir
    retirer l'instance du service tant qu'elle ne peut pas honorer ses deux
    endpoints de prediction.
    """
    manquantes = [famille for famille in FABRIQUES if not MODELES.get(famille)]
    if manquantes:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"familles_non_chargees": manquantes},
        )

    return ReadyResponse(
        ready=True,
        models={famille: sorted(modeles) for famille, modeles in MODELES.items()},
    )


@app.post("/predict-churn", response_model=PredictChurnResponse)
def predict_churn(requete: PredictRequest) -> PredictChurnResponse:
    """Probabilite de churn d'un client, et decision au seuil du modele."""
    modele = _modele("churn", requete.model)
    _verifier(modele, requete.data)

    try:
        probabilite = modele.probabilite(requete.data)
    except (TypeError, ValueError) as erreur:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"colonnes non numeriques : {erreur}",
        ) from erreur

    return PredictChurnResponse(
        model=modele.nom,
        probability=round(probabilite, DECIMALES),
        threshold=modele.seuil,
        churn=probabilite >= modele.seuil,
    )


@app.post("/predict-clv", response_model=PredictClvResponse)
def predict_clv(requete: PredictRequest) -> PredictClvResponse:
    """Valeur vie client estimee, en euros."""
    modele = _modele("clv", requete.model)
    _verifier(modele, requete.data)

    try:
        valeur = modele.valeur(requete.data)
    except (TypeError, ValueError) as erreur:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"colonnes non numeriques : {erreur}",
        ) from erreur

    return PredictClvResponse(
        model=modele.nom, lifetime_value_eur=round(valeur, DECIMALES)
    )


# Monte en dernier : les routes declarees plus haut restent prioritaires.
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
