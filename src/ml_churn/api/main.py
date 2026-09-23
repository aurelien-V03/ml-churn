"""API REST exposant les modeles de prediction du churn.

Trois endpoints :

- `GET /health`  : le service repond, sans rien supposer des modeles.
- `GET /ready`   : les modeles sont charges et peuvent predire.
- `POST /predict`: probabilite de churn d'un client, par le modele demande.

La page de test est servie a la racine. Ouverte directement depuis le disque
elle fonctionne aussi, en visant `127.0.0.1:8000` : c'est ce que CORS autorise
ci-dessous.

La distinction entre `health` et `ready` est celle des sondes Kubernetes : un
service peut etre vivant mais pas encore pret, le temps de charger ses modeles.

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
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from ml_churn.api.registry import Modele, entrainer_tout

# Page de test de l'API, servie a la racine. Elle vit hors du package `api` :
# c'est une interface, pas une brique du service.
UI_DIR = Path(__file__).parent.parent / "ui"

# Decimales de la probabilite renvoyee. La decision, elle, se prend sur la
# valeur complete : arrondir d'abord ferait basculer un client a 0.3951 du bon
# cote d'un seuil a 0.40.
DECIMALES = 2

# Rempli au demarrage. Vide tant que l'entrainement n'a pas abouti, ce que
# `/ready` signale.
MODELES: dict[str, Modele] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Entraine les modeles une fois pour toutes, au demarrage du service."""
    MODELES.update(entrainer_tout())
    yield
    MODELES.clear()


app = FastAPI(
    title="ml-churn",
    description="Prediction du churn a partir des modeles entraines.",
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Le service tourne."""
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Les modeles sont entraines et prets.

    Repond 503 tant qu'aucun modele n'est disponible : un orchestrateur doit
    pouvoir retirer l'instance du service tant qu'elle ne peut pas predire.
    """
    if not MODELES:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="aucun modele disponible"
        )

    return ReadyResponse(ready=True, models=sorted(MODELES))


@app.post("/predict", response_model=PredictResponse)
def predict(requete: PredictRequest) -> PredictResponse:
    """Probabilite de churn d'un client, et decision au seuil du modele."""
    modele = MODELES.get(requete.model)
    if modele is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"modele inconnu : {requete.model!r}, disponibles {sorted(MODELES)}",
        )

    manquantes = modele.colonnes_manquantes(requete.data)
    if manquantes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"colonnes_manquantes": manquantes},
        )

    try:
        probabilite = modele.predict_proba(requete.data)
    except (TypeError, ValueError) as erreur:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"colonnes non numeriques : {erreur}",
        ) from erreur

    return PredictResponse(
        model=modele.nom,
        probability=round(probabilite, DECIMALES),
        threshold=modele.seuil,
        churn=probabilite >= modele.seuil,
    )


# Monte en dernier : les routes declarees plus haut restent prioritaires.
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
