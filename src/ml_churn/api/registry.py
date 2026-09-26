"""Preparation des modeles servis par l'API.

Les modeles sont entraines au demarrage du service, a partir de la couche
gold : rien n'est relu depuis `artifacts/`. Le service depend donc de la base
PostgreSQL pour demarrer, et chaque instance reentraine ses propres modeles.

Deux familles, une par endpoint : `churn` classe, `clv` estime une valeur en
euros. Leurs configurations sont figees ci-dessous -- seuil et hyperparametres
retenus par les recherches, recopies en dur plutot que relus d'un artefact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml_churn.training.classification.final.classification_xgboost_training import (
    COLONNES_RETIREES,
    exclusions_sans,
    train_classification_xgboost,
)
from ml_churn.training.regression.final.regression_xgboost_training import (
    train_regression_xgboost,
)

# --- Modele de churn ---------------------------------------------------------

# Seuil de decision retenu : plus bas que le defaut de 0.5, conformement au
# postulat metier -- un churner manque coute plus cher qu'une fausse alerte.
SEUIL_CHURN = 0.40

HYPERPARAMETRES_CHURN = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.96,
    "colsample_bytree": 0.76,
    "min_child_weight": 8,
    "gamma": 4.57,
    "reg_alpha": 0.00,
    "reg_lambda": 4.62,
}

# --- Modele de valeur vie client ---------------------------------------------

HYPERPARAMETRES_CLV = {
    "n_estimators": 550,
    "max_depth": 8,
    "learning_rate": 0.01,
    "subsample": 0.68,
    "colsample_bytree": 0.82,
    "min_child_weight": 6,
    "gamma": 1.29,
    "reg_alpha": 1.32,
    "reg_lambda": 0.00,
}


@dataclass
class Modele:
    """Un modele pret a predire, avec ce qu'il faut pour l'interroger.

    `seuil` n'existe que pour un modele de classification : une regression
    rend directement sa valeur, sans regle de decision.
    """

    nom: str
    features: list[str]
    pipeline: Any = field(repr=False)
    seuil: float | None = None

    def _ligne(self, donnees: dict[str, Any]) -> pd.DataFrame:
        """Les colonnes sont remises dans l'ordre d'entrainement : un modele
        scikit-learn identifie ses features par position, pas par nom."""
        return pd.DataFrame(
            [[donnees[nom] for nom in self.features]], columns=self.features
        ).astype(float)

    def probabilite(self, donnees: dict[str, Any]) -> float:
        """Probabilite de churn pour un client."""
        return float(self.pipeline.predict_proba(self._ligne(donnees))[0, 1])

    def valeur(self, donnees: dict[str, Any]) -> float:
        """Valeur vie client estimee, en euros."""
        return float(self.pipeline.predict(self._ligne(donnees))[0])

    def colonnes_manquantes(self, donnees: dict[str, Any]) -> list[str]:
        return [nom for nom in self.features if nom not in donnees]


def _xgboost_churn() -> Modele:
    """XGBoost de classification, sans les colonnes sans apport."""
    resultat = train_classification_xgboost(
        seuil=SEUIL_CHURN,
        hyperparametres=HYPERPARAMETRES_CHURN,
        exclusions=exclusions_sans(COLONNES_RETIREES),
        enregistrer=False,
        echo=False,
    )

    return Modele(
        nom="xgboost",
        features=resultat.features,
        pipeline=resultat.model,
        seuil=resultat.seuil,
    )


def _xgboost_clv() -> Modele:
    """XGBoost de regression, sur toutes les colonnes gold."""
    resultat = train_regression_xgboost(
        hyperparametres=HYPERPARAMETRES_CLV,
        enregistrer=False,
        echo=False,
    )

    return Modele(nom="xgboost", features=resultat.features, pipeline=resultat.model)


# Famille -> nom expose par l'API -> fonction qui entraine le modele.
MODELES: dict[str, dict[str, Callable[[], Modele]]] = {
    "churn": {"xgboost": _xgboost_churn},
    "clv": {"xgboost": _xgboost_clv},
}


def entrainer(famille: str, nom: str) -> Modele:
    """Entraine un modele declare dans `MODELES`."""
    return MODELES[famille][nom]()


def entrainer_tout() -> dict[str, dict[str, Modele]]:
    """Entraine tous les modeles declares, au demarrage du service."""
    return {
        famille: {nom: fabrique() for nom, fabrique in modeles.items()}
        for famille, modeles in MODELES.items()
    }
