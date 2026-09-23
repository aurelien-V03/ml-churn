"""Preparation des modeles servis par l'API.

Les modeles sont entraines au demarrage du service, a partir de la couche
gold : rien n'est relu depuis `artifacts/`. Le service depend donc de la base
PostgreSQL pour demarrer, et chaque instance reentraine son propre modele.

La configuration servie est figee ci-dessous : le seuil et les
hyperparametres retenus par la recherche, recopies en dur plutot que relus
d'un artefact ou recherches a chaque demarrage.
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


@dataclass
class Modele:
    """Un modele pret a predire, avec ce qu'il faut pour l'interroger."""

    nom: str
    seuil: float
    features: list[str]
    pipeline: Any = field(repr=False)

    def predict_proba(self, donnees: dict[str, Any]) -> float:
        """Probabilite de churn pour un client.

        Les colonnes sont remises dans l'ordre d'entrainement : un modele
        scikit-learn identifie ses features par position, pas par nom.
        """
        ligne = pd.DataFrame(
            [[donnees[nom] for nom in self.features]], columns=self.features
        )
        return float(self.pipeline.predict_proba(ligne.astype(float))[0, 1])

    def colonnes_manquantes(self, donnees: dict[str, Any]) -> list[str]:
        return [nom for nom in self.features if nom not in donnees]


# Seuil de decision retenu : plus bas que le defaut de 0.5, conformement au
# postulat metier -- un churner manque coute plus cher qu'une fausse alerte.
SEUIL = 0.40

# Hyperparametres retenus par la recherche Optuna, figes pour le service.
HYPERPARAMETRES = {
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


def _xgboost() -> Modele:
    """XGBoost sur les features retenues, sans les colonnes sans apport."""
    resultat = train_classification_xgboost(
        seuil=SEUIL,
        hyperparametres=HYPERPARAMETRES,
        exclusions=exclusions_sans(COLONNES_RETIREES),
        enregistrer=False,
        echo=False,
    )

    return Modele(
        nom="xgboost",
        seuil=resultat.seuil,
        features=resultat.features,
        pipeline=resultat.model,
    )


# Nom expose par l'API -> fonction qui entraine le modele correspondant.
MODELES: dict[str, Callable[[], Modele]] = {
    "xgboost": _xgboost,
}


def entrainer(nom: str) -> Modele:
    """Entraine un modele declare dans `MODELES`."""
    return MODELES[nom]()


def entrainer_tout() -> dict[str, Modele]:
    """Entraine tous les modeles declares, au demarrage du service."""
    return {nom: entrainer(nom) for nom in MODELES}
