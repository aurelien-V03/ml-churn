"""Modele baseline de classification : regression logistique sur la couche gold.

Cible : `churn`. Les colonnes categorielles ne sont pas reprises telles quelles
-- leur encodage one-hot, calcule en gold, l'est a leur place.

`sante_compte_fin_periode` est exclue par `EXCLUSIONS_COMMUNES` : mesuree en fin
de periode, elle n'est pas disponible au moment de la prediction. L'inclure
ferait grimper l'AUC a 0.99 sans valeur en production.

Usage :
    uv run python -m ml_churn.training.scripts.classification.baseline.classification_baseline_training
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_churn.training.scripts.common.data import (
    EXCLUSIONS_COMMUNES,
    feature_columns,
    load_gold,
)
from ml_churn.training.scripts.common.logs import log_classification_training
from ml_churn.training.scripts.common.metrics import classification_metrics

TARGET = "churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# La cible du modele de regression n'est pas une feature de classification.
EXCLUSIONS: dict[str, str] = {
    **EXCLUSIONS_COMMUNES,
    "valeur_vie_client_eur": "cible du modele de regression",
}


@dataclass
class Result:
    """Modele entraine et metriques mesurees sur le jeu de test."""

    model: Pipeline
    features: list[str]
    metrics: dict[str, float]
    confusion: list[list[int]]
    coefficients: pd.Series = field(repr=False)


def build_pipeline() -> Pipeline:
    """Regression logistique seule : les donnees gold sont deja pretes.

    Pas d'imputation : la couche gold ne contient aucune valeur manquante.
    Pas de mise a l'echelle : le solveur `newton-cholesky` s'en passe et
    converge en quelques iterations, la ou `lbfgs` (le defaut) epuise 12 000
    iterations sans converger sur des features allant de csat (1 a 5) au
    revenu mensuel (jusqu'a 89 000).

    `class_weight="balanced"` compense le desequilibre des classes (28 % de
    churn) : les erreurs sur les churners pesent 1.79 contre 0.69 pour les
    clients fideles. Le rappel passe de 0.56 a 0.80, au prix de la precision.
    """
    return Pipeline(
        [
            (
                "model",
                LogisticRegression(
                    solver="newton-cholesky",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_classification_baseline(*, echo: bool = True) -> Result:
    """Entraine la baseline et l'evalue sur un jeu de test tenu a l'ecart."""
    df = load_gold()
    features = feature_columns(TARGET, EXCLUSIONS)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    # Stratifie : le jeu de test conserve la proportion de churn de l'ensemble.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions)
    metrics = classification_metrics(matrix)
    # L'AUC se calcule sur les probabilites, pas sur la matrice : elle ne
    # depend d'aucun seuil de decision.
    metrics["auc"] = roc_auc_score(y_test, probabilities)

    coefficients = pd.Series(
        model.named_steps["model"].coef_[0], index=features
    ).sort_values(key=abs, ascending=False)

    if echo:
        log_classification_training(
            df=df,
            features=features,
            exclusions=EXCLUSIONS,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            metrics=metrics,
            matrix=matrix,
            weights=coefficients,
        )

    return Result(
        model=model,
        features=features,
        metrics=metrics,
        confusion=matrix.tolist(),
        coefficients=coefficients,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    train_classification_baseline()


if __name__ == "__main__":
    app()
