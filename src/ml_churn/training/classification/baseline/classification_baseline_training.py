"""Modele baseline de classification : regression logistique sur la couche gold.

Cible : `churn`. Les colonnes categorielles ne sont pas reprises telles quelles
-- leur encodage one-hot, calcule en gold, l'est a leur place.

`sante_compte_fin_periode` est exclue par `EXCLUSIONS_COMMUNES` : mesuree en fin
de periode, elle n'est pas disponible au moment de la prediction. L'inclure
ferait grimper l'AUC a 0.99 sans valeur en production.

Usage :
    uv run python -m ml_churn.training.classification.baseline.classification_baseline_training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from ml_churn.training.common.artifacts import save_model
from ml_churn.training.common.data import (
    EXCLUSIONS_COMMUNES,
    RANDOM_STATE,
    Split,
    feature_columns,
    load_gold,
    split_train_validation_test,
)
from ml_churn.training.common.logs import log_classification_training
from ml_churn.training.common.metrics import classification_metrics

TARGET = "churn"

# Experience MLflow commune a l'entrainement et a la recherche de seuil.
EXPERIMENT = "classification-baseline"

# Seuil a partir duquel une probabilite devient une alerte. 0.5 est le defaut
# de scikit-learn ; `classification_baseline_tuning.py` cherche le meilleur.
SEUIL_DEFAUT = 0.5

# La cible du modele de regression n'est pas une feature de classification.
EXCLUSIONS: dict[str, str] = {
    **EXCLUSIONS_COMMUNES,
    "valeur_vie_client_eur": "cible du modele de regression",
}


# Sous-dossier d'`artifacts/` et prefixe des fichiers enregistres.
ARTIFACTS_DOSSIER = "baseline"
ARTIFACTS_NOM = "classification_baseline"


@dataclass
class Result:
    """Modele entraine et metriques mesurees sur le jeu de test."""

    model: Pipeline
    chemin: Path
    seuil: float
    features: list[str]
    metrics: dict[str, float]
    confusion: list[list[int]]
    coefficients: pd.Series = field(repr=False)
    # Conserve pour expliquer le modele apres coup (SHAP) sans refaire le split.
    X_test: pd.DataFrame = field(repr=False, default=None)


def build_baseline_pipeline() -> Pipeline:
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


def training_extracts(
    df: pd.DataFrame, features: list[str], split: Split
) -> dict[str, pd.DataFrame]:
    """Extrait gold reellement consomme, un DataFrame par jeu de donnees.

    Ecrire les trois jeux separement fige le decoupage : sans cela, rejouer
    l'entrainement depuis l'extrait supposerait que `train_test_split` decoupe
    toujours a l'identique, ce qui n'est vrai qu'a version de scikit-learn
    constante. `client_id` sert a remonter a la ligne source, pas a predire.
    """
    colonnes = [colonne for colonne in ("client_id", TARGET) if colonne in df.columns]

    return {
        jeu: df.loc[indices, [*colonnes, *features]]
        for jeu, indices in (
            ("train", split.X_train.index),
            ("validation", split.X_validation.index),
            ("test", split.X_test.index),
        )
    }


def train_classification_baseline(
    *, seuil: float = SEUIL_DEFAUT, echo: bool = True
) -> Result:
    """Entraine la baseline et l'evalue sur un jeu de test tenu a l'ecart.

    `seuil` fixe la probabilite a partir de laquelle un client est signale.
    """
    df = load_gold()
    features = feature_columns(TARGET, EXCLUSIONS)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    split = split_train_validation_test(X, y)

    model = build_baseline_pipeline()
    model.fit(split.X_train, split.y_train)

    probabilities = model.predict_proba(split.X_test)[:, 1]
    predictions = (probabilities >= seuil).astype(int)

    matrix = confusion_matrix(split.y_test, predictions)
    metrics = classification_metrics(matrix)
    # L'AUC se calcule sur les probabilites, pas sur la matrice : elle ne
    # depend d'aucun seuil de decision.
    metrics["auc"] = roc_auc_score(split.y_test, probabilities)

    coefficients = pd.Series(
        model.named_steps["model"].coef_[0], index=features
    ).sort_values(key=abs, ascending=False)

    chemin = save_model(
        model,
        datasets=training_extracts(df, features, split),
        dossier=ARTIFACTS_DOSSIER,
        nom=ARTIFACTS_NOM,
        metadonnees={
            "threshold": seuil,
            "target": TARGET,
            "features": features,
            "sizes": split.tailles,
            "test_metrics": {nom: round(valeur, 4) for nom, valeur in metrics.items()},
        },
    )

    if echo:
        print(f"[SEUIL] {seuil:.2f}")
        log_classification_training(
            df=df,
            features=features,
            exclusions=EXCLUSIONS,
            tailles=split.tailles,
            y_test=split.y_test,
            metrics=metrics,
            matrix=matrix,
            weights=coefficients,
        )

    if echo:
        print(f"\n[MODELE ENREGISTRE] {chemin.relative_to(Path.cwd())}")

    return Result(
        model=model,
        chemin=chemin,
        seuil=seuil,
        features=features,
        metrics=metrics,
        confusion=matrix.tolist(),
        coefficients=coefficients,
        X_test=split.X_test,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main(
    seuil: float = typer.Option(SEUIL_DEFAUT, help="seuil de decision, entre 0 et 1"),
) -> None:
    train_classification_baseline(seuil=seuil)


if __name__ == "__main__":
    app()
