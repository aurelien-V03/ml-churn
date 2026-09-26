"""Modele final de regression : XGBoost sur la valeur vie client.

Meme cible et meme decoupage que la baseline lineaire. Les features, elles,
different : le modele lineaire se limite aux dix colonnes les plus correlees a
la cible, les arbres prennent les soixante-quatre. Une variable secondaire
deteriore une droite en se partageant le credit avec ses voisines ; un arbre
s'en sert quand elle aide et l'ignore sinon.

Deux consequences sur une cible aussi etalee que la valeur vie client. Les
arbres ne peuvent pas predire en dehors de l'intervalle vu a l'entrainement,
donc aucune valeur negative ; et ils captent les effets de seuil qu'une droite
ne represente pas.

Pas de standardisation : un arbre decoupe sur des seuils, l'echelle des
features lui est indifferente.

Usage :
    uv run python -m ml_churn.training.regression.final.regression_xgboost_training
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ml_churn.training.common.artifacts import save_model
from ml_churn.training.common.data import (
    RANDOM_STATE,
    feature_columns,
    load_gold,
    split_train_validation_test,
    strates_quantiles,
    training_extracts,
)
from ml_churn.training.common.logs import log_regression_training
from ml_churn.training.common.metrics import regression_metrics
from ml_churn.training.regression.baseline.regression_baseline_training import (
    EXCLUSIONS,
    TARGET,
)

# Sous-dossier d'`artifacts/` et prefixe des fichiers enregistres.
ARTIFACTS_TACHE = "regression"
ARTIFACTS_NIVEAU = "final"
ARTIFACTS_NOM = "regression_xgboost"

# Arbres peu profonds et apprentissage lent : sur 3000 lignes, les reglages
# par defaut de XGBoost surapprennent les quelques comptes a plusieurs
# centaines de milliers d'euros.
HYPERPARAMETRES: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@dataclass
class Result:
    """Modele entraine et erreurs mesurees sur le jeu de test."""

    model: Pipeline
    chemin: Path | None
    features: list[str]
    hyperparametres: dict[str, Any]
    metrics: dict[str, float]
    importances: pd.Series = field(repr=False)
    X_test: pd.DataFrame = field(repr=False, default=None)
    y_test: pd.Series = field(repr=False, default=None)


def build_xgboost_regression_pipeline(
    hyperparametres: dict[str, Any] | None = None,
) -> Pipeline:
    """XGBoost seul : la couche gold est deja numerique et complete."""
    return Pipeline(
        [
            (
                "model",
                XGBRegressor(
                    **{**HYPERPARAMETRES, **(hyperparametres or {})},
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_regression_xgboost(
    *,
    features: Sequence[str] | None = None,
    hyperparametres: dict[str, Any] | None = None,
    enregistrer: bool = True,
    echo: bool = True,
) -> Result:
    """Entraine XGBoost sur la valeur vie client et l'evalue sur le jeu de test."""
    df = load_gold()
    # Toutes les colonnes gold par defaut : contrairement au modele lineaire,
    # les arbres tirent parti des variables secondaires sans en souffrir.
    candidates = feature_columns(TARGET, EXCLUSIONS)
    features = list(features or candidates)

    inconnues = set(features) - set(candidates)
    if inconnues:
        raise ValueError(f"features absentes de gold ou exclues : {sorted(inconnues)}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    # Stratifier sur les deciles de la cible : chaque jeu recoit sa part de
    # petits et de gros comptes, ce qu'une cible continue ne permet pas
    # directement.
    split = split_train_validation_test(X, y, stratify=strates_quantiles(y))

    retenus = {**HYPERPARAMETRES, **(hyperparametres or {})}
    model = build_xgboost_regression_pipeline(retenus)
    model.fit(split.X_train, split.y_train)

    metrics = regression_metrics(split.y_test, model.predict(split.X_test))

    importances = pd.Series(
        model.named_steps["model"].feature_importances_, index=features
    ).sort_values(ascending=False)

    chemin = (
        save_model(
            model,
            datasets=training_extracts(df, features, split, target=TARGET),
            tache=ARTIFACTS_TACHE,
            niveau=ARTIFACTS_NIVEAU,
            nom=ARTIFACTS_NOM,
            metadonnees={
                "target": TARGET,
                "hyperparameters": retenus,
                "features": features,
                "sizes": split.tailles,
                "test_metrics": metrics,
            },
        )
        if enregistrer
        else None
    )

    if echo:
        log_regression_training(
            df=df,
            features=features,
            tailles=split.tailles,
            y_test=split.y_test,
            metrics=metrics,
        )
        if chemin is not None:
            print(f"\n[MODELE ENREGISTRE] {chemin.relative_to(Path.cwd())}")

    return Result(
        model=model,
        chemin=chemin,
        features=features,
        hyperparametres=retenus,
        metrics=metrics,
        importances=importances,
        X_test=split.X_test,
        y_test=split.y_test,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    train_regression_xgboost()


if __name__ == "__main__":
    app()
