"""Modele final de classification : XGBoost sur la couche gold.

Cible : `churn`. Meme decoupage et memes exclusions que la baseline, pour que
les deux modeles restent comparables. Ce qui change tient au modele : la
regression logistique suppose un effet lineaire de chaque feature, XGBoost
enchaine des arbres qui capturent seuils et interactions.

Pas de mise a l'echelle ni d'imputation : les arbres decoupent sur des seuils,
l'echelle des features leur est indifferente.

Usage :
    uv run python -m ml_churn.training.classification.final.classification_xgboost_training
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml_churn.training.common.artifacts import save_model
from ml_churn.training.common.data import (
    EXCLUSIONS_COMMUNES,
    RANDOM_STATE,
    Split,
    feature_columns,
    load_gold,
    split_train_validation_test,
    training_extracts,
)
from ml_churn.training.common.logs import log_classification_training
from ml_churn.training.common.metrics import evaluate_at_threshold

TARGET = "churn"

# Experience MLflow commune a l'entrainement et a la recherche d'hyperparametres.
EXPERIMENT = "classification-xgboost"

# Seuil a partir duquel une probabilite devient une alerte.
SEUIL_DEFAUT = 0.5

# La cible du modele de regression n'est pas une feature de classification.
EXCLUSIONS: dict[str, str] = {
    **EXCLUSIONS_COMMUNES,
    "valeur_vie_client_eur": "cible du modele de regression",
}

# Sous-dossier d'`artifacts/` et prefixe des fichiers enregistres.
ARTIFACTS_TACHE = "classification"
ARTIFACTS_NIVEAU = "final"
ARTIFACTS_NOM = "classification_xgboost"

# Colonnes categorielles dont le modele reduit se passe : leur contribution SHAP
# est negligeable dans le modele complet. `exclusions_sans` retire toutes leurs
# modalites one-hot.
COLONNES_RETIREES = (
    "groupe_experimentation",
    "code_datacenter",
    "couleur_theme_interface",
    "plan",
    "taille_entreprise",
    "pays",
    "jour_souscription",
    "secteur",
)


def exclusions_sans(colonnes: Sequence[str]) -> dict[str, str]:
    """`EXCLUSIONS` augmentee de toutes les modalites one-hot des colonnes citees.

    Une colonne categorielle n'existe en gold que par ses modalites encodees :
    retirer `pays` veut dire retirer `pays_fr`, `pays_es` et les autres. Le
    decoupage des trois jeux n'en depend pas -- il ne porte que sur les lignes
    et la cible -- donc les modeles restent comparables.
    """
    return {
        **EXCLUSIONS,
        **{
            feature: f"retiree avec {colonne}"
            for colonne in colonnes
            for feature in feature_columns(TARGET, EXCLUSIONS)
            if feature.startswith(f"{colonne}_")
        },
    }


# Point de depart raisonnable, avant toute recherche : arbres peu profonds et
# pas d'apprentissage lent, ce qui limite le surapprentissage sur 3000 lignes.
# `classification_xgboost_tuning.py` cherche mieux.
HYPERPARAMETRES: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@dataclass
class Result:
    """Modele entraine et metriques mesurees sur le jeu de test."""

    model: Pipeline
    # `None` quand l'entrainement n'a pas ete enregistre.
    chemin: Path | None
    seuil: float
    hyperparametres: dict[str, Any]
    features: list[str]
    metrics: dict[str, float]
    confusion: list[list[int]]
    importances: pd.Series = field(repr=False)
    # Conserves pour reevaluer ou expliquer le modele sans refaire le split.
    X_test: pd.DataFrame = field(repr=False, default=None)
    y_test: pd.Series = field(repr=False, default=None)


def poids_classe_positive(y_train: pd.Series) -> float:
    """Rapport entre les deux classes, pour `scale_pos_weight`.

    Equivalent XGBoost du `class_weight="balanced"` de la baseline : le
    gradient des churners est multiplie par ce facteur, faute de quoi le
    modele se contente de la classe majoritaire (72 % des clients).
    """
    positifs = int((y_train == 1).sum())
    return float((len(y_train) - positifs) / positifs) if positifs else 1.0


def build_xgboost_pipeline(
    hyperparametres: dict[str, Any] | None = None, *, scale_pos_weight: float = 1.0
) -> Pipeline:
    """XGBoost seul : la couche gold est deja numerique et complete."""
    return Pipeline(
        [
            (
                "model",
                XGBClassifier(
                    **{**HYPERPARAMETRES, **(hyperparametres or {})},
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_split(
    exclusions: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], Split]:
    """Charge gold, selectionne les features et decoupe les trois jeux."""
    df = load_gold()
    features = feature_columns(TARGET, exclusions or EXCLUSIONS)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    return df, features, split_train_validation_test(X, y)


def train_classification_xgboost(
    *,
    seuil: float = SEUIL_DEFAUT,
    hyperparametres: dict[str, Any] | None = None,
    exclusions: dict[str, str] | None = None,
    nom: str = ARTIFACTS_NOM,
    enregistrer: bool = True,
    echo: bool = True,
) -> Result:
    """Entraine XGBoost et l'evalue sur un jeu de test tenu a l'ecart.

    `exclusions` remplace la liste par defaut des colonnes ecartees et `nom`
    prefixe les fichiers ecrits. `enregistrer=False` mesure un modele sans
    rien deposer dans `artifacts/` : utile pour comparer des variantes sans
    encombrer le dossier du jour.
    """
    retenues = exclusions or EXCLUSIONS
    df, features, split = prepare_split(retenues)

    retenus = {**HYPERPARAMETRES, **(hyperparametres or {})}
    model = build_xgboost_pipeline(
        retenus, scale_pos_weight=poids_classe_positive(split.y_train)
    )
    model.fit(split.X_train, split.y_train)

    metrics, matrix = evaluate_at_threshold(
        model, split.X_test, split.y_test, seuil=seuil
    )

    # Gain moyen apporte par chaque feature aux decoupages qui l'utilisent.
    importances = pd.Series(
        model.named_steps["model"].feature_importances_, index=features
    ).sort_values(ascending=False)

    chemin = (
        save_model(
            model,
            datasets=training_extracts(df, features, split, target=TARGET),
            tache=ARTIFACTS_TACHE,
            niveau=ARTIFACTS_NIVEAU,
            nom=nom,
            metadonnees={
                "threshold": seuil,
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
        print(f"[ENTRAINEMENT] XGBoost sur {split.tailles['n_train']} lignes de train")
        print(f"\n[SEUIL] {seuil:.2f}")
        print("\n[HYPERPARAMETRES]")
        for parametre, valeur in retenus.items():
            affichage = f"{valeur:.2f}" if isinstance(valeur, float) else str(valeur)
            print(f"  {parametre:<20} {affichage}")
        log_classification_training(
            df=df,
            features=features,
            exclusions=retenues,
            tailles=split.tailles,
            y_test=split.y_test,
            metrics=metrics,
            matrix=matrix,
            weights=importances,
            weights_titre="IMPORTANCES",
            weights_direction=False,
        )
        if chemin is not None:
            print(f"\n[MODELE ENREGISTRE] {chemin.relative_to(Path.cwd())}")

    return Result(
        model=model,
        chemin=chemin,
        seuil=seuil,
        hyperparametres=retenus,
        features=features,
        metrics=metrics,
        confusion=matrix.tolist(),
        importances=importances,
        X_test=split.X_test,
        y_test=split.y_test,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main(
    seuil: float = typer.Option(SEUIL_DEFAUT, help="seuil de decision, entre 0 et 1"),
) -> None:
    train_classification_xgboost(seuil=seuil)


if __name__ == "__main__":
    app()
