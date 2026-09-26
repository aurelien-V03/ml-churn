"""Modele baseline de regression : estimation de la valeur vie client.

Cible : `valeur_vie_client_eur`, en euros. C'est la seule difference de fond
avec les modeles de classification, qui predisent `churn` : la sortie est
continue, l'erreur se mesure donc en euros plutot qu'en cas bien classes.

`churn` est exclue des features : elle est constatee en fin de periode, au
meme titre que `sante_compte_fin_periode`, et ne serait pas disponible au
moment ou l'on veut estimer la valeur d'un client.

Regression lineaire aux parametres par defaut, sans regularisation : un point
de comparaison, pas un modele final. Les features sont standardisees, ce qui ne
change pas les predictions mais rend les coefficients comparables entre eux --
et prepare le passage a une regression penalisee, ou la mise a l'echelle n'est
plus optionnelle.

Usage :
    uv run python -m ml_churn.training.regression.baseline.regression_baseline_training
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_churn.training.common.artifacts import save_model
from ml_churn.training.common.data import (
    EXCLUSIONS_COMMUNES,
    feature_columns,
    load_gold,
    split_train_validation_test,
    strates_quantiles,
    training_extracts,
)
from ml_churn.training.common.logs import log_regression_training
from ml_churn.training.common.metrics import regression_metrics

TARGET = "valeur_vie_client_eur"

# La cible des modeles de classification n'est pas une feature de regression.
EXCLUSIONS: dict[str, str] = {
    **EXCLUSIONS_COMMUNES,
    "churn": "cible des modeles de classification",
}

# Features retenues : les dix colonnes gold les plus correlees a la cible. Les
# modalites one-hot en font partie -- elles n'apparaissent qu'apres encodage,
# une matrice de correlation lue sur la silver les manquerait.
FEATURES_RETENUES: tuple[str, ...] = (
    "revenu_mensuel_recurrent_eur",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "taille_entreprise_ge",
    "plan_ent",
    "fonctionnalites_utilisees",
    "delai_reponse_support_h",
    "taille_entreprise_tpe",
    "plan_str",
    "taille_entreprise_pme",
)

# Sous-dossier d'`artifacts/` et prefixe des fichiers enregistres.
ARTIFACTS_TACHE = "regression"
ARTIFACTS_NIVEAU = "baseline"
ARTIFACTS_NOM = "regression_baseline"


@dataclass
class Result:
    """Modele entraine et erreurs mesurees sur le jeu de test."""

    model: Pipeline
    chemin: Path | None
    features: list[str]
    metrics: dict[str, float]
    coefficients: pd.Series = field(repr=False)
    # Conserves pour reevaluer ou expliquer le modele sans refaire le split.
    X_test: pd.DataFrame = field(repr=False, default=None)
    y_test: pd.Series = field(repr=False, default=None)


def build_regression_baseline_pipeline() -> Pipeline:
    """Standardisation puis regression lineaire.

    `LinearRegression` n'a aucun parametre a regler : les moindres carres se
    resolvent par decomposition. La standardisation ne change pas ses
    predictions -- multiplier une feature par une constante divise son
    coefficient d'autant -- mais elle ramene les coefficients a une echelle
    commune : sans elle, un coefficient classe les features par unite de mesure
    plutot que par influence.
    """
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


def standardisation(
    model: Pipeline, features: list[str]
) -> dict[str, dict[str, float]]:
    """Moyenne et ecart-type appris par le `StandardScaler`, feature par feature.

    Enregistres avec le modele : ce sont eux qui transforment une ligne brute
    en entree exploitable, un modele serialise seul ne dit pas sur quelles
    statistiques il a ete centre.
    """
    scaler = model.named_steps["scaler"]

    return {
        "method": type(scaler).__name__,
        "mean": dict(zip(features, map(float, scaler.mean_), strict=True)),
        "scale": dict(zip(features, map(float, scaler.scale_), strict=True)),
    }


def train_regression_baseline(
    *,
    features: Sequence[str] | None = None,
    enregistrer: bool = True,
    echo: bool = True,
) -> Result:
    """Entraine la baseline de regression et l'evalue sur le jeu de test.

    `features` remplace la liste retenue : de quoi mesurer ce qu'apporte -- ou
    non -- une colonne, sans toucher au modele de reference.
    """
    df = load_gold()
    features = list(features or FEATURES_RETENUES)

    inconnues = set(features) - set(feature_columns(TARGET, EXCLUSIONS))
    if inconnues:
        raise ValueError(f"features absentes de gold ou exclues : {sorted(inconnues)}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    # Stratifier sur les deciles de la cible : chaque jeu recoit sa part de
    # petits et de gros comptes, ce qu'une cible continue ne permet pas
    # directement.
    split = split_train_validation_test(X, y, stratify=strates_quantiles(y))

    model = build_regression_baseline_pipeline()
    model.fit(split.X_train, split.y_train)

    predictions = model.predict(split.X_test)
    metrics = regression_metrics(split.y_test, predictions)

    coefficients = pd.Series(
        model.named_steps["model"].coef_, index=features
    ).sort_values(key=abs, ascending=False)

    chemin = (
        save_model(
            model,
            datasets=training_extracts(df, features, split, target=TARGET),
            tache=ARTIFACTS_TACHE,
            niveau=ARTIFACTS_NIVEAU,
            nom=ARTIFACTS_NOM,
            metadonnees={
                "target": TARGET,
                "features": features,
                "sizes": split.tailles,
                "test_metrics": metrics,
                "standardization": standardisation(model, features),
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
        metrics=metrics,
        coefficients=coefficients,
        X_test=split.X_test,
        y_test=split.y_test,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    train_regression_baseline()


if __name__ == "__main__":
    app()
