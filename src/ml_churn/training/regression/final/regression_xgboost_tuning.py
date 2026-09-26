"""Recherche des hyperparametres de XGBoost pour la valeur vie client.

Meme principe que la recherche cote classification : chaque essai reentraine un
modele, les hyperparametres sont choisis sur la validation, et le test ne sert
qu'a la mesure finale.

L'objectif par defaut est la MAE et non le R2. Le R2 derive de l'erreur au
carre, dominee par les quelques comptes a plusieurs centaines de milliers
d'euros : l'optimiser revient a soigner les gros clients et a abandonner les
autres. La MAE traite chaque euro d'erreur a egalite.

Usage :
    uv run python -m ml_churn.training.regression.final.regression_xgboost_tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna
import pandas as pd
import typer

from ml_churn.training.common.data import (
    feature_columns,
    load_gold,
    split_train_validation_test,
    strates_quantiles,
)
from ml_churn.training.common.logs import log_carbon_footprint
from ml_churn.training.common.metrics import DIRECTIONS_REGRESSION, regression_metrics
from ml_churn.training.common.tracking import mlflow_tracking
from ml_churn.training.common.tracking.carbon import track_emissions
from ml_churn.training.regression.baseline.regression_baseline_training import (
    EXCLUSIONS,
    TARGET,
)
from ml_churn.training.regression.final.regression_xgboost_training import (
    build_xgboost_regression_pipeline,
)

EXPERIMENT = "regression-xgboost"

# Nom des runs MLflow, suffixe par le numero d'essai.
RUN_PREFIX = "regression_xgboost_trial"

# Metrique optimisee. Voir `DIRECTIONS_REGRESSION` pour les valeurs admises.
OBJECTIF = "mae"

# Nombre d'essais : chacun reentraine un modele.
N_ESSAIS = 30

# Types que skops doit accepter pour serialiser le pipeline dans MLflow.
TYPES_XGBOOST = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]

# Colonnes du suivi essai par essai.
EN_TETE_ESSAIS = f"{'essai':>7} {'r2':>7} {'mae':>11} {'rmse':>12} {'mape':>8}"


def espace_de_recherche(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparametres tires par Optuna a chaque essai."""
    return {
        # Capacite du modele.
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # Echantillonnage : deux formes de regularisation par le hasard.
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        # Contraintes sur les decoupages.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        # Penalites L1 et L2 sur les poids des feuilles.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


@dataclass
class Tuning:
    """Hyperparametres retenus et erreurs associees sur le jeu de test."""

    hyperparametres: dict[str, Any]
    score_validation: float
    metrics_test: dict[str, float]
    essais: pd.DataFrame
    empreinte: dict[str, float] = field(default_factory=dict)


def _ligne_essai(
    numero: int, total: int, metrics: dict[str, float], progres: bool
) -> str:
    """Une ligne par essai, marquee quand elle ameliore le meilleur score."""
    return (
        f"  {f'{numero + 1}/{total}':>7} {metrics['r2']:>7.2f} {metrics['mae']:>11,.0f} "
        f"{metrics['rmse']:>12,.0f} {metrics['mape']:>7.0f}%"
        f"{'  <- meilleur' if progres else ''}"
    )


def tune_regression_xgboost(
    *, objectif: str = OBJECTIF, n_essais: int = N_ESSAIS, echo: bool = True
) -> Tuning:
    """Cherche les hyperparametres, en mesurant l'empreinte carbone de la recherche."""
    with track_emissions("recherche-regression-xgboost") as empreinte:
        resultat = _rechercher(objectif=objectif, n_essais=n_essais, echo=echo)

    resultat.empreinte = empreinte
    if echo:
        log_carbon_footprint(empreinte, titre="de la recherche complete")

    return resultat


def _rechercher(*, objectif: str, n_essais: int, echo: bool) -> Tuning:
    """Explore l'espace des hyperparametres et enregistre chaque essai dans MLflow."""
    if objectif not in DIRECTIONS_REGRESSION:
        raise ValueError(
            f"objectif inconnu : {objectif!r}, attendu {tuple(DIRECTIONS_REGRESSION)}"
        )

    df = load_gold()
    features = feature_columns(TARGET, EXCLUSIONS)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])
    # Stratifier sur les deciles de la cible : chaque jeu recoit sa part de
    # petits et de gros comptes, ce qu'une cible continue ne permet pas
    # directement.
    split = split_train_validation_test(X, y, stratify=strates_quantiles(y))

    direction = DIRECTIONS_REGRESSION[objectif]
    if echo:
        print(f"[RECHERCHE] hyperparametres XGBoost, {n_essais} essais")
        detail = " / ".join(
            f"{nom.removeprefix('n_')} {taille}"
            for nom, taille in split.tailles.items()
        )
        print(f"  {detail}  ({len(features)} features)")
        print(f"\n[OBJECTIF] {objectif} a {direction}")
        print(f"\n  {EN_TETE_ESSAIS}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    essais: list[dict[str, float]] = []
    meilleur = {"score": float("inf") if direction == "minimize" else float("-inf")}

    def objective(trial: optuna.Trial) -> float:
        hyperparametres = espace_de_recherche(trial)

        model = build_xgboost_regression_pipeline(hyperparametres)
        model.fit(split.X_train, split.y_train)
        metrics = regression_metrics(
            split.y_validation, model.predict(split.X_validation)
        )
        score = metrics[objectif]

        with mlflow_tracking.run(
            EXPERIMENT,
            f"{RUN_PREFIX}_{trial.number:02d}",
            params={
                **hyperparametres,
                "objectif": objectif,
                "modele": "XGBRegressor",
                **split.tailles,
            },
            tags={"etape": "tuning", "cible": TARGET},
        ):
            mlflow_tracking.log_metrics(metrics)

        essais.append({"essai": trial.number, **metrics})

        progres = (
            score < meilleur["score"]
            if direction == "minimize"
            else score > meilleur["score"]
        )
        if progres:
            meilleur["score"] = score
        if echo:
            print(_ligne_essai(trial.number, n_essais, metrics, progres))

        return score

    etude = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
    )
    etude.optimize(objective, n_trials=n_essais)

    retenus = dict(etude.best_params)
    if echo:
        print(
            f"\n[REENTRAINEMENT] essai {etude.best_trial.number + 1} retenu, "
            "modele reconstruit sur le train"
        )

    model = build_xgboost_regression_pipeline(retenus)
    model.fit(split.X_train, split.y_train)
    metrics_test = regression_metrics(split.y_test, model.predict(split.X_test))

    with mlflow_tracking.run(
        EXPERIMENT,
        f"{RUN_PREFIX}_{etude.best_trial.number:02d}_retenu",
        params={**retenus, "objectif": objectif, **split.tailles},
        tags={"etape": "retenu", "cible": TARGET},
    ):
        mlflow_tracking.log_metrics(
            {
                "score_validation": etude.best_value,
                **{f"test_{nom}": valeur for nom, valeur in metrics_test.items()},
            }
        )
        mlflow_tracking.log_model(
            model,
            "modele",
            input_example=split.X_train.head(5),
            trusted_types=TYPES_XGBOOST,
        )

    resultat = Tuning(
        hyperparametres=retenus,
        score_validation=etude.best_value,
        metrics_test=metrics_test,
        essais=pd.DataFrame(essais).sort_values(
            objectif, ascending=direction == "minimize"
        ),
    )

    if echo:
        _log(resultat, objectif)

    return resultat


def _log(resultat: Tuning, objectif: str) -> None:
    print(
        f"\n[HYPERPARAMETRES RETENUS] {objectif} = "
        f"{resultat.score_validation:,.2f} en validation"
    )
    for nom, valeur in resultat.hyperparametres.items():
        affichage = f"{valeur:.2f}" if isinstance(valeur, float) else str(valeur)
        print(f"  {nom:<20} {affichage}")

    print("\n[PERFORMANCE] sur le jeu de test")
    for nom, valeur in resultat.metrics_test.items():
        print(f"  {nom:<21} {valeur:>12,.2f}")

    print("\n[MEILLEURS ESSAIS] les 5 premiers")
    print(f"  {EN_TETE_ESSAIS}")
    for _, ligne in resultat.essais.head(5).iterrows():
        print(
            f"  {int(ligne['essai']) + 1:>7} {ligne['r2']:>7.2f} {ligne['mae']:>11,.0f} "
            f"{ligne['rmse']:>12,.0f} {ligne['mape']:>7.0f}%"
        )

    print(f"\n  runs enregistres dans {mlflow_tracking.TRACKING_DB} (uv run mlflow ui)")


app = typer.Typer(help=__doc__)


@app.command()
def main(
    objectif: str = typer.Option(OBJECTIF, help="r2, mae, rmse ou mape"),
    n_essais: int = typer.Option(N_ESSAIS, help="nombre d'essais Optuna"),
) -> None:
    tune_regression_xgboost(objectif=objectif, n_essais=n_essais)


if __name__ == "__main__":
    app()
