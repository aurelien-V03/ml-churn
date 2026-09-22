"""Recherche des hyperparametres de XGBoost, avec Optuna et MLflow.

Contrairement a la baseline ou seul le seuil variait, chaque essai reentraine
un modele : les hyperparametres changent la structure des arbres, donc le
modele lui-meme. Le seuil de decision fait partie de la recherche, puisque
c'est lui qui transforme une probabilite en alerte.

Trois jeux disjoints : les arbres apprennent sur le train, les hyperparametres
sont choisis sur la validation, et le test ne sert qu'a la mesure finale.

`TPESampler` plutot que `GridSampler` : l'espace de recherche est continu et
trop vaste pour etre explore exhaustivement, l'echantillonnage bayesien
concentre les essais sur les zones prometteuses.

Usage :
    uv run python -m ml_churn.training.classification.final.classification_xgboost_tuning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
import typer
from sklearn.metrics import roc_auc_score

from ml_churn.training.classification.final.classification_xgboost_training import (
    EXPERIMENT,
    TARGET,
    build_xgboost_pipeline,
    poids_classe_positive,
    prepare_split,
)
from ml_churn.training.common.explain import log_shap_figures
from ml_churn.training.common.metrics import (
    classification_metrics,
    confusion_at_threshold,
    objective_score,
)
from ml_churn.training.common.plots import confusion_matrix_figure
from ml_churn.training.common.tracking import mlflow_tracking

# Nom des runs MLflow, suffixe par le numero d'essai.
RUN_PREFIX = "classification_xgboost_trial"

# Objectif par defaut : le F1 equilibre precision et rappel sans supposer de
# cout metier. `recall_sous_contrainte` privilegie la detection des churners.
OBJECTIF = "f1"

# Nombre d'essais : chacun reentraine un modele, le cout n'est plus negligeable.
N_ESSAIS = 30

# Types que skops doit accepter pour serialiser le pipeline dans MLflow.
TYPES_XGBOOST = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


# Colonnes du suivi essai par essai.
EN_TETE_ESSAIS = (
    f"{'essai':>7} {'score':>8} {'seuil':>7} {'recall':>8} {'precision':>10} {'auc':>7}"
)


def _ligne_essai(
    numero: int,
    total: int,
    seuil: float,
    score: float,
    metrics: dict[str, float],
    progres: bool,
) -> str:
    """Une ligne par essai, marquee quand elle ameliore le meilleur score."""
    return (
        f"  {f'{numero + 1}/{total}':>7} {score:>8.2f} {seuil:>7.2f} "
        f"{metrics['recall']:>8.2f} {metrics['precision']:>10.2f} "
        f"{metrics['auc']:>7.2f}"
        f"{'  <- meilleur' if progres else ''}"
    )


@dataclass
class Tuning:
    """Hyperparametres retenus et performance associee sur le jeu de test."""

    seuil: float
    hyperparametres: dict[str, Any]
    score_validation: float
    metrics_test: dict[str, float]
    essais: pd.DataFrame


def espace_de_recherche(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparametres tires par Optuna a chaque essai.

    Le catalogue complet de XGBoost est bien plus large ; ces neuf-la sont
    ceux qui pesent sur un jeu tabulaire de cette taille. Les autres
    (`booster`, `max_delta_step`, `colsample_bylevel`...) restent a leur
    valeur par defaut.
    """
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


def tune_xgboost(
    *, objectif: str = OBJECTIF, n_essais: int = N_ESSAIS, echo: bool = True
) -> Tuning:
    """Cherche les meilleurs hyperparametres et enregistre chaque essai dans MLflow."""
    _, features, split = prepare_split()
    poids = poids_classe_positive(split.y_train)

    if echo:
        print(
            f"[RECHERCHE] hyperparametres XGBoost, objectif '{objectif}', "
            f"{n_essais} essais"
        )
        detail = " / ".join(
            f"{nom.removeprefix('n_')} {taille}"
            for nom, taille in split.tailles.items()
        )
        print(f"  {detail}  ({len(features)} features, scale_pos_weight {poids:.2f})")
        print(f"\n  {EN_TETE_ESSAIS}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    essais: list[dict[str, float]] = []
    # Suivi du meilleur score pour signaler les essais qui font progresser.
    meilleur = {"score": float("-inf")}

    def objective(trial: optuna.Trial) -> float:
        hyperparametres = espace_de_recherche(trial)
        seuil = trial.suggest_float("seuil", 0.1, 0.9, step=0.05)

        model = build_xgboost_pipeline(hyperparametres, scale_pos_weight=poids)
        model.fit(split.X_train, split.y_train)

        proba = model.predict_proba(split.X_validation)[:, 1]
        matrice = confusion_at_threshold(split.y_validation, proba, seuil)
        metrics = classification_metrics(matrice)
        metrics["auc"] = roc_auc_score(split.y_validation, proba)
        score = objective_score(metrics, objectif)

        # Un run independant par essai : ils se comparent directement dans
        # MLflow, sans run parent pour les regrouper.
        with mlflow_tracking.run(
            EXPERIMENT,
            f"{RUN_PREFIX}_{trial.number:02d}",
            params={
                **hyperparametres,
                "seuil": seuil,
                "objectif": objectif,
                "modele": "XGBClassifier",
                "scale_pos_weight": round(poids, 4),
                **split.tailles,
            },
            tags={"etape": "tuning", "cible": TARGET},
        ):
            mlflow_tracking.log_metrics({**metrics, "score": score})
            mlflow_tracking.log_figure(
                confusion_matrix_figure(
                    matrice,
                    titre=f"Matrice de confusion — validation, essai {trial.number}",
                ),
                "matrice_confusion_validation.png",
            )
            # Le modele change a chaque essai : ces figures aussi.
            log_shap_figures(
                model,
                split.X_validation,
                contexte=f"validation, essai {trial.number + 1}",
            )

        essais.append(
            {"essai": trial.number, "seuil": seuil, "score": score, **metrics}
        )

        progres = score > meilleur["score"]
        meilleur["score"] = max(meilleur["score"], score)
        if echo:
            print(_ligne_essai(trial.number, n_essais, seuil, score, metrics, progres))

        return score

    etude = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    etude.optimize(objective, n_trials=n_essais)

    retenus = dict(etude.best_params)
    seuil = retenus.pop("seuil")

    if echo:
        print(
            f"\n[REENTRAINEMENT] essai {etude.best_trial.number + 1} retenu, "
            "modele reconstruit sur le train"
        )

    # Le modele retenu est reentraine, puis mesure sur le test jamais utilise.
    model = build_xgboost_pipeline(retenus, scale_pos_weight=poids)
    model.fit(split.X_train, split.y_train)

    proba_test = model.predict_proba(split.X_test)[:, 1]
    matrice_test = confusion_at_threshold(split.y_test, proba_test, seuil)
    metrics_test = classification_metrics(matrice_test)
    metrics_test["auc"] = roc_auc_score(split.y_test, proba_test)

    with mlflow_tracking.run(
        EXPERIMENT,
        f"{RUN_PREFIX}_{etude.best_trial.number:02d}_retenu",
        params={**retenus, "seuil": seuil, "objectif": objectif, **split.tailles},
        tags={"etape": "retenu", "cible": TARGET},
    ):
        mlflow_tracking.log_metrics(
            {
                "score_validation": etude.best_value,
                **{f"test_{nom}": valeur for nom, valeur in metrics_test.items()},
            }
        )
        mlflow_tracking.log_figure(
            confusion_matrix_figure(
                matrice_test, titre=f"Matrice de confusion — test, seuil {seuil:g}"
            ),
            "matrice_confusion_test.png",
        )
        log_shap_figures(model, split.X_test, contexte=f"test, seuil {seuil:g}")
        mlflow_tracking.log_model(
            model,
            "modele",
            input_example=split.X_train.head(5),
            trusted_types=TYPES_XGBOOST,
        )

    resultat = Tuning(
        seuil=seuil,
        hyperparametres=retenus,
        score_validation=etude.best_value,
        metrics_test=metrics_test,
        essais=pd.DataFrame(essais).sort_values("score", ascending=False),
    )

    if echo:
        _log(resultat, objectif)

    return resultat


def _log(resultat: Tuning, objectif: str) -> None:
    print(
        f"\n[HYPERPARAMETRES RETENUS] {objectif} = "
        f"{resultat.score_validation:.2f} en validation"
    )
    print(f"  seuil {resultat.seuil:.2f}")
    for nom, valeur in resultat.hyperparametres.items():
        affichage = f"{valeur:.2f}" if isinstance(valeur, float) else str(valeur)
        print(f"  {nom:<20} {affichage}")

    print("\n[PERFORMANCE] sur le jeu de test, au seuil retenu")
    for nom, valeur in resultat.metrics_test.items():
        print(f"  {nom:<21} {valeur:.2f}")

    print("\n[MEILLEURS ESSAIS] les 5 premiers")
    print(f"  {EN_TETE_ESSAIS}")
    for _, ligne in resultat.essais.head(5).iterrows():
        print(
            f"  {int(ligne['essai']) + 1:>7} {ligne['score']:>8.2f} "
            f"{ligne['seuil']:>7.2f} {ligne['recall']:>8.2f} "
            f"{ligne['precision']:>10.2f} {ligne['auc']:>7.2f}"
        )

    print(f"\n  runs enregistres dans {mlflow_tracking.TRACKING_DB} (uv run mlflow ui)")


app = typer.Typer(help=__doc__)


@app.command()
def main(
    objectif: str = typer.Option(OBJECTIF, help="f1 ou recall_sous_contrainte"),
    n_essais: int = typer.Option(N_ESSAIS, help="nombre d'essais Optuna"),
) -> None:
    tune_xgboost(objectif=objectif, n_essais=n_essais)


if __name__ == "__main__":
    app()
