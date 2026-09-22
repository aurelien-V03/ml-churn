"""Recherche du seuil de decision du modele baseline, avec Optuna et MLflow.

Le modele lui-meme ne change pas : seul change le seuil a partir duquel une
probabilite devient une alerte. Ce seuil arbitre entre churners manques et
fausses alertes, arbitrage que l'accuracy ne capture pas.

Trois jeux disjoints : le modele apprend sur le train, le seuil est choisi sur
la validation, et le test ne sert qu'a la mesure finale. Optimiser le seuil sur
le test reviendrait a s'y ajuster, et la performance annoncee serait
surestimee.

Usage :
    uv run python -m ml_churn.training.classification.baseline.classification_baseline_tuning
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix, roc_auc_score

from ml_churn.training.classification.baseline.classification_baseline_training import (
    EXCLUSIONS,
    EXPERIMENT,
    TARGET,
    build_baseline_pipeline,
)
from ml_churn.training.common.data import (
    feature_columns,
    load_gold,
    split_train_validation_test,
)
from ml_churn.training.common.explain import shap_bar_figure, shap_summary_figure
from ml_churn.training.common.metrics import classification_metrics
from ml_churn.training.common.plots import confusion_matrix_figure
from ml_churn.training.common.tracking import mlflow_tracking

# Nom des runs MLflow, suffixe par le seuil teste.
RUN_PREFIX = "classification_base_model_threshold"

# Grille testee : de 0.0 a 1.0 par pas de 0.1, soit 11 seuils.
PAS_SEUIL = 0.1
SEUILS: list[float] = [round(i * PAS_SEUIL, 1) for i in range(int(1 / PAS_SEUIL) + 1)]

# Objectif par defaut : le F1 equilibre precision et rappel sans supposer de
# cout metier. `recall_sous_contrainte` privilegie la detection des churners.
OBJECTIF = "f1"
PRECISION_MINIMALE = 0.60


@dataclass
class Tuning:
    """Seuil retenu et performance associee sur le jeu de test."""

    seuil: float
    score_validation: float
    metrics_test: dict[str, float]
    essais: pd.DataFrame


def _score(metrics: dict[str, float], objectif: str) -> float:
    """Valeur a maximiser pour un jeu de metriques donne."""
    precision, recall = metrics["precision"], metrics["recall"]

    if objectif == "recall_sous_contrainte":
        # Un modele qui alerte sur tout le monde a un rappel parfait : la
        # contrainte de precision evite cette solution degeneree.
        return recall if precision >= PRECISION_MINIMALE else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _matrice_au_seuil(y_true: pd.Series, probabilities, seuil: float) -> np.ndarray:
    return confusion_matrix(y_true, (probabilities >= seuil).astype(int))


def _metrics_au_seuil(
    y_true: pd.Series, probabilities, seuil: float
) -> dict[str, float]:
    return classification_metrics(_matrice_au_seuil(y_true, probabilities, seuil))


def tune_baseline_threshold(*, objectif: str = OBJECTIF, echo: bool = True) -> Tuning:
    """Cherche le seuil optimal et enregistre chaque essai dans MLflow."""
    df = load_gold()
    features = feature_columns(TARGET, EXCLUSIONS)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET])

    split = split_train_validation_test(X, y)

    # Le modele n'a pas vu la validation : les probabilites sur lesquelles le
    # seuil est choisi sont donc hors echantillon.
    model = build_baseline_pipeline().fit(split.X_train, split.y_train)
    proba_validation = model.predict_proba(split.X_validation)[:, 1]

    # L'AUC se calcule sur les probabilites : elle ne depend d'aucun seuil et
    # vaut donc la meme chose dans les onze runs. Elle y figure pour qu'un run
    # porte la performance du modele autant que celle de son seuil.
    auc_validation = roc_auc_score(split.y_validation, proba_validation)

    if echo:
        print(
            f"[RECHERCHE] seuil, objectif '{objectif}', {len(SEUILS)} seuils "
            f"de {SEUILS[0]:g} a {SEUILS[-1]:g} (pas {PAS_SEUIL:g})"
        )
        detail = " / ".join(
            f"{nom.removeprefix('n_')} {taille}"
            for nom, taille in split.tailles.items()
        )
        print(f"  {detail}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    essais: list[dict[str, float]] = []

    def objective(trial: optuna.Trial) -> float:
        seuil = trial.suggest_float("seuil", SEUILS[0], SEUILS[-1], step=PAS_SEUIL)
        matrice = _matrice_au_seuil(split.y_validation, proba_validation, seuil)
        metrics = classification_metrics(matrice)
        score = _score(metrics, objectif)

        # Un run independant par seuil : ils se comparent directement dans
        # MLflow, sans run parent pour les regrouper.
        with mlflow_tracking.run(
            EXPERIMENT,
            f"{RUN_PREFIX}_{seuil:g}",
            params={
                "seuil": seuil,
                "objectif": objectif,
                "precision_minimale": PRECISION_MINIMALE,
                "modele": "LogisticRegression",
                **split.tailles,
            },
            tags={"etape": "tuning", "cible": TARGET},
        ):
            mlflow_tracking.log_metrics(
                {**metrics, "auc": auc_validation, "score": score}
            )
            # La matrice change a chaque seuil : c'est ce que le run illustre.
            mlflow_tracking.log_figure(
                confusion_matrix_figure(
                    matrice,
                    titre=f"Matrice de confusion — validation, seuil {seuil:g}",
                ),
                "matrice_confusion_validation.png",
            )
            _log_visuels_shap(model, split.X_validation, "validation", seuil)

        essais.append({"seuil": seuil, "score": score, **metrics})
        return score

    # GridSampler plutot que TPE : la grille est petite, autant l'explorer
    # exhaustivement plutot que de l'echantillonner.
    etude = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"seuil": SEUILS}),
    )
    etude.optimize(objective, n_trials=len(SEUILS))

    seuil = etude.best_params["seuil"]

    # Le test, jamais utilise jusqu'ici, mesure le couple modele + seuil.
    proba_test = model.predict_proba(split.X_test)[:, 1]
    matrice_test = _matrice_au_seuil(split.y_test, proba_test, seuil)
    metrics_test = classification_metrics(matrice_test)
    metrics_test["auc"] = roc_auc_score(split.y_test, proba_test)

    # Le modele et sa performance sur le test sont rattaches au run du seuil
    # retenu, celui qui sera reutilise.
    with mlflow_tracking.run(
        EXPERIMENT,
        f"{RUN_PREFIX}_{seuil:g}_retenu",
        params={"seuil": seuil, "objectif": objectif, **split.tailles},
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
        _log_visuels_shap(model, split.X_test, "test", seuil)
        mlflow_tracking.log_model(model, "modele", input_example=split.X_train.head(5))

    resultat = Tuning(
        seuil=seuil,
        score_validation=etude.best_value,
        metrics_test=metrics_test,
        essais=pd.DataFrame(essais).sort_values("score", ascending=False),
    )

    if echo:
        _log(resultat, objectif)

    return resultat


def _log_visuels_shap(model, X, jeu: str, seuil: float) -> None:
    """Attache au run en cours les deux lectures des contributions SHAP.

    Le modele est le meme d'un seuil a l'autre : ces figures sont donc
    identiques dans tous les runs. Elles y sont malgre tout, pour qu'un run
    consulte seul porte l'explication du modele qu'il mesure.
    """
    mlflow_tracking.log_figure(
        shap_bar_figure(
            model,
            X,
            titre=f"SHAP — importance globale ({jeu}, seuil {seuil:g})",
            max_features=None,
        ),
        "shap_importance_globale.png",
    )
    mlflow_tracking.log_figure(
        shap_summary_figure(
            model, X, titre=f"SHAP — effet par client ({jeu}, seuil {seuil:g})"
        ),
        "shap_effet_par_client.png",
    )


def _log(resultat: Tuning, objectif: str) -> None:
    print(
        f"\n[SEUIL RETENU] {resultat.seuil:.4f}  "
        f"({objectif} = {resultat.score_validation:.4f} en validation)"
    )

    print("\n[PERFORMANCE] sur le jeu de test, au seuil retenu")
    for nom, valeur in resultat.metrics_test.items():
        print(f"  {nom:<21} {valeur:.4f}")

    print("\n[TOUS LES SEUILS TESTES]")
    print(f"  {'seuil':>7} {'score':>7} {'recall':>8} {'precision':>10} {'FPR':>7}")
    for _, ligne in resultat.essais.sort_values("seuil").iterrows():
        print(
            f"  {ligne['seuil']:>7.4f} {ligne['score']:>7.4f} {ligne['recall']:>8.4f} "
            f"{ligne['precision']:>10.4f} {ligne['false_positive_rate']:>7.4f}"
        )

    print(f"\n  runs enregistres dans {mlflow_tracking.TRACKING_DB} (uv run mlflow ui)")


app = typer.Typer(help=__doc__)


@app.command()
def main(
    objectif: str = typer.Option(OBJECTIF, help="f1 ou recall_sous_contrainte"),
) -> None:
    tune_baseline_threshold(objectif=objectif)


if __name__ == "__main__":
    app()
