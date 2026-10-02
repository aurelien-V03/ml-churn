"""Random forest : recherche des hyperparametres, entrainement et test.

Troisieme modele de classification, apres la regression logistique et XGBoost.
L'interet de la comparaison : la foret et le boosting sont tous deux des
ensembles d'arbres, mais ils se trompent differemment. XGBoost construit ses
arbres en serie, chacun corrigeant les erreurs du precedent, ce qui le rend
puissant mais sensible au surapprentissage. La foret construit des arbres
profonds et independants puis moyenne leurs votes : elle sur-apprend peu, au
prix d'un biais plus eleve.

Meme protocole que pour XGBoost, pour que les resultats se comparent : trois
jeux disjoints, seuil de decision inclus dans la recherche, objectif F2, et
`TPESampler` avec la meme graine.

Usage :
    uv run python -m ml_churn.training.classification.final.classification_randomforest_tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml_churn.training.classification.final.classification_xgboost_training import (
    EXCLUSIONS,
    TARGET,
    prepare_split,
)
from ml_churn.training.common.artifacts import save_model
from ml_churn.training.common.data import training_extracts
from ml_churn.training.common.explain import log_shap_figures
from ml_churn.training.common.logs import log_carbon_footprint, log_objective
from ml_churn.training.common.metrics import (
    OBJECTIFS,
    classification_metrics,
    confusion_at_threshold,
    objective_score,
    rank_metrics,
)
from ml_churn.training.common.plots import confusion_matrix_figure
from ml_churn.training.common.tracking import mlflow_tracking
from ml_churn.training.common.tracking.carbon import track_emissions

# Experience MLflow propre a ce modele : ses runs ne se melangent pas a ceux
# de XGBoost, mais restent comparables metrique par metrique.
EXPERIMENT = "classification-randomforest"
RUN_PREFIX = "classification_randomforest_trial"

# Memes valeurs que pour XGBoost : la comparaison n'aurait pas de sens sinon.
OBJECTIF = "f2"
N_ESSAIS = 30
GRAINE = 42

# Sous-dossier d'`artifacts/` et prefixe des fichiers enregistres.
ARTIFACTS_TACHE = "classification"
ARTIFACTS_NIVEAU = "final"
ARTIFACTS_NOM = "classification_randomforest"

# Hyperparametres fixes, hors recherche.
HYPERPARAMETRES: dict[str, Any] = {
    # La classe positive pese 28 % : sans reponderation, la foret se contente
    # de la majorite. Equivalent du `scale_pos_weight` de XGBoost.
    "class_weight": "balanced",
    "random_state": GRAINE,
    "n_jobs": -1,
}

# Colonnes du suivi essai par essai.
EN_TETE_ESSAIS = (
    f"{'essai':>7} {'score':>8} {'seuil':>7} {'recall':>8} {'precision':>10} "
    f"{'FPR':>7} {'roc_auc':>8} {'pr_auc':>8}"
)


@dataclass
class Tuning:
    """Hyperparametres retenus et performance associee sur le jeu de test."""

    seuil: float
    hyperparametres: dict[str, Any]
    score_validation: float
    metrics_test: dict[str, float]
    essais: pd.DataFrame
    chemin: Path | None = None
    # Energie et CO2 de la recherche, remplis par `tune_randomforest`.
    empreinte: dict[str, float] = field(default_factory=dict)


def build_randomforest_pipeline(
    hyperparametres: dict[str, Any] | None = None,
) -> Pipeline:
    """La foret seule : la couche gold est deja numerique et complete.

    Pas de standardisation : un arbre decoupe sur des seuils, l'echelle des
    variables ne change donc ni la structure ni les predictions.
    """
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    **{**HYPERPARAMETRES, **(hyperparametres or {})}
                ),
            )
        ]
    )


def espace_de_recherche(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparametres tires par Optuna a chaque essai.

    Ils se repartissent en deux familles : la taille de la foret, et les
    contraintes qui limitent la croissance de chaque arbre. `max_features`
    est le parametre propre a la foret -- c'est lui qui decorrele les arbres
    entre eux, et donc ce qui fait la difference avec un simple bagging.
    """
    return {
        # Taille de la foret : plus d'arbres ne sur-apprend pas, mais coute.
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        # Croissance des arbres. `None` les laisse pousser jusqu'au bout, ce
        # qui est le reglage historique de Breiman.
        "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # Nombre de features tirees a chaque decoupage : le coeur de la foret.
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]
        ),
        # Sans bootstrap, chaque arbre voit tout le train : ils se ressemblent.
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        # Elagage par complexite, la regularisation la plus directe.
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
    }


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
        f"{metrics['false_positive_rate']:>7.2f} "
        f"{metrics['roc_auc']:>8.2f} {metrics['pr_auc']:>8.2f}"
        f"{'  <- meilleur' if progres else ''}"
    )


def tune_randomforest(
    *,
    objectif: str = OBJECTIF,
    n_essais: int = N_ESSAIS,
    enregistrer: bool = True,
    echo: bool = True,
) -> Tuning:
    """Recherche, reentrainement et mesure sur le test, en un appel.

    L'empreinte carbone porte sur la recherche entiere -- une trentaine
    d'entrainements enchaines -- et non sur chacun : un `fit` isole est trop
    bref pour que codecarbon lui attribue une energie fiable.
    """
    with track_emissions("recherche-randomforest") as empreinte:
        resultat = _rechercher(
            objectif=objectif, n_essais=n_essais, enregistrer=enregistrer, echo=echo
        )

    resultat.empreinte = empreinte
    if echo:
        log_carbon_footprint(empreinte, titre="de la recherche complete")

    return resultat


def _rechercher(
    *, objectif: str, n_essais: int, enregistrer: bool, echo: bool
) -> Tuning:
    """Explore l'espace des hyperparametres et enregistre chaque essai dans MLflow."""
    df, features, split = prepare_split()

    if echo:
        print(f"[RECHERCHE] hyperparametres RandomForest, {n_essais} essais")
        detail = " / ".join(
            f"{nom.removeprefix('n_')} {taille}"
            for nom, taille in split.tailles.items()
        )
        print(f"  {detail}  ({len(features)} features, class_weight balanced)")
        log_objective(objectif)
        print(f"\n  {EN_TETE_ESSAIS}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    essais: list[dict[str, float]] = []
    # Suivi du meilleur score pour signaler les essais qui font progresser.
    meilleur = {"score": float("-inf")}

    def objective(trial: optuna.Trial) -> float:
        hyperparametres = espace_de_recherche(trial)
        seuil = trial.suggest_float("seuil", 0.1, 0.9, step=0.05)

        model = build_randomforest_pipeline(hyperparametres)
        model.fit(split.X_train, split.y_train)

        proba = model.predict_proba(split.X_validation)[:, 1]
        matrice = confusion_at_threshold(split.y_validation, proba, seuil)
        metrics = classification_metrics(matrice) | rank_metrics(
            split.y_validation, proba
        )
        score = objective_score(metrics, objectif)

        with mlflow_tracking.run(
            EXPERIMENT,
            f"{RUN_PREFIX}_{trial.number:02d}",
            params={
                **hyperparametres,
                "seuil": seuil,
                "objectif": objectif,
                "modele": "RandomForestClassifier",
                **split.tailles,
            },
            tags={"etape": "tuning", "cible": TARGET},
        ):
            mlflow_tracking.log_metrics(
                {
                    **metrics,
                    "score F2": objective_score(metrics, "f2"),
                    "score": score,
                }
            )
            mlflow_tracking.log_figure(
                confusion_matrix_figure(
                    matrice,
                    titre=f"Matrice de confusion — validation, essai {trial.number}",
                ),
                "matrice_confusion_validation.png",
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
        sampler=optuna.samplers.TPESampler(seed=GRAINE),
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
    model = build_randomforest_pipeline(retenus)
    model.fit(split.X_train, split.y_train)

    proba_test = model.predict_proba(split.X_test)[:, 1]
    matrice_test = confusion_at_threshold(split.y_test, proba_test, seuil)
    metrics_test = classification_metrics(matrice_test) | rank_metrics(
        split.y_test, proba_test
    )

    with mlflow_tracking.run(
        EXPERIMENT,
        f"{RUN_PREFIX}_{etude.best_trial.number:02d}_retenu",
        params={**retenus, "seuil": seuil, "objectif": objectif, **split.tailles},
        tags={"etape": "retenu", "cible": TARGET},
    ):
        mlflow_tracking.log_metrics(
            {
                "score_validation": etude.best_value,
                "test score F2": objective_score(metrics_test, "f2"),
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
        mlflow_tracking.log_model(model, "modele", input_example=split.X_train.head(5))

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
                "exclusions": sorted(EXCLUSIONS),
                "threshold": seuil,
                "objective": objectif,
                "hyperparameters": retenus,
                "sizes": split.tailles,
                "validation_score": etude.best_value,
                "test_metrics": metrics_test,
            },
        )
        if enregistrer
        else None
    )

    resultat = Tuning(
        seuil=seuil,
        hyperparametres=retenus,
        score_validation=etude.best_value,
        metrics_test=metrics_test,
        essais=pd.DataFrame(essais).sort_values("score", ascending=False),
        chemin=chemin,
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
            f"{ligne['precision']:>10.2f} {ligne['false_positive_rate']:>7.2f} "
            f"{ligne['roc_auc']:>8.2f} {ligne['pr_auc']:>8.2f}"
        )

    if resultat.chemin is not None:
        print(f"\n  modele enregistre dans {resultat.chemin}")
    print(f"  runs enregistres dans {mlflow_tracking.TRACKING_DB} (uv run mlflow ui)")


app = typer.Typer(help=__doc__)


@app.command()
def main(
    objectif: str = typer.Option(OBJECTIF, help=f"un de {OBJECTIFS}"),
    n_essais: int = typer.Option(N_ESSAIS, help="nombre d'essais Optuna"),
) -> None:
    tune_randomforest(objectif=objectif, n_essais=n_essais)


if __name__ == "__main__":
    app()
