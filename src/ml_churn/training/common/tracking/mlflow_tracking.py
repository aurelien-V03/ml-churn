"""Enregistrement des runs dans MLflow.

Commun a tous les modeles : chaque entrainement ou recherche
d'hyperparametres passe par `run`, qui garantit la meme structure
d'experimentation (parametres, metriques, tags) d'un modele a l'autre.

Le suivi est stocke dans `mlflow.db` (SQLite) a la racine du projet, et les
artefacts dans `mlartifacts/`. Depuis MLflow 3, le backend fichier (`mlruns/`)
est en maintenance et refuse les nouvelles experimentations.

Pour consulter les runs :
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Doit preceder l'import de mlflow : le hint de l'agent est emis au chargement.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib.pyplot as plt
import mlflow


def _racine_projet() -> Path:
    """Remonte jusqu'au dossier contenant `pyproject.toml`.

    Un simple `parents[n]` casserait silencieusement au moindre deplacement du
    module dans l'arborescence, en pointant vers un dossier voisin.
    """
    for dossier in Path(__file__).resolve().parents:
        if (dossier / "pyproject.toml").exists():
            return dossier
    raise RuntimeError("pyproject.toml introuvable : racine du projet inconnue")


PROJECT_ROOT = _racine_projet()
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

# Nombre de decimales des metriques enregistrees.
DECIMALES = 2
ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"


def configure(experiment: str) -> None:
    """Pointe MLflow vers le stockage local du projet.

    Une experimentation supprimee depuis l'interface reste en base, en etat
    `deleted`, et MLflow refuse alors de l'activer. On la restaure plutot que
    d'echouer : la suppression depuis l'interface ne doit pas casser les
    scripts.
    """
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")

    existante = mlflow.get_experiment_by_name(experiment)
    if existante is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
    elif existante.lifecycle_stage == "deleted":
        mlflow.MlflowClient().restore_experiment(existante.experiment_id)

    mlflow.set_experiment(experiment)


@contextmanager
def run(
    experiment: str,
    run_name: str,
    *,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun]:
    """Ouvre un run MLflow, avec ses parametres et ses tags.

    `nested=True` rattache le run a celui en cours : c'est ce que fait Optuna
    pour chaque essai, sous un run parent qui porte la recherche complete.
    """
    configure(experiment)

    with mlflow.start_run(run_name=run_name, nested=nested) as actif:
        if params:
            mlflow.log_params(_arrondir(params))
        if tags:
            mlflow.set_tags(tags)
        yield actif


def _arrondir(params: dict[str, Any]) -> dict[str, Any]:
    """Arrondit les parametres flottants, les autres types passent tels quels.

    Un hyperparametre tire sur une echelle logarithmique peut tomber sous le
    centieme : il est alors enregistre comme 0.0, la valeur exacte restant
    dans l'objet Optuna.
    """
    return {
        nom: round(valeur, DECIMALES) if isinstance(valeur, float) else valeur
        for nom, valeur in params.items()
    }


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Enregistre les metriques du run en cours, arrondies.

    L'arrondi ne concerne que ce qui est stocke dans MLflow : les decisions
    (choix du seuil, comparaison des essais) se prennent sur les valeurs
    completes, en memoire.
    """
    mlflow.log_metrics(
        {nom: round(float(valeur), DECIMALES) for nom, valeur in metrics.items()},
        step=step,
    )


def log_figure(figure: Any, nom: str) -> None:
    """Enregistre une figure matplotlib comme artefact du run en cours.

    La figure est fermee ensuite : une fois dans MLflow elle n'a plus d'usage,
    et une figure laissee ouverte reste la figure courante de matplotlib, dans
    laquelle les graphiques traces plus tard viendraient se superposer.
    """
    mlflow.log_figure(figure, nom)
    plt.close(figure)


def log_model(
    model: Any,
    nom: str,
    *,
    input_example: Any = None,
    trusted_types: list[str] | None = None,
) -> None:
    """Enregistre le modele entraine comme artefact du run en cours.

    MLflow serialise les pipelines scikit-learn avec skops, qui refuse par
    defaut les types venus d'autres bibliotheques. `trusted_types` autorise
    ceux du modele enregistre -- `xgboost.sklearn.XGBClassifier`, par exemple.
    """
    mlflow.sklearn.log_model(
        model,
        name=nom,
        input_example=input_example,
        skops_trusted_types=trusted_types,
    )
