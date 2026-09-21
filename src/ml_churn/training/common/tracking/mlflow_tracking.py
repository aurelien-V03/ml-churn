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
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield actif


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Enregistre les metriques du run en cours."""
    mlflow.log_metrics(
        {nom: float(valeur) for nom, valeur in metrics.items()}, step=step
    )


def log_figure(figure: Any, nom: str) -> None:
    """Enregistre une figure matplotlib comme artefact du run en cours."""
    mlflow.log_figure(figure, nom)


def log_model(model: Any, nom: str, *, input_example: Any = None) -> None:
    """Enregistre le modele entraine comme artefact du run en cours."""
    mlflow.sklearn.log_model(model, name=nom, input_example=input_example)
