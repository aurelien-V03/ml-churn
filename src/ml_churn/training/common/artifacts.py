"""Sauvegarde des modeles entraines sur disque.

Les modeles sont versionnes dans `artifacts/<modele>/<date>/`, a cote d'un
fichier de metadonnees. Ce dernier n'est pas cosmetique : un modele serialise seul est
inutilisable si les versions de scikit-learn ou de numpy ont change entre
l'entrainement et le chargement.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas as pd
import sklearn


def _racine_projet() -> Path:
    """Remonte jusqu'au dossier contenant `pyproject.toml`."""
    for dossier in Path(__file__).resolve().parents:
        if (dossier / "pyproject.toml").exists():
            return dossier
    raise RuntimeError("pyproject.toml introuvable : racine du projet inconnue")


ARTIFACTS_DIR = _racine_projet() / "artifacts"


def save_model(
    model: Any,
    *,
    dossier: str,
    nom: str,
    metadonnees: dict[str, Any] | None = None,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Enregistre le modele dans `artifacts/<dossier>/<date>/<nom>.joblib`.

    Retourne le chemin du modele. Un `.json` de meme nom l'accompagne, avec
    les versions des bibliotheques necessaires a son rechargement. Chaque
    entree de `datasets` donne un `<nom>_<jeu>.csv` : l'extrait gold qui a
    servi a l'entrainement, un fichier par jeu de donnees.
    """
    jour = datetime.now().astimezone().date().isoformat()
    cible = ARTIFACTS_DIR / dossier / jour
    cible.mkdir(parents=True, exist_ok=True)

    chemin = cible / f"{nom}.joblib"
    joblib.dump(model, chemin)

    donnees: dict[str, Any] = {}
    for jeu, extrait in (datasets or {}).items():
        fichier = cible / f"{nom}_{jeu}.csv"
        extrait.to_csv(fichier, index=False)
        donnees.setdefault("datasets", {})[jeu] = {
            "file": fichier.name,
            "rows": len(extrait),
            "columns": len(extrait.columns),
        }

    (chemin.with_suffix(".json")).write_text(
        json.dumps(
            {
                "model": nom,
                "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "versions": {
                    "python": f"{__import__('sys').version_info.major}."
                    f"{__import__('sys').version_info.minor}",
                    "scikit-learn": sklearn.__version__,
                    "numpy": numpy.__version__,
                    "joblib": joblib.__version__,
                },
                **(metadonnees or {}),
                **donnees,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    return chemin
