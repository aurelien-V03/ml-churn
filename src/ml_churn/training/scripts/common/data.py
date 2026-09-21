"""Lecture de la couche gold et selection des features."""

from __future__ import annotations

import pandas as pd

from ml_churn.ingestion.db import get_engine
from ml_churn.ingestion.models import ChurnSaasGold

# Colonnes jamais utilisables comme features, quel que soit le modele.
EXCLUSIONS_COMMUNES: dict[str, str] = {
    "client_id": "identifiant",
    "date_souscription": "date, non exploitable telle quelle",
    # Categorielles brutes : leur encodage one-hot est utilise a la place.
    "jour_souscription": "encodee en one-hot",
    "secteur": "encodee en one-hot",
    "pays": "encodee en one-hot",
    "taille_entreprise": "encodee en one-hot",
    "plan": "encodee en one-hot",
    "couleur_theme_interface": "encodee en one-hot",
    "code_datacenter": "encodee en one-hot",
    "groupe_experimentation": "encodee en one-hot",
    "polarite_csm": "encodee en one-hot",
    "niveau_anciennete": "encodee en one-hot",
    # Fuite de donnees : connue seulement apres la periode observee.
    "sante_compte_fin_periode": "fuite de donnees",
}


def load_gold() -> pd.DataFrame:
    """Lit la table gold des clients."""
    table = ChurnSaasGold.__table__.fullname
    return pd.read_sql(f"select * from {table}", get_engine())


def feature_columns(target: str, exclusions: dict[str, str]) -> list[str]:
    """Colonnes de gold retenues comme features, dans l'ordre du modele."""
    return [
        colonne.key
        for colonne in ChurnSaasGold.__table__.columns
        if colonne.key != target
        and colonne.key not in exclusions
        and not colonne.key.startswith("_")
    ]
