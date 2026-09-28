"""Lecture de la couche gold et selection des features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_churn.ingestion.db import get_engine
from ml_churn.ingestion.models import ChurnSaasGold, ChurnSaasSilver

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
    # Colonne de decoupage, pas une caracteristique du client.
    "jeu": "appartenance train / validation / test",
}


def load_gold() -> pd.DataFrame:
    """Lit la table gold des clients."""
    table = ChurnSaasGold.__table__.fullname
    return pd.read_sql(f"select * from {table}", get_engine())


def load_silver() -> pd.DataFrame:
    """Lit la table silver des clients."""
    table = ChurnSaasSilver.__table__.fullname
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


# Colonne portant l'appartenance de chaque ligne, posee en couche silver.
COLONNE_JEU = "jeu"

# Valeurs de cette colonne, dans l'ordre des attributs de `Split`.
JEUX: dict[str, str] = {"train": "TRAIN", "validation": "VAL", "test": "TEST"}

# Graine des modeles. Le decoupage, lui, ne depend plus d'elle.
RANDOM_STATE = 42


@dataclass(frozen=True)
class Split:
    """Les trois jeux, chacun avec son role.

    train      : le modele y apprend ses coefficients.
    validation : on y choisit les hyperparametres (dont le seuil).
    test       : mesure finale, utilise une seule fois.
    """

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    @property
    def tailles(self) -> dict[str, int]:
        return {
            "n_train": len(self.X_train),
            "n_validation": len(self.X_validation),
            "n_test": len(self.X_test),
        }


def split_par_jeu(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Split:
    """Decoupe selon la colonne `jeu`, posee lors de l'ingestion silver.

    Le partage ne se rejoue plus a l'entrainement : il est fige dans la donnee,
    stratifie sur le churn croise aux deciles de valeur vie client. Tous les
    modeles voient donc exactement les memes clients, quelle que soit leur
    cible, et un extrait exporte reste interpretable sans rejouer le tirage.
    """
    manquantes = set(JEUX.values()) - set(df[COLONNE_JEU].dropna().unique())
    if manquantes:
        raise ValueError(
            f"jeux absents de la couche gold : {sorted(manquantes)} "
            "-- relancer l'ingestion silver puis gold"
        )

    lignes = {
        role: df.index[df[COLONNE_JEU] == valeur] for role, valeur in JEUX.items()
    }

    return Split(
        **{f"X_{role}": X.loc[index] for role, index in lignes.items()},
        **{f"y_{role}": y.loc[index] for role, index in lignes.items()},
    )


def training_extracts(
    df: pd.DataFrame, features: list[str], split: Split, *, target: str
) -> dict[str, pd.DataFrame]:
    """Extrait gold reellement consomme, un DataFrame par jeu de donnees.

    Ecrire les trois jeux separement fige le decoupage : sans cela, rejouer
    l'entrainement depuis l'extrait supposerait que `train_test_split` decoupe
    toujours a l'identique, ce qui n'est vrai qu'a version de scikit-learn
    constante. `client_id` sert a remonter a la ligne source, pas a predire.
    """
    colonnes = [colonne for colonne in ("client_id", target) if colonne in df.columns]

    return {
        jeu: df.loc[indices, [*colonnes, *features]]
        for jeu, indices in (
            ("train", split.X_train.index),
            ("validation", split.X_validation.index),
            ("test", split.X_test.index),
        )
    }
