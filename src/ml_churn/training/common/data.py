"""Lecture de la couche gold et selection des features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

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


# Trois jeux disjoints : 60 % train, 20 % validation, 20 % test.
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
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


def split_train_validation_test(X: pd.DataFrame, y: pd.Series) -> Split:
    """Decoupe en trois jeux stratifies, en deux temps.

    Le meme decoupage pour tous les modeles : leurs metriques restent
    comparables, et aucun n'a vu le test avant la mesure finale.
    """
    X_reste, X_test, y_reste, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # La part de validation est exprimee sur le total, d'ou le reajustement.
    part_validation = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_reste,
        y_reste,
        test_size=part_validation,
        random_state=RANDOM_STATE,
        stratify=y_reste,
    )

    return Split(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
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
