"""Genere les clients de demonstration servis par la page de test.

Le principe : tirer de vraies lignes du jeu de test, perturber les colonnes
**sources**, puis recalculer les colonnes qui en derivent. Bruiter chaque
colonne independamment romprait les relations qui les lient -- le taux
d'adoption cesserait d'etre le rapport des utilisateurs actifs aux sieges, le
revenu de correspondre au plan -- et produirait des clients decrivant des
situations qui n'existent pas.

Usage :
    uv run python -m ml_churn.ui.generer_clients
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sqlalchemy import select

from ml_churn.ingestion.db import get_engine
from ml_churn.ingestion.models import CatalogueGold
from ml_churn.ingestion.scripts.ingest_gold import (
    NIVEAUX_ANCIENNETE,
    SEUIL_INACTIVITE_JOURS,
)
from ml_churn.training.common.data import COLONNE_JEU, feature_columns, load_gold
from ml_churn.training.regression.baseline.regression_baseline_training import (
    EXCLUSIONS,
    TARGET,
)

DESTINATION = Path(__file__).parent / "clients.js"

NOMBRE_CLIENTS = 100
GRAINE = 7

# Amplitude du bruit applique aux colonnes sources.
BRUIT = 0.12

# Colonnes perturbees : celles qui decrivent le client sans etre calculees a
# partir d'une autre. Les entieres sont arrondies apres bruitage.
SOURCES_ENTIERES: tuple[str, ...] = (
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "connexions_30j",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "csat",
    "retards_paiement_12m",
)
SOURCES_DECIMALES: tuple[str, ...] = ("heures_usage_30j", "delai_reponse_support_h")

# Bornes metier, reprises des regles de la couche silver.
BORNES: dict[str, tuple[float, float]] = {
    "anciennete_mois": (1, 36),
    "csat": (1, 5),
}


def _bruiter(valeurs: pd.Series, tirage: np.random.Generator) -> pd.Series:
    """Applique un bruit multiplicatif, borne par ce que la colonne admet."""
    facteurs = 1 + tirage.uniform(-BRUIT, BRUIT, len(valeurs))
    return valeurs.astype(float) * facteurs


def perturber(clients: pd.DataFrame, tirage: np.random.Generator) -> pd.DataFrame:
    """Bruite les colonnes sources, en respectant leurs bornes."""
    clients = clients.copy()

    for colonne in (*SOURCES_ENTIERES, *SOURCES_DECIMALES):
        valeurs = _bruiter(pd.to_numeric(clients[colonne]), tirage)
        if colonne in BORNES:
            valeurs = valeurs.clip(*BORNES[colonne])
        clients[colonne] = (
            valeurs.round().astype(int)
            if colonne in SOURCES_ENTIERES
            else valeurs.round(2)
        )

    # Un compte ne peut pas avoir plus d'utilisateurs actifs que de sieges.
    clients["utilisateurs_actifs"] = clients[
        ["utilisateurs_actifs", "sieges_souscrits"]
    ].min(axis=1)

    return clients


def recalculer(clients: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les colonnes derivees, avec les formules de l'ingestion."""
    clients = clients.copy()
    anciennete = clients["anciennete_mois"].astype(float)
    derniere_connexion = clients["derniere_connexion_jours"].astype(float)

    # Silver : le taux d'adoption est le rapport des actifs aux sieges.
    clients["taux_adoption_pct"] = (
        clients["utilisateurs_actifs"] / clients["sieges_souscrits"] * 100
    ).round(2)

    # Silver : le revenu suit le prix catalogue du plan souscrit.
    catalogue = pd.read_sql(
        select(
            CatalogueGold.plan,
            CatalogueGold.prix_mensuel_par_siege_eur,
            CatalogueGold.fonctionnalites_incluses,
        ),
        get_engine(),
    ).set_index("plan")
    plan = _plan(clients)
    clients["revenu_mensuel_recurrent_eur"] = (
        clients["sieges_souscrits"]
        * plan.map(catalogue["prix_mensuel_par_siege_eur"].astype(float))
    ).round(2)

    # Gold : les trois colonnes derivees et le niveau d'anciennete.
    clients["inactivite_relative"] = (
        derniere_connexion / (anciennete * 30).replace(0, np.nan)
    ).round(2)
    clients["inactif_30j"] = (derniere_connexion >= SEUIL_INACTIVITE_JOURS).astype(int)
    clients["taux_fonctionnalites"] = (
        clients["fonctionnalites_utilisees"]
        / plan.map(catalogue["fonctionnalites_incluses"].astype(float))
    ).round(2)

    niveaux = pd.cut(
        anciennete,
        bins=[0.0, *(borne for borne, _ in NIVEAUX_ANCIENNETE)],
        labels=[libelle for _, libelle in NIVEAUX_ANCIENNETE],
    )
    for _, libelle in NIVEAUX_ANCIENNETE:
        clients[f"niveau_anciennete_{libelle.lower()}"] = (niveaux == libelle).astype(
            int
        )

    return clients


def _plan(clients: pd.DataFrame) -> pd.Series:
    """Retrouve le plan depuis ses colonnes one-hot."""
    colonnes = [colonne for colonne in clients.columns if colonne.startswith("plan_")]
    codes = clients[colonnes].idxmax(axis=1).str.removeprefix("plan_").str.upper()
    return codes


def generer(nombre: int = NOMBRE_CLIENTS, graine: int = GRAINE) -> pd.DataFrame:
    """Tire des clients du jeu de test, les perturbe et recalcule les derivees."""
    gold = load_gold()
    test = gold[gold[COLONNE_JEU] == "TEST"]

    clients = test.sample(nombre, random_state=graine).reset_index(drop=True)

    # Les features du modele de valeur vie client couvrent aussi celles du
    # modele de churn : un seul jeu de colonnes sert les deux endpoints.
    colonnes = ["client_id", *feature_columns(TARGET, EXCLUSIONS)]

    tirage = np.random.default_rng(graine)
    return recalculer(perturber(clients, tirage))[colonnes]


def ecrire(clients: pd.DataFrame, destination: Path = DESTINATION) -> Path:
    """Ecrit le fichier JavaScript lu par la page."""
    lignes = json.loads(clients.to_json(orient="records"))

    destination.write_text(
        "// Twenty sample clients, drawn from the test set then perturbed on their\n"
        "// source columns; the derived ones are recomputed, so every client stays\n"
        "// coherent. Generated by `ml_churn.ui.generer_clients`.\n"
        f"const CLIENTS = {json.dumps(lignes, indent=2)};\n"
    )
    return destination


app = typer.Typer(help=__doc__)


@app.command()
def main(nombre: int = NOMBRE_CLIENTS, graine: int = GRAINE) -> None:
    clients = generer(nombre, graine)
    chemin = ecrire(clients)
    print(f"{len(clients)} clients ecrits dans {chemin.relative_to(Path.cwd())}")


if __name__ == "__main__":
    app()
