"""Repartition de la cible `churn` dans la couche gold.

Le desequilibre entre classes conditionne le choix des metriques : un modele
qui predirait "personne ne resilie" aurait deja une exactitude egale a la part
de clients actifs.

Usage :
    uv run python -m ml_churn.training.scripts.analyze_target
"""

from __future__ import annotations

import pandas as pd
import typer

from ml_churn.ingestion.db import get_engine
from ml_churn.ingestion.models import ChurnSaasGold

LIBELLES = {0: "actif", 1: "resilie"}


def analyze_target(*, echo: bool = True) -> dict[int, int]:
    """Compte les clients par valeur de `churn`.

    Retourne le nombre de clients pour chaque valeur de la cible.
    """
    table = ChurnSaasGold.__table__.fullname
    churn = pd.read_sql(f"select churn from {table}", get_engine())["churn"]
    effectifs = churn.value_counts().sort_index()

    if echo:
        for valeur, nombre in effectifs.items():
            libelle = LIBELLES.get(valeur, "?")
            part = nombre / len(churn)
            print(f"churn = {valeur} ({libelle:<8}) : {nombre} clients ({part:.2%})")

        print(f"\ntaux de churn : {churn.mean():.2%} sur {len(churn)} clients")

    return {int(valeur): int(nombre) for valeur, nombre in effectifs.items()}


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    analyze_target()


if __name__ == "__main__":
    app()
