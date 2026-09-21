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

LABELS = {0: "actif", 1: "resilie"}


def analyze_target(*, echo: bool = True) -> dict[int, int]:
    """Compte les clients par valeur de `churn`.

    Retourne le nombre de clients pour chaque valeur de la cible.
    """
    table = ChurnSaasGold.__table__.fullname
    churn = pd.read_sql(f"select churn from {table}", get_engine())["churn"]
    counts = churn.value_counts().sort_index()

    if echo:
        for value, count in counts.items():
            label = LABELS.get(value, "?")
            share = count / len(churn)
            print(f"churn = {value} ({label:<8}) : {count} clients ({share:.2%})")

        print(f"\ntaux de churn : {churn.mean():.2%} sur {len(churn)} clients")

    return {int(value): int(count) for value, count in counts.items()}


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    analyze_target()


if __name__ == "__main__":
    app()
