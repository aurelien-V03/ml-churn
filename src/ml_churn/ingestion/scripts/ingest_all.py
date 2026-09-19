"""Ingestion de bout en bout : docs/*.csv -> bronze -> silver -> gold.

Enchaine les trois couches dans l'ordre, chacune lisant ce que la precedente a
ecrit. Equivalent a lancer les trois scripts a la suite.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_all
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from ml_churn.ingestion.logs import log_total
from ml_churn.ingestion.scripts.ingest_bronze import ingest_bronze
from ml_churn.ingestion.scripts.ingest_gold import ingest_gold
from ml_churn.ingestion.scripts.ingest_silver import ingest_silver

Ingestion = Callable[..., dict[str, int]]

COUCHES: tuple[tuple[str, str, Ingestion], ...] = (
    ("🥉", "BRONZE", ingest_bronze),
    ("🥈", "SILVER", ingest_silver),
    ("🥇", "GOLD", ingest_gold),
)


def ingest_all(*, echo: bool = True) -> dict[str, int]:
    """Execute les trois couches a la suite.

    Retourne le nombre de lignes inserees par table, toutes couches confondues.
    """
    resultats: dict[str, int] = {}

    for emoji, nom, ingestion in COUCHES:
        if echo:
            print(f"\n{'=' * 60}\n{emoji}  {nom}\n{'=' * 60}")
        resultats.update(ingestion(echo=echo))

    if echo:
        print(f"\n{'=' * 60}\nRECAPITULATIF")
        log_total(resultats)

    return resultats


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    ingest_all()


if __name__ == "__main__":
    app()
