"""Ingestion de la couche gold : schema `silver` -> schema `gold`.

Les tables gold reprennent la structure de silver a l'identique. La couche est
entierement rechargee a chaque execution.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_gold
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import ensure_schema, get_engine, get_session
from ml_churn.ingestion.logs import log_total
from ml_churn.ingestion.models import (
    GOLD_SCHEMA,
    Base,
    CatalogueGold,
    CatalogueSilver,
    ChurnSaasGold,
    ChurnSaasSilver,
)

BATCH_SIZE = 1_000


@dataclass(frozen=True)
class TableGold:
    """Une table silver et la table gold qui la recoit."""

    source: type[Base]
    cible: type[Base]

    @property
    def colonnes(self) -> list[str]:
        """Colonnes metier communes aux deux tables (hors colonnes techniques)."""
        return [
            colonne.key
            for colonne in self.cible.__table__.columns
            if not colonne.key.startswith("_")
        ]


TABLES: tuple[TableGold, ...] = (
    TableGold(CatalogueSilver, CatalogueGold),
    TableGold(ChurnSaasSilver, ChurnSaasGold),
)


def _charger(session: Session, table: TableGold, *, echo: bool) -> int:
    colonnes = [getattr(table.source, nom) for nom in table.colonnes]
    df = pd.read_sql(select(*colonnes), session.connection())

    if echo:
        print(f"{table.source.__table__.fullname} : {len(df)} lignes lues")

    # La table est reconstruite pour suivre le modele, y compris si les types changent.
    table.cible.__table__.drop(get_engine(), checkfirst=True)
    Base.metadata.create_all(get_engine())

    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    for debut in range(0, len(rows), BATCH_SIZE):
        session.execute(
            table.cible.__table__.insert(), rows[debut : debut + BATCH_SIZE]
        )
    session.commit()

    # Comptage apres commit : ce qui est reellement persiste.
    inserted = session.scalar(select(func.count()).select_from(table.cible.__table__))

    if echo:
        print(f"{table.cible.__table__.fullname} : {inserted} lignes inserees")
        if inserted != len(rows):
            print(
                f"WARNING : {len(rows)} lignes a inserer mais {inserted} en base "
                f"(ecart de {inserted - len(rows):+d})"
            )

    return inserted


def ingest_gold(*, echo: bool = True) -> dict[str, int]:
    """Recopie les tables silver vers gold. Retourne le nombre de lignes par table."""
    ensure_schema(GOLD_SCHEMA)

    resultats: dict[str, int] = {}
    with get_session() as session:
        for index, table in enumerate(TABLES):
            if echo and index:
                print()
            resultats[table.cible.__table__.fullname] = _charger(
                session, table, echo=echo
            )

    if echo:
        log_total(resultats)

    return resultats


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    ingest_gold()


if __name__ == "__main__":
    app()
