"""Ingestion de la couche gold : schema `silver` -> schema `gold`.

Les tables gold reprennent la structure de silver a l'identique. La couche est
entierement rechargee a chaque execution.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_gold
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import ensure_schema, get_engine, get_session
from ml_churn.ingestion.logs import log_total
from ml_churn.ingestion.models import (
    GOLD_SCHEMA,
    MODALITES_ONE_HOT,
    Base,
    CatalogueGold,
    CatalogueSilver,
    ChurnSaasGold,
    ChurnSaasSilver,
    nom_colonne_one_hot,
)

BATCH_SIZE = 1_000

Transformation = Callable[[pd.DataFrame, bool], pd.DataFrame]


@dataclass(frozen=True)
class TableGold:
    """Une table silver et la table gold qui la recoit."""

    source: type[Base]
    cible: type[Base]
    transformations: tuple[Transformation, ...] = field(default_factory=tuple)

    @property
    def colonnes(self) -> list[str]:
        """Colonnes metier communes aux deux tables (hors colonnes techniques).

        Les colonnes propres a gold (one-hot) n'existent pas dans silver : elles
        sont calculees, pas lues.
        """
        return [
            colonne.key
            for colonne in self.cible.__table__.columns
            if not colonne.key.startswith("_")
            and colonne.key in self.source.__table__.columns
        ]


def encoder_categorielles(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Une colonne binaire (0/1) par modalite, nommee [nom_categorie]_valeur.

    Les colonnes d'origine sont conservees : l'encodage les complete, il ne les
    remplace pas.
    """
    df = df.copy()
    creees = 0

    for colonne, modalites in MODALITES_ONE_HOT.items():
        valeurs = df[colonne].astype("string")

        for modalite in modalites:
            df[nom_colonne_one_hot(colonne, modalite)] = (valeurs == modalite).astype(
                int
            )
            creees += 1

        inconnues = sorted(set(valeurs.dropna().unique()) - set(modalites))
        if echo and inconnues:
            print(
                f"WARNING : {colonne} : modalites absentes du modele, "
                f"non encodees -> {inconnues}"
            )

    if echo:
        print(
            f"encodage one-hot : {len(MODALITES_ONE_HOT)} colonnes categorielles "
            f"-> {creees} colonnes binaires"
        )

    return df


TABLES: tuple[TableGold, ...] = (
    TableGold(CatalogueSilver, CatalogueGold),
    TableGold(
        ChurnSaasSilver,
        ChurnSaasGold,
        transformations=(encoder_categorielles,),
    ),
)


def _charger(session: Session, table: TableGold, *, echo: bool) -> int:
    colonnes = [getattr(table.source, nom) for nom in table.colonnes]
    df = pd.read_sql(select(*colonnes), session.connection())

    if echo:
        print(f"{table.source.__table__.fullname} : {len(df)} lignes lues")

    for transformation in table.transformations:
        df = transformation(df, echo)

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
