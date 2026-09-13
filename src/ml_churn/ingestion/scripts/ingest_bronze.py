"""Ingestion de la couche bronze : docs/*.csv -> schema `bronze`.

Les lignes sont copiees telles quelles (tout en texte, aucune conversion).
Une valeur vide dans le CSV est stockee en NULL.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_bronze
    uv run python -m ml_churn.ingestion.scripts.ingest_bronze --append
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import typer
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import PROJECT_ROOT, ensure_schema, get_engine, get_session
from ml_churn.ingestion.models import (
    BRONZE_SCHEMA,
    Base,
    CatalogueBronze,
    ChurnSaasCompletBronze,
    ChurnSaasEchantillonBronze,
)

DOCS_DIR = PROJECT_ROOT / "docs"
BATCH_SIZE = 1_000
TECHNICAL_COLUMNS = {"id", "_source_file", "_source_line", "_ingested_at"}


@dataclass(frozen=True)
class BronzeSource:
    """Un CSV source et la table bronze qui le recoit."""

    csv_name: str
    model: type[Base]

    @property
    def csv_path(self) -> Path:
        return DOCS_DIR / self.csv_name

    @property
    def table_name(self) -> str:
        return self.model.__tablename__


@dataclass(frozen=True)
class IngestionReport:
    """Bilan du chargement d'un CSV dans sa table bronze."""

    table_name: str
    csv_name: str
    csv_rows: int
    inserted_rows: int

    @property
    def difference(self) -> int:
        """Ecart inserees - lues. Doit valoir 0 : sinon des lignes sont perdues."""
        return self.inserted_rows - self.csv_rows


SOURCES: tuple[BronzeSource, ...] = (
    BronzeSource("catalogue_plans.csv", CatalogueBronze),
    BronzeSource("churn_saas_complet.csv", ChurnSaasCompletBronze),
    BronzeSource("churn_saas_echantillon.csv", ChurnSaasEchantillonBronze),
)


def _model_columns(model: type[Base]) -> list[str]:
    return [c.key for c in model.__table__.columns if c.key not in TECHNICAL_COLUMNS]


def _read_rows(source: BronzeSource) -> list[dict[str, str | None]]:
    """Lit le CSV en texte brut. utf-8-sig retire le BOM present sur ces fichiers."""
    expected = _model_columns(source.model)

    with source.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if header != expected:
            raise ValueError(
                f"{source.csv_name} : entetes inattendues.\n"
                f"  CSV    : {header}\n"
                f"  modele : {expected}"
            )
        # start=2 : la ligne 1 du fichier est l'entete.
        return [
            {
                **{column: (row[column] or None) for column in expected},
                "_source_file": source.csv_name,
                "_source_line": line_number,
            }
            for line_number, row in enumerate(reader, start=2)
        ]


def _count_rows(session: Session, source: BronzeSource) -> int:
    return session.scalar(select(func.count()).select_from(source.model.__table__))


def _load(
    session: Session, source: BronzeSource, *, append: bool, echo: bool
) -> IngestionReport:
    """Charge un CSV dans sa table et compare le lu a l'insere."""
    rows = _read_rows(source)
    if echo:
        print(f"{source.csv_name} : {len(rows)} lignes lues")

    if not append:
        session.execute(
            text(
                f'TRUNCATE TABLE "{BRONZE_SCHEMA}"."{source.table_name}" RESTART IDENTITY'
            )
        )

    rows_before = _count_rows(session, source)
    for start in range(0, len(rows), BATCH_SIZE):
        session.execute(
            source.model.__table__.insert(), rows[start : start + BATCH_SIZE]
        )
    session.commit()

    # Comptage apres commit : on mesure ce qui est persiste, pas ce qu'on a envoye.
    inserted = _count_rows(session, source) - rows_before
    report = IngestionReport(
        table_name=source.table_name,
        csv_name=source.csv_name,
        csv_rows=len(rows),
        inserted_rows=inserted,
    )

    if echo:
        print(f"{BRONZE_SCHEMA}.{source.table_name} : {inserted} lignes inserees")
        if report.difference != 0:
            print(
                f"WARNING : {source.csv_name} -> {BRONZE_SCHEMA}.{source.table_name} : "
                f"{report.csv_rows} lignes lues mais {inserted} inserees "
                f"(ecart de {report.difference:+d})"
            )
        print()

    return report


def _print_summary(reports: list[IngestionReport]) -> None:
    """Bilan global : soit tout est passe, soit on signale l'ecart."""
    csv_total = sum(report.csv_rows for report in reports)
    inserted_total = sum(report.inserted_rows for report in reports)
    difference = inserted_total - csv_total

    if difference == 0:
        print(
            f"TOTAL : {csv_total} lignes csv, {inserted_total} lignes inserees en base"
        )
        return

    tables_en_ecart = ", ".join(
        f"{BRONZE_SCHEMA}.{report.table_name} ({report.difference:+d})"
        for report in reports
        if report.difference != 0
    )
    print(
        f"WARNING : TOTAL {csv_total} lignes csv mais {inserted_total} lignes inserees "
        f"en base (ecart de {difference:+d}) -> {tables_en_ecart}"
    )


def ingest_bronze(*, append: bool = False, echo: bool = True) -> list[IngestionReport]:
    """Charge les trois CSV dans le schema bronze.

    Par defaut chaque table est videe avant rechargement : relancer l'ingestion
    ne cree pas de doublons. `append` conserve l'existant.

    Retourne un rapport par table (lignes lues, lignes inserees, ecart).
    """
    ensure_schema(BRONZE_SCHEMA)
    Base.metadata.create_all(get_engine())

    with get_session() as session:
        reports = [
            _load(session, source, append=append, echo=echo) for source in SOURCES
        ]

    if echo:
        _print_summary(reports)

    return reports


app = typer.Typer(help=__doc__)


@app.command()
def main(
    append: bool = typer.Option(
        False,
        "--append",
        help="Conserver les lignes existantes au lieu de vider les tables.",
    ),
) -> None:
    ingest_bronze(append=append)


if __name__ == "__main__":
    app()
