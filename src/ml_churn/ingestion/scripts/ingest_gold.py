"""Ingestion de la couche gold : schema `silver` -> schema `gold`.

Les tables gold reprennent la structure de silver a l'identique. La couche est
entierement rechargee a chaque execution.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_gold
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import pandas as pd
import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import ensure_schema, get_engine, get_session
from ml_churn.ingestion.logs import (
    PREFIXE_ECRITURE_BASE,
    PREFIXE_LECTURE_BASE,
    log_action,
    log_total,
    reinitialiser_groupes,
)
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

# Au-dela de ce nombre de jours sans connexion, le compte est considere inactif.
SEUIL_INACTIVITE_JOURS = 30

# Bornes hautes des niveaux d'anciennete, en mois : onboarding, premiere annee,
# puis au-dela du premier renouvellement.
NIVEAUX_ANCIENNETE: tuple[tuple[float, str], ...] = (
    (3, "RECENT"),
    (12, "ETABLI"),
    (float("inf"), "ANCIEN"),
)


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


def ajouter_colonnes_derivees(
    df: pd.DataFrame, echo: bool = True, *, catalogue: pd.DataFrame
) -> pd.DataFrame:
    """Ajoute les colonnes calculees a partir des colonnes existantes.

    `niveau_anciennete` est encodee en one-hot par l'action suivante, au meme
    titre que les autres colonnes categorielles.
    """
    df = df.copy()
    anciennete = pd.to_numeric(df["anciennete_mois"], errors="coerce")
    derniere_connexion = pd.to_numeric(df["derniere_connexion_jours"], errors="coerce")

    jours_de_vie = (anciennete * 30).replace(0, pd.NA)
    inactivite = (derniere_connexion / jours_de_vie).astype(float).round(2)
    df["inactivite_relative"] = inactivite.map(
        lambda valeur: float(valeur) if pd.notna(valeur) else None
    ).astype(object)

    if echo:
        log_action(
            "derivation inactivite_relative",
            int(inactivite.notna().sum()),
            len(df),
            "valeurs",
            resultat="calculees",
            description="jours sans connexion / jours de vie du compte",
        )

    df["inactif_30j"] = (derniere_connexion >= SEUIL_INACTIVITE_JOURS).astype(int)
    inactifs = int(df["inactif_30j"].sum())

    if echo:
        log_action(
            "derivation inactif_30j",
            int(df["inactif_30j"].notna().sum()),
            len(df),
            "valeurs",
            resultat="calculees",
            description=f"1 au-dela de {SEUIL_INACTIVITE_JOURS} jours sans connexion",
            detail=f" ({inactifs} clients inactifs)",
        )

    # Utilisation rapportee au plan : 3 fonctionnalites sur 8 (Starter) et sur
    # 40 (Enterprise) ne decrivent pas le meme niveau d'adoption.
    utilisees = pd.to_numeric(df["fonctionnalites_utilisees"], errors="coerce")
    fonctionnalites_par_plan = catalogue.set_index("plan")["fonctionnalites_incluses"]
    total = df["plan"].map(fonctionnalites_par_plan.astype(float)).replace(0, pd.NA)
    taux = (utilisees / total).astype(float).round(2)
    df["taux_fonctionnalites"] = taux.map(
        lambda valeur: float(valeur) if pd.notna(valeur) else None
    ).astype(object)

    if echo:
        log_action(
            "derivation taux_fonctionnalites",
            int(taux.notna().sum()),
            len(df),
            "valeurs",
            resultat="calculees",
            description="fonctionnalites utilisees / incluses dans le plan",
        )

    bornes = [0.0, *(borne for borne, _ in NIVEAUX_ANCIENNETE)]
    df["niveau_anciennete"] = pd.cut(
        anciennete,
        bins=bornes,
        labels=[libelle for _, libelle in NIVEAUX_ANCIENNETE],
    ).astype(object)

    if echo:
        repartition = df["niveau_anciennete"].value_counts()
        detail = ", ".join(f"{nom} {nombre}" for nom, nombre in repartition.items())
        bornes_lisibles = ", ".join(
            f"{libelle} jusqu'a {borne:g} mois"
            if borne != float("inf")
            else f"{libelle} au-dela"
            for borne, libelle in NIVEAUX_ANCIENNETE
        )
        log_action(
            "derivation niveau_anciennete",
            int(df["niveau_anciennete"].notna().sum()),
            len(df),
            "valeurs",
            resultat="calculees",
            description=f"anciennete -> {bornes_lisibles}",
            detail=f" ({detail})",
        )

    return df


def encoder_categorielles(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Une colonne binaire (0/1) par modalite, nommee [nom_categorie]_valeur.

    Les colonnes d'origine sont conservees : l'encodage les complete, il ne les
    remplace pas.
    """
    df = df.copy()
    creees = 0
    renseignees = 0
    encodees = 0

    for colonne, modalites in MODALITES_ONE_HOT.items():
        valeurs = df[colonne].astype("string")
        renseignees += int(valeurs.notna().sum())
        encodees += int(valeurs.isin(modalites).sum())

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
        log_action(
            "encodage one-hot",
            encodees,
            renseignees,
            "valeurs",
            resultat="encodees",
            description=f"{len(MODALITES_ONE_HOT)} colonnes categorielles "
            f"-> {creees} colonnes binaires",
        )

    return df


TABLES: tuple[TableGold, ...] = (
    TableGold(CatalogueSilver, CatalogueGold),
    TableGold(
        ChurnSaasSilver,
        ChurnSaasGold,
        transformations=(ajouter_colonnes_derivees, encoder_categorielles),
    ),
)


def _transformations(
    table: TableGold, catalogue: pd.DataFrame
) -> tuple[Transformation, ...]:
    """Les transformations de la table, catalogue injecte dans celle qui en depend."""
    return tuple(
        partial(transformation, catalogue=catalogue)
        if transformation is ajouter_colonnes_derivees
        else transformation
        for transformation in table.transformations
    )


def _lire(session: Session, table: TableGold, *, echo: bool) -> pd.DataFrame:
    """Charge la table silver source."""
    colonnes = [getattr(table.source, nom) for nom in table.colonnes]
    df = pd.read_sql(select(*colonnes), session.connection())

    if echo:
        print(
            f"{PREFIXE_LECTURE_BASE} {table.source.__table__.fullname} : {len(df)} lignes"
        )

    return df


def _ecrire(session: Session, table: TableGold, df: pd.DataFrame, *, echo: bool) -> int:
    """Remplace la table gold et retourne le nombre de lignes inserees."""
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
        print(
            f"{PREFIXE_ECRITURE_BASE} {table.cible.__table__.fullname} : "
            f"{inserted} lignes"
        )
        if inserted != len(rows):
            print(
                f"WARNING : {len(rows)} lignes a inserer mais {inserted} en base "
                f"(ecart de {inserted - len(rows):+d})"
            )

    return inserted


def ingest_gold(*, echo: bool = True) -> dict[str, int]:
    """Recopie les tables silver vers gold. Retourne le nombre de lignes par table."""
    reinitialiser_groupes()
    ensure_schema(GOLD_SCHEMA)

    with get_session() as session:
        # Toutes les lectures silver d'abord, toutes les ecritures gold a la fin :
        # les transformations tournent entre les deux, sur des DataFrames.
        lues = [(table, _lire(session, table, echo=echo)) for table in TABLES]
        catalogue = next(df for table, df in lues if table.cible is CatalogueGold)

        if echo:
            print()

        transformees: list[tuple[TableGold, pd.DataFrame]] = []
        for table, df in lues:
            for transformation in _transformations(table, catalogue):
                df = transformation(df, echo)
            transformees.append((table, df))

        if echo:
            print()

        resultats = {
            table.cible.__table__.fullname: _ecrire(session, table, df, echo=echo)
            for table, df in transformees
        }

    if echo:
        log_total(resultats)

    return resultats


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    ingest_gold()


if __name__ == "__main__":
    app()
