"""Ingestion de la couche silver : schema `bronze` -> schema `silver`.

Chaque action de nettoyage est une fonction independante prenant et retournant
un DataFrame. Pour en ajouter une : ecrire la fonction, puis la referencer dans
TRANSFORMATIONS -- elles sont appliquees dans l'ordre de ce tuple.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_silver
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import typer
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import ensure_schema, get_engine, get_session
from ml_churn.ingestion.models import (
    SILVER_SCHEMA,
    Base,
    ChurnSaasCompletBronze,
    ChurnSaasSilver,
)

SOURCE_MODEL = ChurnSaasCompletBronze
TARGET_MODEL = ChurnSaasSilver
BATCH_SIZE = 1_000

Transformation = Callable[[pd.DataFrame, bool], pd.DataFrame]


# --------------------------------------------------------------------------
# Actions de nettoyage : une fonction par action.
# --------------------------------------------------------------------------


def dedupliquer_clients(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Un client_id ne doit apparaitre qu'une fois.

    On garde la premiere occurrence rencontree dans le CSV source (ordre de
    _source_line), faute d'un critere metier pour departager les doublons.
    """
    avant = len(df)
    dedupliques = df.drop_duplicates(subset="client_id", keep="first")
    apres = len(dedupliques)

    if echo:
        _log_action("deduplication client_id", avant, apres, "lignes")

    return dedupliques


# --- Standardisation des colonnes ---------------------------------------

# Formats rencontres dans le CSV, essayes dans cet ordre.
FORMATS_DATE: tuple[tuple[str, str], ...] = (
    ("AAAA-MM-JJ", "%Y-%m-%d"),
    ("JJ/MM/AAAA", "%d/%m/%Y"),
    ("JJ mois AAAA", "%d %b %Y"),
)

JOURS = {
    "lundi": "L",
    "mardi": "M",
    "mercredi": "ME",
    "jeudi": "J",
    "vendredi": "V",
    "samedi": "S",
    "dimanche": "D",
}

SECTEURS = {
    "tech": "TE",
    "finance": "FI",
    "commerce": "CO",
    "santé": "SA",
    "industrie": "IN",
    "public": "PB",
    "éducation": "EN",
}

# Codes ISO 3166-1 alpha-2.
PAYS = {
    "france": "FR",
    "espagne": "ES",
    "canada": "CA",
    "allemagne": "DE",
    "suisse": "CH",
    "belgique": "BE",
}

TAILLES_ENTREPRISE = {"tpe": "TPE", "pme": "PME", "eti": "ETI", "ge": "GE"}

PLANS = {"pro": "PRO", "business": "BUS", "starter": "STR", "enterprise": "ENT"}

COULEURS_THEME = {
    "clair": "C",
    "vert": "VE",
    "bleu": "B",
    "violet": "V",
    "sombre": "S",
}

GROUPES_EXPERIMENTATION = {"a": "A", "b": "B", "control": "C"}


def _nb_valeurs(valeurs: pd.Series) -> int:
    """Valeurs renseignees : une baisse signale des valeurs perdues par l'action."""
    brutes = valeurs.astype("string")
    return int((brutes.notna() & (brutes.str.strip() != "")).sum())


# Famille de la derniere action loguee, pour aerer entre deux familles.
_groupe_precedent: str | None = None


def _separer_groupe(groupe: str) -> None:
    """Ligne vide au passage d'une famille d'actions a une autre."""
    global _groupe_precedent
    if _groupe_precedent is not None and groupe != _groupe_precedent:
        print()
    _groupe_precedent = groupe


def _log_action(
    action: str, avant: int, apres: int, unite: str, *, difference: bool = False
) -> None:
    """Format commun a toutes les actions : [ACTION] : avant -> apres."""
    _separer_groupe(action.split()[0])
    ecart = f" ({apres - avant})" if difference else ""
    print(f"[{action.upper()}] : {avant} {unite} -> {apres} {unite}{ecart}")


def _valeurs_normalisees(df: pd.DataFrame, colonne: str) -> pd.Series:
    """Casse et espaces harmonises, valeurs vides ramenees a NA."""
    valeurs = df[colonne].astype("string").str.strip().str.casefold()
    return valeurs.where(valeurs.notna() & (valeurs != ""), pd.NA)


def _standardiser_categorie(
    df: pd.DataFrame,
    colonne: str,
    correspondances: dict[str, str],
    echo: bool,
) -> pd.DataFrame:
    """Applique une table de correspondance et signale les valeurs inattendues."""
    avant = _nb_valeurs(df[colonne])
    valeurs = _valeurs_normalisees(df, colonne)
    converties = valeurs.map(correspondances)

    inconnues = sorted(valeurs[converties.isna() & valeurs.notna()].unique())

    df = df.copy()
    df[colonne] = converties

    if echo:
        _log_action(
            f"standardisation {colonne}",
            avant,
            _nb_valeurs(converties),
            "valeurs",
        )
        if inconnues:
            print(
                f"WARNING : {colonne} : valeurs non reconnues, mises a NULL -> {inconnues}"
            )

    return df


def standardiser_date_souscription(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Ramene les trois formats de date rencontres au seul format AAAA-MM-JJ."""
    colonne = "date_souscription"
    avant = _nb_valeurs(df[colonne])
    brut = _valeurs_normalisees(df, colonne)
    dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for _libelle, format_source in FORMATS_DATE:
        restantes = brut.where(dates.isna() & brut.notna())
        essai = pd.to_datetime(restantes, format=format_source, errors="coerce")
        dates = dates.fillna(essai)

    non_converties = sorted(brut[dates.isna() & brut.notna()].unique())

    df = df.copy()
    df[colonne] = dates.dt.strftime("%Y-%m-%d").astype("string")

    if echo:
        _log_action(
            f"standardisation {colonne}",
            avant,
            _nb_valeurs(df[colonne]),
            "valeurs",
        )
        if non_converties:
            print(
                f"WARNING : {colonne} : {len(non_converties)} valeurs non reconnues, "
                f"mises a NULL -> {non_converties[:5]}"
            )

    return df


def standardiser_jour_souscription(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """lundi -> L, mardi -> M, mercredi -> ME, jeudi -> J, vendredi -> V, samedi -> S, dimanche -> D."""
    return _standardiser_categorie(df, "jour_souscription", JOURS, echo)


def standardiser_secteur(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Tech -> TE, Finance -> FI, Commerce -> CO, Sante -> SA, Industrie -> IN, Public -> PB, Education -> EN."""
    return _standardiser_categorie(df, "secteur", SECTEURS, echo)


def standardiser_pays(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Nom du pays -> code ISO 3166-1 alpha-2 (France -> FR, Allemagne -> DE...)."""
    return _standardiser_categorie(df, "pays", PAYS, echo)


def standardiser_taille_entreprise(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Harmonise la casse : tpe / " TPE " -> TPE, idem PME, ETI, GE."""
    return _standardiser_categorie(df, "taille_entreprise", TAILLES_ENTREPRISE, echo)


def standardiser_plan(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Pro -> PRO, Business -> BUS, Starter -> STR, Enterprise -> ENT."""
    return _standardiser_categorie(df, "plan", PLANS, echo)


def standardiser_couleur_theme_interface(
    df: pd.DataFrame, echo: bool = True
) -> pd.DataFrame:
    """clair -> C, vert -> VE, bleu -> B, violet -> V, sombre -> S."""
    return _standardiser_categorie(df, "couleur_theme_interface", COULEURS_THEME, echo)


def standardiser_groupe_experimentation(
    df: pd.DataFrame, echo: bool = True
) -> pd.DataFrame:
    """A -> A, B -> B, control -> C."""
    return _standardiser_categorie(
        df, "groupe_experimentation", GROUPES_EXPERIMENTATION, echo
    )


# --- Regles metier ------------------------------------------------------

# Bornes attendues, valeur incluses.
BORNES: dict[str, tuple[float, float]] = {
    "anciennete_mois": (1, 36),
    "sieges_souscrits": (1, 898),
    "utilisateurs_actifs": (0, 829),
    "taux_adoption_pct": (0, 100),
    "csat": (1, 5),
    "sante_compte_fin_periode": (0, 100),
}

FORMAT_CLIENT_ID = r"^CLI-\d+$"


def _numerique(df: pd.DataFrame, colonne: str) -> pd.Series:
    """Lecture numerique tolerante : virgule decimale, signe % et espaces.

    Ces ecarts de format ne sont pas des violations de regle metier :
    "20.0%" vaut 20.0 et doit etre acceptee.
    """
    valeurs = _valeurs_normalisees(df, colonne)
    # On ne garde que chiffres, separateurs et signe : le CSV note aussi les
    # unites ("20.0%", "3.1 h", "280.62 €") qui ne changent pas la valeur.
    valeurs = valeurs.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(valeurs.str.replace(",", ".", regex=False), errors="coerce")


def _renseignee(df: pd.DataFrame, colonne: str) -> pd.Series:
    return _valeurs_normalisees(df, colonne).notna()


def _dans_bornes(df: pd.DataFrame, colonne: str) -> pd.Series:
    """True = ligne conservee. Une valeur absente n'est pas une violation."""
    mini, maxi = BORNES[colonne]
    return ~_renseignee(df, colonne) | _numerique(df, colonne).between(mini, maxi)


@dataclass(frozen=True)
class RegleMetier:
    """Une regle : la colonne concernee et le predicat des lignes conservees."""

    colonne: str
    description: str
    conforme: Callable[[pd.DataFrame], pd.Series]


REGLES_METIER: tuple[RegleMetier, ...] = (
    RegleMetier(
        "client_id",
        "format CLI-<chiffres>",
        # Un client_id absent ou mal forme est une violation : c'est la cle.
        lambda df: (
            df["client_id"]
            .astype("string")
            .str.strip()
            .str.match(FORMAT_CLIENT_ID, na=False)
        ),
    ),
    RegleMetier(
        "anciennete_mois",
        "entre 1 et 36",
        lambda df: _dans_bornes(df, "anciennete_mois"),
    ),
    RegleMetier(
        "sieges_souscrits",
        "entre 1 et 898",
        lambda df: _dans_bornes(df, "sieges_souscrits"),
    ),
    RegleMetier(
        "utilisateurs_actifs",
        "entre 0 et 829, et <= sieges_souscrits",
        lambda df: (
            _dans_bornes(df, "utilisateurs_actifs")
            & (
                ~(
                    _renseignee(df, "utilisateurs_actifs")
                    & _renseignee(df, "sieges_souscrits")
                )
                | (
                    _numerique(df, "utilisateurs_actifs")
                    <= _numerique(df, "sieges_souscrits")
                )
            )
        ),
    ),
    RegleMetier(
        "taux_adoption_pct",
        "entre 0 et 100",
        lambda df: _dans_bornes(df, "taux_adoption_pct"),
    ),
    RegleMetier("csat", "entre 1 et 5", lambda df: _dans_bornes(df, "csat")),
    RegleMetier(
        "sante_compte_fin_periode",
        "entre 0 et 100",
        lambda df: _dans_bornes(df, "sante_compte_fin_periode"),
    ),
    RegleMetier(
        "churn",
        "vaut 0 ou 1",
        lambda df: ~_renseignee(df, "churn") | _numerique(df, "churn").isin([0, 1]),
    ),
)


def appliquer_regles_metier(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Supprime les lignes violant une regle metier, colonne par colonne."""
    for regle in REGLES_METIER:
        avant = len(df)
        df = df[regle.conforme(df).fillna(False)]
        apres = len(df)

        if echo:
            _log_action(
                f"regle {regle.colonne}", avant, apres, "lignes", difference=True
            )

    return df


# --- Typage des colonnes ------------------------------------------------

COLONNES_ENTIERES: tuple[str, ...] = (
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "connexions_30j",
    "fonctionnalites_total",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "csat",
    "retards_paiement_12m",
    "sante_compte_fin_periode",
    "churn",
)

COLONNES_DECIMALES: tuple[str, ...] = (
    "taux_adoption_pct",
    "heures_usage_30j",
    "delai_reponse_support_h",
    "revenu_mensuel_recurrent_eur",
    "valeur_vie_client_eur",
)

COLONNE_DATE = "date_souscription"

# Le reste des colonnes metier est du texte (deduit du modele, pas d'oubli possible).
COLONNES_TEXTE: tuple[str, ...] = tuple(
    colonne.key
    for colonne in TARGET_MODEL.__table__.columns
    if not colonne.key.startswith("_")
    and colonne.key not in {*COLONNES_ENTIERES, *COLONNES_DECIMALES, COLONNE_DATE}
)


def typer_colonnes(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Convertit les colonnes texte vers les types du modele silver.

    Les valeurs sont produites en types Python natifs (int, float, date, None)
    pour etre inserees telles quelles. Une valeur absente reste None.
    """
    df = df.copy()
    avant = 0
    apres = 0

    dates_texte = _valeurs_normalisees(df, COLONNE_DATE)
    avant += int(dates_texte.notna().sum())
    dates = pd.to_datetime(dates_texte, format="%Y-%m-%d", errors="coerce")
    df[COLONNE_DATE] = dates.dt.date.astype(object).where(dates.notna(), None)
    apres += int(dates.notna().sum())

    for colonne in COLONNES_ENTIERES:
        avant += _nb_valeurs(df[colonne])
        valeurs = _numerique(df, colonne)

        arrondies = valeurs.dropna() % 1 != 0
        if echo and arrondies.any():
            print(
                f"WARNING : {colonne} : {int(arrondies.sum())} valeurs decimales "
                f"arrondies pour tenir dans un entier"
            )

        df[colonne] = valeurs.map(
            lambda valeur: round(valeur) if pd.notna(valeur) else None
        ).astype(object)
        apres += int(valeurs.notna().sum())

    for colonne in COLONNES_DECIMALES:
        avant += _nb_valeurs(df[colonne])
        valeurs = _numerique(df, colonne)
        df[colonne] = valeurs.map(
            lambda valeur: float(valeur) if pd.notna(valeur) else None
        ).astype(object)
        apres += int(valeurs.notna().sum())

    for colonne in COLONNES_TEXTE:
        valeurs = df[colonne].astype("string").str.strip()
        valeurs = valeurs.where(valeurs.notna() & (valeurs != ""), None)
        df[colonne] = valeurs.astype(object)

    if echo:
        _log_action("typage colonnes", avant, apres, "valeurs")

    return df


TRANSFORMATIONS: tuple[Transformation, ...] = (
    dedupliquer_clients,
    standardiser_date_souscription,
    standardiser_jour_souscription,
    standardiser_secteur,
    standardiser_pays,
    standardiser_taille_entreprise,
    standardiser_plan,
    standardiser_couleur_theme_interface,
    standardiser_groupe_experimentation,
    appliquer_regles_metier,
    typer_colonnes,
)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _log_donnees_manquantes(df: pd.DataFrame) -> None:
    """Part de valeurs NULL une fois toutes les actions appliquees.

    Toutes les colonnes metier du modele sont listees, des plus incompletes aux
    plus completes. A taux egal, l'ordre du modele est conserve.
    """
    colonnes = [
        colonne.key
        for colonne in TARGET_MODEL.__table__.columns
        if not colonne.key.startswith("_")
    ]
    manquantes = df[colonnes].isna().sum()
    cellules = len(df) * len(colonnes)
    total = int(manquantes.sum())

    print()
    print(
        f"[DONNEES MANQUANTES] : {total / cellules:.1%} des valeurs "
        f"({total} / {cellules})"
    )

    largeur = max(len(colonne) for colonne in colonnes)
    # kind="stable" : les colonnes a egalite restent dans l'ordre du modele.
    for colonne, nombre in manquantes.sort_values(
        ascending=False, kind="stable"
    ).items():
        print(f"  {colonne:<{largeur}} : {int(nombre) / len(df):>5.1%} ({int(nombre)})")


def lire_bronze(session: Session, *, echo: bool = True) -> pd.DataFrame:
    """Charge la table bronze source, triee pour que la deduplication soit stable."""
    colonnes = [
        column
        for column in SOURCE_MODEL.__table__.columns
        if column.key in TARGET_MODEL.__table__.columns
    ]
    statement = select(*colonnes, SOURCE_MODEL._source_line).order_by(
        SOURCE_MODEL._source_line
    )
    df = pd.read_sql(statement, session.connection()).drop(columns=["_source_line"])

    if echo:
        print(f"{SOURCE_MODEL.__table__.fullname} : {len(df)} lignes lues\n")

    return df


def ecrire_silver(session: Session, df: pd.DataFrame, *, echo: bool = True) -> int:
    """Remplace le contenu de la table silver et retourne le nombre de lignes inserees."""
    session.execute(
        text(f'TRUNCATE TABLE "{SILVER_SCHEMA}"."{TARGET_MODEL.__tablename__}"')
    )

    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    for start in range(0, len(rows), BATCH_SIZE):
        session.execute(
            TARGET_MODEL.__table__.insert(), rows[start : start + BATCH_SIZE]
        )
    session.commit()

    inserted = session.scalar(
        text(f'SELECT count(*) FROM "{SILVER_SCHEMA}"."{TARGET_MODEL.__tablename__}"')
    )

    if echo:
        print(f"\n{TARGET_MODEL.__table__.fullname} : {inserted} lignes inserees")
        if inserted != len(rows):
            print(
                f"WARNING : {len(rows)} lignes a inserer mais {inserted} en base "
                f"(ecart de {inserted - len(rows):+d})"
            )

    return inserted


def ingest_silver(*, echo: bool = True) -> int:
    """Applique les transformations sur bronze et recharge la table silver."""
    global _groupe_precedent
    _groupe_precedent = None

    ensure_schema(SILVER_SCHEMA)
    # La table est entierement rechargee a chaque execution : on la recree pour
    # qu'elle suive le modele, y compris quand les types changent.
    TARGET_MODEL.__table__.drop(get_engine(), checkfirst=True)
    Base.metadata.create_all(get_engine())

    with get_session() as session:
        df = lire_bronze(session, echo=echo)

        for transformation in TRANSFORMATIONS:
            df = transformation(df, echo)

        if echo:
            _log_donnees_manquantes(df)

        inserted = ecrire_silver(session, df, echo=echo)

    if echo:
        print(f"\nTOTAL : {inserted} lignes dans {TARGET_MODEL.__table__.fullname}")

    return inserted


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    ingest_silver()


if __name__ == "__main__":
    app()
