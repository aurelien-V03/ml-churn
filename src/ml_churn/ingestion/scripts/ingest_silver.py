"""Ingestion de la couche silver : schema `bronze` -> schema `silver`.

Chaque action de nettoyage est une fonction independante prenant et retournant
un DataFrame. Pour en ajouter une : ecrire la fonction, puis la referencer dans
TRANSFORMATIONS -- elles sont appliquees dans l'ordre de ce tuple.

Usage :
    uv run python -m ml_churn.ingestion.scripts.ingest_silver
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import typer
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ml_churn.ingestion.db import ensure_schema, get_engine, get_session
from ml_churn.ingestion.logs import log_total
from ml_churn.ingestion.models import (
    SILVER_SCHEMA,
    Base,
    CatalogueBronze,
    CatalogueSilver,
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


def _reinitialiser_groupes() -> None:
    """Repart d'un etat neutre : deux executions successives restent lisibles."""
    global _groupe_precedent
    _groupe_precedent = None


def _log_action(
    action: str,
    avant: int,
    apres: int,
    unite: str,
    *,
    difference: bool = False,
    detail: str = "",
) -> None:
    """Format commun a toutes les actions : [ACTION] : avant -> apres."""
    _separer_groupe(action.split()[0])
    ecart = f" ({apres - avant})" if difference else ""
    print(f"[{action.upper()}] : {avant} {unite} -> {apres} {unite}{ecart}{detail}")


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


# --- Analyse de polarite du commentaire CSM -----------------------------

# Les commentaires proviennent d'une liste fermee de formulations courtes.
# Un TF-IDF sur des phrases aussi breves regroupe sur le vocabulaire partage
# ("Client satisfait" avec "Client insatisfait") : on passe donc par un
# lexique de polarite, plus sur sur ce type de corpus.
TERMES_NEGATIFS: tuple[str, ...] = (
    "insatisfait",
    "mecontentement",
    "risque",
    "baisse",
    "friction",
    "limite",
    "relance",
    "sollicite",
    "multiples tickets",
    "depart",
)
TERMES_POSITIFS: tuple[str, ...] = ("satisfait", "engage", "ambassadeur", "actif")

# Inversent la polarite du terme qui suit : "peu actif", "faible adoption".
MODIFICATEURS_NEGATIFS: tuple[str, ...] = ("peu", "faible")

# Aucun commentaire n'est une information en soi : modalite a part entiere,
# distincte de NEUTRE (27.4 % de churn contre 22.0 %).
POLARITE_ABSENTE = "ABSENT"


def _polarite(commentaire: str) -> str:
    """ALERTE, POSITIF ou NEUTRE selon les termes reperes dans le commentaire."""
    texte = unicodedata.normalize("NFKD", commentaire.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    mots = texte.split()

    score = sum(terme in texte for terme in TERMES_NEGATIFS)
    score += sum(modificateur in mots for modificateur in MODIFICATEURS_NEGATIFS)
    score -= sum(
        terme in texte
        and not re.search(
            rf"(?:{'|'.join(MODIFICATEURS_NEGATIFS)})\s+\w*\s*{terme}", texte
        )
        for terme in TERMES_POSITIFS
    )

    if score > 0:
        return "ALERTE"
    if score < 0:
        return "POSITIF"
    return "NEUTRE"


def deriver_polarite_csm(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Ajoute `polarite_csm` a partir du commentaire, sans toucher au texte.

    La colonne est toujours renseignee : les clients sans commentaire portent
    la modalite ABSENT.
    """
    df = df.copy()
    commentaires = _valeurs_normalisees(df, "commentaire_csm")
    df["polarite_csm"] = (
        commentaires.map(_polarite, na_action="ignore")
        .fillna(POLARITE_ABSENTE)
        .astype(object)
    )

    if echo:
        repartition = df["polarite_csm"].value_counts()
        detail = ", ".join(f"{nom} {nombre}" for nom, nombre in repartition.items())
        _log_action(
            "derivation polarite_csm",
            int(commentaires.notna().sum()),
            int(df["polarite_csm"].notna().sum()),
            "valeurs",
            detail=f" ({detail})",
        )

    return df


# --- Regles metier ------------------------------------------------------

# Bornes attendues, valeur incluses.
BORNES: dict[str, tuple[float, float]] = {
    "anciennete_mois": (1, 36),
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


def _utilisateurs_actifs_valides(df: pd.DataFrame) -> pd.Series:
    """Positif, et plafonne au nombre de sieges souscrits par le client.

    Le plafond n'est pas une constante : c'est la valeur de `sieges_souscrits`
    de la ligne. Si l'une des deux colonnes manque, la comparaison est
    impossible et la ligne est conservee.
    """
    actifs = _numerique(df, "utilisateurs_actifs")
    sieges = _numerique(df, "sieges_souscrits")
    renseigne = _renseignee(df, "utilisateurs_actifs")

    positif = ~renseigne | (actifs >= 0)
    sous_plafond = ~(renseigne & _renseignee(df, "sieges_souscrits")) | (
        actifs <= sieges
    )
    return positif & sous_plafond


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
        "utilisateurs_actifs",
        "entre 0 et sieges_souscrits",
        lambda df: _utilisateurs_actifs_valides(df),
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


# --- Imputation ---------------------------------------------------------

# Colonnes dont les valeurs manquantes sont remplacees par la mediane.
IMPUTATIONS_MEDIANE: tuple[str, ...] = (
    "delai_reponse_support_h",
    "csat",
    "heures_usage_30j",
    "taux_adoption_pct",
    "retards_paiement_12m",
    "nb_integrations",
)

# Colonnes categorielles : valeurs manquantes remplacees par la modalite la
# plus frequente.
IMPUTATIONS_MODE: tuple[str, ...] = ("secteur", "pays")


def imputer_revenu_par_catalogue(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Reconstitue le revenu manquant : sieges souscrits x prix du plan.

    Le prix vient de `catalogue_silver`, deja chargee a ce stade. Sur les lignes
    ou le revenu est connu, ce calcul le retrouve a environ 9 % pres (remises
    commerciales), contre 91 % pour une imputation par la mediane.
    """
    colonne = "revenu_mensuel_recurrent_eur"
    df = df.copy()

    prix_par_plan = pd.read_sql(
        select(CATALOGUE_CIBLE.plan, CATALOGUE_CIBLE.prix_mensuel_par_siege_eur),
        get_engine(),
    ).set_index("plan")["prix_mensuel_par_siege_eur"]

    revenus = pd.to_numeric(df[colonne], errors="coerce")
    avant = int(revenus.notna().sum())

    calcules = pd.to_numeric(df["sieges_souscrits"], errors="coerce") * df["plan"].map(
        prix_par_plan.astype(float)
    )
    completes = revenus.fillna(calcules)

    df[colonne] = completes.map(
        lambda valeur: round(float(valeur), 2) if pd.notna(valeur) else None
    ).astype(object)

    if echo:
        _log_action(
            f"imputation {colonne}",
            avant,
            int(completes.notna().sum()),
            "valeurs",
            detail=" (sieges x prix du plan)",
        )
        restantes = int(completes.isna().sum())
        if restantes:
            print(
                f"WARNING : {colonne} : {restantes} valeurs non calculables "
                f"(sieges ou plan manquant)"
            )

    return df


def imputer_valeurs_manquantes(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Comble les valeurs manquantes : mediane pour les numeriques, mode pour
    les categorielles.

    Mediane et mode sont calcules sur l'ensemble des lignes : si un decoupage
    train/test intervient plus tard, ils devront etre recalcules sur le train
    seul pour ne pas y faire fuiter le test.
    """
    df = df.copy()

    for colonne in IMPUTATIONS_MEDIANE:
        valeurs = pd.to_numeric(df[colonne], errors="coerce")
        avant = int(valeurs.notna().sum())
        mediane = valeurs.median()

        if pd.isna(mediane):
            if echo:
                print(f"WARNING : {colonne} : aucune valeur, imputation impossible")
            continue

        # Une colonne entiere doit le rester : la mediane peut tomber sur x.5.
        if colonne in COLONNES_ENTIERES:
            mediane = round(float(mediane))
            remplies = valeurs.fillna(mediane).round().astype(int)
        else:
            mediane = float(mediane)
            remplies = valeurs.fillna(mediane).astype(float)

        df[colonne] = remplies.astype(object)

        if echo:
            _log_action(
                f"imputation {colonne}",
                avant,
                int(remplies.notna().sum()),
                "valeurs",
                detail=f" (mediane = {mediane:g})",
            )

    for colonne in IMPUTATIONS_MODE:
        valeurs = df[colonne].astype("string")
        avant = int(valeurs.notna().sum())
        modes = valeurs.mode()

        if modes.empty:
            if echo:
                print(f"WARNING : {colonne} : aucune valeur, imputation impossible")
            continue

        mode = modes.iloc[0]
        remplies = valeurs.fillna(mode)
        df[colonne] = remplies.astype(object)

        if echo:
            _log_action(
                f"imputation {colonne}",
                avant,
                int(remplies.notna().sum()),
                "valeurs",
                detail=f" (mode = {mode})",
            )

    return df


# --- Analyse des valeurs extremes ---------------------------------------

# Critere usuel du boxplot : au-dela de 1.5 x IQR de part et d'autre des
# quartiles, la valeur est consideree extreme.
FACTEUR_IQR = 1.5


def analyser_outliers_iqr(df: pd.DataFrame, echo: bool = True) -> pd.DataFrame:
    """Signale les valeurs extremes de chaque colonne numerique.

    Action de diagnostic : elle ne modifie ni ne supprime aucune ligne. Les
    bornes sont Q1 - 1.5 x IQR et Q3 + 1.5 x IQR.
    """
    if not echo:
        return df

    colonnes = (*COLONNES_ENTIERES, *COLONNES_DECIMALES)
    lignes: list[tuple[str, int, float, str, str]] = []
    total = 0

    for colonne in colonnes:
        valeurs = pd.to_numeric(df[colonne], errors="coerce").dropna()
        if valeurs.empty:
            continue

        premier, troisieme = valeurs.quantile([0.25, 0.75])
        ecart = troisieme - premier
        borne_basse = premier - FACTEUR_IQR * ecart
        borne_haute = troisieme + FACTEUR_IQR * ecart

        inferieurs = valeurs[valeurs < borne_basse]
        superieurs = valeurs[valeurs > borne_haute]
        nombre = len(inferieurs) + len(superieurs)
        total += nombre

        lignes.append(
            (
                colonne,
                nombre,
                nombre / len(valeurs),
                _intervalle(inferieurs),
                _intervalle(superieurs),
            )
        )

    print()
    print(f"[OUTLIERS IQR] : {total} valeurs extremes (seuil {FACTEUR_IQR:g} x IQR)")

    largeur = max(len(colonne) for colonne, *_ in lignes)
    entete = (
        f"  {'colonne':<{largeur}} {'volume':>7} {'part':>8}  "
        f"{'inferieurs (min..max)':<24} superieurs (min..max)"
    )
    print(entete)
    for colonne, nombre, part, bas, haut in sorted(lignes, key=lambda l: -l[2]):
        print(f"  {colonne:<{largeur}} {nombre:>7} {part:>7.2%}  {bas:<24} {haut}")

    return df


def _intervalle(valeurs: pd.Series) -> str:
    """ "min..max" des valeurs extremes d'un cote, ou "-" s'il n'y en a aucune."""
    if valeurs.empty:
        return "-"
    return f"{valeurs.min():g}..{valeurs.max():g}"


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
    deriver_polarite_csm,
    appliquer_regles_metier,
    typer_colonnes,
    imputer_revenu_par_catalogue,
    imputer_valeurs_manquantes,
    analyser_outliers_iqr,
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
        f"[DONNEES MANQUANTES] : {total / cellules:.2%} des valeurs "
        f"({total} / {cellules})"
    )

    largeur = max(len(colonne) for colonne in colonnes)
    # kind="stable" : les colonnes a egalite restent dans l'ordre du modele.
    for colonne, nombre in manquantes.sort_values(
        ascending=False, kind="stable"
    ).items():
        print(f"  {colonne:<{largeur}} : {int(nombre) / len(df):>7.2%} ({int(nombre)})")


# --- Catalogue des plans ------------------------------------------------

CATALOGUE_SOURCE = CatalogueBronze
CATALOGUE_CIBLE = CatalogueSilver

CATALOGUE_ENTIERS: tuple[str, ...] = (
    "fonctionnalites_incluses",
    "sla_reponse_h",
    "quota_stockage_go",
)
CATALOGUE_DECIMAUX: tuple[str, ...] = ("prix_mensuel_par_siege_eur",)

BOOLEENS = {"oui": True, "non": False}


def ingerer_catalogue(session: Session, *, echo: bool = True) -> int:
    """bronze.catalogue_bronze -> silver.catalogue_silver.

    Le plan recoit le meme code que dans `churn_saas_silver`, condition pour
    pouvoir joindre les deux tables.
    """
    colonnes = [
        colonne
        for colonne in CATALOGUE_SOURCE.__table__.columns
        if colonne.key in CATALOGUE_CIBLE.__table__.columns
    ]
    df = pd.read_sql(
        select(*colonnes).order_by(CATALOGUE_SOURCE._source_line), session.connection()
    )

    if echo:
        print(f"{CATALOGUE_SOURCE.__table__.fullname} : {len(df)} lignes lues\n")

    df = _standardiser_categorie(df, "plan", PLANS, echo)

    for colonne in CATALOGUE_ENTIERS:
        valeurs = _numerique(df, colonne)
        df[colonne] = valeurs.map(
            lambda valeur: round(valeur) if pd.notna(valeur) else None
        ).astype(object)

    for colonne in CATALOGUE_DECIMAUX:
        valeurs = _numerique(df, colonne)
        df[colonne] = valeurs.map(
            lambda valeur: float(valeur) if pd.notna(valeur) else None
        ).astype(object)

    # "Oui" / "Non" -> booleen.
    supports = _valeurs_normalisees(df, "support_dedie")
    inconnus = sorted(supports[supports.notna() & ~supports.isin(BOOLEENS)].unique())
    df["support_dedie"] = supports.map(BOOLEENS).astype(object)

    if echo:
        _log_action(
            "typage catalogue",
            int(supports.notna().sum()),
            int(df["support_dedie"].notna().sum()),
            "valeurs",
        )
        if inconnus:
            print(f"WARNING : support_dedie : valeurs non reconnues -> {inconnus}")

    CATALOGUE_CIBLE.__table__.drop(get_engine(), checkfirst=True)
    Base.metadata.create_all(get_engine())

    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    session.execute(CATALOGUE_CIBLE.__table__.insert(), rows)
    session.commit()

    inserted = session.scalar(
        text(
            f'SELECT count(*) FROM "{SILVER_SCHEMA}"."{CATALOGUE_CIBLE.__tablename__}"'
        )
    )

    if echo:
        print(f"\n{CATALOGUE_CIBLE.__table__.fullname} : {inserted} lignes inserees")

    return inserted


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


def ingest_silver(*, echo: bool = True) -> dict[str, int]:
    """Recharge les tables silver a partir de bronze.

    Retourne le nombre de lignes inserees par table.
    """
    _reinitialiser_groupes()

    ensure_schema(SILVER_SCHEMA)
    # La table est entierement rechargee a chaque execution : on la recree pour
    # qu'elle suive le modele, y compris quand les types changent.
    TARGET_MODEL.__table__.drop(get_engine(), checkfirst=True)
    Base.metadata.create_all(get_engine())

    with get_session() as session:
        lignes_catalogue = ingerer_catalogue(session, echo=echo)

        if echo:
            print()
        _reinitialiser_groupes()

        df = lire_bronze(session, echo=echo)

        for transformation in TRANSFORMATIONS:
            df = transformation(df, echo)

        if echo:
            _log_donnees_manquantes(df)

        inserted = ecrire_silver(session, df, echo=echo)

    resultats = {
        CATALOGUE_CIBLE.__table__.fullname: lignes_catalogue,
        TARGET_MODEL.__table__.fullname: inserted,
    }

    if echo:
        log_total(resultats)

    return resultats


app = typer.Typer(help=__doc__)


@app.command()
def main() -> None:
    ingest_silver()


if __name__ == "__main__":
    app()
