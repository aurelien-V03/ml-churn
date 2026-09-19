"""Couche gold : donnees pretes a l'usage, exposees aux consommateurs.

Les tables reprennent la structure de silver (memes colonnes, memes types),
completee par les colonnes calculees pour la modelisation. Les definitions sont
independantes : chaque couche peut evoluer sans entrainer l'autre.

`commentaire_csm` n'est volontairement pas repris : `polarite_csm` le resume.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from sqlalchemy import Boolean, Date, DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ml_churn.ingestion.models.base import Base

GOLD_SCHEMA = "gold"


def nom_colonne_one_hot(colonne: str, modalite: str) -> str:
    """Convention de nommage : [nom_categorie]_valeur.

    La modalite est ramenee a un identifiant SQL valide : minuscules, et tout
    caractere non alphanumerique remplace par un underscore ("eu-w1" ->
    "eu_w1").
    """
    return f"{colonne}_{re.sub(r'[^0-9a-z]+', '_', modalite.lower())}"


# Modalites encodees en one-hot, dans l'ordre des colonnes generees.
MODALITES_ONE_HOT: dict[str, tuple[str, ...]] = {
    "jour_souscription": ("L", "M", "ME", "J", "V", "S", "D"),
    "secteur": ("TE", "FI", "CO", "SA", "IN", "PB", "EN"),
    "pays": ("FR", "ES", "CA", "DE", "CH", "BE"),
    "taille_entreprise": ("TPE", "PME", "ETI", "GE"),
    "plan": ("STR", "PRO", "BUS", "ENT"),
    "couleur_theme_interface": ("C", "VE", "B", "V", "S"),
    "code_datacenter": ("eu-w1", "eu-w3", "us-e1", "ap-s1"),
    "groupe_experimentation": ("A", "B", "C"),
    # Colonne derivee (cf. ajouter_colonnes_derivees).
    "niveau_anciennete": ("RECENT", "ETABLI", "ANCIEN"),
}


class CatalogueGold(Base):
    """Catalogue des offres : un plan par ligne.

    `plan` porte le meme code que `ChurnSaasGold.plan` (STR, PRO, BUS, ENT),
    ce qui permet de joindre les deux tables.
    """

    __tablename__ = "catalogue_gold"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": GOLD_SCHEMA}

    plan: Mapped[str] = mapped_column(String(3), primary_key=True)
    prix_mensuel_par_siege_eur: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    fonctionnalites_incluses: Mapped[int | None] = mapped_column(Integer)
    sla_reponse_h: Mapped[int | None] = mapped_column(Integer)
    quota_stockage_go: Mapped[int | None] = mapped_column(Integer)
    support_dedie: Mapped[bool | None] = mapped_column(Boolean)

    _transformed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChurnSaasGold(Base):
    """Un client par ligne, `client_id` unique."""

    __tablename__ = "churn_saas_gold"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": GOLD_SCHEMA}

    client_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    date_souscription: Mapped[date | None] = mapped_column(Date)

    # Codes issus de la standardisation (L/M/ME..., TE/FI/CO..., FR/ES/CA...).
    jour_souscription: Mapped[str | None] = mapped_column(String(2))
    secteur: Mapped[str | None] = mapped_column(String(2))
    pays: Mapped[str | None] = mapped_column(String(2))
    taille_entreprise: Mapped[str | None] = mapped_column(String(3))
    plan: Mapped[str | None] = mapped_column(String(3))

    anciennete_mois: Mapped[int | None] = mapped_column(Integer)
    sieges_souscrits: Mapped[int | None] = mapped_column(Integer)
    utilisateurs_actifs: Mapped[int | None] = mapped_column(Integer)
    taux_adoption_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    connexions_30j: Mapped[int | None] = mapped_column(Integer)
    heures_usage_30j: Mapped[Decimal | None] = mapped_column(Numeric(7, 2))
    fonctionnalites_utilisees: Mapped[int | None] = mapped_column(Integer)
    nb_integrations: Mapped[int | None] = mapped_column(Integer)
    derniere_connexion_jours: Mapped[int | None] = mapped_column(Integer)
    tickets_support_90j: Mapped[int | None] = mapped_column(Integer)
    delai_reponse_support_h: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    csat: Mapped[int | None] = mapped_column(Integer)
    retards_paiement_12m: Mapped[int | None] = mapped_column(Integer)

    # Montants en euros : Numeric plutot que float, pas d'arrondi binaire.
    revenu_mensuel_recurrent_eur: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    valeur_vie_client_eur: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))

    couleur_theme_interface: Mapped[str | None] = mapped_column(String(2))
    code_datacenter: Mapped[str | None] = mapped_column(String(16))
    groupe_experimentation: Mapped[str | None] = mapped_column(String(1))
    # Le commentaire brut n'est pas repris en gold : seule sa polarite l'est.
    polarite_csm: Mapped[str | None] = mapped_column(String(8))
    sante_compte_fin_periode: Mapped[int | None] = mapped_column(Integer)
    churn: Mapped[int | None] = mapped_column(Integer)

    # --- Colonnes derivees ---
    # Latence de connexion rapportee a la duree de vie du compte : 15 jours
    # sans connexion ne pesent pas pareil a 1 mois et a 3 ans d'anciennete.
    inactivite_relative: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    inactif_30j: Mapped[int | None] = mapped_column(Integer)
    # Part des fonctionnalites du plan reellement utilisees (0 a 1).
    taux_fonctionnalites: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    niveau_anciennete: Mapped[str | None] = mapped_column(String(8))

    # --- Encodage one-hot des colonnes categorielles ---
    # jour_souscription
    jour_souscription_l: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_m: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_me: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_j: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_v: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_s: Mapped[int | None] = mapped_column(Integer)
    jour_souscription_d: Mapped[int | None] = mapped_column(Integer)
    # secteur
    secteur_te: Mapped[int | None] = mapped_column(Integer)
    secteur_fi: Mapped[int | None] = mapped_column(Integer)
    secteur_co: Mapped[int | None] = mapped_column(Integer)
    secteur_sa: Mapped[int | None] = mapped_column(Integer)
    secteur_in: Mapped[int | None] = mapped_column(Integer)
    secteur_pb: Mapped[int | None] = mapped_column(Integer)
    secteur_en: Mapped[int | None] = mapped_column(Integer)
    # pays
    pays_fr: Mapped[int | None] = mapped_column(Integer)
    pays_es: Mapped[int | None] = mapped_column(Integer)
    pays_ca: Mapped[int | None] = mapped_column(Integer)
    pays_de: Mapped[int | None] = mapped_column(Integer)
    pays_ch: Mapped[int | None] = mapped_column(Integer)
    pays_be: Mapped[int | None] = mapped_column(Integer)
    # taille_entreprise
    taille_entreprise_tpe: Mapped[int | None] = mapped_column(Integer)
    taille_entreprise_pme: Mapped[int | None] = mapped_column(Integer)
    taille_entreprise_eti: Mapped[int | None] = mapped_column(Integer)
    taille_entreprise_ge: Mapped[int | None] = mapped_column(Integer)
    # plan
    plan_str: Mapped[int | None] = mapped_column(Integer)
    plan_pro: Mapped[int | None] = mapped_column(Integer)
    plan_bus: Mapped[int | None] = mapped_column(Integer)
    plan_ent: Mapped[int | None] = mapped_column(Integer)
    # couleur_theme_interface
    couleur_theme_interface_c: Mapped[int | None] = mapped_column(Integer)
    couleur_theme_interface_ve: Mapped[int | None] = mapped_column(Integer)
    couleur_theme_interface_b: Mapped[int | None] = mapped_column(Integer)
    couleur_theme_interface_v: Mapped[int | None] = mapped_column(Integer)
    couleur_theme_interface_s: Mapped[int | None] = mapped_column(Integer)
    # code_datacenter
    code_datacenter_eu_w1: Mapped[int | None] = mapped_column(Integer)
    code_datacenter_eu_w3: Mapped[int | None] = mapped_column(Integer)
    code_datacenter_us_e1: Mapped[int | None] = mapped_column(Integer)
    code_datacenter_ap_s1: Mapped[int | None] = mapped_column(Integer)
    # groupe_experimentation
    groupe_experimentation_a: Mapped[int | None] = mapped_column(Integer)
    groupe_experimentation_b: Mapped[int | None] = mapped_column(Integer)
    groupe_experimentation_c: Mapped[int | None] = mapped_column(Integer)
    # niveau_anciennete
    niveau_anciennete_recent: Mapped[int | None] = mapped_column(Integer)
    niveau_anciennete_etabli: Mapped[int | None] = mapped_column(Integer)
    niveau_anciennete_ancien: Mapped[int | None] = mapped_column(Integer)

    _transformed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
