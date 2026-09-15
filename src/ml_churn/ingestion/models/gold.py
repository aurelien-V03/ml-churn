"""Couche gold : donnees pretes a l'usage, exposees aux consommateurs.

Les tables reprennent la structure de silver (memes colonnes, memes types),
mais les definitions sont independantes : chaque couche peut evoluer sans
entrainer l'autre.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from sqlalchemy import Boolean, Date, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ml_churn.ingestion.models.base import Base

GOLD_SCHEMA = "gold"


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
    fonctionnalites_total: Mapped[int | None] = mapped_column(Integer)
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
    commentaire_csm: Mapped[str | None] = mapped_column(Text)
    # Polarite derivee du commentaire (cf. deriver_polarite_csm).
    polarite_csm: Mapped[str | None] = mapped_column(String(8))
    sante_compte_fin_periode: Mapped[int | None] = mapped_column(Integer)
    churn: Mapped[int | None] = mapped_column(Integer)

    _transformed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
