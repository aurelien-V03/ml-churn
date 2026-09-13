"""Couche silver : donnees nettoyees et typees a partir de la couche bronze.

Contrairement au bronze (tout en texte), chaque colonne porte ici son vrai
type : la conversion est faite par l'action `typer_colonnes` de
ingestion/scripts/ingest_silver.py.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from sqlalchemy import Date, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ml_churn.ingestion.models.base import Base

SILVER_SCHEMA = "silver"


class ChurnSaasSilver(Base):
    """Un client par ligne : client_id est unique apres deduplication."""

    __tablename__ = "churn_saas_silver"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": SILVER_SCHEMA}

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
    sante_compte_fin_periode: Mapped[int | None] = mapped_column(Integer)
    churn: Mapped[int | None] = mapped_column(Integer)

    _transformed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
