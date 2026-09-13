"""Couche bronze : les CSV de docs/ stockes tels quels.

Aucune conversion de type ni nettoyage a ce niveau : toutes les colonnes
metier sont du texte, y compris les nombres et les dates. Les transformations
(dates heterogenes, virgules decimales, casse, espaces) sont du ressort de la
couche silver.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ml_churn.ingestion.models.base import Base

BRONZE_SCHEMA = "bronze"


class _BronzeMixin:
    """Colonnes techniques communes, prefixees pour ne pas heurter le CSV."""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    _source_file: Mapped[str] = mapped_column(Text, nullable=False)
    _source_line: Mapped[int] = mapped_column(nullable=False)
    _ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CatalogueBronze(_BronzeMixin, Base):
    """docs/catalogue_plans.csv"""

    __tablename__ = "catalogue_bronze"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": BRONZE_SCHEMA}

    plan: Mapped[str | None] = mapped_column(Text)
    prix_mensuel_par_siege_eur: Mapped[str | None] = mapped_column(Text)
    fonctionnalites_incluses: Mapped[str | None] = mapped_column(Text)
    sla_reponse_h: Mapped[str | None] = mapped_column(Text)
    quota_stockage_go: Mapped[str | None] = mapped_column(Text)
    support_dedie: Mapped[str | None] = mapped_column(Text)


class _ChurnSaasBronzeMixin(_BronzeMixin):
    """Colonnes des deux exports churn (dataset complet et echantillon)."""

    client_id: Mapped[str | None] = mapped_column(Text)
    date_souscription: Mapped[str | None] = mapped_column(Text)
    jour_souscription: Mapped[str | None] = mapped_column(Text)
    secteur: Mapped[str | None] = mapped_column(Text)
    pays: Mapped[str | None] = mapped_column(Text)
    taille_entreprise: Mapped[str | None] = mapped_column(Text)
    plan: Mapped[str | None] = mapped_column(Text)
    anciennete_mois: Mapped[str | None] = mapped_column(Text)
    sieges_souscrits: Mapped[str | None] = mapped_column(Text)
    utilisateurs_actifs: Mapped[str | None] = mapped_column(Text)
    taux_adoption_pct: Mapped[str | None] = mapped_column(Text)
    connexions_30j: Mapped[str | None] = mapped_column(Text)
    heures_usage_30j: Mapped[str | None] = mapped_column(Text)
    fonctionnalites_total: Mapped[str | None] = mapped_column(Text)
    fonctionnalites_utilisees: Mapped[str | None] = mapped_column(Text)
    nb_integrations: Mapped[str | None] = mapped_column(Text)
    derniere_connexion_jours: Mapped[str | None] = mapped_column(Text)
    tickets_support_90j: Mapped[str | None] = mapped_column(Text)
    delai_reponse_support_h: Mapped[str | None] = mapped_column(Text)
    csat: Mapped[str | None] = mapped_column(Text)
    retards_paiement_12m: Mapped[str | None] = mapped_column(Text)
    revenu_mensuel_recurrent_eur: Mapped[str | None] = mapped_column(Text)
    couleur_theme_interface: Mapped[str | None] = mapped_column(Text)
    code_datacenter: Mapped[str | None] = mapped_column(Text)
    groupe_experimentation: Mapped[str | None] = mapped_column(Text)
    commentaire_csm: Mapped[str | None] = mapped_column(Text)
    sante_compte_fin_periode: Mapped[str | None] = mapped_column(Text)
    valeur_vie_client_eur: Mapped[str | None] = mapped_column(Text)
    churn: Mapped[str | None] = mapped_column(Text)


class ChurnSaasCompletBronze(_ChurnSaasBronzeMixin, Base):
    """docs/churn_saas_complet.csv"""

    __tablename__ = "churn_saas_complet_bronze"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": BRONZE_SCHEMA}


class ChurnSaasEchantillonBronze(_ChurnSaasBronzeMixin, Base):
    """docs/churn_saas_echantillon.csv"""

    __tablename__ = "churn_saas_echantillon_bronze"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": BRONZE_SCHEMA}
