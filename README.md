## Notes personnelles

### Données

- catalogue_plans : offre de service
- churn_saas_complet : dataset complet (5035 lignes)
- churn_saas_echantillon : echantillon (50 lignes) ≈ 1%

### Acteurs

- Editeur SaaS
- Equipe customer
- Clients

### Objectifs

Estimer la probabilité de churn d'un client (résiliation à l'échéance)

livrable :
- modele de classification (cible principal) : le client resilie ou ne resilie pas à l'échéance
- modele de regression (cible secondaire) : estimation de la valeur vie client

### Cadrage

KPI : % de client qui ont résiliés par taille d'entreprise et formule (chiffre à minimiser le plus possible)

### Etapes

# 1. Visualisation des données sous forme graphique

Scripts de génération : `src/visualization/scripts`
Graphiques générés : `src/visualization/graphs`

| Colonne | Distribution |
| --- | --- |
| `date_souscription` | asymétrique |
| `jour_souscription` | normale |
| `anciennete_mois` | asymétrique |
| `sieges_souscrits` | asymétrique |
| `utilisateurs_actifs` | asymétrique |
| `taux_adoption_pct` | asymétrique |
| `connexions_30j` | asymétrique |
| `heures_usage_30j` | asymétrique |
| `nb_integrations` | asymétrique |
| `derniere_connexion_jours` | asymétrique |
| `tickets_support_90j` | asymétrique |
| `delai_reponse_support_h` | asymétrique |
| `csat` | asymétrique |
| `retards_paiement_12m` | asymétrique |
| `sante_compte_fin_periode` | asymétrique |

# 2. Ingestion

On utilise uniquement le fichire churn_saas_complet comme source de donnée, churn_saas_echantillon n'est pas inclu car cela provoquerait des doublons.
Pas besoin d'anonymisation car les données concernent des entreprises.

## 🥈 Silver

Chaque action est une fonction indépendante référencée dans `TRANSFORMATIONS`
(`src/ml_churn/ingestion/scripts/ingest_silver.py`) et appliquée dans cet ordre.
Le catalogue est chargé en amont, ses plans servant à l'imputation du revenu.

| # | Action | Effet |
| --- | --- | --- |
| 1 | `dedupliquer_clients` | Un client_id ne doit apparaitre qu'une fois. |
| 2 | `standardiser_date_souscription` | Ramene les trois formats de date rencontres au seul format AAAA-MM-JJ. |
| 3 | `standardiser_jour_souscription` | lundi -> L, mardi -> M, mercredi -> ME, jeudi -> J, vendredi -> V, samedi -> S, dimanche -> D. |
| 4 | `standardiser_secteur` | Tech -> TE, Finance -> FI, Commerce -> CO, Sante -> SA, Industrie -> IN, Public -> PB, Education -> EN. |
| 5 | `standardiser_pays` | Nom du pays -> code ISO 3166-1 alpha-2 (France -> FR, Allemagne -> DE...). |
| 6 | `standardiser_taille_entreprise` | Harmonise la casse : tpe / " TPE " -> TPE, idem PME, ETI, GE. |
| 7 | `standardiser_plan` | Pro -> PRO, Business -> BUS, Starter -> STR, Enterprise -> ENT. |
| 8 | `standardiser_couleur_theme_interface` | clair -> C, vert -> VE, bleu -> B, violet -> V, sombre -> S. |
| 9 | `standardiser_groupe_experimentation` | A -> A, B -> B, control -> C. |
| 10 | `deriver_polarite_csm` | Ajoute `polarite_csm` a partir du commentaire, sans toucher au texte. |
| 11 | `appliquer_regles_metier` | Supprime les lignes violant une regle metier, colonne par colonne. |
| 12 | `typer_colonnes` | Convertit les colonnes texte vers les types du modele silver. |
| 13 | `imputer_revenu_par_catalogue` | Reconstitue le revenu manquant : sieges souscrits x prix du plan. |
| 14 | `imputer_valeurs_manquantes` | Comble les valeurs manquantes : médiane pour les numériques, mode pour les catégorielles. |

Les sections ci-dessous détaillent chacune de ces étapes.

### 1. Déduplication

`client_id` ne doit apparaître qu'une fois : 5035 lignes → 5000.

### 2. Standardisation

Casse et espaces parasites sont harmonisés avant toute correspondance : le CSV
contient `TPE`, `" TPE "` et `tpe` pour la même valeur. Une valeur vide reste à
`NULL` ; une valeur hors table de correspondance déclenche un `WARNING` et est
mise à `NULL`.

#### Clients (`churn_saas_silver`)

| Colonne | Transformation |
| --- | --- |
| `date_souscription` | `AAAA-MM-JJ`, `JJ/MM/AAAA` et `JJ mois AAAA` → **`AAAA-MM-JJ`** |
| `jour_souscription` | lundi→`L`, mardi→`M`, mercredi→`ME`, jeudi→`J`, vendredi→`V`, samedi→`S`, dimanche→`D` |
| `secteur` | Tech→`TE`, Finance→`FI`, Commerce→`CO`, Santé→`SA`, Industrie→`IN`, Public→`PB`, Éducation→`EN` |
| `pays` | nom → code ISO 3166-1 alpha-2 (`FR`, `ES`, `CA`, `DE`, `CH`, `BE`) |
| `taille_entreprise` | harmonisation de casse → `TPE`, `PME`, `ETI`, `GE` |
| `plan` | Pro→`PRO`, Business→`BUS`, Starter→`STR`, Enterprise→`ENT` |
| `couleur_theme_interface` | clair→`C`, vert→`VE`, bleu→`B`, violet→`V`, sombre→`S` |
| `groupe_experimentation` | A→`A`, B→`B`, control→`C` |

#### Catalogue des plans (`catalogue_silver`)

| Colonne | Transformation |
| --- | --- |
| `plan` | Pro→`PRO`, Business→`BUS`, Starter→`STR`, Enterprise→`ENT` |
| `support_dedie` | `Oui` / `Non` → booléen |

Le plan reçoit le même code que dans `churn_saas_silver` : c'est ce qui permet
de joindre les deux tables (`join silver.catalogue_silver using (plan)`).

### 3. Polarité du commentaire CSM

`commentaire_csm` provient d'une liste fermée de 13 formulations. Une analyse de
polarité par lexique en dérive la colonne `polarite_csm`, sans toucher au texte
d'origine : termes négatifs (*insatisfait*, *risque*, *baisse*…), termes positifs
(*satisfait*, *engagé*, *ambassadeur*) et modificateurs inverseurs (*peu*,
*faible*, comme dans « compte **peu actif** »). Le signe du score décide.

| Polarité | Termes du lexique |
| --- | --- |
| `ALERTE` | `insatisfait`, `mecontentement`, `risque`, `baisse`, `friction`, `limite`, `relance`, `sollicite`, `multiples tickets`, `depart` — ainsi que les modificateurs `peu` et `faible` |
| `POSITIF` | `satisfait`, `engage`, `ambassadeur`, `actif` — sauf précédés d'un modificateur (« peu actif ») |
| `NEUTRE` | aucun terme du lexique reconnu |
| `ABSENT` | pas de commentaire |

Un TF-IDF a été écarté : sur des phrases aussi courtes, il regroupe sur le
vocabulaire partagé et classe « Client très satisfait » avec « Client
insatisfait ».

> À confirmer auprès de l'équipe customer : si ces commentaires sont saisis à la
> résiliation plutôt qu'en amont, la colonne est une fuite de données, au même
> titre que `sante_compte_fin_periode`.

### 4. Règles métier

Toute ligne violant une règle est **supprimée**. Conventions : une valeur absente
n'est pas une violation (seules les valeurs présentes et hors bornes sont
rejetées) ; exception pour `client_id`, où l'absence est une violation puisque
c'est la clé.

| Colonne | Règle |
| --- | --- |
| `client_id` | format `CLI-<chiffres>` |
| `anciennete_mois` | entre 1 et 36 |
| `utilisateurs_actifs` | entre 0 et `sieges_souscrits` de la ligne |
| `taux_adoption_pct` | entre 0 et 100 |
| `csat` | entre 1 et 5 |
| `sante_compte_fin_periode` | entre 0 et 100 |
| `churn` | vaut 0 ou 1 |
    
### 5.Gestion des outliers

L'observation visuelle des graphiques ne denote aucun outliers significatif

### 6. Typage

Les colonnes texte sont converties vers les types du modèle : `date` pour la
souscription, `integer` pour 13 colonnes, `numeric` pour les taux, durées et
montants, `varchar` pour les codes. La lecture numérique tolère les unités du
CSV (`20.0%`, `3.1 h`, `280.62 €`, `40,0`) : sans cela, environ 1260 valeurs
valides seraient perdues.

### 7.Imputation

Appliquée après le typage. Médiane pour les colonnes numériques, mode pour les
catégorielles. La valeur retenue est calculée sur les 5000 lignes et affichée
dans le log de l'ingestion.

| Colonne | Méthode |
| --- | --- |
| `delai_reponse_support_h` | médiane |
| `csat` | médiane |
| `heures_usage_30j` | médiane |
| `taux_adoption_pct` | médiane |
| `retards_paiement_12m` | médiane |
| `secteur` | mode |
| `nb_integrations` | médiane |
| `pays` | mode |
| `revenu_mensuel_recurrent_eur` | `sieges_souscrits` × `prix_mensuel_par_siege_eur` |
| `polarite_csm` | modalité explicite |

> Les valeurs sont calculées sur l'ensemble du dataset. En cas de découpage
> train/test ultérieur, elles devront être recalculées sur le train seul pour
> ne pas y faire fuiter le test.

### Base de données

PostgreSQL via Docker Compose (`docker-compose.yml`) :
On utilise le robust z-score
```bash
cp .env.example .env
docker compose up -d
```

Les schémas `bronze`, `silver` et `gold` sont créés au premier démarrage par
`docker/postgres/init/01-schemas.sql` (rejoué uniquement si le volume est vide :
`docker compose down -v` pour repartir de zéro).

#### pgAdmin

Interface d'administration sur http://localhost:5050 — pas d'écran de connexion
(mode local), le serveur `ml-churn` est déjà déclaré dans l'arbre via
`docker/pgadmin/servers.json`. Au premier clic sur le serveur, pgAdmin demande le
mot de passe de l'utilisateur `mlchurn` (celui du `.env`, `mlchurn` par défaut) ;
cocher « Save password » pour ne plus l'avoir à le saisir.

Accès en ligne de commande sans passer par l'interface :

```bash
docker compose exec postgres psql -U mlchurn -d mlchurn
```

### 🥉 Ingestion — couche bronze

Les CSV de `docs/` sont copiés tels quels dans le schéma `bronze` : **toutes les
colonnes métier sont en `TEXT`**, aucune conversion ni nettoyage (dates
hétérogènes, virgules décimales, casse et espaces restent bruts — c'est le
travail de la couche silver). Chaque ligne porte en plus `_source_file`,
`_source_line` et `_ingested_at`.

| CSV | Table |
| --- | --- |
| `docs/catalogue_plans.csv` | `bronze.catalogue_bronze` |
| `docs/churn_saas_complet.csv` | `bronze.churn_saas_complet_bronze` |
| `docs/churn_saas_echantillon.csv` | `bronze.churn_saas_echantillon_bronze` |

```bash
uv run python -m ml_churn.ingestion.scripts.ingest_bronze
```

Les tables sont vidées puis rechargées : relancer l'ingestion ne crée pas de
doublons (`--append` pour conserver l'existant). Le code est dans
`src/ml_churn/ingestion/` (`models/` pour les modèles SQLAlchemy, `scripts/`
pour les scripts), et la dernière cellule de `churn-notebook.ipynb` lance la
même ingestion.
