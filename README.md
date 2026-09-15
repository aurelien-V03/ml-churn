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

Estimer la probabilité de churn d'un client

livrable :
- modele de classification (cible principal) : le client resilie ou ne resilie pas à l'échéance
- modele de regression (cible secondaire) : estimation de la valeur vie client

### Cadrage

KPI : % de client qui ont résiliés par taille d'entreprise et formule (chiffre à minimiser le plus possible)

### Etapes

- Pipeline données (architecture en médaillon)
    - bronze
    stockage telle quelle
    - silver

#### 1. Déduplication
`client_id` ne doit apparaître qu'une fois. 

#### 2. Standardisation

Casse et espaces parasites sont harmonisés avant toute correspondance : le CSV
contient `TPE`, `" TPE "` et `tpe` pour la même valeur. Une valeur vide reste à
`NULL` ; une valeur hors table de correspondance déclenche un `WARNING` et est
mise à `NULL`.

| Colonne | Transformation |
| --- | --- |
| `date_souscription` | `AAAA-MM-JJ`, `JJ/MM/AAAA` et `JJ mois AAAA` → **`AAAA-MM-JJ`** |
| `jour_souscription` | lundi→`L`, mardi→`M`, mercredi→`ME`, jeudi→`J`, vendredi→`V`, samedi→`S`, dimanche→`D` |
| `secteur` | Tech→`TE`, Finance→`FI`, Commerce→`CO`, Santé→`SA`, Industrie→`IN`, Public→`PB`, Éducation→`EN` |
| `pays` | nom → code ISO 3166-1 alpha-2 (`FR`, `ES`, `CA`, `DE`, `CH`, `BE`) |
| `taille_entreprise` | harmonisation de casse → `TPE`, `PME`, `ETI`, `GE` |
| `plan` | Pro→`PRO`, Business→`BUS`, Starter→`STR`, Enterprise→`ENT` |
| `couleur_theme_interface` | clair→`C`, vert→`V`, bleu→`B`, violet→`V`, sombre→`S` |
| `groupe_experimentation` | A→`A`, B→`B`, control→`C` |

#### 3. Règles métier

Toute ligne violant une règle est **supprimée**. Conventions : une valeur absente
n'est pas une violation (seules les valeurs présentes et hors bornes sont
rejetées) ; exception pour `client_id`, où l'absence est une violation puisque
c'est la clé.

| Colonne | Règle |
| --- | --- |
| `client_id` | format `CLI-<chiffres>` |
| `anciennete_mois` | entre 1 et 36 |
| `sieges_souscrits` | entre 1 et 898 |
| `utilisateurs_actifs` | entre 0 et 829, et `<= sieges_souscrits` |
| `taux_adoption_pct` | entre 0 et 100 |
| `csat` | entre 1 et 5 |
| `sante_compte_fin_periode` | entre 0 et 100 |
| `churn` | vaut 0 ou 1 |

- standardisation des données : 
date_souscription : il y a plusieurs format dans le fichier AAAA-MM-JJ,
JJ/MM/AAAA, JJ mois AAAA, je veux un unique format (AAAA-MM-JJ)
jour_souscription : les valeurs dans le csv sont les suivantes (mardi, jeudi, samedi,
dimanche, vendredi, mercredi, lundi), je veux que tu les convertises au format (L,M,ME,J,V,S,D)
plan: les valeurs (Pro, Business, Starter, Enterprise) sont transformées (PRO, BUS, STR, ENT)
 secteur : tu convertis les valeurs (Tech, Finance, Commerce, Santé, Industrie, Public,Éducation) en (TE, FI, CO, SA, IN, PB, EN)
pays : tu convertis les valeurs (France, Espagne, Canada, Allemagne, Suisse, Belgique) dans le code du pays
plan: les valeurs doivent etre (TPE, PME, ETI, GE), convertis si la case est différent
couleur_theme_interface : les valeurs (clair, vert, bleu, violet, sombre) deviennent (C, V, B, V, S)
groupe_experimentation  : les valeurs (B, A, control) deviennent (B,A,C)

Regles metiers
client_id : doit avoir le format CLI-
anciennete_mois : 1-36
sieges_souscrits : 1-898
utilisateurs_actifs : 0 - 829 et <= sieges_souscrits
taux_adoption_pct : 0 - 100
csat : 1-5
sante_compte_fin_periode : 0-100
churn: 0 ou 1

- typage des propriétés : le type string de bronze est convertis vers son type correspondant dans silver


    - gold
### Base de données

PostgreSQL via Docker Compose (`docker-compose.yml`) :

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

### Ingestion — couche bronze

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
