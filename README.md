## Structure du projet

```
churn-notebook.ipynb        Déroulé complet : exploration, ingestion, entraînement
docker-compose.yml          PostgreSQL + pgAdmin
docs/                       Données sources (CSV) et énoncé du cas d'usage
mlflow.db, mlartifacts/     Suivi des expérimentations (généré)

artifacts/                  Modèles entraînés, versionnés dans git
├── classification/         Une tâche par dossier
│   ├── baseline/           Régression logistique
│   └── final/              XGBoost
└── regression/
    ├── baseline/
    └── final/
        └── AAAA-MM-JJ/     Un dossier par jour d'entraînement
            ├── *.joblib    Le modèle sérialisé
            ├── *.json      Versions des bibliothèques, seuil, features, métriques
            └── *_train.csv Extraits gold consommés (train / validation / test)

src/
├── ml_churn/               Package installé (`uv run python -m ml_churn...`)
│   ├── api/                Service FastAPI : /health, /ready, /predict
│   │   └── data/           Schémas Pydantic des requêtes et réponses
│   ├── ui/                 Page de test : index.html, style.css, app.js, clients.js
│   ├── ingestion/          Médaillon : CSV → bronze → silver → gold
│   │   ├── db.py           Connexion PostgreSQL
│   │   ├── logs.py         Format de log commun aux trois couches
│   │   ├── models/         Modèles SQLAlchemy, un fichier par couche
│   │   └── scripts/        Un script par couche, plus `ingest_all`
│   └── training/
│       ├── common/         Gold, découpage, métriques, figures, SHAP, sauvegarde
│       │   └── tracking/   Suivi MLflow, commun à tous les modèles
│       ├── classification/ Prédiction du churn (cible `churn`)
│       │   ├── baseline/   Régression logistique, seuil fixe à 0.5
│       │   └── final/      XGBoost + recherche d'hyperparamètres
│       └── regression/     Valeur vie client (cible `valeur_vie_client_eur`)
│           ├── baseline/   Régression linéaire, paramètres par défaut
│           └── final/      XGBoost
└── visualization/          Hors package, importé via `sys.path`
    ├── scripts/            Un script par type de graphique, plus `plot_all`
    └── graphs/             PNG générés, un dossier par colonne (généré)
```

Trois responsabilités séparées :

| Dossier | Rôle |
| --- | --- |
| `ingestion/` | Charger et transformer les données, du CSV brut à la table exploitable |
| `visualization/` | Comprendre les données : distributions, valeurs extrêmes, relations au churn |
| `training/` | Entraîner et évaluer les modèles, en suivant les runs dans MLflow |
| `api/` | Servir les modèles entraînés par HTTP, sans jamais réentraîner |

Les scripts sont autonomes, exécutables en ligne de commande comme importables
depuis le notebook. Le code partagé entre
plusieurs scripts vit dans un `common/` — jamais dupliqué d'un modèle à
l'autre.

## Commandes utiles

- docker compose `docker compose up -d`
- serveur FastApi `uv run uvicorn ml_churn.api.main:app --reload`
- mlflow `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

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


### Etapes

# 2. Choix du type de modèle

On connait le label (ce que l'on veut prédire = churn) donc il s'agit d'un problème de machine learning
avec apprentissage supervisé

# 3. Visualisation des données sous forme graphique

Source : la table `bronze.churn_saas_complet_bronze`, où le CSV est stocké tel
quel — l'ingestion bronze doit donc avoir tourné.

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

Observation par rapport aux hypothèses :

- satisfaction client
    - Plus le délai de réponse du support et long plus le taux de churn augmente
    - Plus le nombre de tickets supports au cours des 90j augmente plus le taux de churn augmente
- Bon payeur
    - Plus l'entreprise est grosse plus le churn rate tend à baisser
    - Plus le client a des retards de payement plus le taux de churn augmente

# 4. Ingestion

### 🥉 Bronze

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

On utilise uniquement le fichire churn_saas_complet comme source de donnée, churn_saas_echantillon n'est pas inclu car cela provoquerait des doublons.
Pas besoin d'anonymisation car les données concernent des entreprises.

### 🥈 Silver

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

#### Colonnes écartées

Le filtre se déclare dans le modèle : une colonne absente de `ChurnSaasSilver`
n'est pas lue depuis bronze.

| Colonne | Raison |
| --- | --- |
| `fonctionnalites_total` | Dénormalisation du plan, identique à `catalogue_silver.fonctionnalites_incluses` sur toutes les lignes. Le catalogue est la source de vérité. |

Toutes les autres colonnes du CSV sont reprises.

### 1. Déduplication

`client_id` ne doit apparaître qu'une fois.

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
CSV (`20.0%`, `3.1 h`, `280.62 €`, `40,0`) : sans cela, des valeurs valides
seraient perdues.

### 7.Imputation

Appliquée après le typage. Médiane pour les colonnes numériques, mode pour les
catégorielles. La valeur retenue est calculée sur l'ensemble des lignes et
affichée dans le log de l'ingestion.

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

## 🥇 Gold

`silver` → `gold`, pour les deux tables : `catalogue_gold` et `churn_saas_gold`.
La structure de silver est reprise, complétée par des colonnes calculées
destinées à la modélisation. La couche est entièrement rechargée à chaque
exécution.

#### Colonnes écartées

Même mécanisme qu'en silver : une colonne absente de `ChurnSaasGold` n'est pas
lue depuis silver.

| Colonne | Raison |
| --- | --- |
| `commentaire_csm` | Texte libre sans usage en modélisation ; `polarite_csm` en conserve le signal. |

`fonctionnalites_total`, déjà écartée en silver, n'apparaît donc pas non plus
ici.

Comme en silver, chaque action est une fonction indépendante, listée dans les
`transformations` de la table concernée
(`src/ml_churn/ingestion/scripts/ingest_gold.py`) :

| # | Action | Effet |
| --- | --- | --- |
| 1 | `ajouter_colonnes_derivees` | Calcule les 4 colonnes ci-dessous |
| 2 | `encoder_categorielles` | Encode les colonnes catégorielles en one-hot |

### 1. Colonnes dérivées

| Colonne | Calcul | Intérêt |
| --- | --- | --- |
| `inactivite_relative` | `derniere_connexion_jours / (anciennete_mois × 30)` | 15 jours sans connexion ne pèsent pas pareil à 1 mois et à 3 ans d'ancienneté |
| `inactif_30j` | `derniere_connexion_jours >= 30` | Isole les comptes dormants, dont le risque de résiliation est nettement plus élevé |
| `taux_fonctionnalites` | `fonctionnalites_utilisees / catalogue.fonctionnalites_incluses` | Normalise par le plan : 3 fonctionnalités sur 8 (Starter) ou sur 40 (Enterprise) ne décrivent pas la même adoption |
| `niveau_anciennete` | tranches d'`anciennete_mois` | Segmentation du cycle de vie, encodée en one-hot |

Les seuils sont des constantes en tête du script (`SEUIL_INACTIVITE_JOURS`,
`NIVEAUX_ANCIENNETE`).

Découpage retenu pour `niveau_anciennete` — les bornes suivent le cycle
contractuel (fin d'onboarding, premier renouvellement annuel) :

| Niveau | Ancienneté |
| --- | --- |
| `RECENT` | 1 à 3 mois |
| `ETABLI` | 4 à 12 mois |
| `ANCIEN` | 13 mois et plus |

### 2. Encodage one-hot

Dix colonnes catégorielles donnent **47 colonnes binaires** (`0` / `1`), selon
la convention `[nom_colonne]_valeur` : `pays_fr`, `plan_str`,
`polarite_csm_alerte`, `niveau_anciennete_recent`…

| Colonne encodée | Origine |
| --- | --- |
| `jour_souscription`, `secteur`, `pays`, `taille_entreprise`, `plan`, `couleur_theme_interface`, `code_datacenter`, `groupe_experimentation` | standardisées en silver |
| `polarite_csm` | dérivée du commentaire CSM en silver |
| `niveau_anciennete` | dérivée en gold |

`polarite_csm` donne `polarite_csm_alerte`, `_neutre`, `_positif` et `_absent` :
l'absence de commentaire est une modalité à part entière, distincte d'un
commentaire neutre.

Deux règles de nommage : minuscules, et tout caractère non alphanumérique
remplacé par `_` (`eu-w1` → `code_datacenter_eu_w1`), pour obtenir des
identifiants SQL utilisables sans guillemets. Les colonnes d'origine sont
conservées à côté de leur encodage.

Les modalités sont déclarées dans `MODALITES_ONE_HOT`
(`src/ml_churn/ingestion/models/gold.py`) et non déduites des données : le
schéma de la table reste ainsi stable quel que soit le contenu du lot chargé, et
toute modalité inattendue déclenche un `WARNING` au lieu de casser l'insertion.

# 5. Entrainement des modèles

## 1. Choix des modèles

Modèles disponibles :
- regression linéaire : regression
- regression logistique : classification
- Random Forest : classification
- XG Boost : classification
- Gradient boosting : classification

### Baseline models

- regression linéaire : regression
- regression logistique : classification

Hypothese confirmation
    - La support impacte pas mal la prediction

### Choix des features

Data leakage :
    - sante_compte_fin_periode

    
## 2. Choic des métriques

Il s'agit d'un dataset déséquilibré (28%)

## 3. API de prédiction

Le service entraîne ses modèles au démarrage, à partir de la couche gold : il
ne relit rien depuis `artifacts/`. Il dépend donc de PostgreSQL pour démarrer.
Le seuil et les hyperparamètres retenus par la recherche sont figés en tête de
`api/registry.py`.

```bash
uv run uvicorn ml_churn.api.main:app --reload
```

Page de test sur <http://127.0.0.1:8000/> et documentation interactive sur
<http://127.0.0.1:8000/docs>.

La page (`ui/`) affiche vingt clients tirés au hasard du jeu de test et
légèrement modifiés, stockés dans `clients.js` avec les 64 colonnes gold dont
les deux modèles ont besoin. Dès que les sondes passent au vert, elle appelle
`/predict-churn` et `/predict-clv` une fois par client et remplit les deux
dernières colonnes du tableau, distinguées par leur couleur. Elle est servie
par l'API elle-même : même origine, donc aucune configuration CORS.

| Endpoint | Rôle | Réponse |
| --- | --- | --- |
| `GET /health` | Le service répond | `{"status": "ok"}` |
| `GET /ready` | Les deux familles de modèles sont entraînées | `{"ready": true, "models": {"churn": ["xgboost"], "clv": ["xgboost"]}}`, ou 503 |
| `POST /predict-churn` | Probabilité de churn d'un client | `{"model", "probability", "threshold", "churn"}` |
| `POST /predict-clv` | Valeur vie client estimée | `{"model", "lifetime_value_eur"}` |

`/health` et `/ready` sont distincts comme les sondes Kubernetes : le service
peut répondre avant d'être capable de prédire, le temps de charger ses modèles.

Les deux endpoints de prédiction partagent le même corps, `model` et `data`. Les clés de `data`
sont les colonnes de la couche gold ; celles que le modèle n'utilise pas sont
ignorées, une ligne gold complète peut donc être envoyée telle quelle.

```bash
curl -X POST http://127.0.0.1:8000/predict-churn \
  -H 'content-type: application/json' \
  -d '{"model": "xgboost", "data": {"anciennete_mois": 12, "nb_integrations": 0, ...}}'
```

```json
{"model": "xgboost", "probability": 0.77, "threshold": 0.4, "churn": true}
```

Même requête sur `/predict-clv` :

```json
{"model": "xgboost", "lifetime_value_eur": 264317.94}
```

Codes d'erreur : **404** si le modèle demandé n'existe pas (avec la liste des
modèles disponibles), **422** s'il manque des colonnes (avec leur liste) ou si
une valeur n'est pas numérique.

Les modèles exposés sont déclarés dans `MODELES`, à la fin de
`api/registry.py` : une entrée par famille (`churn`, `clv`), puis une par
modèle, associant le nom public à la fonction qui l'entraîne. Les
configurations servies — seuil et hyperparamètres retenus par les recherches —
sont figées en tête du même fichier.

# 6 TODO

- bien penser a versionner les modeles avec leurs donnes
- ajouter makefile
