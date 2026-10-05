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
│   ├── drift/              Dérive des données : KS et PSI
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

### Etapes

Observation par rapport aux hypothèses :

- satisfaction client
    - Plus le délai de réponse du support et long plus le taux de churn augmente
    - Plus le nombre de tickets supports au cours des 90j augmente plus le taux de churn augmente
- Bon payeur
    - Plus l'entreprise est grosse plus le churn rate tend à baisser
    - Plus le client a des retards de payement plus le taux de churn augmente

# 4. Ingestion

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

La page (`ui/`) affiche cent clients stockés dans `clients.js`, par pages de
vingt, avec les 64 colonnes gold dont les deux modèles ont besoin. Le tri par
en-tête et les prédictions portent sur l'ensemble, pas sur la page courante. Ils sont produits par
`uv run python -m ml_churn.ui.generer_clients` : des lignes réelles du jeu de
test, perturbées de ±12 % **sur leurs seules colonnes sources**, les colonnes
dérivées étant ensuite recalculées avec les formules de l'ingestion. Bruiter
chaque colonne séparément romprait les relations qui les lient — le taux
d'adoption cesserait d'être le rapport des utilisateurs actifs aux sièges, le
revenu de correspondre au plan. Dès que les sondes passent au vert, elle appelle
`/predict-churn` et `/predict-clv` une fois par client et remplit les deux
dernières colonnes du tableau, distinguées par leur couleur. Elle est servie
par l'API elle-même : même origine, donc aucune configuration CORS.

| Endpoint | Rôle | Réponse |
| --- | --- | --- |
| `GET /health` | Le service répond | `{"status": "ok"}` |
| `GET /ready` | Les deux familles de modèles sont entraînées | `{"ready": true, "models": {"churn": ["xgboost"], "clv": ["xgboost"]}}`, ou 503 |
| `GET /ui-config` | Clé d'API remise à la page de test | `{"api_key": "…"}` |
| `POST /drift` | Dérive entre l'entraînement et les lignes transmises | `{"reference", "current", "columns": [{"column", "ks", "p_value", "psi", "verdict"}]}` |
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

### Protections

Trois middlewares dans `api/security/`, traversés dans cet ordre — du contrôle
le moins cher au plus cher :

| Middleware | Règle | Refus |
| --- | --- | --- |
| `BodySizeLimitMiddleware` | corps de 64 ko au maximum, 1 Mo sur `/drift` | **413** |
| `RateLimitMiddleware` | 400 appels/minute sur `/predict-*`, 60 sur le reste, par IP | **429**, avec `retry-after` |
| `ApiKeyMiddleware` | en-tête `X-API-Key` sur les `POST` | **401** |

La clé est lue dans `FAST_API_KEY` (`.env` à la racine, modèle dans
`.env.example`). Sans clé configurée, le service répond **503** à toute
prédiction plutôt que de s'ouvrir par inadvertance. `/health`, `/ready`, la
documentation et la page de test restent accessibles sans clé : une sonde
d'orchestrateur ne s'authentifie pas.

```bash
curl -X POST http://127.0.0.1:8000/predict-churn \
  -H "x-api-key: $FAST_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model": "xgboost", "data": {…}}'
```

Les deux budgets sont comptés séparément : la page interroge les sondes toutes
les 5 secondes, soit 24 appels par minute, qui ne doivent pas grever ceux dont
les prédictions ont besoin. Chaque réponse porte `x-ratelimit-limit` et
`x-ratelimit-remaining` du budget concerné.

Le quota est compté en mémoire du processus : derrière plusieurs workers, la
limite effective est multipliée par leur nombre. Un compteur partagé (Redis)
serait nécessaire pour une limite globale.

La page de test récupère la clé auprès du service, via `GET /ui-config` : elle
n'a pas accès au `.env`. **Cet endpoint est une commodité de développement** —
qui peut joindre le service peut lire la clé, donc la protection ne vaut plus
que contre les clients qui ignorent son existence. À retirer avant toute
exposition réseau.

Les modèles exposés sont déclarés dans `MODELES`, à la fin de
`api/registry.py` : une entrée par famille (`churn`, `clv`), puis une par
modèle, associant le nom public à la fonction qui l'entraîne. Les
configurations servies — seuil et hyperparamètres retenus par les recherches —
sont figées en tête du même fichier.

## 4. Dérive des données

`drift/drifting.py` compare deux populations, colonne par colonne, à partir de
deux DataFrames ou de deux CSV :

```python
from ml_churn.drift import rapport_derive_csv

for mesure in rapport_derive_csv("train.csv", "test.csv"):
    print(mesure.colonne, mesure.ks, mesure.psi, mesure.verdict)
```

| Mesure | Ce qu'elle dit |
| --- | --- |
| **KS** (Kolmogorov-Smirnov) | Écart maximal entre les deux fonctions de répartition, avec sa p-value. Un test : l'écart est-il explicable par le hasard de l'échantillonnage ? |
| **PSI** (Population Stability Index) | Ampleur du déplacement, tranche par tranche. Ne teste rien, mais se lit avec des seuils conventionnels : < 0.10 stable, < 0.25 dérive modérée, au-delà importante. |

Les tranches du PSI viennent de la **référence** — c'est elle qui définit la
normalité à laquelle le courant est comparé. Les colonnes à peu de modalités
(binaires one-hot, `csat`) sont découpées par valeur plutôt que par quantile.

`POST /drift` prend la population courante dans son corps et la compare à
l'extrait `_train.csv` du dernier modèle enregistré. La page de test lui envoie
ses cent clients et affiche le résultat sous leur tableau ; en production, ce
seraient les clients récemment scorés. Ce corps dépasse la limite des autres
routes — cent lignes pèsent 153 ko — d'où la limite propre de 1 Mo.

**Le PSI demande de la matière.** Le nombre de tranches s'adapte au plus petit
des deux échantillons — cinq observations par tranche au minimum — sinon la
moitié se vident et le logarithme du rapport s'emballe : le PSI annonce alors
une dérive massive là où il ne mesure que la petitesse de l'échantillon. Sur
vingt lignes il reste indicatif ; c'est la p-value du KS qui tranche.

## 5. Intégration continue

`.github/workflows/ci.yml` se déclenche à chaque poussée sur `main`, et
manuellement via *Run workflow*. Un seul job, sans base de données :

| Étape | Commande |
| --- | --- |
| Dépendances | `uv sync --locked` |
| Règles | `uv run ruff check src` |
| Format | `uv run ruff format --check src` |

`uv sync --locked` échoue si `uv.lock` ne correspond plus à `pyproject.toml` :
l'environnement de la CI est alors exactement celui du poste.

# 6 TODO
